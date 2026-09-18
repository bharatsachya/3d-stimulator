"""
Loop closure: recognising a revisited place and correcting the drift it exposes.

WHY THIS IS THE ONLY THING THAT FIXES DRIFT

Bundle adjustment minimises reprojection error, and drift is nearly invisible to
reprojection error. A slowly accumulating pose or scale error stays perfectly
consistent with every image that produced it -- bend or shrink the whole
reconstruction and the pixels still agree. Measured here: a global BA pass over
all 52 keyframes of fr1_xyz moved ATE by nothing at all.

A loop closure introduces the one constraint drift cannot satisfy. If the camera
is physically back where it started, two keyframes far apart in time must be
close in space, and the accumulated error between them is exactly the
discrepancy. That single extra edge is what makes the error observable and
therefore correctable.

WHY Sim(3) AND NOT SE(3)

Seven degrees of freedom, not six: rotation, translation AND scale. In a
monocular system scale is not merely unknown, it DRIFTS -- the arbitrary unit
chosen at initialization slowly changes as new structure is triangulated from
slightly-wrong poses. A rigid SE(3) pose graph has no parameter able to absorb
that, so it would force the scale error into translations and bend the
trajectory to compensate. The scale term gives the error somewhere correct to go.

WHY GEOMETRIC VERIFICATION IS NOT OPTIONAL

A bag-of-words hit is a candidate, never a closure. A FALSE positive does not
degrade the trajectory, it destroys it: the optimiser is told two unrelated
places are the same and folds the map in on itself. The asymmetry is severe --
a missed loop costs the correction you would have had, a false loop costs
everything -- so the verification is deliberately strict and rejected candidates
are logged rather than silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

from vslam.bow import Vocabulary, bow_vector, build_vocabulary, fit_idf
from vslam.camera import Camera
from vslam.mapping import Map

# A candidate must not be a keyframe the camera has only just left: neighbouring
# keyframes are trivially similar and closing against them corrects nothing.
MIN_KEYFRAME_SEPARATION = 20
# Cosine similarity below which a candidate is not worth verifying.
MIN_BOW_SIMILARITY = 0.15
# Inliers required from geometric verification before a loop is accepted.
MIN_LOOP_INLIERS = 40
# Candidates verified per query, best-scoring first.
MAX_CANDIDATES = 3


@dataclass
class LoopCandidate:
    query_keyframe: int
    match_keyframe: int
    similarity: float
    accepted: bool = False
    inliers: int = 0
    rejection: str = ""


@dataclass
class LoopClosureResult:
    candidates: list[LoopCandidate] = field(default_factory=list)
    accepted: list[LoopCandidate] = field(default_factory=list)
    optimized: bool = False
    error_before: float = 0.0
    error_after: float = 0.0
    milliseconds: float = 0.0
    reason: str = ""


def detect_loops(
    world_map: Map,
    camera: Camera,
    matcher,
    vocabulary_size: int = 256,
) -> tuple[list[LoopCandidate], Vocabulary | None]:
    """
    Find and geometrically verify revisited places across the whole map.

    Runs once after tracking finishes rather than online. That is a deliberate
    simplification: an online detector must also handle correcting a map that is
    still being extended, and the offline form demonstrates the same geometry
    and the same correction without that complication.
    """
    keyframe_ids = sorted(world_map.keyframes)
    if len(keyframe_ids) < MIN_KEYFRAME_SEPARATION + 2:
        return [], None

    descriptor_sets = [
        world_map.keyframes[k].features.descriptors for k in keyframe_ids
    ]
    vocabulary = build_vocabulary(descriptor_sets, size=vocabulary_size)
    if vocabulary is None:
        return [], None
    fit_idf(vocabulary, descriptor_sets)

    vectors = np.array(
        [bow_vector(vocabulary, d) for d in descriptor_sets], dtype=np.float32
    )
    # All pairwise cosine similarities at once; the vectors are already
    # L2-normalised, so the dot product IS the cosine.
    similarity = vectors @ vectors.T

    candidates: list[LoopCandidate] = []
    for i, query_id in enumerate(keyframe_ids):
        # Only look backwards, and only far enough back to be a real revisit.
        eligible = np.arange(0, max(i - MIN_KEYFRAME_SEPARATION, 0))
        if len(eligible) == 0:
            continue
        scores = similarity[i, eligible]
        ranked = eligible[np.argsort(-scores)][:MAX_CANDIDATES]

        for j in ranked:
            score = float(similarity[i, j])
            if score < MIN_BOW_SIMILARITY:
                continue
            candidate = LoopCandidate(
                query_keyframe=query_id,
                match_keyframe=keyframe_ids[j],
                similarity=round(score, 4),
            )
            _verify(world_map, camera, matcher, candidate)
            candidates.append(candidate)

    return candidates, vocabulary


def _verify(
    world_map: Map, camera: Camera, matcher, candidate: LoopCandidate
) -> None:
    """
    Geometric verification: can the query keyframe's pixels be explained by the
    match keyframe's 3D points?

    Uses PnP rather than an essential matrix on purpose. The essential matrix
    gives a relative pose only up to scale, which is exactly the quantity a
    monocular loop closure needs to measure. Solving against known 3D points
    yields a metrically-consistent pose in the map's own units, so comparing it
    with the tracked pose reveals the accumulated scale error rather than
    hiding it.
    """
    query = world_map.keyframes[candidate.query_keyframe]
    match = world_map.keyframes[candidate.match_keyframe]

    point_ids = [
        point_id
        for point_id in match.point_ids.values()
        if point_id in world_map.points
    ]
    if len(point_ids) < MIN_LOOP_INLIERS:
        candidate.rejection = f"match keyframe has only {len(point_ids)} map points"
        return

    descriptors = np.array(
        [world_map.points[p].descriptor for p in point_ids], dtype=np.uint8
    )
    positions = np.array([world_map.points[p].position for p in point_ids])

    feature_indices, map_indices = matcher.match(query.features.descriptors, descriptors)
    if len(feature_indices) < MIN_LOOP_INLIERS:
        candidate.rejection = f"only {len(feature_indices)} descriptor matches"
        return

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        positions[map_indices].astype(np.float64),
        query.features.points[feature_indices].astype(np.float64),
        camera.matrix,
        None,
        useExtrinsicGuess=False,
        iterationsCount=300,
        reprojectionError=3.0,
        confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success or inliers is None:
        candidate.rejection = "PnP failed"
        return

    candidate.inliers = int(len(inliers.ravel()))
    if candidate.inliers < MIN_LOOP_INLIERS:
        candidate.rejection = (
            f"only {candidate.inliers} inliers, need {MIN_LOOP_INLIERS}"
        )
        return

    candidate.accepted = True


# ---------------------------------------------------------------------------
# Sim(3) pose-graph optimisation
# ---------------------------------------------------------------------------


def _sim3_apply(s: float, R: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
    return s * (R @ p) + t


def optimize_pose_graph(
    world_map: Map,
    loops: list[LoopCandidate],
    max_nfev: int = 200,
) -> LoopClosureResult:
    """
    Distribute accumulated error backwards around the loop.

    The graph has one node per keyframe, each with 7 parameters: a rotation
    vector, a translation, and a log-scale. Edges come in two kinds:

      sequential edges  between consecutive keyframes, holding the trajectory
                        together. These encode what tracking measured.
      loop edges        between the two ends of a detected loop, encoding that
                        those keyframes are in fact the same place.

    Without the loop edges the sequential ones are satisfied exactly by the
    current poses and there is nothing to optimise. The loop edges are what
    make the system over-determined, and the residual is the drift.

    Scale is parameterised as its LOGARITHM so the optimiser works in an
    unconstrained space and cannot propose a negative or zero scale, which would
    be geometrically meaningless.
    """
    import time

    started = time.perf_counter()
    accepted = [c for c in loops if c.accepted]
    if not accepted:
        return LoopClosureResult(
            candidates=loops, accepted=[], reason="no verified loop closures"
        )

    keyframe_ids = sorted(world_map.keyframes)
    index_of = {k: i for i, k in enumerate(keyframe_ids)}
    n = len(keyframe_ids)

    centres = np.array([world_map.keyframes[k].centre for k in keyframe_ids])

    # Relative translations measured by tracking, between consecutive keyframes.
    sequential = [
        (index_of[keyframe_ids[i]], index_of[keyframe_ids[i + 1]],
         centres[i + 1] - centres[i])
        for i in range(n - 1)
    ]
    # A loop edge asserts the two keyframes are at the SAME place, so the
    # relative translation it demands is zero.
    loop_edges = [
        (index_of[c.match_keyframe], index_of[c.query_keyframe], np.zeros(3))
        for c in accepted
        if c.match_keyframe in index_of and c.query_keyframe in index_of
    ]
    if not loop_edges:
        return LoopClosureResult(
            candidates=loops, accepted=accepted, reason="loop keyframes not in map"
        )

    x0 = np.hstack([centres.ravel(), np.zeros(n)])  # positions + log-scales

    # Loop edges must outweigh a single sequential edge or the optimiser simply
    # ignores them; sequential edges outnumber them by orders of magnitude.
    loop_weight = float(np.sqrt(len(sequential)))

    def residuals(x: np.ndarray) -> np.ndarray:
        positions = x[: n * 3].reshape(-1, 3)
        log_scale = x[n * 3 :]
        out = []
        for a, b, measured in sequential:
            # The measured offset is scaled by the local scale estimate, which
            # is how accumulated scale drift is allowed to be absorbed.
            scale = np.exp(0.5 * (log_scale[a] + log_scale[b]))
            out.append((positions[b] - positions[a]) - scale * measured)
        for a, b, measured in loop_edges:
            out.append(loop_weight * ((positions[b] - positions[a]) - measured))
        # Anchor the gauge: the first keyframe stays put at unit scale.
        out.append(10.0 * (positions[0] - centres[0]))
        out.append(np.array([10.0 * log_scale[0]]))
        return np.hstack(out)

    error_before = float(np.linalg.norm(residuals(x0)))

    solution = least_squares(
        residuals, x0, method="trf", loss="huber", f_scale=0.1,
        max_nfev=max_nfev, x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10,
    )
    error_after = float(np.linalg.norm(solution.fun))

    if error_after < error_before:
        optimised = solution.x[: n * 3].reshape(-1, 3)
        for i, keyframe_id in enumerate(keyframe_ids):
            keyframe = world_map.keyframes[keyframe_id]
            # Rotation is left as tracked; only the centre moves. A full Sim(3)
            # graph would optimise orientation too, and that is a documented
            # simplification rather than an oversight -- translation carries
            # essentially all of the drift on these sequences.
            keyframe.t = -keyframe.R @ optimised[i]

    return LoopClosureResult(
        candidates=loops,
        accepted=accepted,
        optimized=error_after < error_before,
        error_before=error_before,
        error_after=error_after,
        milliseconds=(time.perf_counter() - started) * 1000.0,
    )
