"""
Map initialization: find the first pair of frames that can be reconstructed.

THE PROBLEM

A monocular system cannot start from one frame. Depth comes only from seeing the
same point from two different POSITIONS, so initialization needs a pair of frames
with enough baseline between them -- and the video does not say which pair that
is.

WHY NOT SIMPLY TAKE FRAMES 0 AND 5

Because frame distance is not baseline, and the difference is not academic.
Measured on TUM fr1_xyz, where the camera oscillates along each axis in turn:

    frames 0-30   baseline 0.354 m   translation error   1.75 deg
    frames 0-90   baseline 0.048 m   translation error  35.23 deg

Frame 90 is three times further away in time and has a SEVEN times smaller
baseline, because the camera had wandered back near where it started. Choosing
by frame index would have picked the far worse pair. So candidates are tested,
not assumed.

WHY PARALLAX IS MEASURED AS A RAY ANGLE

Pixel disparity is the tempting proxy and it is wrong, because it conflates
rotation with translation. A camera rotating on the spot sweeps every feature
right across the image while producing exactly zero depth information. Measuring
the angle between the two viewing rays at a triangulated point is immune to
this: under pure rotation both rays point the same way and the angle is zero,
which is the truth.

That is what makes the pure-rotation clip fail here rather than silently
producing a garbage map.

ON FAILURE

Fail loudly, with a typed reason. A monocular system asked to reconstruct a
rotation-only clip has no honest answer, and inventing one is worse than saying
so.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.schema import FailureReason, SlamFailure
from vslam.camera import Camera
from vslam.features import Features
from vslam.mapping import Map, median_parallax_degrees, triangulate

# CLAUDE.md's locked values.
MIN_PARALLAX_DEGREES = 1.0
MIN_ESSENTIAL_INLIERS = 50
RANSAC_THRESHOLD_PX = 1.0
RANSAC_CONFIDENCE = 0.999
# How many processed frames past the reference to consider. At 10 fps this is
# three seconds, which is generous for finding some translation.
SEARCH_WINDOW = 30
MAX_REPROJECTION_PX = 4.0

# ---------------------------------------------------------------------------
# Static-camera detection
# ---------------------------------------------------------------------------
#
# A fixed camera on a highway overpass produced a plausible-looking result
# instead of a diagnosis. The camera never moved; the only motion was traffic.
#
# The parallax gate could not catch this, and the reason is conceptual rather
# than a coding error. The gate measures APPARENT feature motion, and apparent
# motion has two possible causes: a moving camera in a static scene, or a static
# camera with independently moving objects. Nothing in a feature displacement
# distinguishes them, and the cars supplied enough apparent motion to read as
# camera translation.
#
# What DOES distinguish them is the shape of the distribution. A translating
# camera moves everything, by varying amounts. A fixed camera moves nothing
# except the handful of features attached to whatever is passing through.
#
# Measured as "fraction of matched features that moved less than one pixel",
# taken across all 30 candidate initialization pairs, on the target instance:
#
#     clip            min      median
#     cars            49%      77%
#     synth_dolly      0%       0%
#     synth_rotate     0%       0%
#     fr1_xyz          0%       0%
#     fr1_desk         0%       0%
#     fr1_desk2        0%       0%
#     fr1_room         0%       0%
#     fr2_desk         0%       1%
#
# Every valid clip sits at zero. The static-camera clip sits at 49% in its most
# favourable pair. The threshold below is placed in the middle of a 48-point
# gap, which is the opposite situation to the homography degeneracy ratio --
# measured at 0.43-0.45 against 0.29-0.34, a 0.09 margin, and deliberately NOT
# branched on. This margin is wide enough to gate on; that one was not.
STATIC_DISPLACEMENT_PX = 1.0
MAX_STATIC_FRACTION = 0.25


class Rejection:
    """
    Why one candidate pair was rejected.

    These are distinguished because they tell the user to do completely
    different things. "Not enough parallax -- walk sideways past the subject" is
    actively wrong advice for someone filming a blank wall, where the problem is
    that there is nothing to detect at all. Reporting the dominant reason across
    all attempts is what makes the final message useful.
    """

    TOO_FEW_MATCHES = "too_few_matches"
    TOO_FEW_INLIERS = "too_few_inliers"
    TOO_FEW_TRIANGULATED = "too_few_triangulated"
    TOO_LITTLE_PARALLAX = "too_little_parallax"
    STATIC_CAMERA = "static_camera"


@dataclass
class InitializationResult:
    world_map: Map
    reference_frame_index: int
    current_frame_index: int
    parallax_degrees: float
    n_inliers: int
    n_points: int
    attempts: int
    # Reported, never gated on. See homography_dominance for the measurement
    # that settled that, and degenerate_geometry_warning for what is done with it.
    homography_dominance: float = 0.0


def try_initialize(
    reference_features: Features,
    reference_frame_index: int,
    candidate_features: Features,
    candidate_frame_index: int,
    matcher,
    camera: Camera,
) -> tuple[InitializationResult | None, str, float]:
    """
    Attempt to initialize from one specific pair.

    Returns (result, rejection_reason, parallax_seen). A failed pair is an
    ordinary outcome, not an error -- but WHY it failed is recorded, so the
    final message can name the real problem.
    """
    query_indices, train_indices = matcher.match(
        reference_features.descriptors, candidate_features.descriptors
    )
    if len(query_indices) < MIN_ESSENTIAL_INLIERS:
        return None, Rejection.TOO_FEW_MATCHES, 0.0

    points_reference = reference_features.points[query_indices]
    points_candidate = candidate_features.points[train_indices]

    # Reject a pair whose features mostly did not move. See the module-level
    # note: this is checked BEFORE the essential matrix, because on a static
    # camera findEssentialMat will happily fit the moving objects' motion and
    # report it as the camera's.
    displacement = np.linalg.norm(points_candidate - points_reference, axis=1)
    static_fraction = float((displacement < STATIC_DISPLACEMENT_PX).mean())
    if static_fraction > MAX_STATIC_FRACTION:
        return None, Rejection.STATIC_CAMERA, 0.0

    # The essential matrix encodes the epipolar geometry between two calibrated
    # views. RANSAC because the matches still contain outliers even after the
    # ratio test -- repetitive structure defeats it sometimes.
    essential, mask = cv2.findEssentialMat(
        points_reference,
        points_candidate,
        camera.matrix,
        method=cv2.RANSAC,
        prob=RANSAC_CONFIDENCE,
        threshold=RANSAC_THRESHOLD_PX,
    )
    if essential is None or essential.shape != (3, 3) or mask is None:
        # A degenerate configuration can yield a stacked or empty matrix.
        return None, Rejection.TOO_FEW_INLIERS, 0.0
    if int(mask.sum()) < MIN_ESSENTIAL_INLIERS:
        return None, Rejection.TOO_FEW_INLIERS, 0.0

    # recoverPose decomposes E into the four possible (R, t) and picks the one
    # putting points in front of both cameras. t is a UNIT vector: the essential
    # matrix determines direction only, never magnitude. That is monocular scale
    # ambiguity appearing at the earliest possible moment, and it never goes
    # away -- it just gets fixed arbitrarily here and carried forward.
    n_pose_inliers, R, t, pose_mask = cv2.recoverPose(
        essential, points_reference, points_candidate, camera.matrix, mask=mask.copy()
    )
    if n_pose_inliers < MIN_ESSENTIAL_INLIERS:
        return None, Rejection.TOO_FEW_INLIERS, 0.0

    inlier_mask = pose_mask.ravel() > 0
    if int(inlier_mask.sum()) < MIN_ESSENTIAL_INLIERS:
        return None, Rejection.TOO_FEW_INLIERS, 0.0

    world_map = Map()
    # The reference keyframe defines the world frame: identity rotation at the
    # origin. Gauge has to be fixed somewhere, and there is nothing else to
    # anchor to.
    reference_keyframe = world_map.add_keyframe(
        reference_frame_index, np.eye(3), np.zeros(3), reference_features
    )
    candidate_keyframe = world_map.add_keyframe(
        candidate_frame_index, R, t.ravel(), candidate_features
    )

    positions, keep = triangulate(
        reference_keyframe,
        candidate_keyframe,
        points_reference[inlier_mask],
        points_candidate[inlier_mask],
        camera,
        MIN_PARALLAX_DEGREES,
        MAX_REPROJECTION_PX,
    )
    if not keep.any():
        # Nothing survived the depth/parallax/reprojection filter. Measure the
        # parallax on everything triangulated, so we can still report how close
        # this pair came.
        parallax = median_parallax_degrees(
            reference_keyframe.centre, candidate_keyframe.centre, positions
        )
        return None, Rejection.TOO_LITTLE_PARALLAX, parallax

    parallax = median_parallax_degrees(
        reference_keyframe.centre, candidate_keyframe.centre, positions[keep]
    )
    if parallax < MIN_PARALLAX_DEGREES:
        return None, Rejection.TOO_LITTLE_PARALLAX, parallax
    if int(keep.sum()) < MIN_ESSENTIAL_INLIERS:
        return None, Rejection.TOO_FEW_TRIANGULATED, parallax

    # Feature indices for the surviving correspondences, so each new map point
    # knows which feature in each keyframe produced it.
    reference_feature_indices = query_indices[inlier_mask][keep]
    candidate_feature_indices = train_indices[inlier_mask][keep]

    for position, reference_index, candidate_index in zip(
        positions[keep], reference_feature_indices, candidate_feature_indices
    ):
        world_map.add_point(
            position,
            reference_features.descriptors[reference_index],
            {
                reference_keyframe.id: int(reference_index),
                candidate_keyframe.id: int(candidate_index),
            },
        )

    return (
        InitializationResult(
            world_map=world_map,
            reference_frame_index=reference_frame_index,
            current_frame_index=candidate_frame_index,
            parallax_degrees=parallax,
            n_inliers=int(inlier_mask.sum()),
            n_points=world_map.n_points,
            attempts=0,
            homography_dominance=homography_dominance(
                points_reference, points_candidate
            ),
        ),
        "",
        parallax,
    )


def homography_dominance(
    points_a: np.ndarray, points_b: np.ndarray
) -> float:
    """
    How well a homography explains these correspondences, relative to an
    essential matrix. Returns a fraction in [0, 1]; high means degenerate.

    WHY THIS DISTINGUISHES PURE ROTATION

    A homography maps one image to another exactly when the two views are
    related by a rotation about the camera centre, OR when everything in view
    lies on a plane. Both are exactly the cases where the essential matrix
    cannot be determined -- there is no baseline in the first, and the epipolar
    geometry is not unique in the second.

    So if a homography explains the matches at least as well as an essential
    matrix does, the pair is degenerate for reconstruction. That is a very
    different diagnosis from "there were no matches", and it deserves different
    advice: move the camera through the scene rather than turning it, or point
    it at something with depth rather than a flat wall.

    This is the same model-selection idea ORB-SLAM uses to choose between a
    homography and a fundamental matrix at initialization.

    IT IS REPORTED, NEVER BRANCHED ON, AND THE EVIDENCE FOR THAT GOT STRONGER.

    First measured on three clips: rotation 0.43-0.45, translation 0.29-0.34, a
    margin of about 0.09 -- judged too thin to gate on. Re-measured across nine
    clips and 226 candidate pairs on the target instance, the margin is
    NEGATIVE:

        pure rotation    min 0.400   median 0.446   max 0.458
        static camera    min 0.396   median 0.471   max 0.485
        valid clips      min 0.000   median 0.314   max 0.429

    Rotation's minimum (0.400) falls BELOW the valid maximum (0.429): the
    distributions overlap, and no per-pair threshold separates them. More data
    made the case for gating worse, which is the usual direction when a margin
    was thin to begin with.

    There is a second, independent reason never to gate on it. A homography
    explains correspondences under pure rotation OR when the scene is planar,
    and a planar scene filmed by a TRANSLATING camera -- a desk filling the
    frame, a road surface -- is a perfectly valid input. Gating here would
    reject it for being flat.

    So the value is surfaced as a confidence warning on the result instead.
    """
    if len(points_a) < 8:
        return 0.0

    _, homography_mask = cv2.findHomography(
        points_a, points_b, cv2.RANSAC, RANSAC_THRESHOLD_PX
    )
    _, fundamental_mask = cv2.findFundamentalMat(
        points_a, points_b, cv2.FM_RANSAC, RANSAC_THRESHOLD_PX, RANSAC_CONFIDENCE
    )

    homography_inliers = int(homography_mask.sum()) if homography_mask is not None else 0
    fundamental_inliers = (
        int(fundamental_mask.sum()) if fundamental_mask is not None else 0
    )
    total = homography_inliers + fundamental_inliers
    if total == 0:
        return 0.0
    return homography_inliers / total


def degenerate_geometry_warning(dominance: float) -> bool:
    """
    Whether the geometry looks degenerate enough to warn about.

    A WARNING, NOT A GATE. The threshold is the midpoint between the valid
    clips' median (0.314) and the rotation/static clips' median (0.446-0.471).
    Because the per-pair distributions overlap, crossing it means "this
    reconstruction deserves suspicion", never "this input is invalid" -- and the
    result is still produced, still rendered, and labelled.
    """
    return dominance >= 0.40


def initialize(
    frames_and_features: list,
    matcher,
    camera: Camera,
    search_window: int = SEARCH_WINDOW,
) -> InitializationResult:
    """
    Find the first workable pair within the search window.

    `frames_and_features` is a list of (frame, features) already extracted.
    Raises SlamFailure if nothing in the window works.
    """
    if len(frames_and_features) < 2:
        raise SlamFailure(
            FailureReason.TOO_FEW_FRAMES,
            f"only {len(frames_and_features)} frames were decoded",
        )

    reference_frame, reference_features = frames_and_features[0]
    attempts = 0
    best_parallax = 0.0
    rejections: dict[str, int] = {}
    match_counts: list[int] = []
    static_fractions: list[float] = []
    # Keep the widest-baseline candidate's correspondences for the degeneracy
    # test below. The last candidate in the window is the furthest in time and
    # so the most likely to show translation if there is any.
    last_correspondence: tuple[np.ndarray, np.ndarray] | None = None

    for frame, features in frames_and_features[1 : search_window + 1]:
        attempts += 1
        result, reason, parallax = try_initialize(
            reference_features,
            reference_frame.index,
            features,
            frame.index,
            matcher,
            camera,
        )
        if result is not None:
            result.attempts = attempts
            return result
        best_parallax = max(best_parallax, parallax)
        rejections[reason] = rejections.get(reason, 0) + 1

        query_indices, train_indices = matcher.match(
            reference_features.descriptors, features.descriptors
        )
        match_counts.append(len(query_indices))
        if len(query_indices) >= 8:
            displacement = np.linalg.norm(
                features.points[train_indices]
                - reference_features.points[query_indices],
                axis=1,
            )
            static_fractions.append(
                float((displacement < STATIC_DISPLACEMENT_PX).mean())
            )
        if len(query_indices) >= 8:
            last_correspondence = (
                reference_features.points[query_indices],
                features.points[train_indices],
            )

    # Nothing worked. The remedies for the possible causes are completely
    # different, so the diagnosis has to be earned rather than guessed.
    median_matches = float(np.median(match_counts)) if match_counts else 0.0

    # Case 1: there was nothing to match. No amount of camera movement helps.
    if median_matches < MIN_ESSENTIAL_INLIERS:
        raise SlamFailure(
            FailureReason.INITIALIZATION_FAILED,
            f"too few reliable correspondences across {attempts} candidate pairs "
            f"(median {median_matches:.0f}, need {MIN_ESSENTIAL_INLIERS}); the scene "
            f"likely has too little texture, or the footage is motion-blurred",
        )

    # Case 1b: the camera itself never moved. Checked before the parallax case
    # because it is a strict subset of it -- a static camera trivially has no
    # parallax -- but the advice differs completely, and "walk sideways past the
    # subject" is useless to someone whose camera is bolted to a bridge.
    if static_fractions and min(static_fractions) > MAX_STATIC_FRACTION:
        raise SlamFailure(
            FailureReason.STATIC_CAMERA,
            f"the camera does not appear to have moved: across {attempts} candidate "
            f"pairs, at best {min(static_fractions) * 100:.0f}% and typically "
            f"{float(np.median(static_fractions)) * 100:.0f}% of matched features "
            f"stayed within {STATIC_DISPLACEMENT_PX:.0f} px, while a minority moved "
            f"far. That is a fixed camera observing independent motion, not a "
            f"camera travelling through a scene",
        )

    # Case 2: there were plenty of matches, so the scene has texture and the
    # features are being found and matched reliably. What is missing is
    # translation. The camera turned, or shook, but it did not travel -- and a
    # single lens recovers depth only from travel.
    dominance = (
        homography_dominance(*last_correspondence)
        if last_correspondence is not None
        else 0.0
    )
    raise SlamFailure(
        FailureReason.INSUFFICIENT_PARALLAX,
        f"features matched well across {attempts} candidate pairs "
        f"(median {median_matches:.0f} matches), but the camera never translated "
        f"enough to triangulate: best median parallax {best_parallax:.2f} deg "
        f"against the {MIN_PARALLAX_DEGREES:.1f} deg needed. A homography explains "
        f"{dominance * 100:.0f}% of the correspondences, consistent with rotation "
        f"about the camera centre or a flat scene",
    )
