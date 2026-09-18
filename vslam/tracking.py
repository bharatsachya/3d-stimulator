"""
Pose estimation against the map.

WHAT THIS DOES AND WHY IT IS THE IMPORTANT PART

Each frame's pose is solved from 3D-to-2D correspondences: map points whose
world positions are already known, matched to pixels in the current frame, fed
to `solvePnPRansac`.

The alternative -- estimating each pose relative to the previous FRAME and
chaining the results -- is simpler and much worse. Every relative estimate
carries error, and chaining composes those errors, so drift grows with the
number of frames without bound. This is the difference between visual odometry
and SLAM, and CLAUDE.md is explicit that it must not be traded away for
simplicity.

Tracking against the map does not eliminate drift; the map accumulates its own
error. But an error in frame N does not become part of frame N+1's reference,
so it grows far more slowly.

WHY RANSAC IS NOT OPTIONAL HERE

Descriptor matching produces wrong matches even after the ratio test. A single
gross outlier will drag a least-squares pose a long way, and PnP is least
squares. RANSAC fits to a minimal random subset repeatedly and keeps the pose
that the most correspondences agree with, so outliers are excluded rather than
averaged in.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vslam.camera import Camera
from vslam.features import LOWE_RATIO, Features, search_by_projection

# CLAUDE.md's locked values.
REPROJECTION_THRESHOLD_PX = 3.0
MIN_TRACKING_INLIERS = 30
PNP_ITERATIONS = 100
PNP_CONFIDENCE = 0.99


@dataclass
class TrackingResult:
    """The outcome of tracking one frame."""

    success: bool
    R: np.ndarray | None = None      # world -> camera
    t: np.ndarray | None = None
    n_matches: int = 0
    n_inliers: int = 0
    # Map point id -> feature index, for the inliers only. This is what lets a
    # new keyframe inherit observations of points it can already see.
    inlier_associations: dict[int, int] = None
    reason: str = ""

    @property
    def inlier_ratio(self) -> float:
        return self.n_inliers / self.n_matches if self.n_matches else 0.0


def track_frame(
    features: Features,
    map_descriptors: np.ndarray,
    map_positions: np.ndarray,
    map_point_ids: list[int],
    matcher,
    camera: Camera,
    previous_R: np.ndarray | None = None,
    previous_t: np.ndarray | None = None,
    timer=None,
    predicted_R: np.ndarray | None = None,
    predicted_t: np.ndarray | None = None,
    projection_radius_px: float = 30.0,
    projection_ratio: float = LOWE_RATIO,
) -> TrackingResult:
    """
    Estimate this frame's pose from the local map.

    `previous_R`/`previous_t` seed the solver with the last known pose. Frames
    are 100 ms apart, so the camera has barely moved, and starting from the
    previous pose converges faster and more reliably than starting from nothing.

    `timer` separates the descriptor matching from the PnP solve. They are two
    very different costs -- matching is brute-force and quadratic in descriptor
    counts, PnP is a small iterative solve -- and reporting them as one number
    would tell the reader nothing about which to tune.
    """
    from contextlib import nullcontext

    stage = timer.stage if timer is not None else (lambda _name: nullcontext())
    if features.descriptors is None or len(map_descriptors) < 4:
        return TrackingResult(
            success=False, reason="no descriptors or too few map points"
        )

    # PROJECTION-GUIDED FIRST, brute force only as a fallback.
    #
    # See vslam/features.search_by_projection for the measurement that motivated
    # this. In short: searching the whole image makes the ratio test compete
    # against ~1500 descriptors, and it discards correct matches for being
    # ambiguous against descriptors nowhere near the point. Searching a small
    # disc around the projection removes that competition.
    #
    # The fallback matters at exactly two moments -- the frame after
    # initialization, and any frame where the motion model is badly wrong --
    # when there is no trustworthy pose to project with.
    with stage("match_map"):
        feature_indices, map_indices = (np.empty(0, dtype=int),) * 2
        used_projection = False
        if predicted_R is not None and predicted_t is not None:
            feature_indices, map_indices = search_by_projection(
                features,
                map_descriptors,
                map_positions,
                predicted_R,
                predicted_t,
                camera,
                radius_px=projection_radius_px,
                ratio=projection_ratio,
            )
            used_projection = True

        # Widen once before giving up. The radius has to cover the motion
        # model's error, and that error is not constant: measured on fr1_xyz the
        # constant-velocity prediction is off by a median of 8-23 px but
        # occasionally by over 100 px when the camera accelerates. Retrying at
        # double the radius costs nothing on the frames that do not need it, and
        # rescues the ones that do -- without fitting a constant to the clips we
        # happen to have.
        if used_projection and len(feature_indices) < MIN_TRACKING_INLIERS:
            feature_indices, map_indices = search_by_projection(
                features,
                map_descriptors,
                map_positions,
                predicted_R,
                predicted_t,
                camera,
                radius_px=projection_radius_px * 2.0,
                ratio=projection_ratio,
            )

        if len(feature_indices) < MIN_TRACKING_INLIERS:
            feature_indices, map_indices = matcher.match(
                features.descriptors, map_descriptors
            )
            used_projection = False
    n_matches = len(feature_indices)

    # PnP needs at least 4 points; below MIN_TRACKING_INLIERS there is no
    # prospect of a trustworthy pose anyway.
    if n_matches < MIN_TRACKING_INLIERS:
        return TrackingResult(
            success=False,
            n_matches=n_matches,
            reason=f"only {n_matches} map matches",
        )

    object_points = map_positions[map_indices].astype(np.float64)
    image_points = features.points[feature_indices].astype(np.float64)

    use_guess = previous_R is not None and previous_t is not None
    rvec = cv2.Rodrigues(previous_R)[0] if use_guess else None
    tvec = previous_t.reshape(3, 1).copy() if use_guess else None

    with stage("pnp"):
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera.matrix,
            None,                  # no distortion coefficients; see camera.py
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=use_guess,
            iterationsCount=PNP_ITERATIONS,
            reprojectionError=REPROJECTION_THRESHOLD_PX,
            confidence=PNP_CONFIDENCE,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

    if not success or inliers is None:
        return TrackingResult(
            success=False, n_matches=n_matches, reason="PnP did not converge"
        )

    inlier_indices = inliers.ravel()
    if len(inlier_indices) < MIN_TRACKING_INLIERS:
        return TrackingResult(
            success=False,
            n_matches=n_matches,
            n_inliers=len(inlier_indices),
            reason=f"only {len(inlier_indices)} PnP inliers, need {MIN_TRACKING_INLIERS}",
        )

    R, _ = cv2.Rodrigues(rvec)

    associations = {
        map_point_ids[map_indices[i]]: int(feature_indices[i]) for i in inlier_indices
    }

    return TrackingResult(
        success=True,
        R=R,
        t=tvec.ravel(),
        n_matches=n_matches,
        n_inliers=len(inlier_indices),
        inlier_associations=associations,
        reason="projection" if used_projection else "brute_force",
    )


def should_insert_keyframe(
    frames_since_last: int,
    translation_since_last: float,
    median_scene_depth: float,
    tracked_ratio: float,
    min_frames_between: int = 3,
    translation_fraction: float = 0.10,
    tracked_ratio_threshold: float = 0.7,
) -> tuple[bool, str]:
    """
    Decide whether this frame should become a keyframe.

    WHY THE TRANSLATION TRIGGER IS RELATIVE, NOT ABSOLUTE

    A monocular map has no units, so "the camera moved 10 cm" is not a sentence
    this system can form. The threshold is therefore a fraction of the median
    scene depth: moving 10% of the distance to what you are looking at produces
    roughly the same parallax whether the scene is a desk or a street, which is
    the quantity that actually matters for triangulating new structure.

    WHY THE TRACKED-RATIO TRIGGER EXISTS TOO

    Translation is not the only way to run out of map. Turning a corner, or the
    camera simply pointing somewhere new, leaves the existing points behind
    without much translation at all. When the fraction of map points still being
    tracked falls, new structure is needed regardless of how far the camera has
    travelled -- and waiting for translation would lose tracking first.
    """
    if frames_since_last < min_frames_between:
        # Keyframes closer than this add cost without adding baseline.
        return False, ""

    if median_scene_depth > 0:
        if translation_since_last > translation_fraction * median_scene_depth:
            return True, "translation"

    if tracked_ratio < tracked_ratio_threshold:
        return True, "tracked_ratio"

    return False, ""


# ---------------------------------------------------------------------------
# Relocalization
# ---------------------------------------------------------------------------
#
# WHY THIS IS THE RIGHT FIX, AND WHY THE OTHERS WERE NOT
#
# Tracking on TUM fr1_xyz died at frame 186 of 798, capping every accuracy
# figure at 23% coverage. Three fixes were tried first and measured:
#
#   the cull eating the map      -> refuted: zero points culled in the whole run
#   keyframes drying up          -> refuted: map healthy, 1596 points still in view
#   projection-guided matching   -> implemented, measured WORSE (23 poses vs 59)
#   denser keyframes / bigger
#     local map                  -> measured, no material change (62 poses vs 59)
#
# What the data actually showed is that the camera accelerates into sustained
# translation and the scene genuinely changes: new structure enters, old
# structure leaves, and descriptor matching against the existing map degrades
# from ~25% of visible points to 5%. That is not a bug to be prevented. It is
# what happens when a camera explores.
#
# So the fix is to survive it rather than avoid it. Instead of terminating on
# loss, match the current frame against the WHOLE map and try to solve a pose
# from scratch. The trajectory then continues with a recorded gap, which is a
# far more useful answer than stopping.
#
# The inlier requirement is deliberately stricter than for normal tracking.
# Ordinary tracking has a prior -- the previous pose -- and a wrong answer is
# corrected by the next frame. Relocalization has no prior at all, and a wrong
# answer is adopted as truth and corrupts everything after it. The asymmetry in
# cost justifies the asymmetry in evidence required.

MIN_RELOCALIZATION_INLIERS = 50


def relocalize(
    features: Features,
    map_descriptors: np.ndarray,
    map_positions: np.ndarray,
    map_point_ids: list[int],
    matcher,
    camera: Camera,
) -> TrackingResult:
    """
    Recover a pose with no prior, by matching against the whole map.

    Deliberately does NOT take a previous pose: after tracking is lost the
    previous pose is exactly the thing that is no longer trustworthy, and
    seeding PnP with it would bias the solution towards where the camera was
    rather than where it is.
    """
    if features.descriptors is None or len(map_descriptors) < 4:
        return TrackingResult(success=False, reason="no map to relocalize against")

    feature_indices, map_indices = matcher.match(
        features.descriptors, map_descriptors
    )
    n_matches = len(feature_indices)
    if n_matches < MIN_RELOCALIZATION_INLIERS:
        return TrackingResult(
            success=False,
            n_matches=n_matches,
            reason=f"only {n_matches} matches against the map",
        )

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        map_positions[map_indices].astype(np.float64),
        features.points[feature_indices].astype(np.float64),
        camera.matrix,
        None,
        useExtrinsicGuess=False,
        # More iterations than normal tracking: with no prior, RANSAC has a
        # harder search and this runs only on frames that already failed.
        iterationsCount=300,
        reprojectionError=REPROJECTION_THRESHOLD_PX,
        confidence=PNP_CONFIDENCE,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success or inliers is None:
        return TrackingResult(
            success=False, n_matches=n_matches, reason="relocalization PnP failed"
        )

    inlier_indices = inliers.ravel()
    if len(inlier_indices) < MIN_RELOCALIZATION_INLIERS:
        return TrackingResult(
            success=False,
            n_matches=n_matches,
            n_inliers=len(inlier_indices),
            reason=(
                f"only {len(inlier_indices)} relocalization inliers, "
                f"need {MIN_RELOCALIZATION_INLIERS}"
            ),
        )

    R, _ = cv2.Rodrigues(rvec)
    associations = {
        map_point_ids[map_indices[i]]: int(feature_indices[i]) for i in inlier_indices
    }
    return TrackingResult(
        success=True,
        R=R,
        t=tvec.ravel(),
        n_matches=n_matches,
        n_inliers=len(inlier_indices),
        inlier_associations=associations,
        reason="relocalized",
    )
