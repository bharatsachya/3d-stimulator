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
from vslam.features import Features

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

    # Query = frame features, train = map points, so indices come back as
    # (feature index, map point index).
    with stage("match_map"):
        feature_indices, map_indices = matcher.match(
            features.descriptors, map_descriptors
        )
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
