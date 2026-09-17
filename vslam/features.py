"""
ORB detection and descriptor matching.

WHY ORB AND NOT SIFT

Binary descriptors compared with a Hamming distance, which is a XOR and a
popcount. SIFT's 128 floats per descriptor and L2 distance are 5-10x slower to
both compute and match, and the 100 ms/frame budget does not have room for that.
ORB is also unencumbered, where SIFT was patented for most of its life.

The cost is robustness: ORB is rotation invariant and, through its pyramid,
scale invariant, but it is not affine invariant and it degrades faster under
motion blur. Those are the documented failure modes.

WHY THE LOWE RATIO TEST AND NOT crossCheck

`BFMatcher(crossCheck=True)` keeps a match only if each descriptor is the
other's best match. That rejects non-mutual matches, which is a weaker and
subtly different condition from what we want.

The ratio test compares the best match to the SECOND best. If a descriptor's two
nearest neighbours are nearly equidistant, then whatever the region looks like,
it looks like that in at least two places -- so the match is AMBIGUOUS and is
discarded no matter how small its absolute distance. Repetitive texture (brick,
carpet, foliage, a row of windows) produces exactly this, and it is where
mismatches come from in practice.

The two are mutually exclusive in OpenCV anyway: crossCheck=True forbids
knnMatch with k=2, and k=2 is what the ratio test needs.

A NOTE ON MATCHING COST, MEASURED

On the target instance matching is 44% of the per-frame floor, co-equal with
ORB, where on an Apple Silicon laptop it was 13%. Brute force is O(n*m) in
descriptor counts, so halving the feature budget quarters the matching work
while only halving the detection work. That asymmetry matters for the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# OpenCV defaults, and CLAUDE.md's locked values. Eight pyramid levels at a 1.2
# scale factor covers roughly a 3.6x size range, which is what lets a feature
# survive the camera approaching the object it sits on.
PYRAMID_LEVELS = 8
PYRAMID_SCALE = 1.2
LOWE_RATIO = 0.75


@dataclass
class Features:
    """Keypoints and descriptors for one frame."""

    keypoints: tuple           # cv2.KeyPoint objects, as returned
    descriptors: np.ndarray | None   # (N, 32) uint8, or None if nothing found
    points: np.ndarray         # (N, 2) float32 pixel coordinates, for geometry

    def __len__(self) -> int:
        return len(self.keypoints)


class FeatureExtractor:
    """
    Wraps a configured ORB detector.

    Kept as an object rather than a function because `cv2.ORB_create` allocates
    its pyramid buffers once; recreating it per frame would pay that on every
    frame of every job.
    """

    def __init__(self, n_features: int = 1000) -> None:
        self.n_features = n_features
        self._orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=PYRAMID_SCALE,
            nlevels=PYRAMID_LEVELS,
        )

    def detect(self, gray: np.ndarray) -> Features:
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        keypoints = tuple(keypoints or ())
        points = (
            np.array([kp.pt for kp in keypoints], dtype=np.float32)
            if keypoints
            else np.empty((0, 2), dtype=np.float32)
        )
        return Features(keypoints=keypoints, descriptors=descriptors, points=points)


class Matcher:
    """Brute-force Hamming matching with the ratio test."""

    def __init__(self, ratio: float = LOWE_RATIO) -> None:
        self.ratio = ratio
        # crossCheck stays False: it is incompatible with knnMatch(k=2), which
        # the ratio test requires. See the module docstring.
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(
        self, query: np.ndarray | None, train: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Match two descriptor sets.

        Returns (query_indices, train_indices) as parallel integer arrays --
        indices rather than cv2.DMatch objects, because every caller
        immediately wants to index into point arrays with them, and NumPy
        indexing is far faster than a Python loop over DMatch attributes.
        """
        empty = (np.empty(0, dtype=int), np.empty(0, dtype=int))
        if query is None or train is None or len(query) < 2 or len(train) < 2:
            return empty

        # k=2 so each query descriptor gets its best and second-best match.
        pairs = self._matcher.knnMatch(query, train, k=2)

        query_indices, train_indices = [], []
        for pair in pairs:
            # A query descriptor can return fewer than 2 neighbours when the
            # train set is tiny. Without a second-best there is nothing to
            # compare against, so the match cannot be shown unambiguous and is
            # dropped rather than trusted.
            if len(pair) < 2:
                continue
            best, second = pair
            if best.distance < self.ratio * second.distance:
                query_indices.append(best.queryIdx)
                train_indices.append(best.trainIdx)

        return (
            np.array(query_indices, dtype=int),
            np.array(train_indices, dtype=int),
        )
