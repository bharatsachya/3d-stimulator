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


# ---------------------------------------------------------------------------
# Projection-guided matching
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS, AND WHAT IT REPLACED
#
# Tracking originally matched every frame descriptor against every map
# descriptor in the local window -- around 1500 of them -- and applied the ratio
# test to the result. That is how tracking died at frame 183 of TUM fr1_xyz.
#
# The failure is not about the map. Measured at the exact frame where tracking
# was lost, with the map healthy and 1116 points projecting inside the image:
#
#     global brute force over 1464 map descriptors ->  90 matches
#     projection-guided search, 8px radius         -> 203 matches   (2.3x)
#     projection-guided search, 15px radius        -> 193 matches
#     projection-guided search, 40px radius        -> 180 matches
#     projection-guided search, 80px radius        -> 162 matches
#
# Match count FALLS as the search radius grows, which is the signature of the
# real problem: ambiguity. Lowe's ratio test discards a match when the second
# best candidate is nearly as close. Searching the whole image means competing
# against ~1500 descriptors, and somewhere among them there is almost always a
# near-tie -- so correct matches are thrown away for being ambiguous against
# descriptors that are nowhere near the point in question.
#
# Restricting the search to keypoints within a few pixels of where the map point
# actually projects removes that competition. It is both more accurate AND
# cheaper, because each map point is compared against roughly 11 candidates
# rather than the whole frame.
#
# This is how ORB-SLAM's tracking works, and the measurement above is why.

# Popcount lookup for Hamming distance on 32-byte ORB descriptors. XOR then sum
# the set bits. A 256-entry table beats np.unpackbits comfortably and keeps the
# inner loop free of allocation.
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)

# Candidate keypoints examined per projected map point. See search_by_projection.
NEIGHBOURS_PER_PROJECTION = 8


def hamming_distances(descriptor: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Hamming distance from one descriptor to each of several candidates."""
    return _POPCOUNT[np.bitwise_xor(descriptor[None, :], candidates)].sum(axis=1)


def search_by_projection(
    features: Features,
    map_descriptors: np.ndarray,
    map_positions: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    camera,
    radius_px: float = 12.0,
    ratio: float = LOWE_RATIO,
    max_distance: int = 70,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match map points to frame keypoints using a predicted pose.

    Each map point is projected into the image with (R, t) and compared only
    against keypoints within `radius_px` of where it lands.

    Returns (feature_indices, map_indices), the same parallel-array shape the
    brute-force matcher returns, so callers are unaffected by which was used.

    `radius_px` must cover the error in the predicted pose. Too small and a
    correct match falls outside the circle; too large and the ambiguity this
    exists to avoid creeps back in. 12 px is a little above the 8 px that
    measured best, for tolerance when the motion model is wrong.
    """
    from scipy.spatial import cKDTree

    empty = (np.empty(0, dtype=int), np.empty(0, dtype=int))
    if features.descriptors is None or len(map_descriptors) == 0:
        return empty
    if len(features.points) == 0:
        return empty

    camera_points = (R @ map_positions.T).T + t
    in_front = camera_points[:, 2] > 1e-6
    if not in_front.any():
        return empty

    u = camera.fx * camera_points[:, 0] / camera_points[:, 2] + camera.cx
    v = camera.fy * camera_points[:, 1] / camera_points[:, 2] + camera.cy
    inside = in_front & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    candidate_map_indices = np.flatnonzero(inside)
    if len(candidate_map_indices) == 0:
        return empty

    # One tree over the frame's keypoints, queried once for all projections.
    #
    # `query` with a fixed k and an upper bound, NOT `query_ball_point`.
    # query_ball_point returns a ragged list of lists, and flattening that costs
    # a Python-level pass over every candidate -- which measured 4x slower than
    # the OpenCV brute-force matcher it was supposed to beat, despite comparing
    # far fewer descriptors. `query` returns rectangular arrays, so everything
    # downstream is pure NumPy.
    #
    # k = 8 is comfortably above the ~11 candidates a 12px radius was measured to
    # contain on average, and the ratio test only ever looks at the best two.
    tree = cKDTree(features.points)
    projected = np.column_stack([u[candidate_map_indices], v[candidate_map_indices]])
    neighbour_distance, neighbour_index = tree.query(
        projected, k=NEIGHBOURS_PER_PROJECTION, distance_upper_bound=radius_px
    )
    if neighbour_index.ndim == 1:
        neighbour_distance = neighbour_distance[:, None]
        neighbour_index = neighbour_index[:, None]

    # Missing neighbours come back as index == len(points) and distance == inf.
    valid = np.isfinite(neighbour_distance)
    if not valid.any():
        return empty
    # Clamp so the out-of-range sentinel can be used as an index safely; those
    # entries are masked out by `valid` before anything depends on them.
    neighbour_index = np.where(valid, neighbour_index, 0)

    rows, columns = np.nonzero(valid)
    pair_map = candidate_map_indices[rows]
    pair_feature = neighbour_index[rows, columns]

    # All Hamming distances in one shot.
    xor = np.bitwise_xor(map_descriptors[pair_map], features.descriptors[pair_feature])
    distances = _POPCOUNT[xor].sum(axis=1)

    # Sort by (map point, distance) so each map point's best and second best sit
    # adjacent, which makes the ratio test a comparison between neighbours.
    order = np.lexsort((distances, pair_map))
    pair_map = pair_map[order]
    pair_feature = pair_feature[order]
    distances = distances[order]

    first_of_group = np.empty(len(pair_map), dtype=bool)
    first_of_group[0] = True
    first_of_group[1:] = pair_map[1:] != pair_map[:-1]
    best_positions = np.flatnonzero(first_of_group)

    best_distance = distances[best_positions]
    group_sizes = np.diff(np.append(best_positions, len(pair_map)))
    has_second = group_sizes > 1
    second_distance = np.full(len(best_positions), np.inf)
    second_distance[has_second] = distances[best_positions[has_second] + 1]

    accept = (best_distance <= max_distance) & (
        (~has_second) | (best_distance < ratio * second_distance)
    )
    accepted = best_positions[accept]
    if len(accepted) == 0:
        return empty

    chosen_feature = pair_feature[accepted]
    chosen_map = pair_map[accepted]
    chosen_distance = distances[accepted]

    # A frame keypoint must not be claimed by two map points: a duplicate
    # association is a guaranteed outlier for PnP. Keep the closest claim.
    keep_order = np.lexsort((chosen_distance, chosen_feature))
    chosen_feature = chosen_feature[keep_order]
    chosen_map = chosen_map[keep_order]
    unique = np.empty(len(chosen_feature), dtype=bool)
    unique[0] = True
    unique[1:] = chosen_feature[1:] != chosen_feature[:-1]

    return chosen_feature[unique], chosen_map[unique]
