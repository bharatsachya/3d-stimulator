"""
The map: keyframes, map points, and triangulation.

WHY A PERSISTENT MAP EXISTS AT ALL

This is the single largest drift reduction available to a monocular system, and
it is the reason this is SLAM rather than visual odometry.

Frame-to-frame odometry estimates each pose relative to the previous frame and
chains them. Every estimate carries error, and chaining multiplies those errors
together, so drift compounds with the number of frames -- fast, and without
bound.

Tracking against a persistent map instead estimates each pose against 3D points
that were triangulated from EARLIER keyframes and have been refined since. The
reference is the map, not the previous frame, so an error in one frame does not
propagate into the next one's reference. Drift still exists -- the map itself
accumulates error -- but it grows far more slowly.

WHAT A MAP POINT KNOWS

Its position, a representative descriptor for matching, and which keyframes have
observed it. The observation count is what lets us distinguish a well-constrained
point seen by six keyframes from a speculative one seen by the minimum two, and
it is what the culling and the viewer both key off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from vslam.camera import Camera
from vslam.features import Features


@dataclass
class Keyframe:
    """A frame whose pose is kept and which contributes structure to the map."""

    id: int
    frame_index: int          # index in the SOURCE video
    R: np.ndarray             # (3,3) world -> camera
    t: np.ndarray             # (3,)   world -> camera
    features: Features
    # feature index -> map point id, for the features that became map points.
    point_ids: dict[int, int] = field(default_factory=dict)

    @property
    def centre(self) -> np.ndarray:
        """
        Camera centre in world coordinates.

        C = -R^T t. Using `t` directly is the classic error: for a world->camera
        transform, t is where the world origin sits in camera coordinates, which
        is not the camera's position.
        """
        return -self.R.T @ self.t

    @property
    def pose_matrix(self) -> np.ndarray:
        """4x4 world -> camera."""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def projection_matrix(self, camera: Camera) -> np.ndarray:
        """P = K [R | t], mapping world points to homogeneous pixels."""
        return camera.matrix @ np.hstack([self.R, self.t.reshape(3, 1)])


@dataclass
class MapPoint:
    """A 3D point, and every keyframe that has seen it."""

    id: int
    position: np.ndarray            # (3,) world coordinates
    descriptor: np.ndarray          # (32,) uint8, for matching against frames
    observations: dict[int, int] = field(default_factory=dict)  # kf id -> feat idx

    @property
    def n_observations(self) -> int:
        return len(self.observations)


class Map:
    """Keyframes and map points, with the bookkeeping that links them."""

    def __init__(self) -> None:
        self.keyframes: dict[int, Keyframe] = {}
        self.points: dict[int, MapPoint] = {}
        self._next_keyframe_id = 0
        self._next_point_id = 0

    # ---------------------------------------------------------------- adding

    def add_keyframe(
        self, frame_index: int, R: np.ndarray, t: np.ndarray, features: Features
    ) -> Keyframe:
        keyframe = Keyframe(
            id=self._next_keyframe_id,
            frame_index=frame_index,
            R=R.copy(),
            t=t.copy(),
            features=features,
        )
        self.keyframes[keyframe.id] = keyframe
        self._next_keyframe_id += 1
        return keyframe

    def add_point(
        self,
        position: np.ndarray,
        descriptor: np.ndarray,
        observations: dict[int, int],
    ) -> MapPoint:
        point = MapPoint(
            id=self._next_point_id,
            position=position.astype(np.float64).copy(),
            descriptor=descriptor.copy(),
            observations=dict(observations),
        )
        self.points[point.id] = point
        self._next_point_id += 1

        # Keep the reverse link in step. Two structures pointing at each other
        # is a bug waiting to happen, so nothing else is allowed to write it.
        for keyframe_id, feature_index in observations.items():
            self.keyframes[keyframe_id].point_ids[feature_index] = point.id
        return point

    def observe(self, point_id: int, keyframe_id: int, feature_index: int) -> None:
        """Record that an existing map point was seen again."""
        self.points[point_id].observations[keyframe_id] = feature_index
        self.keyframes[keyframe_id].point_ids[feature_index] = point_id

    # -------------------------------------------------------------- removing

    def remove_point(self, point_id: int) -> None:
        point = self.points.pop(point_id, None)
        if point is None:
            return
        for keyframe_id, feature_index in point.observations.items():
            keyframe = self.keyframes.get(keyframe_id)
            if keyframe is not None:
                keyframe.point_ids.pop(feature_index, None)

    def cull(self, camera: Camera, max_error_px: float, min_observations: int) -> int:
        """
        Drop points that are badly reprojected or barely observed.

        Both criteria remove the same thing from different directions: a point
        that is probably not real. A large reprojection error means the position
        does not explain the pixels that voted for it; too few observations mean
        too little evidence to tell a real point from a mismatch.

        Returns how many were removed.
        """
        doomed = []
        for point in self.points.values():
            if point.n_observations < min_observations:
                doomed.append(point.id)
                continue
            if self.reprojection_error(point, camera) > max_error_px:
                doomed.append(point.id)

        for point_id in doomed:
            self.remove_point(point_id)
        return len(doomed)

    # ------------------------------------------------------------- measuring

    def reprojection_error(self, point: MapPoint, camera: Camera) -> float:
        """Mean pixel error of a point across the keyframes that observe it."""
        errors = []
        for keyframe_id, feature_index in point.observations.items():
            keyframe = self.keyframes.get(keyframe_id)
            if keyframe is None:
                continue
            camera_point = keyframe.R @ point.position + keyframe.t
            if camera_point[2] <= 1e-6:
                # Behind the camera: not a small error, a wrong point.
                return float("inf")
            u = camera.fx * camera_point[0] / camera_point[2] + camera.cx
            v = camera.fy * camera_point[1] / camera_point[2] + camera.cy
            observed = keyframe.features.points[feature_index]
            errors.append(float(np.hypot(u - observed[0], v - observed[1])))
        return float(np.mean(errors)) if errors else float("inf")

    def mean_reprojection_error(self, camera: Camera) -> float:
        errors = [
            self.reprojection_error(p, camera)
            for p in self.points.values()
        ]
        finite = [e for e in errors if np.isfinite(e)]
        return float(np.mean(finite)) if finite else float("nan")

    def median_depth(self, keyframe: Keyframe) -> float:
        """
        Median depth of visible map points, as seen from one keyframe.

        The keyframe trigger is a fraction of this rather than an absolute
        distance, because there are no units. Ten centimetres of motion is a lot
        in a scene half a metre away and nothing in a scene fifty metres away,
        and a monocular map cannot tell those apart -- so the threshold has to be
        relative to the scene it is in.
        """
        depths = []
        for point_id in keyframe.point_ids.values():
            point = self.points.get(point_id)
            if point is None:
                continue
            depth = float((keyframe.R @ point.position + keyframe.t)[2])
            if depth > 0:
                depths.append(depth)
        return float(np.median(depths)) if depths else 0.0

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_keyframes(self) -> int:
        return len(self.keyframes)

    def local_points(self, window: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """
        Map points observed by the most recent `window` keyframes.

        Matching a frame against EVERY map point would grow quadratically with
        video length -- brute-force matching is O(n*m) and the map only grows.
        Restricting to a local window keeps per-frame cost flat, and points
        outside it are, by construction, ones the camera has moved away from.

        Returns (descriptors, positions, point_ids) ready for the matcher.
        """
        recent = sorted(self.keyframes)[-window:]
        point_ids: list[int] = []
        seen = set()
        for keyframe_id in recent:
            for point_id in self.keyframes[keyframe_id].point_ids.values():
                if point_id not in seen and point_id in self.points:
                    seen.add(point_id)
                    point_ids.append(point_id)

        if not point_ids:
            return (
                np.empty((0, 32), dtype=np.uint8),
                np.empty((0, 3)),
                [],
            )

        descriptors = np.array(
            [self.points[i].descriptor for i in point_ids], dtype=np.uint8
        )
        positions = np.array([self.points[i].position for i in point_ids])
        return descriptors, positions, point_ids


def triangulate(
    keyframe_a: Keyframe,
    keyframe_b: Keyframe,
    points_a: np.ndarray,
    points_b: np.ndarray,
    camera: Camera,
    min_parallax_degrees: float,
    max_reprojection_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Triangulate matched pixels between two keyframes.

    Returns (positions, keep_mask). The mask marks which input correspondences
    produced a point that survived every check, so the caller can keep its own
    indices aligned.

    THREE CHECKS, AND WHY EACH IS NECESSARY

    1. POSITIVE DEPTH IN BOTH VIEWS (cheirality). Triangulation is a linear
       solve that will happily place a point behind a camera. Such a point is
       geometrically consistent and physically impossible.

    2. PARALLAX. This is the important one, and it is measured as the angle
       between the two viewing rays at the point -- not as pixel disparity.
       Pixel disparity conflates rotation with translation: a camera that only
       rotates moves every feature a long way across the image while producing
       no depth information at all. Ray angle is immune to that, because pure
       rotation leaves both rays pointing the same way.

       A small ray angle means the two rays are nearly parallel, so depth is
       poorly conditioned: a tiny pixel error swings the intersection a long
       way down the ray.

    3. REPROJECTION ERROR. Catches correspondences that were simply wrong --
       a mismatch triangulates to a point that projects nowhere near either
       of the pixels that produced it.
    """
    if len(points_a) == 0:
        return np.empty((0, 3)), np.zeros(0, dtype=bool)

    P_a = keyframe_a.projection_matrix(camera)
    P_b = keyframe_b.projection_matrix(camera)

    homogeneous = cv2.triangulatePoints(
        P_a, P_b, points_a.T.astype(np.float64), points_b.T.astype(np.float64)
    )
    # Guard the perspective divide: w == 0 is a point at infinity, which is what
    # a zero-baseline pair produces, and dividing by it yields inf or nan that
    # then quietly poisons every downstream mean.
    w = homogeneous[3]
    safe = np.abs(w) > 1e-12
    positions = np.zeros((homogeneous.shape[1], 3))
    positions[safe] = (homogeneous[:3, safe] / w[safe]).T

    centre_a = keyframe_a.centre
    centre_b = keyframe_b.centre

    depth_a = (keyframe_a.R @ positions.T).T[:, 2] + keyframe_a.t[2]
    depth_b = (keyframe_b.R @ positions.T).T[:, 2] + keyframe_b.t[2]

    ray_a = positions - centre_a
    ray_b = positions - centre_b
    norm_a = np.linalg.norm(ray_a, axis=1)
    norm_b = np.linalg.norm(ray_b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.sum(ray_a * ray_b, axis=1) / (norm_a * norm_b)
    parallax = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    keep = (
        safe
        & np.isfinite(positions).all(axis=1)
        & (depth_a > 0)
        & (depth_b > 0)
        & (parallax >= min_parallax_degrees)
    )

    # Reprojection, checked only on what survived so far.
    if keep.any():
        for keyframe, observed, P in (
            (keyframe_a, points_a, P_a),
            (keyframe_b, points_b, P_b),
        ):
            candidate = positions[keep]
            projected = (P @ np.hstack([candidate, np.ones((len(candidate), 1))]).T).T
            depth = projected[:, 2]
            with np.errstate(invalid="ignore", divide="ignore"):
                pixels = projected[:, :2] / depth[:, None]
            error = np.linalg.norm(pixels - observed[keep], axis=1)
            surviving = np.isfinite(error) & (error <= max_reprojection_px)
            indices = np.flatnonzero(keep)
            keep[indices[~surviving]] = False

    return positions, keep


def median_parallax_degrees(
    centre_a: np.ndarray, centre_b: np.ndarray, positions: np.ndarray
) -> float:
    """Median ray angle at a set of points. See `triangulate` for why this."""
    if len(positions) == 0:
        return 0.0
    ray_a = positions - centre_a
    ray_b = positions - centre_b
    norm_a = np.linalg.norm(ray_a, axis=1)
    norm_b = np.linalg.norm(ray_b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.sum(ray_a * ray_b, axis=1) / (norm_a * norm_b)
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    angles = angles[np.isfinite(angles)]
    return float(np.median(angles)) if len(angles) else 0.0
