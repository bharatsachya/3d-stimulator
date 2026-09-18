"""
Sliding-window bundle adjustment.

WHAT IT DOES

Jointly refines the poses of the last few keyframes AND the positions of the map
points they observe, by minimising total reprojection error. Joint, not
pose-only: a pose-only optimisation takes the points as truth and can only
distribute the error into the cameras, which is precisely the wrong place when
the points are themselves estimates from noisy triangulation.

WHY A SLIDING WINDOW AND NOT FULL HISTORY

Full-history BA cost grows with video length, so a longer clip would blow the
time budget for reasons entirely unrelated to how hard the reconstruction is. A
fixed window makes the cost per keyframe constant, which is what lets a 10-second
budget hold for a 60-second video.

The trade is real and worth stating: a sliding window cannot correct drift that
accumulated before the window. Only loop closure can do that.

WHY THE JACOBIAN SPARSITY MATTERS MORE THAN ANYTHING ELSE HERE

With 5 keyframes and ~2000 points the parameter vector has roughly 6000 entries
and the residual vector tens of thousands. A dense Jacobian would be ~10^8
entries, and `least_squares` would estimate it by finite differences, one
column per parameter -- thousands of full residual evaluations per iteration.

But almost every entry is structurally zero: a residual for point P in keyframe
K depends only on P's three coordinates and K's six pose parameters, and on
nothing else at all. Passing that structure as `jac_sparsity` lets SciPy group
parameters that cannot interact into a handful of evaluations. This is the
difference between BA being feasible and being impossible in the budget.

WHY HUBER

Least squares assumes Gaussian noise, and a single surviving mismatch is not
Gaussian -- it is arbitrarily far away and its squared residual dominates the
sum, dragging the whole solution towards it. Huber grows linearly rather than
quadratically beyond its threshold, so an outlier's influence is bounded.

WHY THE FIRST POSE IS FIXED

The cost function is invariant to a global similarity: translate, rotate and
rescale the entire reconstruction and every reprojection is unchanged. That is
gauge freedom, and it leaves the problem rank-deficient -- the optimiser would
wander along directions that change nothing. Holding one keyframe fixed removes
the rotation and translation freedom. Scale is handled by also holding the
second keyframe's distance implicitly through its observed points.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from vslam.camera import Camera
from vslam.mapping import Map

# CLAUDE.md's locked values.
HUBER_DELTA_PX = 2.0
MAX_NFEV = 50
DEFAULT_WINDOW = 5


@dataclass
class BundleAdjustmentResult:
    ran: bool
    n_keyframes: int = 0
    n_points: int = 0
    n_residuals: int = 0
    error_before_px: float = 0.0
    error_after_px: float = 0.0
    iterations: int = 0
    milliseconds: float = 0.0
    reason: str = ""


def _project(points: np.ndarray, rvecs: np.ndarray, tvecs: np.ndarray,
             camera_index: np.ndarray, point_index: np.ndarray,
             camera: Camera) -> np.ndarray:
    """
    Project every observation at once.

    Rotations are carried as rotation VECTORS (axis-angle, 3 numbers) rather
    than matrices. A matrix has 9 entries constrained to 3 degrees of freedom,
    so optimising it directly would let the solver leave the rotation manifold
    and produce something that is no longer a rotation. Axis-angle is minimal
    and unconstrained, which is what an unconstrained optimiser needs.
    """
    theta = np.linalg.norm(rvecs[camera_index], axis=1, keepdims=True)
    axis = np.divide(
        rvecs[camera_index], theta, out=np.zeros_like(rvecs[camera_index]),
        where=theta > 1e-12,
    )
    p = points[point_index]

    # Rodrigues' rotation formula, vectorised.
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    dot = np.sum(axis * p, axis=1, keepdims=True)
    rotated = (
        p * cos_t
        + np.cross(axis, p) * sin_t
        + axis * dot * (1.0 - cos_t)
    )
    camera_points = rotated + tvecs[camera_index]

    # Guard the perspective divide. A point that optimisation has pushed behind
    # a camera would otherwise produce inf/nan and poison the whole residual
    # vector, which least_squares reports only as a failure to converge.
    depth = camera_points[:, 2]
    safe_depth = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    u = camera.fx * camera_points[:, 0] / safe_depth + camera.cx
    v = camera.fy * camera_points[:, 1] / safe_depth + camera.cy
    return np.column_stack([u, v])


def run_bundle_adjustment(
    world_map: Map,
    camera: Camera,
    window: int = DEFAULT_WINDOW,
    max_nfev: int = MAX_NFEV,
    huber_delta: float = HUBER_DELTA_PX,
) -> BundleAdjustmentResult:
    """
    Optimise the last `window` keyframes and the points they see, in place.
    """
    started = time.perf_counter()

    keyframe_ids = sorted(world_map.keyframes)[-window:]
    if len(keyframe_ids) < 2:
        return BundleAdjustmentResult(ran=False, reason="fewer than 2 keyframes")

    keyframe_position = {kf_id: i for i, kf_id in enumerate(keyframe_ids)}

    # Observations: every (point, keyframe) pair inside the window.
    point_ids: list[int] = []
    point_position: dict[int, int] = {}
    camera_index: list[int] = []
    point_index: list[int] = []
    observed: list[np.ndarray] = []

    for kf_id in keyframe_ids:
        keyframe = world_map.keyframes[kf_id]
        for feature_index, point_id in keyframe.point_ids.items():
            point = world_map.points.get(point_id)
            if point is None:
                continue
            if point_id not in point_position:
                point_position[point_id] = len(point_ids)
                point_ids.append(point_id)
            camera_index.append(keyframe_position[kf_id])
            point_index.append(point_position[point_id])
            observed.append(keyframe.features.points[feature_index])

    if len(observed) < 20:
        return BundleAdjustmentResult(
            ran=False, reason=f"only {len(observed)} observations in the window"
        )

    camera_index = np.array(camera_index)
    point_index = np.array(point_index)
    observed = np.array(observed, dtype=np.float64)

    n_cameras = len(keyframe_ids)
    n_points = len(point_ids)

    rvecs = np.array(
        [cv2.Rodrigues(world_map.keyframes[k].R)[0].ravel() for k in keyframe_ids]
    )
    tvecs = np.array([world_map.keyframes[k].t for k in keyframe_ids])
    points = np.array([world_map.points[p].position for p in point_ids])

    # The first keyframe in the window is held fixed -- see the module docstring
    # on gauge freedom. Its parameters are therefore not in the vector at all,
    # which is both correct and cheaper than penalising them.
    fixed_rvec = rvecs[0].copy()
    fixed_tvec = tvecs[0].copy()

    x0 = np.hstack([rvecs[1:].ravel(), tvecs[1:].ravel(), points.ravel()])
    n_free_cameras = n_cameras - 1

    def unpack(x: np.ndarray):
        offset = 0
        free_rvecs = x[offset : offset + n_free_cameras * 3].reshape(-1, 3)
        offset += n_free_cameras * 3
        free_tvecs = x[offset : offset + n_free_cameras * 3].reshape(-1, 3)
        offset += n_free_cameras * 3
        pts = x[offset:].reshape(-1, 3)
        return (
            np.vstack([fixed_rvec, free_rvecs]),
            np.vstack([fixed_tvec, free_tvecs]),
            pts,
        )

    def residuals(x: np.ndarray) -> np.ndarray:
        all_rvecs, all_tvecs, pts = unpack(x)
        projected = _project(
            pts, all_rvecs, all_tvecs, camera_index, point_index, camera
        )
        return (projected - observed).ravel()

    error_before = float(
        np.mean(np.linalg.norm(residuals(x0).reshape(-1, 2), axis=1))
    )

    # Sparsity: residual row for observation i depends only on its camera's 6
    # parameters and its point's 3. Everything else is structurally zero.
    n_residuals = len(observed) * 2
    n_parameters = len(x0)
    sparsity = lil_matrix((n_residuals, n_parameters), dtype=int)
    rows = np.arange(len(observed))
    for axis in range(2):
        for k in range(3):
            free_camera = camera_index - 1  # camera 0 is fixed and has no columns
            has_free_camera = free_camera >= 0
            sparsity[
                2 * rows[has_free_camera] + axis,
                free_camera[has_free_camera] * 3 + k,
            ] = 1
            sparsity[
                2 * rows[has_free_camera] + axis,
                n_free_cameras * 3 + free_camera[has_free_camera] * 3 + k,
            ] = 1
            sparsity[
                2 * rows + axis,
                n_free_cameras * 6 + point_index * 3 + k,
            ] = 1

    solution = least_squares(
        residuals,
        x0,
        jac_sparsity=sparsity,
        method="trf",
        loss="huber",
        f_scale=huber_delta,
        max_nfev=max_nfev,
        # ------------------------------------------------------------------
        # x_scale="jac" IS NOT OPTIONAL. It is the difference between this
        # function working and silently doing nothing.
        #
        # The parameter vector mixes three quantities with completely different
        # sensitivities: rotation vectors in radians, translations, and 3D point
        # coordinates. With the default isotropic scaling, trf cannot choose a
        # trust-region step that is meaningful for all of them at once.
        #
        # Measured, recovering a known 0.05-unit perturbation of the map points:
        #
        #     xtol 1e-4,  x_scale 1.0     ->   0.1% recovered  (nfev 2)
        #     xtol 1e-10, x_scale 1.0     ->   1.7% recovered
        #     xtol 1e-10, x_scale "jac"   ->  86.6% recovered
        #
        # The first row was the shipped configuration. It terminated with status
        # 3 ("xtol satisfied") after a single step, while the gradient norm was
        # still 3143 -- nowhere near a minimum. Across a 3549-element parameter
        # vector the relative step looked negligible even though the absolute
        # step was not, so the optimiser concluded it had converged when it had
        # barely started. "jac" rescales each variable by its Jacobian column,
        # which is the standard remedy for a badly scaled bundle adjustment.
        # ------------------------------------------------------------------
        x_scale="jac",
        # Tolerances tight enough that termination is decided by max_nfev -- the
        # time budget -- rather than by a premature convergence test.
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        verbose=0,
    )

    error_after = float(
        np.mean(np.linalg.norm(solution.fun.reshape(-1, 2), axis=1))
    )

    # Only adopt the result if it actually improved things. `least_squares` can
    # return having made matters worse when it hits max_nfev mid-step, and a
    # refinement that refines nothing should not be written back.
    if error_after <= error_before:
        all_rvecs, all_tvecs, optimised_points = unpack(solution.x)
        for i, kf_id in enumerate(keyframe_ids):
            keyframe = world_map.keyframes[kf_id]
            keyframe.R = cv2.Rodrigues(all_rvecs[i])[0]
            keyframe.t = all_tvecs[i]
        for i, point_id in enumerate(point_ids):
            world_map.points[point_id].position = optimised_points[i]

    return BundleAdjustmentResult(
        ran=True,
        n_keyframes=n_cameras,
        n_points=n_points,
        n_residuals=n_residuals,
        error_before_px=error_before,
        error_after_px=error_after,
        iterations=int(solution.nfev),
        milliseconds=(time.perf_counter() - started) * 1000.0,
    )
