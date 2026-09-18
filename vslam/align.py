"""
Similarity alignment and trajectory error.

Lives in vslam/ rather than tools/ because it is library code: the CLI uses it,
and the measurement sweep will use it too. Like everything in this package it
imports nothing from the web layer.

THE CENTRAL POINT

A monocular reconstruction is correct only up to a similarity transform. The
world origin, its orientation, and its SCALE are all unobservable from a single
lens -- a small scene filmed close and a large one filmed far produce identical
images. Comparing an estimated trajectory to metric ground truth without first
solving for that transform would measure an arbitrary choice, not an error.

So: solve for the best (scale, rotation, translation) between the two paths,
then measure what is left. What remains is the part genuinely got wrong -- the
SHAPE of the path -- which is precisely what drift distorts.
"""

from __future__ import annotations

import numpy as np


def umeyama_similarity(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Least-squares similarity transform mapping `source` onto `target`.

    Both are (3, N). Returns (scale, R, t) minimizing
        || target - (scale * R @ source + t) ||^2

    Umeyama's closed-form solution -- the method Horn's absolute orientation is
    usually implemented with. No iteration and no initial guess to get wrong.
    """
    if source.shape != target.shape:
        raise ValueError(
            f"trajectories must match in shape, got {source.shape} and {target.shape}"
        )
    n = source.shape[1]

    mu_source = source.mean(axis=1, keepdims=True)
    mu_target = target.mean(axis=1, keepdims=True)

    source_centred = source - mu_source
    target_centred = target - mu_target

    covariance = (target_centred @ source_centred.T) / n
    U, singular_values, Vt = np.linalg.svd(covariance)

    # Guard against a reflection. If the SVD's natural solution has determinant
    # -1 it is a mirror, not a rotation. Mirroring a trajectory to make it fit
    # is not a legitimate alignment: it would hide a genuinely mirrored result,
    # which is a real bug class when a coordinate convention gets flipped.
    correction = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        correction[2, 2] = -1.0

    rotation = U @ correction @ Vt

    variance_source = (source_centred ** 2).sum() / n
    scale = float(np.trace(np.diag(singular_values) @ correction) / variance_source)

    translation = mu_target - scale * rotation @ mu_source
    return scale, rotation, translation


def absolute_trajectory_error(estimated: np.ndarray, truth: np.ndarray) -> dict:
    """ATE after Sim(3) alignment. Both arrays are (3, N) camera centres."""
    scale, rotation, translation = umeyama_similarity(estimated, truth)
    aligned = scale * rotation @ estimated + translation

    errors = np.linalg.norm(truth - aligned, axis=0)
    path_length = float(np.linalg.norm(np.diff(truth, axis=1), axis=0).sum())

    return {
        "n_poses": int(errors.size),
        # Not a nuisance parameter: this is how far the arbitrary reconstruction
        # scale sits from metric truth.
        "scale_factor": scale,
        "ate_rmse": float(np.sqrt((errors ** 2).mean())),
        "ate_mean": float(errors.mean()),
        "ate_median": float(np.median(errors)),
        "ate_min": float(errors.min()),
        "ate_max": float(errors.max()),
        # Error quoted without path length is close to meaningless: 5cm over a
        # 1m path is poor, over a 50m path it is excellent.
        "truth_path_length": path_length,
        "rmse_over_path_pct": (
            100.0 * float(np.sqrt((errors ** 2).mean())) / path_length
            if path_length > 1e-9
            else None
        ),
    }


def centres_from_camera_to_world(poses: list, undo_y_flip: bool = False) -> np.ndarray:
    """
    Camera centres from 4x4 CAMERA-TO-WORLD matrices.

    For camera-to-world the translation column IS the camera centre in world
    coordinates, so no inversion is needed.

    `undo_y_flip` reverses the Y-up conversion applied at export. That matters:
    the flip is a REFLECTION (determinant -1), and umeyama_similarity refuses to
    absorb reflections by design. Comparing exported Y-up poses against Y-down
    ground truth without undoing it produces an enormous ATE that looks like
    catastrophic drift and is really a coordinate convention.
    """
    centres = [np.array(pose, dtype=float)[:3, 3] for pose in poses if pose is not None]
    array = np.array(centres).T
    if undo_y_flip:
        array = array.copy()
        array[1, :] *= -1.0
    return array


def centres_from_world_to_camera(poses: list) -> tuple[np.ndarray, list[int]]:
    """
    Camera centres from 4x4 WORLD-TO-CAMERA matrices (the ground-truth form).

    Here the centre must be recovered by inversion: C = -R^T t. Taking the
    translation column directly is the single most common way to get a
    trajectory comparison silently and plausibly wrong.

    Also returns which indices had a pose, since ground truth can have gaps.
    """
    centres, kept = [], []
    for index, pose in enumerate(poses):
        if pose is None:
            continue
        matrix = np.array(pose, dtype=float)
        centres.append(-matrix[:3, :3].T @ matrix[:3, 3])
        kept.append(index)
    return np.array(centres).T, kept


def evaluate_segments(
    exported: dict,
    truth_centres: np.ndarray,
    column_of_frame: dict,
) -> dict:
    """
    ATE over a trajectory that may contain several map segments.

    EACH SEGMENT IS ALIGNED SEPARATELY, and that is not a convenience.

    When tracking cannot be recovered the pipeline starts a fresh map. A new map
    has a new origin, a new orientation and -- because monocular scale is
    unobservable -- a completely new arbitrary unit. Nothing measured by the
    system relates the new segment's scale to the old one's. Aligning the whole
    trajectory with a single similarity transform would therefore charge the
    system for a discontinuity it cannot even observe, and would understate its
    accuracy by an amount that depends only on how unlucky the scale jump was.

    Aligning per segment reports what the system actually knows: the shape of
    each reconstructed stretch. The number of segments is reported alongside,
    because a trajectory in twenty pieces is a worse result than the same error
    in one piece, and the reader must be able to see that.

    The combined figure is the RMS across all poses -- pose-weighted, so a long
    segment counts for more than a short one.
    """
    segments = exported.get("segments") or [0] * len(exported["poses"])
    frame_indices = exported["frame_indices"]

    squared_total = 0.0
    pose_total = 0
    path_total = 0.0
    per_segment = []

    for segment in sorted(set(segments)):
        rows = [i for i, s in enumerate(segments) if s == segment]
        pairs = [
            (i, column_of_frame[frame_indices[i]])
            for i in rows
            if frame_indices[i] in column_of_frame
        ]
        if len(pairs) < 3:
            continue

        estimated = centres_from_camera_to_world(
            [exported["poses"][i] for i, _ in pairs], undo_y_flip=True
        )
        metrics = absolute_trajectory_error(
            estimated, truth_centres[:, [c for _, c in pairs]]
        )
        squared_total += (metrics["ate_rmse"] ** 2) * metrics["n_poses"]
        pose_total += metrics["n_poses"]
        path_total += metrics["truth_path_length"]
        per_segment.append(
            {
                "segment": segment,
                "poses": metrics["n_poses"],
                "path_length": round(metrics["truth_path_length"], 3),
                "ate_rmse": metrics["ate_rmse"],
                "scale_factor": metrics["scale_factor"],
            }
        )

    if pose_total == 0:
        return {"n_poses": 0, "segments_evaluated": 0}

    rmse = float(np.sqrt(squared_total / pose_total))
    return {
        "n_poses": pose_total,
        "segments_evaluated": len(per_segment),
        "ate_rmse": rmse,
        "truth_path_length": path_total,
        "rmse_over_path_pct": 100.0 * rmse / path_total if path_total > 1e-9 else None,
        "per_segment": per_segment,
    }
