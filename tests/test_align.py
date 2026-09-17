"""
Alignment tests.

Two properties matter and both are easy to get wrong silently:
  1. an exact similarity must be recovered exactly, and
  2. a MIRRORED trajectory must NOT be alignable, because absorbing a
     reflection would hide a flipped-coordinate bug rather than expose it.
"""

from __future__ import annotations

import numpy as np

from vslam.align import (
    absolute_trajectory_error,
    centres_from_camera_to_world,
    centres_from_world_to_camera,
    umeyama_similarity,
)


def _rotation_z(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_exact_similarity_is_recovered() -> None:
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(3, 60))

    scale_true, rotation_true = 2.5, _rotation_z(0.7)
    translation_true = np.array([[1.0], [-2.0], [3.0]])
    source = rotation_true.T @ (truth - translation_true) / scale_true

    scale, rotation, translation = umeyama_similarity(source, truth)

    assert scale == 0 or abs(scale - scale_true) < 1e-9
    assert np.abs(rotation - rotation_true).max() < 1e-9
    assert np.abs(translation - translation_true).max() < 1e-9


def test_perfect_trajectory_has_zero_ate() -> None:
    rng = np.random.default_rng(1)
    truth = np.cumsum(rng.normal(size=(3, 40)), axis=1)
    # Same path at a different arbitrary scale -- which is exactly what a
    # correct monocular reconstruction produces.
    metrics = absolute_trajectory_error(truth * 7.3, truth)

    assert metrics["ate_rmse"] < 1e-9
    assert abs(metrics["scale_factor"] - 1 / 7.3) < 1e-9


def test_mirrored_trajectory_is_not_aligned_away() -> None:
    rng = np.random.default_rng(2)
    truth = np.cumsum(rng.normal(size=(3, 40)), axis=1)
    mirrored = truth.copy()
    mirrored[1] *= -1.0

    scale, rotation, _ = umeyama_similarity(mirrored, truth)
    # A proper rotation, never a reflection.
    assert np.linalg.det(rotation) > 0
    assert absolute_trajectory_error(mirrored, truth)["ate_rmse"] > 0.1


def test_world_to_camera_centres_are_inverted_not_copied() -> None:
    """C = -R^T t. Using the translation column directly is the classic bug."""
    rotation = _rotation_z(0.4)
    centre = np.array([3.0, -1.0, 2.0])
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = -rotation @ centre  # world->camera translation

    centres, kept = centres_from_world_to_camera([pose.tolist()])
    assert kept == [0]
    assert np.allclose(centres[:, 0], centre)


def test_y_flip_is_undone_for_camera_to_world() -> None:
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    flipped = centres_from_camera_to_world([pose.tolist()], undo_y_flip=True)
    assert np.allclose(flipped[:, 0], [1.0, -2.0, 3.0])


def test_ground_truth_gaps_are_skipped() -> None:
    pose = np.eye(4).tolist()
    centres, kept = centres_from_world_to_camera([pose, None, pose])
    assert kept == [0, 2]
    assert centres.shape == (3, 2)
