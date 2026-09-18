"""
Bundle adjustment tests.

The important one is `test_recovers_a_known_perturbation`. The first working
version of BA passed every structural check -- it ran, it returned, its Jacobian
sparsity was correct -- while recovering 0.1% of a deliberately injected error,
because `x_scale` left the optimiser unable to take a useful step. Only a test
that demands it actually FIX something would have caught that.
"""

from __future__ import annotations

import numpy as np
import pytest

from vslam.ba import run_bundle_adjustment
from vslam.camera import Camera
from vslam.features import Features
from vslam.mapping import Map


def _toy_map(seed: int = 0) -> tuple[Map, Camera]:
    """A small synthetic reconstruction whose true solution is known."""
    rng = np.random.default_rng(seed)
    camera = Camera.guess(640, 480)
    world_map = Map()

    points = rng.uniform(-2, 2, size=(80, 3)) + np.array([0.0, 0.0, 6.0])

    for index in range(4):
        # Cameras translate along x, looking straight ahead.
        R = np.eye(3)
        t = np.array([-0.3 * index, 0.0, 0.0])
        projected = (R @ points.T).T + t
        u = camera.fx * projected[:, 0] / projected[:, 2] + camera.cx
        v = camera.fy * projected[:, 1] / projected[:, 2] + camera.cy
        pixels = np.column_stack([u, v]).astype(np.float32)

        features = Features(
            keypoints=tuple(range(len(points))),
            descriptors=np.zeros((len(points), 32), dtype=np.uint8),
            points=pixels,
        )
        world_map.add_keyframe(index, R, t, features)

    for point_index, position in enumerate(points):
        world_map.add_point(
            position,
            np.zeros(32, dtype=np.uint8),
            {kf: point_index for kf in world_map.keyframes},
        )
    return world_map, camera


def test_perfect_reconstruction_has_near_zero_error() -> None:
    world_map, camera = _toy_map()
    # float32 pixel coordinates limit the achievable precision to ~1e-5 px.
    assert world_map.mean_reprojection_error(camera) < 1e-4


def test_recovers_a_known_perturbation() -> None:
    """
    THE regression test for the x_scale bug.

    With `x_scale=1.0` this recovered 0.1% and the assertion below fails.
    """
    world_map, camera = _toy_map()
    rng = np.random.default_rng(1)
    for point in world_map.points.values():
        point.position = point.position + rng.normal(0, 0.03, 3)

    result = run_bundle_adjustment(world_map, camera, window=4, max_nfev=200)

    assert result.ran, result.reason
    assert result.error_before_px > 1.0, "perturbation should be visible"
    recovered = 1.0 - result.error_after_px / result.error_before_px
    assert recovered > 0.5, (
        f"BA recovered only {recovered:.1%} of a known perturbation; "
        "check x_scale and the termination tolerances"
    )


def test_does_not_write_back_a_worse_solution() -> None:
    world_map, camera = _toy_map()
    before = {p.id: p.position.copy() for p in world_map.points.values()}
    # One evaluation cannot improve anything, so nothing should change.
    run_bundle_adjustment(world_map, camera, window=4, max_nfev=1)
    for point in world_map.points.values():
        assert np.allclose(point.position, before[point.id])


def test_declines_when_there_is_nothing_to_optimise() -> None:
    world_map = Map()
    camera = Camera.guess(640, 480)
    result = run_bundle_adjustment(world_map, camera)
    assert not result.ran
    assert "keyframes" in result.reason
