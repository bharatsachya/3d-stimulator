"""
Export: pipeline result -> JSON for the viewer.

TWO CONVERSIONS HAPPEN HERE, AND ONLY HERE

1. WORLD-TO-CAMERA BECOMES CAMERA-TO-WORLD.
   The pipeline works in world->camera throughout, because that is what every
   OpenCV call expects. The viewer wants the opposite: to place a camera in the
   world it needs the camera's position and orientation IN the world, which is
   the inverse. Doing this once, here, keeps a single convention inside the
   pipeline and a single convention in the viewer, with one documented boundary
   between them.

2. Y-DOWN BECOMES Y-UP.
   OpenCV's image coordinates put +y downward, and the camera frame inherits it.
   Three.js is +y up. Without the flip the reconstruction renders upside down.

   Note this flip is a REFLECTION, not a rotation: its determinant is -1. That
   matters when comparing an exported trajectory against ground truth, because
   Sim(3) alignment deliberately refuses to absorb reflections -- so the flip
   must be undone before evaluating, which vslam/align.py does explicitly.

UNITS ARE ARBITRARY

Stated in the output and rendered on the page. A single lens cannot observe
scale: the same images are produced by a small scene nearby and a large one far
away. There is no metre to report, and printing one would be a lie.
"""

from __future__ import annotations

import numpy as np

from app.schema import FLAG_MESSAGES, ResultFlag, SlamResult
from vslam.pipeline import PipelineResult

# Flip Y to convert between OpenCV's y-down and Three.js's y-up. Applied as a
# similarity on both poses and points so the two stay consistent with each other.
Y_FLIP = np.diag([1.0, -1.0, 1.0])


def to_slam_result(result: PipelineResult) -> SlamResult:
    poses: list[list[list[float]]] = []
    frame_indices: list[int] = []
    keyframe_indices: list[int] = []

    for position, tracked in enumerate(result.poses):
        # Invert world->camera to camera->world.
        R_wc = tracked.R.T
        centre = -tracked.R.T @ tracked.t

        # Apply the Y flip in world space, to both the rotation and the centre.
        R_flipped = Y_FLIP @ R_wc @ Y_FLIP
        centre_flipped = Y_FLIP @ centre

        matrix = np.eye(4)
        matrix[:3, :3] = R_flipped
        matrix[:3, 3] = centre_flipped

        poses.append([[float(v) for v in row] for row in matrix])
        frame_indices.append(int(tracked.frame_index))
        if tracked.is_keyframe:
            keyframe_indices.append(position)

    points: list[list[float]] = []
    observations: list[int] = []
    for point in result.world_map.points.values():
        flipped = Y_FLIP @ point.position
        points.append([float(v) for v in flipped])
        observations.append(int(point.n_observations))

    flag_messages = [
        FLAG_MESSAGES[ResultFlag(flag)]
        for flag in result.flags
        if flag in {f.value for f in ResultFlag}
    ]

    return SlamResult(
        poses=poses,
        frame_indices=frame_indices,
        keyframe_indices=keyframe_indices,
        points=points,
        observations=observations,
        flags=list(result.flags),
        flag_messages=flag_messages,
    )
