"""
Orchestration: video in, trajectory and point cloud out.

This is the only module that knows the whole sequence, and it owns the timer, so
every stage is measured the same way and the breakdown that reaches the README
is the breakdown the code actually performed.

THE SEQUENCE

    decode ->  features  ->  initialize (once)
                         ->  track each frame against the map
                         ->  insert keyframes and triangulate new structure
                         ->  cull bad points
                         ->  export

Bundle adjustment slots into the keyframe step at stage 4; the structure here
leaves a place for it rather than needing rearranging.

ON PARTIAL RESULTS

If tracking is lost part way, the trajectory up to that point is returned and
flagged. That is a real answer to a real question -- where the camera went
before the reconstruction broke down -- and discarding it in favour of an error
would be throwing away work the user can use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.schema import FailureReason, ResultFlag, SlamFailure
from vslam.camera import Camera
from vslam.features import FeatureExtractor, Matcher
from vslam.initialize import SEARCH_WINDOW, initialize, try_initialize
from vslam.mapping import Map, triangulate
from vslam.timing import StageTimer
from vslam.tracking import should_insert_keyframe, track_frame
from vslam.video import iter_frames, probe_video

# How many recent keyframes supply the map points a frame is matched against.
# Bounded so per-frame matching cost stays flat as the map grows.
LOCAL_MAP_KEYFRAMES = 8
# Applied when culling after each keyframe.
MAX_MAP_POINT_ERROR_PX = 5.0
MIN_OBSERVATIONS_TO_KEEP = 2
# New structure is only triangulated where the rays are separated enough to
# make depth well conditioned. Same reasoning as at initialization.
TRIANGULATION_MIN_PARALLAX_DEGREES = 1.0
TRIANGULATION_MAX_REPROJECTION_PX = 4.0


@dataclass
class TrackedPose:
    frame_index: int
    R: np.ndarray
    t: np.ndarray
    is_keyframe: bool = False
    n_inliers: int = 0


@dataclass
class PipelineResult:
    poses: list[TrackedPose]
    world_map: Map
    camera: Camera
    timing: dict
    flags: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def run_pipeline(
    video_path: str,
    processed_fps: float = 10.0,
    working_width: int = 640,
    n_features: int = 1000,
    focal_px: float | None = None,
    max_duration_seconds: float | None = None,
    progress=None,
) -> PipelineResult:
    timer = StageTimer()
    extractor = FeatureExtractor(n_features)
    matcher = Matcher()

    info = probe_video(video_path, processed_fps, working_width)
    expected_frames = max(int(info.source_frame_count / max(info.stride, 1)), 1)

    poses: list[TrackedPose] = []
    flags: list[str] = []
    world_map: Map | None = None
    camera: Camera | None = None

    # Enough frames to cover the initialization search window before committing.
    initialization_buffer: list = []
    initialized = False
    last_keyframe_id = 0
    frames_since_keyframe = 0
    previous_R = previous_t = None
    lost_at_frame: int | None = None

    with timer.run():
        for processed_index, frame in enumerate(
            iter_frames(
                video_path,
                processed_fps,
                working_width,
                max_duration_seconds=max_duration_seconds,
                on_decode_time=lambda seconds: timer.add("decode", seconds),
            )
        ):
            if progress is not None and processed_index % 5 == 0:
                progress(
                    "initializing" if not initialized else "tracking",
                    processed_index,
                    expected_frames,
                )

            if camera is None:
                height, width = frame.gray.shape
                camera = Camera.guess(width, height, focal_px=focal_px)

            with timer.stage("orb"):
                features = extractor.detect(frame.gray)

            # ---------------------------------------------------- initialize
            if not initialized:
                initialization_buffer.append((frame, features))
                if len(initialization_buffer) < 2:
                    continue

                # Try the FIRST frame against the newest one, on every new
                # frame, so the earliest pair with enough baseline wins.
                #
                # Testing incrementally rather than buffering the whole window
                # and choosing afterwards matters: initialization should happen
                # as soon as it can, because every frame spent searching is a
                # frame missing from the trajectory.
                reference_frame, reference_features = initialization_buffer[0]
                with timer.stage("initialize"):
                    result, _, _ = try_initialize(
                        reference_features,
                        reference_frame.index,
                        features,
                        frame.index,
                        matcher,
                        camera,
                    )

                if result is None:
                    if len(initialization_buffer) > SEARCH_WINDOW:
                        # The window is exhausted. Re-run over the whole buffer
                        # purely to raise a properly diagnosed failure, which
                        # distinguishes "no texture" from "no translation".
                        with timer.stage("initialize"):
                            initialize(initialization_buffer, matcher, camera)
                    continue

                world_map = result.world_map
                initialized = True
                reference_keyframe = world_map.keyframes[0]
                current_keyframe = world_map.keyframes[1]
                poses.append(
                    TrackedPose(
                        reference_keyframe.frame_index,
                        reference_keyframe.R,
                        reference_keyframe.t,
                        is_keyframe=True,
                    )
                )
                poses.append(
                    TrackedPose(
                        current_keyframe.frame_index,
                        current_keyframe.R,
                        current_keyframe.t,
                        is_keyframe=True,
                        n_inliers=result.n_inliers,
                    )
                )
                previous_R, previous_t = current_keyframe.R, current_keyframe.t
                last_keyframe_id = current_keyframe.id
                frames_since_keyframe = 0
                continue

            # ------------------------------------------------------- track
            with timer.stage("local_map"):
                descriptors, positions, point_ids = world_map.local_points(
                    LOCAL_MAP_KEYFRAMES
                )

            tracked = track_frame(
                features,
                descriptors,
                positions,
                point_ids,
                matcher,
                camera,
                previous_R,
                previous_t,
                timer=timer,
            )

            if not tracked.success:
                # CLAUDE.md: stop, return the partial trajectory, flag it. No
                # relocalization -- that is explicitly out of scope.
                lost_at_frame = frame.index
                flags.append(ResultFlag.TRACKING_LOST.value)
                break

            poses.append(
                TrackedPose(
                    frame.index, tracked.R, tracked.t, n_inliers=tracked.n_inliers
                )
            )
            previous_R, previous_t = tracked.R, tracked.t
            frames_since_keyframe += 1

            # --------------------------------------------------- keyframe?
            last_keyframe = world_map.keyframes[last_keyframe_id]
            camera_centre = -tracked.R.T @ tracked.t
            translation = float(np.linalg.norm(camera_centre - last_keyframe.centre))
            median_depth = world_map.median_depth(last_keyframe)

            insert, why = should_insert_keyframe(
                frames_since_keyframe,
                translation,
                median_depth,
                tracked.inlier_ratio,
            )
            if not insert:
                continue

            with timer.stage("keyframe"):
                keyframe = world_map.add_keyframe(
                    frame.index, tracked.R, tracked.t, features
                )
                poses[-1].is_keyframe = True

                # Points this keyframe already sees keep their identity, which
                # is what links the map across keyframes rather than growing a
                # fresh disconnected cloud each time.
                for point_id, feature_index in tracked.inlier_associations.items():
                    world_map.observe(point_id, keyframe.id, feature_index)

            with timer.stage("triangulate"):
                _triangulate_new_points(
                    world_map, last_keyframe, keyframe, matcher, camera
                )

            with timer.stage("cull"):
                world_map.cull(
                    camera, MAX_MAP_POINT_ERROR_PX, MIN_OBSERVATIONS_TO_KEEP
                )

            last_keyframe_id = keyframe.id
            frames_since_keyframe = 0

    if not initialized or world_map is None or camera is None:
        raise SlamFailure(
            FailureReason.INITIALIZATION_FAILED,
            "the video ended before a usable pair of frames was found",
        )

    if world_map.n_points < 50:
        flags.append(ResultFlag.FEW_MAP_POINTS.value)

    timing = timer.report(frames=len(poses))
    stats = {
        "video": info.to_dict(),
        "camera": camera.to_dict(),
        "focal_source": "user override" if focal_px is not None else "heuristic 0.9*width",
        "n_poses": len(poses),
        "n_keyframes": world_map.n_keyframes,
        "n_map_points": world_map.n_points,
        "mean_reprojection_error_px": round(
            world_map.mean_reprojection_error(camera), 3
        ),
        "tracking_lost_at_frame": lost_at_frame,
    }

    return PipelineResult(
        poses=poses,
        world_map=world_map,
        camera=camera,
        timing=timing,
        flags=flags,
        stats=stats,
    )


def _triangulate_new_points(
    world_map: Map, previous_keyframe, new_keyframe, matcher, camera: Camera
) -> int:
    """
    Create map points from features the two keyframes share that are not yet
    mapped. This is how the map grows as the camera explores.
    """
    query_indices, train_indices = matcher.match(
        previous_keyframe.features.descriptors, new_keyframe.features.descriptors
    )
    if len(query_indices) == 0:
        return 0

    # Only correspondences where NEITHER feature already belongs to a map point.
    # Re-triangulating an existing point would create a duplicate that competes
    # with the original in matching and splits its observation count.
    fresh = [
        i
        for i in range(len(query_indices))
        if int(query_indices[i]) not in previous_keyframe.point_ids
        and int(train_indices[i]) not in new_keyframe.point_ids
    ]
    if not fresh:
        return 0

    query_indices = query_indices[fresh]
    train_indices = train_indices[fresh]

    positions, keep = triangulate(
        previous_keyframe,
        new_keyframe,
        previous_keyframe.features.points[query_indices],
        new_keyframe.features.points[train_indices],
        camera,
        TRIANGULATION_MIN_PARALLAX_DEGREES,
        TRIANGULATION_MAX_REPROJECTION_PX,
    )
    if not keep.any():
        return 0

    added = 0
    for position, query_index, train_index in zip(
        positions[keep], query_indices[keep], train_indices[keep]
    ):
        world_map.add_point(
            position,
            previous_keyframe.features.descriptors[query_index],
            {
                previous_keyframe.id: int(query_index),
                new_keyframe.id: int(train_index),
            },
        )
        added += 1
    return added
