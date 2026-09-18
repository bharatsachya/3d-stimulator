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
from vslam.tracking import relocalize, should_insert_keyframe, track_frame
from vslam.video import iter_frames, probe_video

def _pose_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def predict_pose(
    previous: tuple[np.ndarray, np.ndarray] | None,
    current: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Constant-velocity prediction of where the camera will be next.

    Projection-guided matching needs a pose BEFORE PnP has run, which is
    circular unless the pose is predicted. Frames are 100 ms apart, so assuming
    the motion between the last two frames repeats is a good approximation and
    a standard one.

    The prediction only has to be good enough to land the projection within the
    search radius -- roughly 12 px. It is never used as an answer, only as a
    hint, and PnP corrects whatever it gets wrong.

    Falls back to the current pose (assume stationary) when there is no history,
    which is still far better than no prediction at all.
    """
    if current is None:
        return None
    if previous is None:
        return current

    T_previous = _pose_matrix(*previous)
    T_current = _pose_matrix(*current)
    # Motion between the last two frames, applied again.
    delta = T_current @ np.linalg.inv(T_previous)
    predicted = delta @ T_current
    return predicted[:3, :3], predicted[:3, 3]


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


def _count_visible(world_map: Map, R: np.ndarray, t: np.ndarray, camera: Camera) -> int:
    """
    How many map points project inside the image from this pose.

    Diagnostic only. It answers a question the match and inlier counts cannot:
    whether the camera still has map in front of it at all. A healthy map that
    the camera has walked out of looks identical, from the inlier count alone,
    to a map that has been culled away -- and the two need opposite fixes.
    """
    if not world_map.points:
        return 0
    positions = np.array([p.position for p in world_map.points.values()])
    cam = (R @ positions.T).T + t
    in_front = cam[:, 2] > 1e-6
    if not in_front.any():
        return 0
    projected = cam[in_front]
    u = camera.fx * projected[:, 0] / projected[:, 2] + camera.cx
    v = camera.fy * projected[:, 1] / projected[:, 2] + camera.cy
    inside = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    return int(inside.sum())


def run_pipeline(
    video_path: str,
    processed_fps: float = 10.0,
    working_width: int = 640,
    n_features: int = 1000,
    focal_px: float | None = None,
    max_duration_seconds: float | None = None,
    progress=None,
    diagnostics: list | None = None,
    # Projection-guided matching is OFF by default: measured worse than brute
    # force on TUM fr1_xyz (23 poses against 59). See docs/measurements.md.
    projection_radius_px: float = 0.0,
    projection_ratio: float = 0.9,
    local_map_keyframes: int = LOCAL_MAP_KEYFRAMES,
    min_frames_between_keyframes: int = 3,
    keyframe_translation_fraction: float = 0.10,
    keyframe_tracked_ratio: float = 0.7,
    max_map_point_error_px: float = MAX_MAP_POINT_ERROR_PX,
    cull_window_keyframes: int = 8,
    enable_relocalization: bool = True,
    max_consecutive_lost_frames: int = 60,
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
    # One further step of history, for the constant-velocity motion model.
    older_R = older_t = None
    lost_at_frame: int | None = None
    consecutive_lost = 0
    relocalizations: list[dict] = []
    lost_frames = 0
    lost_since: int | None = None

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
                    local_map_keyframes
                )

            predicted = predict_pose(
                (older_R, older_t) if older_R is not None else None,
                (previous_R, previous_t) if previous_R is not None else None,
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
                predicted_R=predicted[0] if predicted else None,
                predicted_t=predicted[1] if predicted else None,
                projection_radius_px=projection_radius_px,
                projection_ratio=projection_ratio,
            )

            if not tracked.success:
                if diagnostics is not None:
                    # The failing frame is the most informative one in the run,
                    # so record it before breaking out.
                    diagnostics.append(
                        {
                            "frame_index": frame.index,
                            "n_features": len(features),
                            "n_map_points": world_map.n_points,
                            "n_local_points": len(point_ids),
                            "n_visible": _count_visible(
                                world_map, previous_R, previous_t, camera
                            ),
                            "n_matches": tracked.n_matches,
                            "n_inliers": tracked.n_inliers,
                            "inlier_ratio": round(tracked.inlier_ratio, 4),
                            "is_keyframe": False,
                            "n_culled": 0,
                            "n_triangulated": 0,
                            "median_depth": 0.0,
                            "translation_since_kf": 0.0,
                            "n_keyframes": world_map.n_keyframes,
                            "tracking_failed": True,
                            "failure_reason": tracked.reason,
                        }
                    )
                if not enable_relocalization:
                    lost_at_frame = frame.index
                    flags.append(ResultFlag.TRACKING_LOST.value)
                    break

                # Try to recover against the WHOLE map, not the local window:
                # the local window is built from recent keyframes, which are
                # exactly the ones the camera has just failed to match.
                with timer.stage("relocalize"):
                    all_descriptors, all_positions, all_ids = world_map.local_points(
                        world_map.n_keyframes
                    )
                    recovered = relocalize(
                        features, all_descriptors, all_positions, all_ids,
                        matcher, camera,
                    )

                if not recovered.success:
                    consecutive_lost += 1
                    lost_frames += 1
                    if lost_since is None:
                        lost_since = frame.index
                    if consecutive_lost >= max_consecutive_lost_frames:
                        lost_at_frame = frame.index
                        flags.append(ResultFlag.TRACKING_LOST.value)
                        break
                    # Skip this frame and try the next one. No pose is emitted,
                    # so the trajectory has an honest hole rather than a guess.
                    continue

                relocalizations.append(
                    {
                        "frame_index": frame.index,
                        "lost_since": lost_since,
                        "gap_frames": consecutive_lost,
                        "inliers": recovered.n_inliers,
                    }
                )
                consecutive_lost = 0
                lost_since = None
                tracked = recovered
                # The motion model is meaningless across a gap; start it again.
                older_R = older_t = None
                previous_R, previous_t = recovered.R, recovered.t

            poses.append(
                TrackedPose(
                    frame.index, tracked.R, tracked.t, n_inliers=tracked.n_inliers
                )
            )
            older_R, older_t = previous_R, previous_t
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
                min_frames_between=min_frames_between_keyframes,
                translation_fraction=keyframe_translation_fraction,
                tracked_ratio_threshold=keyframe_tracked_ratio,
            )

            record = None
            if diagnostics is not None:
                record = {
                    "frame_index": frame.index,
                    "n_features": len(features),
                    "n_map_points": world_map.n_points,
                    "n_local_points": len(point_ids),
                    "n_visible": _count_visible(
                        world_map, tracked.R, tracked.t, camera
                    ),
                    "n_matches": tracked.n_matches,
                    "n_inliers": tracked.n_inliers,
                    "inlier_ratio": round(tracked.inlier_ratio, 4),
                    "is_keyframe": bool(insert),
                    "n_culled": 0,
                    "n_triangulated": 0,
                    "median_depth": round(median_depth, 4),
                    "translation_since_kf": round(translation, 4),
                    "n_keyframes": world_map.n_keyframes,
                    "tracking_failed": False,
                    "failure_reason": tracked.reason,
                }
                diagnostics.append(record)

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
                n_triangulated = _triangulate_new_points(
                    world_map, last_keyframe, keyframe, matcher, camera
                )

            with timer.stage("cull"):
                n_culled = world_map.cull(
                    camera,
                    max_map_point_error_px,
                    MIN_OBSERVATIONS_TO_KEEP,
                    window=cull_window_keyframes,
                )

            if record is not None:
                record["n_triangulated"] = n_triangulated
                record["n_culled"] = n_culled
                record["n_map_points"] = world_map.n_points
                record["n_keyframes"] = world_map.n_keyframes

            last_keyframe_id = keyframe.id
            frames_since_keyframe = 0

    if not initialized or world_map is None or camera is None:
        raise SlamFailure(
            FailureReason.INITIALIZATION_FAILED,
            "the video ended before a usable pair of frames was found",
        )

    if world_map.n_points < 50:
        flags.append(ResultFlag.FEW_MAP_POINTS.value)
    if relocalizations:
        flags.append(ResultFlag.RELOCALIZED.value)

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
        "n_relocalizations": len(relocalizations),
        "relocalizations": relocalizations,
        "frames_lost": lost_frames,
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
