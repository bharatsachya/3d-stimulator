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
from vslam.ba import run_bundle_adjustment
from vslam.camera import Camera
from vslam.features import FeatureExtractor, Matcher
from vslam.loop import detect_loops, optimize_pose_graph
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
    # Which map segment this pose belongs to. See the note on re-initialization
    # in run_pipeline: each segment has its own arbitrary scale and origin, so
    # poses are only comparable with others carrying the same segment number.
    segment: int = 0


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
    # Re-initialize a fresh map after this many consecutive unrecoverable
    # frames. See the block comment where this is used.
    enable_reinitialization: bool = True,
    # Eight processed frames -- 0.8 s at 10 fps. Swept across three sequences on
    # the target instance; this is the value the benchmark chose, not a guess:
    #
    #   wait  fr1_xyz %path   fr1_desk cover/%path   fr1_room cover/%path
    #   off        0.30          6.0% / 7.99            7.6% / 5.66
    #     8        0.32        100.0% / 0.77           99.9% / 0.55
    #    12        0.42        100.0% / 0.88           95.9% / 0.95
    #    20        0.31        100.0% / 1.21           99.9% / 0.84
    #    35        0.30         83.8% / 2.06           95.7% / 1.18
    #
    # Waiting longer preserves a single coordinate frame when tracking WILL
    # recover, which is why fr1_xyz mildly prefers it. Waiting less rescues the
    # sequences where it never will. Eight costs fr1_xyz 0.02 percentage points
    # and takes the other two from single-digit coverage to essentially complete.
    frames_before_reinitialization: int = 8,
    enable_loop_closure: bool = False,
    enable_bundle_adjustment: bool = True,
    ba_window: int = 5,
    # Every SECOND keyframe, not every one. Measured end to end on TUM fr1_xyz:
    # every keyframe at nfev=50 costs 106 ms/frame -- over the 100 ms budget --
    # and is WORSE (0.57% of path) than every second keyframe at 70.8 ms/frame
    # (0.45%). More optimisation is not monotonically better when it competes
    # with the frames that feed it.
    ba_every_n_keyframes: int = 2,
    ba_max_nfev: int = 50,
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
    segment = 0
    segment_starts: list[dict] = []
    all_maps: list = []
    relocalizations: list[dict] = []
    ba_runs: list[dict] = []
    keyframes_since_ba = 0
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
                        if segment == 0:
                            # The FIRST initialization failing means the video
                            # cannot be reconstructed at all. Re-run over the
                            # whole buffer purely to raise a properly diagnosed
                            # failure, distinguishing "no texture" from "no
                            # translation".
                            with timer.stage("initialize"):
                                initialize(initialization_buffer, matcher, camera)
                        else:
                            # A LATER segment failing is not fatal: earlier
                            # segments are valid results that must not be thrown
                            # away. Slide the buffer forward and keep trying, so
                            # a stretch of unreconstructable video is skipped
                            # rather than ending the run.
                            initialization_buffer = initialization_buffer[-2:]
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
                        segment=segment,
                    )
                )
                poses.append(
                    TrackedPose(
                        current_keyframe.frame_index,
                        current_keyframe.R,
                        current_keyframe.t,
                        is_keyframe=True,
                        n_inliers=result.n_inliers,
                        segment=segment,
                    )
                )
                if world_map not in all_maps:
                    all_maps.append(world_map)
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

                    # ----------------------------------------------------------
                    # RE-INITIALIZE RATHER THAN GIVE UP.
                    #
                    # Measured across the TUM benchmark, skip-and-retry alone
                    # reaches 99.7% coverage on fr1_xyz but only 6-11% on desk,
                    # desk2, room and fr2_desk. The reason is a death spiral: the
                    # map cannot grow while tracking is lost, because keyframes
                    # are only inserted from tracked frames -- so a camera that
                    # explores AWAY from its initial map can never re-acquire it,
                    # and waiting is futile. fr1_xyz only survives because its
                    # camera oscillates in one small volume and keeps coming back.
                    #
                    # So after a short wait we start a fresh map from the current
                    # frames and carry on. The new segment has its OWN arbitrary
                    # scale and origin -- monocular scale is unobservable, and
                    # nothing links the new segment's units to the old one's. The
                    # segment number is recorded on every pose and evaluation
                    # aligns each segment separately. That is an honest
                    # representation of what a monocular system can know after
                    # losing track, not a stitched trajectory pretending to a
                    # continuity it cannot establish.
                    # ----------------------------------------------------------
                    if (
                        enable_reinitialization
                        and consecutive_lost >= frames_before_reinitialization
                    ):
                        segment += 1
                        segment_starts.append(
                            {"segment": segment, "frame_index": frame.index,
                             "lost_since": lost_since}
                        )
                        initialized = False
                        initialization_buffer = [(frame, features)]
                        world_map = Map()
                        previous_R = previous_t = None
                        older_R = older_t = None
                        consecutive_lost = 0
                        lost_since = None
                        keyframes_since_ba = 0
                        continue

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
                    frame.index, tracked.R, tracked.t,
                    n_inliers=tracked.n_inliers, segment=segment,
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

            keyframes_since_ba += 1
            if (
                enable_bundle_adjustment
                and keyframes_since_ba >= ba_every_n_keyframes
            ):
                with timer.stage("bundle_adjustment"):
                    ba = run_bundle_adjustment(
                        world_map, camera, window=ba_window, max_nfev=ba_max_nfev
                    )
                keyframes_since_ba = 0
                if ba.ran:
                    ba_runs.append(
                        {
                            "keyframe": keyframe.id,
                            "error_before_px": round(ba.error_before_px, 4),
                            "error_after_px": round(ba.error_after_px, 4),
                            "n_points": ba.n_points,
                            "n_residuals": ba.n_residuals,
                            "iterations": ba.iterations,
                            "ms": round(ba.milliseconds, 2),
                        }
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

    # Fail only if NOTHING was reconstructed. With segment re-initialization the
    # final segment can end mid-initialization -- the video simply ran out while
    # a fresh map was being started -- and that must not discard the segments
    # that already succeeded.
    if not poses or camera is None:
        raise SlamFailure(
            FailureReason.INITIALIZATION_FAILED,
            "the video ended before a usable pair of frames was found",
        )
    if world_map is None:
        world_map = all_maps[-1] if all_maps else Map()

    # Loop closure runs once after tracking, over the finished map. See
    # vslam/loop.py for why this is offline rather than online.
    loop_result = None
    if enable_loop_closure and world_map.n_keyframes > 20:
        loop_result = optimize_pose_graph(
            world_map, detect_loops(world_map, camera, matcher)[0]
        )
        if loop_result.optimized:
            # Keyframe poses moved, so the emitted trajectory must follow them.
            for pose in poses:
                for keyframe in world_map.keyframes.values():
                    if keyframe.frame_index == pose.frame_index:
                        pose.R, pose.t = keyframe.R, keyframe.t
                        break

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
        "n_keyframes": sum(m.n_keyframes for m in all_maps) or world_map.n_keyframes,
        "n_map_points": sum(m.n_points for m in all_maps) or world_map.n_points,
        "mean_reprojection_error_px": round(
            world_map.mean_reprojection_error(camera), 3
        ),
        "tracking_lost_at_frame": lost_at_frame,
        "n_segments": segment + 1,
        "segment_starts": segment_starts,
        "n_ba_runs": len(ba_runs),
        "ba_mean_ms": round(float(np.mean([b["ms"] for b in ba_runs])), 2) if ba_runs else 0.0,
        "ba_mean_error_before_px": round(float(np.mean([b["error_before_px"] for b in ba_runs])), 4) if ba_runs else 0.0,
        "ba_mean_error_after_px": round(float(np.mean([b["error_after_px"] for b in ba_runs])), 4) if ba_runs else 0.0,
        "loop_candidates": len(loop_result.candidates) if loop_result else 0,
        "loops_accepted": len(loop_result.accepted) if loop_result else 0,
        "loop_optimized": bool(loop_result.optimized) if loop_result else False,
        "loop_error_before": round(loop_result.error_before, 4) if loop_result else 0.0,
        "loop_error_after": round(loop_result.error_after, 4) if loop_result else 0.0,
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
