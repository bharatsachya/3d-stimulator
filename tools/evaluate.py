"""
Run the pipeline on a clip and score it against ground truth.

This is the command behind every accuracy number in the README. It is a tool
rather than a notebook because the README invites the reader to check the
claims, and a claim you cannot re-run is not evidence.

    python tools/evaluate.py samples/tum_fr1_xyz.mp4 samples/tum_fr1_xyz.truth.json

WHAT "ATE" MEANS HERE, AND WHY IT NEEDS AN ALIGNMENT

Absolute Trajectory Error is the RMS distance between estimated and true camera
positions after the two trajectories have been aligned. The alignment is a
SIMILARITY transform -- rotation, translation and a single scale factor, seven
degrees of freedom -- not a rigid one.

The scale term is not a convenience. A monocular reconstruction is only defined
up to scale, because a small scene viewed closely and a large one viewed from
far away produce identical images. Comparing to metric ground truth without
solving for scale would measure that arbitrary choice rather than any error the
system made. Solving for it is the standard procedure for monocular evaluation
and is what the ORB-SLAM papers report.

COVERAGE IS PART OF THE RESULT

The summary prints how many frames were actually tracked, because an ATE figure
without its coverage is misleading: a system that tracks 8% of a sequence and
then stops will report a far better number than one that tracks all of it, for
the worst possible reason.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vslam.align import (  # noqa: E402
    absolute_trajectory_error,
    centres_from_camera_to_world,
    centres_from_world_to_camera,
)
from vslam.export import to_slam_result  # noqa: E402
from vslam.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ATE against ground truth.")
    parser.add_argument("video")
    parser.add_argument("truth")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--features", type=int, default=1000)
    parser.add_argument("--focal", type=float, default=None, help="override fx in px")
    parser.add_argument("--runs", type=int, default=1, help="repeat, to show variance")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    truth = json.loads(Path(args.truth).read_text())
    truth_centres, kept = centres_from_world_to_camera(truth["poses"])
    column_of_frame = {frame: column for column, frame in enumerate(kept)}

    records = []
    for run in range(args.runs):
        result = run_pipeline(
            args.video,
            processed_fps=args.fps,
            working_width=args.width,
            n_features=args.features,
            focal_px=args.focal,
        )
        exported = to_slam_result(result).to_dict()

        pairs = [
            (position, column_of_frame[frame])
            for position, frame in enumerate(exported["frame_indices"])
            if frame in column_of_frame
        ]
        if len(pairs) < 3:
            raise SystemExit(f"only {len(pairs)} poses could be paired with ground truth")

        estimated = centres_from_camera_to_world(
            [exported["poses"][p] for p, _ in pairs], undo_y_flip=True
        )
        metrics = absolute_trajectory_error(
            estimated, truth_centres[:, [c for _, c in pairs]]
        )

        source_frames = result.stats["video"]["source_frames"]
        last_tracked = result.poses[-1].frame_index

        records.append(
            {
                "run": run,
                **metrics,
                "timing": result.timing,
                "stats": result.stats,
                "source_frames": source_frames,
                "last_tracked_frame": last_tracked,
                "coverage_pct": 100.0 * (last_tracked + 1) / max(source_frames, 1),
            }
        )

        if run == 0:
            print(f"video              {args.video}")
            print(
                f"settings           {args.fps:g} fps, {args.width}px, "
                f"{args.features} features, "
                f"focal {'override ' + str(args.focal) if args.focal else 'heuristic 0.9*width'}"
            )
            print()
            print("--- coverage " + "-" * 50)
            print(f"source frames      {source_frames}")
            print(f"poses produced     {len(result.poses)}")
            print(f"last tracked frame {last_tracked}")
            print(
                f"coverage           {records[0]['coverage_pct']:.1f}% of the sequence"
            )
            lost = result.stats["tracking_lost_at_frame"]
            print(f"tracking lost at   {lost if lost is not None else 'not lost'}")
            print()
            print("--- map " + "-" * 55)
            print(f"keyframes          {result.stats['n_keyframes']}")
            print(f"map points         {result.stats['n_map_points']}")
            print(
                f"mean reprojection  {result.stats['mean_reprojection_error_px']} px"
            )
            print()
            print("--- timing " + "-" * 52)
            timing = result.timing
            print(f"{'stage':<14}{'total ms':>10}{'ms/frame':>11}{'% wall':>9}")
            for stage in timing["stages"]:
                print(
                    f"{stage['stage']:<14}{stage['total_ms']:>10.1f}"
                    f"{stage['ms_per_frame']:>11.2f}{stage['pct_of_wall']:>9.1f}"
                )
            print(
                f"{'unaccounted':<14}{timing['unaccounted_ms']:>10.1f}"
                f"{'':>11}{timing['unaccounted_pct']:>9.1f}"
            )
            print(
                f"{'WALL':<14}{timing['wall_ms']:>10.1f}"
                f"{timing['ms_per_frame']:>11.2f}{100.0:>9.1f}"
            )
            print()
            print("--- accuracy (Sim(3)-aligned) " + "-" * 33)
            print(
                f"paired poses       {metrics['n_poses']}  "
                f"(ground-truth path {metrics['truth_path_length']:.3f} m over this segment)"
            )
            print(f"recovered scale    {metrics['scale_factor']:.4f}")
            print(f"ATE RMSE           {metrics['ate_rmse'] * 100:.2f} cm")
            print(f"ATE median         {metrics['ate_median'] * 100:.2f} cm")
            print(f"ATE max            {metrics['ate_max'] * 100:.2f} cm")
            print(f"RMSE / path length {metrics['rmse_over_path_pct']:.2f}%")

    if args.runs > 1:
        rmses = np.array([r["ate_rmse"] for r in records])
        poses = np.array([r["n_poses"] for r in records])
        print()
        print(f"--- over {args.runs} runs " + "-" * 45)
        print(
            f"ATE RMSE           mean {rmses.mean() * 100:.3f} cm, "
            f"sd {rmses.std() * 100:.3f} cm, "
            f"range {rmses.min() * 100:.3f}-{rmses.max() * 100:.3f}"
        )
        print(f"paired poses       range {poses.min()}-{poses.max()}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(records, indent=2, default=float))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
