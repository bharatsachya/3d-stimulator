"""
Evaluate the pipeline across several TUM RGB-D sequences.

WHY MORE THAN ONE SEQUENCE

One good number on one sequence says very little. A system can be tuned, however
unintentionally, until it works on the clip it was developed against, and a
single-sequence result cannot distinguish that from a system that generalises.
The per-sequence tracking coverage matters as much as the error: a system that
quietly tracks 20% of a hard sequence and reports a small ATE looks better than
one that tracks all of it and reports a larger one.

WHICH SEQUENCES, AND WHICH ARE EXCLUDED

The ORB-SLAM authors note that several TUM sequences are unsuitable for
monocular systems -- those dominated by rotation without translation, those
with no texture, and those with no motion -- because a monocular system cannot
initialise without parallax, which is a property of the sensor rather than of
the implementation (Mur-Artal, Montiel and Tardos, "ORB-SLAM: A Versatile and
Accurate Monocular SLAM System", IEEE T-RO 2015).

This benchmark therefore excludes `fr1_360` (rotation-dominated) and
`fr1_floor` (low texture), and SAYS SO rather than quietly omitting them. The
low-texture case is separately demonstrated with fr3 nostructure_notexture_far,
where the required behaviour is a clear diagnosis rather than a reconstruction.

ALIGNMENT

Sim(3), 7 degrees of freedom, because monocular scale is unobservable. This is
the standard procedure for monocular evaluation and what the ORB-SLAM paper
reports.

    python tools/benchmark.py --dataset-root ~/tum --out out/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vslam.align import (  # noqa: E402
    absolute_trajectory_error,
    centres_from_camera_to_world,
    centres_from_world_to_camera,
)
from vslam.export import to_slam_result  # noqa: E402
from vslam.pipeline import run_pipeline  # noqa: E402

SEQUENCES = [
    "rgbd_dataset_freiburg1_xyz",
    "rgbd_dataset_freiburg1_desk",
    "rgbd_dataset_freiburg1_desk2",
    "rgbd_dataset_freiburg1_room",
    "rgbd_dataset_freiburg2_desk",
]

EXCLUDED = {
    "rgbd_dataset_freiburg1_360": "rotation-dominated; a monocular system cannot "
    "initialise without translation",
    "rgbd_dataset_freiburg1_floor": "low texture; too few repeatable keypoints",
}


def evaluate_one(video: Path, truth_path: Path, **kwargs) -> dict:
    truth = json.loads(truth_path.read_text())
    truth_centres, kept = centres_from_world_to_camera(truth["poses"])
    column_of_frame = {frame: column for column, frame in enumerate(kept)}

    started = time.perf_counter()
    result = run_pipeline(str(video), **kwargs)
    elapsed = time.perf_counter() - started

    exported = to_slam_result(result).to_dict()
    pairs = [
        (position, column_of_frame[frame])
        for position, frame in enumerate(exported["frame_indices"])
        if frame in column_of_frame
    ]
    if len(pairs) < 3:
        return {"error": f"only {len(pairs)} poses paired with ground truth"}

    estimated = centres_from_camera_to_world(
        [exported["poses"][p] for p, _ in pairs], undo_y_flip=True
    )
    metrics = absolute_trajectory_error(
        estimated, truth_centres[:, [c for _, c in pairs]]
    )

    source_frames = result.stats["video"]["source_frames"]
    last_tracked = result.poses[-1].frame_index

    return {
        "source_frames": source_frames,
        "poses": len(result.poses),
        "last_tracked_frame": last_tracked,
        "coverage_pct": round(100.0 * (last_tracked + 1) / max(source_frames, 1), 1),
        "frames_lost": result.stats.get("frames_lost", 0),
        "relocalizations": result.stats.get("n_relocalizations", 0),
        "keyframes": result.stats["n_keyframes"],
        "map_points": result.stats["n_map_points"],
        "reprojection_px": result.stats["mean_reprojection_error_px"],
        "path_length_m": round(metrics["truth_path_length"], 3),
        "ate_rmse_cm": round(metrics["ate_rmse"] * 100, 2),
        "ate_median_cm": round(metrics["ate_median"] * 100, 2),
        "ate_max_cm": round(metrics["ate_max"] * 100, 2),
        "rmse_over_path_pct": round(metrics["rmse_over_path_pct"], 2),
        "scale_factor": round(metrics["scale_factor"], 5),
        "ms_per_frame": round(result.timing["ms_per_frame"], 1),
        "wall_s": round(elapsed, 1),
        "stages": {
            s["stage"]: s["ms_per_frame"] for s in result.timing["stages"]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark across TUM sequences.")
    parser.add_argument("--dataset-root", default="~/tum")
    parser.add_argument("--samples", default="samples")
    parser.add_argument("--out", default="out/benchmark.json")
    parser.add_argument("--no-ba", action="store_true")
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser()
    samples = Path(args.samples)
    samples.mkdir(parents=True, exist_ok=True)

    kwargs = {"enable_bundle_adjustment": not args.no_ba}
    results: dict[str, dict] = {}

    for name in SEQUENCES:
        short = name.replace("rgbd_dataset_", "")
        video = samples / f"tum_{short}.mp4"
        truth = video.with_suffix(".truth.json")

        if not video.is_file():
            source = root / name
            if not source.is_dir():
                print(f"{short:<26} SKIPPED (not downloaded)")
                results[short] = {"skipped": "sequence not present"}
                continue
            # Convert once; the video and its ground truth are then reusable.
            subprocess.run(
                [
                    sys.executable, str(Path(__file__).parent / "tum_to_video.py"),
                    str(source), "--out", str(video),
                ],
                check=True, capture_output=True,
            )

        print(f"{short:<26} running...", flush=True)
        try:
            results[short] = evaluate_one(video, truth, **kwargs)
        except Exception as exc:  # noqa: BLE001 - one bad sequence must not stop the sweep
            results[short] = {"error": f"{type(exc).__name__}: {exc}"}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results, "excluded": EXCLUDED}, indent=2))

    print()
    header = (
        f"{'sequence':<22}{'frames':>8}{'cover%':>8}{'path m':>8}"
        f"{'ATE cm':>8}{'%path':>7}{'kf':>5}{'points':>8}{'ms/fr':>8}"
    )
    print(header)
    print("-" * len(header))
    for short, r in results.items():
        if "error" in r or "skipped" in r:
            print(f"{short:<22}{r.get('error', r.get('skipped'))}")
            continue
        print(
            f"{short:<22}{r['source_frames']:>8}{r['coverage_pct']:>8.1f}"
            f"{r['path_length_m']:>8.2f}{r['ate_rmse_cm']:>8.2f}"
            f"{r['rmse_over_path_pct']:>7.2f}{r['keyframes']:>5}"
            f"{r['map_points']:>8}{r['ms_per_frame']:>8.1f}"
        )
    print()
    print("excluded, per the ORB-SLAM authors' note on monocular-unsuitable sequences:")
    for name, why in EXCLUDED.items():
        print(f"  {name.replace('rgbd_dataset_', ''):<24} {why}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
