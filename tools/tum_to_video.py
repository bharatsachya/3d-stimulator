"""
Convert a TUM RGB-D sequence into a video plus ground truth in our own format.

WHY CONVERT TO VIDEO RATHER THAN READ THE FOLDER

The deliverable accepts an uploaded video. If the benchmark ran through a
special "dataset mode" that read PNGs off disk, it would be measuring a code
path the reviewer never exercises -- different decode cost, different colour
handling, no container overhead. Encoding to mp4 first means the benchmark and
the demo travel the identical route, and the resulting file is something a
reviewer can drag onto the page themselves.

The re-encode does add compression artifacts. That is not a distortion of the
test: phone video is compressed too, and the alternative (pristine PNGs) would
be the less representative input.

WHY TUM AT ALL

It supplies what neither phone footage nor our synthetic scene can supply at
once: REAL imagery with REAL motion-capture ground truth, 100 Hz, externally
measured. That is what lets "minimizes drift" be a number rather than a claim.

It also supplies true intrinsics, which sets up an experiment worth running --
the pipeline's focal-length heuristic guesses 0.9 * width = 576 for these
640x480 frames, where the measured value is 535.4. An 7.6% error we can put a
cost on, instead of hand-waving about it in the limitations section.

USAGE

  python tools/tum_to_video.py ~/Downloads/rgbd_dataset_freiburg3_... \
      --out samples/tum_nostructure.mp4
  python tools/tum_to_video.py <dir> --seconds 10   # trim to the budget case
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Published TUM intrinsics per camera. The dataset directory name carries which
# one applies, so it is detected rather than asked for -- getting this wrong is
# a silent error that shows up as a subtly wrong trajectory.
#
# fr3's images are supplied already undistorted, which happens to suit a
# pipeline that ignores lens distortion by design.
TUM_INTRINSICS = {
    "freiburg1": {"fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3},
    "freiburg2": {"fx": 520.9, "fy": 521.0, "cx": 325.1, "cy": 249.7},
    "freiburg3": {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6},
}

# TUM's own association tolerance. Colour frames and mocap poses are logged by
# different clocks, so a pose is matched to the nearest frame within 20 ms.
ASSOCIATION_TOLERANCE_S = 0.02


def read_list(path: Path) -> list[tuple[float, str]]:
    """Parse a TUM index file: '# comment' lines, then 'timestamp value'."""
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        timestamp, _, value = line.partition(" ")
        entries.append((float(timestamp), value.strip()))
    return entries


def read_groundtruth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read 'timestamp tx ty tz qx qy qz qw'.

    TUM's convention: this is the CAMERA-TO-WORLD transform. The translation is
    the camera's position in the world frame, not the world's position in the
    camera frame. Confusing the two produces a trajectory that is inverted and
    wrong in a way that still looks plausible, so it is worth being explicit.
    """
    timestamps, poses = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        timestamps.append(float(parts[0]))
        poses.append([float(p) for p in parts[1:8]])
    return np.array(timestamps), np.array(poses)


def quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def detect_intrinsics(dataset_dir: Path) -> dict:
    name = dataset_dir.name
    for key, values in TUM_INTRINSICS.items():
        if key in name:
            return values
    raise SystemExit(
        f"could not tell which Freiburg camera '{name}' came from; "
        f"expected one of {sorted(TUM_INTRINSICS)} in the directory name"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="TUM RGB-D sequence -> mp4 + ground truth")
    parser.add_argument("dataset_dir")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seconds", type=float, default=None, help="trim to this length")
    parser.add_argument("--fps", type=float, default=None, help="override the encoded fps")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser()
    if not (dataset_dir / "rgb.txt").is_file():
        raise SystemExit(f"{dataset_dir} does not look like a TUM sequence (no rgb.txt)")

    intrinsics = detect_intrinsics(dataset_dir)
    frames = read_list(dataset_dir / "rgb.txt")
    gt_timestamps, gt_poses = read_groundtruth(dataset_dir / "groundtruth.txt")

    if args.seconds is not None:
        cutoff = frames[0][0] + args.seconds
        frames = [f for f in frames if f[0] <= cutoff]

    # Derive fps from the timestamps rather than assuming 30: TUM sequences are
    # nominally 30 Hz but actually drift, and an fps that disagrees with the
    # timestamps would silently misalign every later comparison.
    deltas = np.diff([t for t, _ in frames])
    fps = args.fps or float(1.0 / np.median(deltas))

    out_path = Path(args.out or f"samples/tum_{dataset_dir.name.split('_', 2)[-1]}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first = cv2.imread(str(dataset_dir / frames[0][1]))
    if first is None:
        raise SystemExit(f"could not read {frames[0][1]}")
    height, width = first.shape[:2]

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise SystemExit(f"could not open VideoWriter for {out_path}")

    poses_out, frame_timestamps, unmatched = [], [], 0
    for timestamp, relative in frames:
        image = cv2.imread(str(dataset_dir / relative))
        if image is None:
            continue
        writer.write(image)
        frame_timestamps.append(timestamp)

        # Nearest pose in time, rejected if no pose is close enough. A frame
        # with no trustworthy ground truth must be excluded from the comparison
        # rather than matched to whatever was nearest.
        index = int(np.argmin(np.abs(gt_timestamps - timestamp)))
        if abs(gt_timestamps[index] - timestamp) > ASSOCIATION_TOLERANCE_S:
            poses_out.append(None)
            unmatched += 1
            continue

        tx, ty, tz, qx, qy, qz, qw = gt_poses[index]
        R_wc = quaternion_to_matrix(qx, qy, qz, qw)
        t_wc = np.array([tx, ty, tz])

        # Invert camera-to-world into the world-to-camera form the rest of the
        # project uses, matching make_synthetic.py's output.
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc

        T = np.eye(4)
        T[:3, :3] = R_cw
        T[:3, 3] = t_cw
        poses_out.append(T.tolist())

    writer.release()

    K = [
        [intrinsics["fx"], 0.0, intrinsics["cx"]],
        [0.0, intrinsics["fy"], intrinsics["cy"]],
        [0.0, 0.0, 1.0],
    ]
    heuristic_focal = 0.9 * width

    truth = {
        "source": "TUM RGB-D",
        "sequence": dataset_dir.name,
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "K": K,
        "K_is_measured": True,
        "heuristic_focal": heuristic_focal,
        "heuristic_focal_error_pct": round(
            100.0 * (heuristic_focal - intrinsics["fx"]) / intrinsics["fx"], 2
        ),
        "note": "poses are 4x4 world->camera, from 100Hz motion capture. Metric "
        "units (metres) -- but a monocular reconstruction cannot recover them, so "
        "comparison requires a similarity alignment that solves for scale.",
        "frame_timestamps": frame_timestamps,
        "poses": poses_out,
    }
    truth_path = out_path.with_suffix(".truth.json")
    truth_path.write_text(json.dumps(truth))

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"wrote {out_path}  ({len(frame_timestamps)} frames @ {fps:.2f} fps, {size_mb:.1f} MB)")
    print(f"wrote {truth_path}")
    print(
        f"intrinsics       measured fx={intrinsics['fx']}, "
        f"heuristic 0.9*width={heuristic_focal:.0f} "
        f"({truth['heuristic_focal_error_pct']:+.1f}%)"
    )
    if unmatched:
        print(f"note             {unmatched} frames had no pose within {ASSOCIATION_TOLERANCE_S}s")


if __name__ == "__main__":
    main()
