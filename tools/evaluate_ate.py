"""
Absolute Trajectory Error against a ground-truth path.

WHY ALIGNMENT IS REQUIRED, AND WHY IT MUST SOLVE FOR SCALE

A monocular reconstruction is correct only up to a similarity transform. The
world origin, its orientation, and crucially its SCALE are all unobservable from
one lens -- a small scene filmed close up and a large one filmed far away
produce identical images. So comparing our trajectory to metric ground truth
directly would measure nothing but that arbitrary choice.

The standard remedy, and the one TUM's own evaluation script uses, is to solve
for the best similarity transform (rotation, translation, scale) between the two
trajectories first, then measure what is left over. What remains is the part the
reconstruction genuinely got wrong: the SHAPE of the path, which is exactly what
drift distorts.

The recovered scale factor is reported too, and it is not a nuisance parameter.
It is the number that says how far the arbitrary scale sits from metric truth.

WHY THE Y FLIP IS UNDONE FIRST -- a subtle trap

The pipeline exports Y-up for Three.js, converting once from OpenCV's Y-down.
That conversion is a REFLECTION, and a reflection is not a rotation: its
determinant is -1. Umeyama alignment constrains its rotation to be proper
(determinant +1) precisely so it cannot "fix" a mirrored trajectory by
cheating. Feed it a Y-flipped path against an unflipped ground truth and it
cannot align them, producing an enormous ATE that looks like catastrophic drift
and is actually a coordinate convention.

So the flip is undone here, and the comparison happens in the pipeline's native
OpenCV frame.

USAGE

  python tools/evaluate_ate.py result.json samples/tum_xyz.truth.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# The maths lives in vslam/ because the sweep runner needs it too; this file is
# only the command-line front end.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vslam.align import (  # noqa: E402
    absolute_trajectory_error,
    centres_from_camera_to_world,
    centres_from_world_to_camera,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Absolute Trajectory Error, Sim(3)-aligned")
    parser.add_argument("estimate", help="job result JSON (or a bare list of poses)")
    parser.add_argument("truth", help="*.truth.json from make_synthetic or tum_to_video")
    parser.add_argument(
        "--no-undo-y-flip",
        action="store_true",
        help="estimate is already in OpenCV Y-down (not exported for Three.js)",
    )
    args = parser.parse_args()

    estimate_doc = json.loads(Path(args.estimate).read_text())
    # Accept either a full job document or a bare result.
    result = estimate_doc.get("result", estimate_doc)
    estimated_poses = result["poses"]
    frame_indices = result.get("frame_indices") or list(range(len(estimated_poses)))

    truth_doc = json.loads(Path(args.truth).read_text())
    truth_centres_all, truth_kept = centres_from_world_to_camera(truth_doc["poses"])

    # The estimate covers only the frames that were processed and tracked, so
    # ground truth is subsampled to match, by frame index.
    truth_index_of = {index: position for position, index in enumerate(truth_kept)}
    paired_estimate, paired_truth = [], []
    for position, frame_index in enumerate(frame_indices):
        if frame_index in truth_index_of:
            paired_estimate.append(estimated_poses[position])
            paired_truth.append(truth_index_of[frame_index])

    if len(paired_estimate) < 3:
        raise SystemExit(
            f"only {len(paired_estimate)} frames could be paired with ground truth; "
            "cannot align a trajectory from that"
        )

    estimated_centres = centres_from_camera_to_world(
        paired_estimate, undo_y_flip=not args.no_undo_y_flip
    )
    truth_centres = truth_centres_all[:, paired_truth]

    metrics = absolute_trajectory_error(estimated_centres, truth_centres)

    print(f"paired poses       {metrics['n_poses']}")
    print(f"ground-truth path  {metrics['truth_path_length']:.3f} (metres, if truth is metric)")
    print(f"recovered scale    {metrics['scale_factor']:.4f}")
    print()
    print(f"ATE RMSE           {metrics['ate_rmse']:.4f}")
    print(f"ATE mean           {metrics['ate_mean']:.4f}")
    print(f"ATE median         {metrics['ate_median']:.4f}")
    print(f"ATE max            {metrics['ate_max']:.4f}")
    print()
    print(f"RMSE / path length {metrics['rmse_over_path_pct']:.2f}%")
    print(
        "\nScale was solved for, not assumed: a single lens cannot observe it, so "
        "\ncomparing without a similarity alignment would measure an arbitrary "
        "\nchoice rather than the reconstruction."
    )


if __name__ == "__main__":
    main()
