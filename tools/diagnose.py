"""
Per-frame diagnostics for a tracking run, as CSV and as a plot.

WHY THIS EXISTS

Tracking stops at a particular frame and the single failing frame tells you
almost nothing: every cause looks the same from there -- not enough inliers.
What distinguishes them is the SHAPE of the series leading up to it.

    map size collapsing while triangulation stalls  -> the cull is eating the map
    keyframes stop being inserted                   -> map stops growing
    visible points healthy, inliers decaying        -> matching, not the map
    inliers hovering just above the threshold       -> the threshold is the problem

The plot is rendered with OpenCV rather than matplotlib deliberately: OpenCV is
already a dependency, and adding a plotting library to a project whose whole
argument is about a constrained compute budget would be a poor trade for a
diagnostic that runs a handful of times.

    python tools/diagnose.py samples/tum_fr1_xyz.mp4 --out out/diagnose
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vslam.pipeline import run_pipeline  # noqa: E402
from vslam.tracking import MIN_TRACKING_INLIERS  # noqa: E402

PANEL_HEIGHT = 150
PANEL_WIDTH = 1100
MARGIN_LEFT = 74
MARGIN_BOTTOM = 26


def draw_panel(
    canvas: np.ndarray,
    top: int,
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    title: str,
    keyframe_positions: list[int],
    failure_position: int | None,
    threshold: float | None = None,
) -> None:
    """One stacked panel with a shared x axis of processed-frame position."""
    height, width = PANEL_HEIGHT, PANEL_WIDTH
    plot_w = width - MARGIN_LEFT - 12
    plot_h = height - MARGIN_BOTTOM - 20

    # Panel background and frame.
    cv2.rectangle(
        canvas, (MARGIN_LEFT, top + 18), (MARGIN_LEFT + plot_w, top + 18 + plot_h),
        (32, 32, 36), -1,
    )
    cv2.putText(
        canvas, title, (8, top + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
        (210, 210, 215), 1, cv2.LINE_AA,
    )

    all_values = [v for _, values, _ in series for v in values]
    if threshold is not None:
        all_values.append(threshold)
    top_value = max(all_values) if all_values else 1.0
    top_value = max(top_value * 1.12, 1.0)

    n = max(len(series[0][1]), 2)

    def to_xy(i: int, value: float) -> tuple[int, int]:
        x = MARGIN_LEFT + int(i * plot_w / (n - 1))
        y = top + 18 + plot_h - int(value / top_value * plot_h)
        return x, y

    # Keyframe insertions as faint vertical lines, so every other series can be
    # read against when the map was last extended.
    for position in keyframe_positions:
        x, _ = to_xy(position, 0)
        cv2.line(canvas, (x, top + 18), (x, top + 18 + plot_h), (58, 58, 64), 1)

    if threshold is not None:
        _, y = to_xy(0, threshold)
        for x in range(MARGIN_LEFT, MARGIN_LEFT + plot_w, 8):
            cv2.line(canvas, (x, y), (x + 4, y), (70, 120, 200), 1)

    if failure_position is not None:
        x, _ = to_xy(failure_position, 0)
        cv2.line(canvas, (x, top + 18), (x, top + 18 + plot_h), (60, 60, 200), 2)

    for label, values, colour in series:
        points = [to_xy(i, v) for i, v in enumerate(values)]
        for a, b in zip(points, points[1:]):
            cv2.line(canvas, a, b, colour, 1, cv2.LINE_AA)

    # y axis: only top and zero, which is all a diagnostic needs.
    cv2.putText(
        canvas, f"{top_value:.0f}", (6, top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
        (150, 150, 155), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "0", (6, top + 18 + plot_h), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
        (150, 150, 155), 1, cv2.LINE_AA,
    )

    legend_x = MARGIN_LEFT + 6
    for label, _, colour in series:
        cv2.putText(
            canvas, label, (legend_x, top + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
            colour, 1, cv2.LINE_AA,
        )
        legend_x += 9 * len(label) + 16


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose a tracking run.")
    parser.add_argument("video")
    parser.add_argument("--out", default="out/diagnose")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--features", type=int, default=1000)
    args = parser.parse_args()

    records: list[dict] = []
    result = run_pipeline(
        args.video,
        processed_fps=args.fps,
        working_width=args.width,
        n_features=args.features,
        diagnostics=records,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    def column(name: str) -> list[float]:
        return [float(r[name]) for r in records]

    keyframe_positions = [i for i, r in enumerate(records) if r["is_keyframe"]]
    failure_position = next(
        (i for i, r in enumerate(records) if r["tracking_failed"]), None
    )

    panels = [
        (
            "map size and visibility",
            [
                ("map_points", column("n_map_points"), (120, 200, 120)),
                ("local_window", column("n_local_points"), (200, 180, 90)),
                ("visible_in_frame", column("n_visible"), (200, 120, 200)),
            ],
            None,
        ),
        (
            "matching and PnP inliers",
            [
                ("matches", column("n_matches"), (200, 180, 90)),
                ("inliers", column("n_inliers"), (120, 200, 255)),
            ],
            float(MIN_TRACKING_INLIERS),
        ),
        (
            "map churn per keyframe",
            [
                ("triangulated", column("n_triangulated"), (120, 220, 160)),
                ("culled", column("n_culled"), (110, 110, 240)),
            ],
            None,
        ),
        (
            "inlier ratio and keyframe rate",
            [
                ("inlier_ratio x100", [v * 100 for v in column("inlier_ratio")], (120, 200, 255)),
                ("keyframes", column("n_keyframes"), (200, 200, 200)),
            ],
            70.0,
        ),
    ]

    canvas = np.full(
        (PANEL_HEIGHT * len(panels) + 34, PANEL_WIDTH, 3), 18, dtype=np.uint8
    )
    for index, (title, series, threshold) in enumerate(panels):
        draw_panel(
            canvas, index * PANEL_HEIGHT, series, title,
            keyframe_positions, failure_position, threshold,
        )

    footer = (
        f"{Path(args.video).name}  |  {len(records)} processed frames  |  "
        f"grey = keyframe, red = tracking lost, blue dashes = threshold"
    )
    cv2.putText(
        canvas, footer, (8, PANEL_HEIGHT * len(panels) + 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 175), 1, cv2.LINE_AA,
    )

    png_path = out.with_suffix(".png")
    cv2.imwrite(str(png_path), canvas)

    # A compact text summary, so the diagnosis does not depend on reading pixels.
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    print()
    lost = result.stats["tracking_lost_at_frame"]
    print(f"processed frames   {len(records)}")
    print(f"tracking lost at   {lost if lost is not None else 'not lost'}")
    if failure_position is not None:
        print(f"failure reason     {records[failure_position]['failure_reason']}")
    print()
    print(
        f"{'pos':>4} {'frame':>6} {'map':>6} {'local':>6} {'vis':>5} "
        f"{'match':>6} {'inl':>5} {'ratio':>6} {'tri':>5} {'cull':>5} {'kf':>3}"
    )
    tail = records[-28:] if len(records) > 28 else records
    for record in tail:
        marker = "KF" if record["is_keyframe"] else ("XX" if record["tracking_failed"] else "")
        print(
            f"{records.index(record):>4} {record['frame_index']:>6} "
            f"{record['n_map_points']:>6} {record['n_local_points']:>6} "
            f"{record['n_visible']:>5} {record['n_matches']:>6} "
            f"{record['n_inliers']:>5} {record['inlier_ratio']:>6.3f} "
            f"{record['n_triangulated']:>5} {record['n_culled']:>5} {marker:>3}"
        )


if __name__ == "__main__":
    main()
