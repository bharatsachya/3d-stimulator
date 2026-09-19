"""
Measure apparent-motion statistics across candidate initialization pairs.

WHY THIS EXISTS

A fixed camera on a highway overpass produced a plausible-looking reconstruction
instead of a diagnosis. The camera never moved; the only motion was traffic. The
existing parallax gate could not catch it, and the reason is conceptual rather
than a coding error: the gate measures APPARENT feature motion, and apparent
motion has two possible causes -- a moving camera in a static scene, or a static
camera with independently moving objects. The gate cannot tell them apart, and
the cars supplied enough apparent motion to read as camera translation.

This tool reports the statistics that might separate the two, ACROSS SEVERAL
CLIPS, so a threshold can be chosen on evidence or rejected for lack of margin.
It deliberately adds no gate of its own.

WHAT IT MEASURES, AND WHY EACH

  displacement distribution -- median, p90, max, and the fraction of matches
      that moved less than a pixel. A translating camera moves features broadly
      across the frame. A static camera with moving objects produces a large
      stationary mode plus a small cluster of large displacements, so the median
      sits near zero while the max is enormous.

  inlier spatial spread -- the convex-hull area of the essential matrix's
      inliers, as a fraction of the image. Monocular SLAM assumes a rigid static
      world. When that assumption is violated, findEssentialMat can happily fit
      the MOVING OBJECTS' motion and report it as camera motion; the giveaway is
      that the inliers supporting it occupy one small region rather than
      spreading across the frame.

    python tools/motion_stats.py samples/cars.mp4 samples/tum_fr1_xyz.mp4 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vslam.camera import Camera  # noqa: E402
from vslam.features import FeatureExtractor, Matcher  # noqa: E402
from vslam.video import iter_frames  # noqa: E402


def convex_hull_fraction(points: np.ndarray, width: int, height: int) -> float:
    """Convex-hull area of a point set, as a fraction of the image area."""
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull) / (width * height))


def pair_statistics(
    features_a, features_b, matcher, camera: Camera
) -> dict | None:
    query, train = matcher.match(features_a.descriptors, features_b.descriptors)
    if len(query) < 8:
        return None

    points_a = features_a.points[query]
    points_b = features_b.points[train]
    displacement = np.linalg.norm(points_b - points_a, axis=1)

    stats = {
        "matches": int(len(query)),
        "median_px": float(np.median(displacement)),
        "p90_px": float(np.percentile(displacement, 90)),
        "max_px": float(displacement.max()),
        "frac_under_1px": float((displacement < 1.0).mean()),
        "frac_under_2px": float((displacement < 2.0).mean()),
    }

    # Where do the essential matrix's supporters live?
    essential, mask = cv2.findEssentialMat(
        points_a, points_b, camera.matrix,
        method=cv2.RANSAC, prob=0.999, threshold=1.0,
    )
    if essential is None or essential.shape != (3, 3) or mask is None:
        stats.update(inliers=0, inlier_hull_frac=0.0, all_hull_frac=0.0)
        return stats

    inlier = mask.ravel() > 0
    stats["inliers"] = int(inlier.sum())
    stats["inlier_hull_frac"] = convex_hull_fraction(
        points_a[inlier], camera.width, camera.height
    )
    # The hull of ALL matches, for reference: a small inlier hull inside a large
    # match hull is the interesting case, not a small hull everywhere.
    stats["all_hull_frac"] = convex_hull_fraction(
        points_a, camera.width, camera.height
    )
    return stats


def analyse(video: str, gaps: list[int], fps: float, width: int, features: int) -> dict:
    extractor = FeatureExtractor(features)
    matcher = Matcher()

    needed = max(gaps) + 1
    frames = list(iter_frames(video, fps, width, max_frames=needed))
    if len(frames) < 2:
        return {"error": "too few frames"}

    detected = [extractor.detect(f.gray) for f in frames]
    height, image_width = frames[0].gray.shape
    camera = Camera.guess(image_width, height)

    rows = {}
    for gap in gaps:
        if gap >= len(frames):
            continue
        stats = pair_statistics(detected[0], detected[gap], matcher, camera)
        if stats is not None:
            rows[gap] = stats
    return {"pairs": rows, "resolution": f"{image_width}x{height}"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Apparent-motion statistics.")
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--gaps", default="1,3,10,20,30")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--features", type=int, default=1000)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    gaps = [int(g) for g in args.gaps.split(",")]
    everything = {}

    header = (
        f"{'clip':<26}{'gap':>5}{'match':>7}{'median':>9}{'p90':>9}{'max':>10}"
        f"{'<1px':>8}{'inliers':>9}{'inl hull':>10}{'all hull':>10}"
    )
    print(header)
    print("-" * len(header))

    for video in args.videos:
        name = Path(video).stem
        result = analyse(video, gaps, args.fps, args.width, args.features)
        everything[name] = result
        if "error" in result:
            print(f"{name:<26}{result['error']}")
            continue
        for gap, s in result["pairs"].items():
            print(
                f"{name:<26}{gap:>5}{s['matches']:>7}{s['median_px']:>9.2f}"
                f"{s['p90_px']:>9.2f}{s['max_px']:>10.2f}"
                f"{s['frac_under_1px'] * 100:>7.0f}%{s.get('inliers', 0):>9}"
                f"{s.get('inlier_hull_frac', 0) * 100:>9.1f}%"
                f"{s.get('all_hull_frac', 0) * 100:>9.1f}%"
            )
        print()

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(everything, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
