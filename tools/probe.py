"""
The budget probe.

WHAT THIS ANSWERS

The brief allows 10 seconds of processing for a 10-second video. At 10 fps that
is ~100 processed frames, so ~100 ms per frame, on 2 vCPUs with no GPU. Before
any pipeline architecture is built on top of that assumption, we should know
what the unavoidable per-frame work actually costs on the target machine.

This probe measures only the three stages that run on EVERY frame and that no
design choice can remove:

    decode   pull the frame out of the container and downscale it
    orb      detect keypoints and compute descriptors
    match    brute-force Hamming against the previous frame + Lowe ratio test

It deliberately does NOT measure PnP, triangulation or bundle adjustment. Those
stages do not exist yet, and two of them run only on keyframes. What matters
right now is the floor: whatever these three cost, every later stage has to fit
in the remainder.

HOW TO READ THE RESULT

If the floor is ~50 ms/frame, there is ~50 ms/frame of headroom for tracking and
BA and the locked parameters stand. If the floor is already ~100 ms/frame, the
parameters have to change -- fewer features, smaller working resolution, or a
lower processed frame rate -- and it is much cheaper to learn that now than
after the map and the optimizer are written against them.

This must be run on the TARGET instance. A laptop with 8 performance cores will
report numbers that have nothing to do with an m7i-flex.large, and quoting a dev
laptop's timings in the README would make the whole measurement section false.

USAGE

  python tools/probe.py samples/synth_dolly.mp4
  python tools/probe.py samples/synth_dolly.mp4 --threads 2 --json out/probe.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Import from the package rather than duplicating the timer here: the probe and
# the real pipeline must measure the same way, or their numbers are not comparable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vslam.timing import StageTimer  # noqa: E402


def environment(threads_requested: int | None) -> dict:
    """
    Everything the README's 'test environment' line has to state.

    A caveat on `cv2_threads`: on at least some OpenCV builds `setNumThreads`
    genuinely changes execution time while `getNumThreads` keeps reporting the
    default. Measured here: 1 thread gives ~30 ms/frame and 2 gives ~18, yet
    getNumThreads() reports 8 in both cases. So `cv2_threads` is recorded as
    what OpenCV claims, and `threads_requested` as what we actually asked for.
    Quote the latter in the README; the former is not trustworthy.
    """
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "cpu_count": cv2.getNumberOfCPUs(),
        "cv2_threads": cv2.getNumThreads(),
        "threads_requested": threads_requested,
    }


def probe(
    video_path: str,
    target_fps: float,
    working_width: int,
    n_features: int,
    ratio: float,
    max_frames: int | None,
) -> tuple[StageTimer, dict]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise SystemExit(f"could not open {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Take every Nth frame to hit the target processed rate. A 30 fps source at
    # 10 fps processing means stride 3.
    stride = max(int(round(source_fps / target_fps)), 1)

    orb = cv2.ORB_create(nfeatures=n_features)
    # crossCheck must stay False: knnMatch(k=2) is what the ratio test needs, and
    # crossCheck=True is incompatible with k=2. The ratio test is the stronger
    # filter anyway -- it rejects AMBIGUOUS matches, where cross-checking only
    # rejects non-mutual ones.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    timer = StageTimer()
    previous_descriptors = None
    processed = 0
    keypoint_counts: list[int] = []
    match_counts: list[int] = []

    with timer.run():
        source_index = 0
        while True:
            if max_frames is not None and processed >= max_frames:
                break

            decode_start = time.perf_counter()
            # grab() advances without handing back a decoded frame, which is the
            # cheap way to skip. It is not free -- inter-frame compression means
            # skipped frames still have to be decoded to reconstruct later ones --
            # but it avoids the colour conversion and the copy.
            ok = capture.grab()
            if not ok:
                break
            take = source_index % stride == 0
            if not take:
                timer.add("decode", time.perf_counter() - decode_start)
                source_index += 1
                continue

            ok, frame = capture.retrieve()
            if not ok:
                break

            # Downscale to the working resolution and drop colour. ORB ignores
            # colour entirely, so carrying three channels is pure waste.
            scale = working_width / frame.shape[1]
            if scale < 1.0:
                frame = cv2.resize(
                    frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timer.add("decode", time.perf_counter() - decode_start)

            with timer.stage("orb"):
                keypoints, descriptors = orb.detectAndCompute(gray, None)
            keypoint_counts.append(len(keypoints))

            if previous_descriptors is not None and descriptors is not None:
                with timer.stage("match"):
                    pairs = matcher.knnMatch(previous_descriptors, descriptors, k=2)
                    good = [
                        m
                        for m, n in (p for p in pairs if len(p) == 2)
                        if m.distance < ratio * n.distance
                    ]
                match_counts.append(len(good))

            previous_descriptors = descriptors
            processed += 1
            source_index += 1

    capture.release()

    stats = {
        "source_fps": round(source_fps, 2),
        "source_frames": source_count,
        "stride": stride,
        "processed_frames": processed,
        "working_width": working_width,
        "n_features": n_features,
        "ratio": ratio,
        "median_keypoints": int(np.median(keypoint_counts)) if keypoint_counts else 0,
        "median_matches": int(np.median(match_counts)) if match_counts else 0,
    }
    return timer, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the per-frame cost floor.")
    parser.add_argument("video")
    parser.add_argument("--fps", type=float, default=10.0, help="processed frame rate")
    parser.add_argument("--width", type=int, default=640, help="working long edge")
    parser.add_argument("--features", type=int, default=1000)
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio threshold")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="pin OpenCV thread count, e.g. 2 to imitate the target instance",
    )
    parser.add_argument("--json", default=None, help="also write the report here")
    args = parser.parse_args()

    if args.threads is not None:
        # Useful on a dev machine to approximate the target, but it is an
        # approximation only: thread count is not core speed, and it does not
        # reproduce the instance's memory bandwidth or cache.
        cv2.setNumThreads(args.threads)

    timer, stats = probe(
        args.video, args.fps, args.width, args.features, args.ratio, args.max_frames
    )

    env = environment(args.threads)
    report = timer.report(frames=stats["processed_frames"])

    print(f"video            {args.video}")
    print(f"source           {stats['source_frames']} frames @ {stats['source_fps']} fps")
    print(
        f"processed        {stats['processed_frames']} frames "
        f"(every {stats['stride']}), {args.width}px wide, {args.features} features"
    )
    print(
        f"features         median {stats['median_keypoints']} keypoints, "
        f"{stats['median_matches']} matches after ratio test"
    )
    print(f"environment      {env['platform']}")
    thread_note = (
        f"{env['threads_requested']} requested"
        if env["threads_requested"] is not None
        else f"{env['cv2_threads']} reported (default)"
    )
    print(
        f"                 {env['cpu_count']} CPUs, OpenCV {env['opencv']}, "
        f"threads: {thread_note}"
    )
    print()
    print(timer.format_table(frames=stats["processed_frames"]))
    print()

    # The verdict. The budget is 10s of processing for 10s of video, so the
    # per-frame allowance is (1000 / processed_fps) ms.
    budget_ms = 1000.0 / args.fps
    floor = report["ms_per_frame"] or 0.0
    headroom = budget_ms - floor
    print(f"budget           {budget_ms:.0f} ms/frame at {args.fps:g} fps processing")
    print(f"measured floor   {floor:.1f} ms/frame (decode + orb + match only)")
    if headroom > 0:
        print(
            f"headroom         {headroom:.1f} ms/frame remains for PnP, "
            f"triangulation and bundle adjustment"
        )
    else:
        print(
            f"OVER BUDGET      by {-headroom:.1f} ms/frame before tracking or BA "
            f"has run at all -- the locked parameters need revisiting"
        )

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "video": args.video,
                    "settings": stats,
                    "environment": env,
                    "timing": report,
                    "budget_ms_per_frame": budget_ms,
                    "headroom_ms_per_frame": round(headroom, 2),
                },
                indent=2,
            )
        )
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
