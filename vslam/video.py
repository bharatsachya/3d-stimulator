"""
Decoding and frame sampling.

THREE THINGS HAPPEN HERE, ALL OF THEM COST-DRIVEN

1. SUBSAMPLE TO ~10 fps. A 30 fps source gives every third frame. Consecutive
   frames at 30 fps have almost no baseline between them, so the extra frames
   cost full ORB and matching time while contributing nearly no parallax.

2. DOWNSCALE TO A 640px LONG EDGE. ORB cost scales with pixel count, and on the
   target instance so does matching, indirectly, through how many features a
   larger image yields. INTER_AREA is used for downscaling specifically: it
   averages over the source region rather than sampling it, so it does not
   alias. Aliased edges produce spurious FAST corners that do not repeat
   between frames, which is the worst possible input to a matcher.

3. DROP COLOUR. ORB is computed on intensity, so carrying three channels is
   pure waste.

WHY grab() AND retrieve() ARE SEPARATE

`grab()` advances to the next frame without handing it back; `retrieve()`
decodes the grabbed frame into an array. For skipped frames only `grab()` is
called, which avoids the colour conversion and the copy.

It is not free: inter-frame compression means a skipped frame still has to be
decoded to reconstruct the frames that follow it. Measured on the target, decode
is 1.7 ms per processed frame, ~13% of the per-frame floor. Cheap, not zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

from app.schema import FailureReason, SlamFailure


@dataclass
class Frame:
    """One frame, ready for feature detection."""

    index: int          # index in the SOURCE video, so results map back to it
    timestamp: float    # seconds from the start of the video
    gray: np.ndarray    # working resolution, single channel


@dataclass
class VideoInfo:
    """What the container claims about itself."""

    source_width: int
    source_height: int
    source_fps: float
    source_frame_count: int
    duration_seconds: float
    stride: int
    scale: float

    def to_dict(self) -> dict:
        return {
            "source_resolution": f"{self.source_width}x{self.source_height}",
            "source_fps": round(self.source_fps, 2),
            "source_frames": self.source_frame_count,
            "duration_seconds": round(self.duration_seconds, 2),
            "stride": self.stride,
            "working_scale": round(self.scale, 4),
        }


def open_video(
    path: str, max_duration_seconds: float | None = None
) -> tuple[cv2.VideoCapture, int, int, float, int, float]:
    """
    Open a video and work out how to sample it.

    Raises SlamFailure with a typed reason rather than returning None, so the
    worker can turn it into a message the user can act on.
    """
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise SlamFailure(
            FailureReason.DECODE_FAILED, "the container could not be opened"
        )

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        capture.release()
        raise SlamFailure(
            FailureReason.DECODE_FAILED, "the video reported no usable resolution"
        )

    # A container can lie, or simply not know. Both happen with phone video and
    # with streams remuxed by a messaging app.
    if not fps or fps <= 0 or fps > 240:
        fps = 30.0
    duration = count / fps if count > 0 else 0.0

    # SERVER-SIDE duration enforcement. The browser checks this too, but a
    # client-side check is a courtesy, not a control -- a crafted request skips
    # it entirely.
    if max_duration_seconds is not None and duration > max_duration_seconds:
        capture.release()
        raise SlamFailure(
            FailureReason.DECODE_FAILED,
            f"the video is {duration:.1f}s, over the {max_duration_seconds:.0f}s limit",
        )

    return capture, width, height, fps, count, duration


def iter_frames(
    path: str,
    processed_fps: float,
    working_width: int,
    max_duration_seconds: float | None = None,
    max_frames: int | None = None,
    on_decode_time=None,
) -> Iterator[Frame]:
    """
    Yield sampled, downscaled, grayscale frames.

    A generator rather than a list: a 60-second clip is ~600 processed frames,
    and holding them all at 640x480 would be ~180 MB of arrays for no reason,
    since each is finished with before the next is needed.

    `on_decode_time` receives the seconds spent decoding each yielded frame.
    Decode cannot be timed with a simple `with` block because it is interleaved
    with the caller's work inside a generator, so the timing is handed out.
    """
    capture, width, height, fps, count, duration = open_video(
        path, max_duration_seconds
    )

    stride = max(int(round(fps / processed_fps)), 1)
    scale = min(working_width / width, 1.0)  # never upscale; it invents no detail

    try:
        source_index = 0
        yielded = 0
        while True:
            if max_frames is not None and yielded >= max_frames:
                break

            import time

            started = time.perf_counter()

            if not capture.grab():
                break

            if source_index % stride != 0:
                if on_decode_time is not None:
                    on_decode_time(time.perf_counter() - started)
                source_index += 1
                continue

            ok, frame = capture.retrieve()
            if not ok:
                break

            if scale < 1.0:
                frame = cv2.resize(
                    frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if on_decode_time is not None:
                on_decode_time(time.perf_counter() - started)

            yield Frame(
                index=source_index, timestamp=source_index / fps, gray=gray
            )
            yielded += 1
            source_index += 1
    finally:
        # Released even if the consumer abandons the generator part-way, which
        # happens on tracking loss.
        capture.release()


def probe_video(path: str, processed_fps: float, working_width: int) -> VideoInfo:
    """Read the header without decoding the whole file."""
    capture, width, height, fps, count, duration = open_video(path)
    capture.release()
    return VideoInfo(
        source_width=width,
        source_height=height,
        source_fps=fps,
        source_frame_count=count,
        duration_seconds=duration,
        stride=max(int(round(fps / processed_fps)), 1),
        scale=min(working_width / width, 1.0),
    )
