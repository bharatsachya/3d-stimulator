"""
The shapes the API returns, and the vocabulary of failure.

WHY FAILURES ARE TYPED

CLAUDE.md is explicit that a failure becomes a flagged result, never a 500, and
that the pipeline fails loudly rather than emitting garbage. Both of those need
a fixed vocabulary of reasons, because:

  * the frontend has to say something useful, and "500 Internal Server Error"
    is not useful to someone who filmed a wall;
  * several of these failures are EXPECTED and are documented failure modes of
    monocular SLAM, not bugs. Pure rotation cannot be initialized. That is
    physics. It should read as a clear diagnosis, not a crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class FailureReason(str, Enum):
    """Why a job failed. Each maps to one sentence the user can act on."""

    # The container could not be opened, or held no decodable video.
    DECODE_FAILED = "decode_failed"
    # Fewer frames than initialization needs, e.g. a one-second clip.
    TOO_FEW_FRAMES = "too_few_frames"
    # The camera rotated but never translated. No parallax, so no depth, so no
    # map. This is the documented pure-rotation failure mode.
    INSUFFICIENT_PARALLAX = "insufficient_parallax"
    # Parallax was adequate but too few correspondences survived the essential
    # matrix. Usually low texture or motion blur.
    INITIALIZATION_FAILED = "initialization_failed"
    # A bug, not a diagnosis. Distinguished from the above precisely so that
    # the expected failures are not mistaken for one.
    INTERNAL_ERROR = "internal_error"


class ResultFlag(str, Enum):
    """
    Something the user should know about a result that still succeeded.

    Distinct from FailureReason: a flagged job produced usable output. Tracking
    loss mid-sequence returns the trajectory up to that point, which is a real
    partial answer, so it is a flag rather than a failure.
    """

    TRACKING_LOST = "tracking_lost"
    RELOCALIZED = "relocalized"
    FEW_MAP_POINTS = "few_map_points"
    BA_SKIPPED_FOR_TIME = "ba_skipped_for_time"


# Every reason rendered as something a person can act on. Kept beside the enum
# so adding a reason without a message is a visible omission.
FAILURE_MESSAGES: dict[FailureReason, str] = {
    FailureReason.DECODE_FAILED: (
        "The video could not be decoded. Try an MP4 or MOV recorded on a phone."
    ),
    FailureReason.TOO_FEW_FRAMES: (
        "The clip is too short to reconstruct. Around five seconds of motion works well."
    ),
    FailureReason.INSUFFICIENT_PARALLAX: (
        "Not enough parallax - the camera may have rotated in place rather than "
        "moving through the scene. A single lens recovers depth only from motion "
        "that changes viewpoint, so try walking sideways past the subject."
    ),
    FailureReason.INITIALIZATION_FAILED: (
        "Could not establish an initial map. This usually means too little texture "
        "(a blank wall or plain floor) or motion blur from moving too fast."
    ),
    FailureReason.INTERNAL_ERROR: (
        "Processing failed unexpectedly. This one is on us, not on your video."
    ),
}

FLAG_MESSAGES: dict[ResultFlag, str] = {
    ResultFlag.TRACKING_LOST: (
        "Tracking was lost partway through; the trajectory shown stops at that point."
    ),
    ResultFlag.RELOCALIZED: (
        "Tracking was lost and recovered at least once. The trajectory continues "
        "across the gap, but the frames in between have no pose."
    ),
    ResultFlag.FEW_MAP_POINTS: (
        "The map is sparse, so the point cloud may look thin. More texture in the "
        "scene usually helps."
    ),
    ResultFlag.BA_SKIPPED_FOR_TIME: (
        "Bundle adjustment was run less often than usual to stay within the time "
        "budget, so drift may be slightly higher."
    ),
}


class SlamFailure(Exception):
    """
    An expected, diagnosable failure. Carries a reason the API can surface.

    Raised by pipeline stages. The worker catches it and records a failed job;
    anything else that escapes becomes INTERNAL_ERROR, which is how a genuine
    bug stays distinguishable from a video we correctly refused.
    """

    def __init__(self, reason: FailureReason, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason.value}: {detail}" if detail else reason.value)


@dataclass
class SlamResult:
    """
    What a completed job hands back.

    Coordinates are Y-UP here, converted once at export. Units are ARBITRARY --
    a single lens cannot observe scale, so there is no metre to report, and the
    frontend labels the axes accordingly. See the README.
    """

    # 4x4 camera-to-world matrices, one per processed frame that was tracked.
    poses: list[list[list[float]]] = field(default_factory=list)
    # Which source frame index each pose corresponds to.
    frame_indices: list[int] = field(default_factory=list)
    # Which of those poses are keyframes, as indices into `poses`.
    keyframe_indices: list[int] = field(default_factory=list)
    # [x, y, z] per map point.
    points: list[list[float]] = field(default_factory=list)
    # How many keyframes observed each point. Higher means better constrained;
    # the viewer uses it to fade weakly-observed points.
    observations: list[int] = field(default_factory=list)

    flags: list[str] = field(default_factory=list)
    flag_messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "poses": self.poses,
            "frame_indices": self.frame_indices,
            "keyframe_indices": self.keyframe_indices,
            "points": self.points,
            "observations": self.observations,
            "n_poses": len(self.poses),
            "n_points": len(self.points),
            "flags": self.flags,
            "flag_messages": self.flag_messages,
            "units": "arbitrary",
            "coordinate_system": "y-up, right-handed (converted from OpenCV y-down)",
        }
