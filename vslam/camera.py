"""
Camera intrinsics.

WHAT WE KNOW AND WHAT WE GUESS

The principal point is assumed to be the image centre. That is very nearly true
for any normal camera and the error is small.

The focal length is GUESSED, at 0.9 x image width. This is the honest weak point
of accepting arbitrary uploads: focal length is a property of the lens, it is
knowable in principle -- unlike scale, which is not observable from one lens at
all -- but we are handed a file from an unknown camera with no calibration.

Measured against real cameras the guess is mediocre but not absurd:
    TUM freiburg1  fx = 517.3, guess 576  -> +11.3%
    TUM freiburg3  fx = 535.4, guess 576  ->  +7.6%

So the user can override it, and the README reports what the error costs.

LENS DISTORTION IS IGNORED

Correcting it requires calibration parameters we do not have. Uncorrected radial
distortion bends straight lines near the frame edge, which biases the features
furthest from the centre -- exactly the ones with the most parallax information.
It is a documented limitation, not an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Multiplier on image width. Corresponds to a horizontal field of view of about
# 58 degrees, which is in the middle of the range typical phone main cameras
# occupy (roughly 60-75 degrees).
DEFAULT_FOCAL_RATIO = 0.9


@dataclass(frozen=True)
class Camera:
    """Pinhole intrinsics for a specific working resolution."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def guess(
        cls,
        width: int,
        height: int,
        focal_ratio: float = DEFAULT_FOCAL_RATIO,
        focal_px: float | None = None,
    ) -> "Camera":
        """
        Build intrinsics for an unknown camera.

        `focal_px` overrides the heuristic outright, for a user who knows their
        camera or for a dataset that publishes measured intrinsics.

        Square pixels (fx == fy) are assumed. Non-square pixels are rare in
        consumer cameras, and nothing here could estimate the ratio anyway.
        """
        focal = focal_px if focal_px is not None else focal_ratio * width
        return cls(
            width=width,
            height=height,
            fx=focal,
            fy=focal,
            cx=width / 2.0,
            cy=height / 2.0,
        )

    @property
    def matrix(self) -> np.ndarray:
        """The 3x3 K that every OpenCV geometry call wants."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def scaled(self, scale: float) -> "Camera":
        """
        Intrinsics for a resized image.

        Every intrinsic parameter scales linearly with resolution. Forgetting to
        rescale K after resizing a frame is a classic silent error: the geometry
        still solves, it just solves the wrong problem, producing a trajectory
        that looks plausible and is systematically wrong.
        """
        return Camera(
            width=int(round(self.width * scale)),
            height=int(round(self.height * scale)),
            fx=self.fx * scale,
            fy=self.fy * scale,
            cx=self.cx * scale,
            cy=self.cy * scale,
        )

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fx": round(self.fx, 3),
            "fy": round(self.fy, 3),
            "cx": round(self.cx, 3),
            "cy": round(self.cy, 3),
        }
