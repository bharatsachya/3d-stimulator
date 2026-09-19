"""
The four documented failure modes must stay DISTINCT, and must not swallow
valid input.

WHY THIS FILE EXISTS

A fixed camera on a highway overpass produced a plausible-looking reconstruction
instead of a diagnosis: 35 poses, 11 segments, ZERO map points, and a cosmetic
"the map is sparse" flag. The viewer rendered the bare frusta and nothing said
they were meaningless.

Adding a gate to catch that is the easy half. The hard half is not breaking the
other three diagnoses, or the five benchmark sequences, in the process -- and
the first version of the fix did exactly that, condemning fr1_desk2 (3015 points
across 13 healthy segments) as degenerate because it read the reprojection error
from the LAST segment's map, which is legitimately empty when a video ends
mid-initialization.

These tests are skipped when the clips are absent, since they are generated or
downloaded rather than committed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.schema import FailureReason, SlamFailure
from vslam.initialize import (
    MAX_STATIC_FRACTION,
    STATIC_DISPLACEMENT_PX,
    degenerate_geometry_warning,
)

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def clip(name: str) -> Path:
    return SAMPLES / name


def requires(name: str):
    return pytest.mark.skipif(
        not clip(name).is_file(), reason=f"{name} not present (generated/downloaded)"
    )


# --------------------------------------------------------------------------
# The statistic itself, tested without needing any video.
# --------------------------------------------------------------------------


def static_fraction(displacements: np.ndarray) -> float:
    return float((displacements < STATIC_DISPLACEMENT_PX).mean())


def test_static_fraction_separates_the_two_motion_patterns() -> None:
    """
    A translating camera moves everything by varying amounts; a fixed camera
    moves nothing except what passes through. These are the measured shapes.
    """
    # cars.mp4, gap 10: median 0.00 px, max 261 px, 80% under a pixel.
    static_camera = np.concatenate([np.zeros(800), np.linspace(50, 261, 200)])
    # fr1_xyz, gap 10: median 75 px, nothing under a pixel.
    moving_camera = np.random.default_rng(0).normal(75, 20, 1000).clip(1.5, None)

    assert static_fraction(static_camera) > MAX_STATIC_FRACTION
    assert static_fraction(moving_camera) < MAX_STATIC_FRACTION


def test_the_threshold_sits_in_the_measured_gap() -> None:
    """
    Measured minima across 30 candidate pairs: cars 0.49, every valid clip
    0.00-0.01. The threshold must sit between them with real room on both sides,
    which is what distinguishes this from the homography ratio below.
    """
    worst_valid_clip = 0.01
    best_static_pair = 0.49
    assert worst_valid_clip < MAX_STATIC_FRACTION < best_static_pair
    assert MAX_STATIC_FRACTION - worst_valid_clip > 0.1
    assert best_static_pair - MAX_STATIC_FRACTION > 0.1


def test_homography_dominance_is_a_warning_not_a_gate() -> None:
    """
    Re-measured across nine clips, rotation's minimum (0.400) falls BELOW the
    valid maximum (0.429) -- the distributions overlap. It may only warn.
    """
    assert degenerate_geometry_warning(0.45) is True
    assert degenerate_geometry_warning(0.31) is False
    # And it must never be wired to a failure reason.
    assert not hasattr(FailureReason, "DEGENERATE_GEOMETRY")


# --------------------------------------------------------------------------
# End to end, when the clips are available.
# --------------------------------------------------------------------------


def run(name: str):
    from vslam.pipeline import run_pipeline

    return run_pipeline(str(clip(name)))


@requires("cars.mp4")
def test_static_camera_is_diagnosed() -> None:
    with pytest.raises(SlamFailure) as raised:
        run("cars.mp4")
    assert raised.value.reason is FailureReason.STATIC_CAMERA
    # The advice must be specific to a camera that cannot move, not the
    # "walk sideways" advice given for pure rotation.
    assert "moved" in raised.value.detail


@requires("tum_nostructure_notexture.mp4")
def test_low_texture_keeps_its_own_diagnosis() -> None:
    """The new gate must not start claiming blank walls are static cameras."""
    with pytest.raises(SlamFailure) as raised:
        run("tum_nostructure_notexture.mp4")
    assert raised.value.reason is FailureReason.INITIALIZATION_FAILED
    assert raised.value.reason is not FailureReason.STATIC_CAMERA


@requires("synth_dolly.mp4")
def test_a_valid_clip_is_not_swallowed_by_the_static_gate() -> None:
    result = run("synth_dolly.mp4")
    assert len(result.poses) > 20
    assert result.stats["n_map_points"] > 100


@requires("synth_dolly.mp4")
def test_a_result_is_never_returned_without_structure() -> None:
    """
    The overpass clip reached 'done' with zero map points. Any successful run
    must carry an actual reconstruction and a finite reprojection error.
    """
    result = run("synth_dolly.mp4")
    assert result.stats["n_map_points"] >= 25
    assert np.isfinite(result.stats["mean_reprojection_error_px"])
