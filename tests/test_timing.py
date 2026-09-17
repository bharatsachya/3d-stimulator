"""
StageTimer tests.

The one that matters is `unaccounted`: the whole point of that row is that time
spent outside any measured stage cannot disappear from the report.
"""

from __future__ import annotations

import time

from vslam.timing import StageTimer


def test_stages_accumulate_across_calls() -> None:
    timer = StageTimer()
    with timer.run():
        for _ in range(4):
            with timer.stage("orb"):
                time.sleep(0.005)

    report = timer.report(frames=4)
    orb = next(stage for stage in report["stages"] if stage["stage"] == "orb")
    assert orb["calls"] == 4
    assert orb["total_ms"] >= 20.0
    assert orb["mean_ms"] >= 5.0


def test_unaccounted_time_is_reported() -> None:
    timer = StageTimer()
    with timer.run():
        with timer.stage("measured"):
            time.sleep(0.005)
        time.sleep(0.02)  # deliberately outside any stage

    report = timer.report(frames=1)
    # Roughly 20ms went unmeasured and the report must say so.
    assert report["unaccounted_ms"] >= 15.0
    assert report["unaccounted_pct"] > 50.0


def test_a_failing_stage_still_records_its_time() -> None:
    """Timings for a failed run are exactly when you most want them."""
    timer = StageTimer()
    with timer.run():
        try:
            with timer.stage("explodes"):
                time.sleep(0.005)
                raise RuntimeError("boom")
        except RuntimeError:
            pass

    report = timer.report(frames=1)
    stage = next(s for s in report["stages"] if s["stage"] == "explodes")
    assert stage["total_ms"] >= 5.0


def test_ms_per_frame_uses_processed_frame_count() -> None:
    timer = StageTimer()
    with timer.run():
        with timer.stage("ba"):
            time.sleep(0.01)

    # BA runs on keyframes, not frames, so its ms_per_frame must be its total
    # divided by frames processed -- much less than its mean per call.
    report = timer.report(frames=10)
    stage = report["stages"][0]
    assert stage["mean_ms"] > stage["ms_per_frame"]
