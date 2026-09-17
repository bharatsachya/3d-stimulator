"""
Stage timing.

This module exists before anything it measures, on purpose. The brief requires
"measured processing time", the client's stated interest is performance under
constrained hardware, and retrofitting instrumentation means guessing at which
boundaries you wish you had measured. So the timer is written first and every
stage of the pipeline reports through it.

WHAT IT MEASURES

A stage is a named span of work -- decode, orb, match, pnp, triangulate, ba.
Each is entered many times (usually once per frame), so for each stage we keep:

    total elapsed, and how many times it was entered

From those two numbers plus the frame count, everything the README needs falls
out: total ms, mean ms per call, ms per frame, and share of wall clock.

WHY WALL CLOCK IS TRACKED SEPARATELY

The stages are summed, but the run is *also* timed end to end, and the report
shows the difference as `unaccounted`. That row is the honest one. If the stages
add to 6s and the wall clock says 9s, three seconds are going somewhere
unmeasured -- JSON serialization, a file write, an accidental copy -- and a
report that silently omitted them would be a report that lies by construction.

WHY perf_counter

`time.perf_counter` is monotonic and the highest resolution clock available.
`time.time` can step backwards when NTP adjusts the system clock, which turns a
timing measurement into a negative number at random.

THREAD SAFETY

None, deliberately. One StageTimer belongs to one job, and a job is processed by
one task. Sharing a timer across threads would need a lock, and a lock in the
hot path of a per-frame measurement would distort the thing being measured.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class StageStats:
    """Accumulated timing for one named stage."""

    total_seconds: float = 0.0
    calls: int = 0

    @property
    def total_ms(self) -> float:
        return self.total_seconds * 1000.0

    @property
    def mean_ms(self) -> float:
        """Mean cost of one entry into this stage."""
        if self.calls == 0:
            return 0.0
        return self.total_ms / self.calls


class StageTimer:
    """
    Accumulates per-stage timings for a single run.

    Usage:

        timer = StageTimer()
        with timer.run():                 # starts the wall clock
            for frame in frames:
                with timer.stage("orb"):
                    keypoints = detect(frame)
        report = timer.report(frames=len(frames))

    `stage()` is re-entrant across iterations -- entering "orb" a hundred times
    accumulates into one row with calls=100.
    """

    def __init__(self) -> None:
        self._stages: dict[str, StageStats] = {}
        self._wall_start: float | None = None
        self._wall_seconds: float = 0.0
        # Insertion order is preserved by dict, and stages are first entered in
        # pipeline order, so the report comes out in the order work happens
        # rather than alphabetically. That makes it readable as a narrative.

    @contextmanager
    def run(self):
        """Time the whole run. Everything else is measured inside this."""
        self._wall_start = time.perf_counter()
        try:
            yield self
        finally:
            self._wall_seconds = time.perf_counter() - self._wall_start

    @contextmanager
    def stage(self, name: str):
        """
        Time one entry into a named stage.

        The `finally` matters: a stage that raises still records the time it
        burned before failing. A failed job's timings are exactly when you most
        want to know where the time went.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            stats = self._stages.get(name)
            if stats is None:
                stats = self._stages[name] = StageStats()
            stats.total_seconds += elapsed
            stats.calls += 1

    def add(self, name: str, seconds: float) -> None:
        """
        Record a stage timed elsewhere.

        Needed where the work is not a clean block -- e.g. decode time measured
        inside a generator that yields frames one at a time.
        """
        stats = self._stages.get(name)
        if stats is None:
            stats = self._stages[name] = StageStats()
        stats.total_seconds += seconds
        stats.calls += 1

    @property
    def wall_ms(self) -> float:
        """Wall clock for the run. Live if still running, final once closed."""
        if self._wall_seconds:
            return self._wall_seconds * 1000.0
        if self._wall_start is not None:
            return (time.perf_counter() - self._wall_start) * 1000.0
        return 0.0

    def report(self, frames: int = 0) -> dict:
        """
        Build the JSON-serializable breakdown returned with a job and rendered
        on the results page.

        `frames` is the number of frames actually *processed* (after fps
        subsampling), not the number in the source video. ms/frame against the
        source count would flatter us by a factor of three.
        """
        wall = self.wall_ms
        stages = []
        measured = 0.0

        for name, stats in self._stages.items():
            measured += stats.total_ms
            stages.append(
                {
                    "stage": name,
                    "total_ms": round(stats.total_ms, 2),
                    "calls": stats.calls,
                    "mean_ms": round(stats.mean_ms, 3),
                    # Per processed frame. For a stage that runs once per frame
                    # this equals mean_ms; for one that runs only on keyframes
                    # it is much smaller, which is the honest way to compare a
                    # keyframe-only cost against a per-frame one.
                    "ms_per_frame": round(stats.total_ms / frames, 3) if frames else None,
                    "pct_of_wall": round(100.0 * stats.total_ms / wall, 1) if wall else None,
                }
            )

        # See the module docstring: this row is the point.
        unaccounted = wall - measured

        return {
            "wall_ms": round(wall, 2),
            "frames": frames,
            "ms_per_frame": round(wall / frames, 2) if frames else None,
            "stages": stages,
            "unaccounted_ms": round(unaccounted, 2),
            "unaccounted_pct": round(100.0 * unaccounted / wall, 1) if wall else None,
        }

    def format_table(self, frames: int = 0) -> str:
        """Plain-text table, for probe.py and the sweep runner."""
        report = self.report(frames=frames)
        lines = [
            f"{'stage':<14} {'total ms':>10} {'calls':>7} {'mean ms':>9} "
            f"{'ms/frame':>9} {'% wall':>7}",
            "-" * 60,
        ]
        for row in report["stages"]:
            lines.append(
                f"{row['stage']:<14} {row['total_ms']:>10.1f} {row['calls']:>7} "
                f"{row['mean_ms']:>9.2f} "
                f"{(row['ms_per_frame'] if row['ms_per_frame'] is not None else 0):>9.2f} "
                f"{(row['pct_of_wall'] if row['pct_of_wall'] is not None else 0):>7.1f}"
            )
        lines.append("-" * 60)
        lines.append(
            f"{'unaccounted':<14} {report['unaccounted_ms']:>10.1f} "
            f"{'':>7} {'':>9} {'':>9} "
            f"{(report['unaccounted_pct'] if report['unaccounted_pct'] is not None else 0):>7.1f}"
        )
        lines.append(
            f"{'WALL':<14} {report['wall_ms']:>10.1f} {'':>7} {'':>9} "
            f"{(report['ms_per_frame'] if report['ms_per_frame'] is not None else 0):>9.2f} "
            f"{100.0:>7.1f}"
        )
        return "\n".join(lines)
