"""
Configuration.

Everything tunable lives here and reads from the environment, so the same code
runs on a laptop and on the EC2 instance without edits. That matters more than
usual for this project: five of these values are the parameters CLAUDE.md says
must be tuned by measurement, and the sweep runner drives them through exactly
this mechanism rather than by editing source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # --- Upload limits ---------------------------------------------------
    # 25 MB. A 10-second phone clip is a few MB, and anything larger is
    # downsampled to the working resolution before a single feature is
    # detected -- so the extra bytes buy nothing and cost upload time and disk.
    max_upload_bytes: int = _int("MAX_UPLOAD_BYTES", 25 * 1024 * 1024)
    # Checked after decode opens the file, because container metadata is the
    # only place duration lives and we do not trust the client to report it.
    max_duration_seconds: float = _float("MAX_DURATION_SECONDS", 30.0)

    # --- The five parameters that get swept ------------------------------
    # Starting points, not conclusions. See docs/measurements.md.
    processed_fps: float = _float("PROCESSED_FPS", 10.0)
    working_width: int = _int("WORKING_WIDTH", 640)
    n_features: int = _int("N_FEATURES", 1000)
    ba_window: int = _int("BA_WINDOW", 5)
    ba_every_n_keyframes: int = _int("BA_EVERY_N_KEYFRAMES", 1)

    # --- Concurrency ------------------------------------------------------
    # ONE. This is not timidity, it is the whole performance argument.
    #
    # The box has 2 vCPUs and the measured workload already stops scaling past
    # 2 threads, so a single job can use the machine effectively on its own.
    # Running two jobs concurrently would not make either finish sooner; it
    # would make both miss the 10-second budget, and the budget is the
    # deliverable. A second uploader waits a few seconds instead.
    max_concurrent_jobs: int = _int("MAX_CONCURRENT_JOBS", 1)

    # --- Storage ----------------------------------------------------------
    # Where uploads spool. Empty means the system temp directory.
    upload_dir: str = os.getenv("UPLOAD_DIR", "")
    # Completed jobs retained in memory. The store is a dict, so this is what
    # stops a long-lived process growing without bound.
    max_jobs_retained: int = _int("MAX_JOBS_RETAINED", 50)


settings = Settings()
