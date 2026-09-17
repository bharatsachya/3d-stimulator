"""
The background worker.

WHY A JOB QUEUE AT ALL

The budget is ten seconds of processing for a ten-second video, and a reviewer
may upload something longer. Holding an HTTP request open for that is asking for
trouble: nginx's default proxy_read_timeout is 60s, browsers abandon fetches,
and when the connection dies every frame of completed work dies with it because
it only ever existed in that request's memory.

So POST returns 202 with a job id immediately, this worker runs detached, and
the client polls. The user watches a progress bar rather than a spinner that
eventually fails.

WHY THE PIPELINE RUNS IN A THREAD -- the part that differs from Assignment 1

Assignment 1's worker was I/O-bound: it awaited HTTP calls to a model server, so
the event loop was free between awaits. This pipeline is the opposite. Decode,
ORB, matching, PnP and bundle adjustment are CPU-bound synchronous code with no
await points anywhere. Calling it directly from a coroutine would block the
event loop for the entire run -- every poll from every client would hang until
the job finished, which defeats the point of polling.

`asyncio.to_thread` moves it to a worker thread. This works well specifically
because the heavy stages are OpenCV and NumPy, which release the GIL around
their native code, so the event loop really does get scheduled.

MEMORY

The video is read from disk inside the semaphore, not when it is accepted, so
peak memory is bounded by concurrency rather than by how many people upload at
once. With max_concurrent_jobs = 1 that is one video's frames at a time.
"""

from __future__ import annotations

import asyncio
import time

from app.config import settings
from app.schema import FailureReason, JobStatus, SlamFailure, SlamResult
from app.store import store
from app.uploads import SpooledVideo, cleanup_job_dir

# asyncio holds only a WEAK reference to tasks from create_task, so a task with
# no other reference can be garbage-collected mid-flight. Keeping the set is the
# documented way to prevent that.
_running: set[asyncio.Task] = set()

# One semaphore for the process. See config.max_concurrent_jobs for why it is 1.
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """
    Created lazily, on the running loop.

    A module-level Semaphore() would bind to whichever loop happened to be
    current at import time, which is not necessarily the loop that later runs
    the server -- a classic source of "attached to a different loop" errors.
    """
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)
    return _semaphore


def _run_slam(video_path: str, progress) -> tuple[SlamResult, dict, dict]:
    """
    Run the real pipeline.

    Imported inside the function rather than at module scope so that importing
    `app.jobs` -- which the API does at startup -- does not pull in OpenCV,
    NumPy and SciPy. That keeps the web layer's import graph independent of the
    algorithm's, which is the same separation vslam/ maintains by never
    importing FastAPI.
    """
    from vslam.export import to_slam_result
    from vslam.pipeline import run_pipeline

    result = run_pipeline(
        video_path,
        processed_fps=settings.processed_fps,
        working_width=settings.working_width,
        n_features=settings.n_features,
        max_duration_seconds=settings.max_duration_seconds,
        progress=progress,
    )
    return to_slam_result(result), result.timing, result.stats


async def run_job(job_id: str, video: SpooledVideo, job_dir: str) -> None:
    """
    Process one video. Never raises.

    This is a detached task, so an escaping exception would vanish into the
    event loop and leave the job stuck on "running" forever. Every exit path
    therefore ends in either finish() or fail().
    """
    try:
        async with _get_semaphore():
            await store.update(job_id, status=JobStatus.RUNNING, stage="starting")

            loop = asyncio.get_running_loop()

            def progress(stage: str, done: int, total: int) -> None:
                # Called from the worker THREAD. It must not touch the store
                # directly: the store's asyncio.Lock belongs to the event loop
                # and is not thread-safe. run_coroutine_threadsafe schedules the
                # update ON the loop and hands back a concurrent.futures.Future.
                #
                # That future is deliberately NOT awaited. Progress is advisory;
                # blocking the pipeline thread on the event loop just to record
                # "frame 37 of 100" would make the measurement worse than the
                # thing it measures.
                asyncio.run_coroutine_threadsafe(
                    store.set_progress(job_id, stage, done, total), loop
                )

            result, timing, stats = await asyncio.to_thread(
                _run_slam, video.path, progress
            )
            # The reconstruction statistics travel with the timing block, so the
            # results page can show what was measured alongside how long it took.
            timing = {**timing, "stats": stats}
            await store.finish(job_id, result, timing)

    except SlamFailure as failure:
        # An expected, diagnosable outcome -- not a bug. Recorded as a typed
        # reason the frontend turns into a sentence the user can act on.
        await store.fail(job_id, failure.reason, failure.detail)
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all, see docstring
        await store.fail(
            job_id, FailureReason.INTERNAL_ERROR, f"{type(exc).__name__}: {exc}"
        )
    finally:
        # Release the disk as soon as the job ends, not when the process exits.
        cleanup_job_dir(job_dir)


def schedule_job(job_id: str, video: SpooledVideo, job_dir: str) -> None:
    """Fire the worker into the background and return immediately."""
    task = asyncio.create_task(run_job(job_id, video, job_dir))
    _running.add(task)
    task.add_done_callback(_running.discard)
