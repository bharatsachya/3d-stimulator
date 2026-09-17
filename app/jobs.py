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


def _stub_pipeline(video_path: str, progress) -> tuple[SlamResult, dict]:
    """
    STAGE 0 PLACEHOLDER. Replaced by the real pipeline at stage 7.

    It exists so the whole path -- upload, spool, 202, background task, polling,
    progress, render -- can be deployed and proven end to end before any SLAM
    code is written. If deployment is going to fail on a security group or a
    missing wheel, it should fail today, not on the deadline.

    It fabricates a small helix of poses and a ring of points so the viewer has
    something to draw. The numbers are meaningless and the result is labelled
    as a stub.
    """
    import math

    total = 40
    poses, frame_indices, points, observations = [], [], [], []

    for i in range(total):
        progress("stub", i, total)
        time.sleep(0.02)  # stand in for per-frame work

        angle = i * 0.15
        # 4x4 camera-to-world. Y-up already, as the real export will be.
        pose = [
            [math.cos(angle), 0.0, math.sin(angle), math.sin(angle) * 2.0],
            [0.0, 1.0, 0.0, i * 0.02],
            [-math.sin(angle), 0.0, math.cos(angle), math.cos(angle) * 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        poses.append(pose)
        frame_indices.append(i * 3)

    for i in range(300):
        angle = i * 0.21
        radius = 3.0 + (i % 7) * 0.1
        points.append(
            [math.cos(angle) * radius, (i % 40) * 0.05 - 1.0, math.sin(angle) * radius]
        )
        observations.append(2 + (i % 5))

    result = SlamResult(
        poses=poses,
        frame_indices=frame_indices,
        keyframe_indices=list(range(0, total, 5)),
        points=points,
        observations=observations,
        flags=[],
        flag_messages=["STUB RESULT - the SLAM pipeline is not wired in yet."],
    )
    timing = {
        "wall_ms": 800.0,
        "frames": total,
        "ms_per_frame": 20.0,
        "stages": [],
        "unaccounted_ms": 0.0,
        "unaccounted_pct": 0.0,
        "stub": True,
    }
    return result, timing


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

            result, timing = await asyncio.to_thread(
                _stub_pipeline, video.path, progress
            )
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
