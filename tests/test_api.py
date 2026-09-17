"""
API tests.

A NOTE ON TestClient AND BACKGROUND WORK -- worth reading before adding tests.

Starlette's TestClient creates its event-loop portal per request. A detached
`asyncio.create_task` therefore only advances while a request is in flight, so a
job scheduled by POST /api/jobs never reliably reaches "done" under TestClient,
no matter how many times you poll. This was hit for real during stage 0: the job
sat at 39/40 frames forever under TestClient and completed in 0.96s when the
worker was driven directly.

So the split here is deliberate:
  * TestClient covers the synchronous request/response contract.
  * The worker is exercised directly on an event loop, where it behaves.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.schema import FailureReason, JobStatus
from app.store import store
from app.uploads import SpooledVideo
from app import jobs


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_reports_settings(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    # The README quotes timings taken at specific settings, so /health has to
    # report what is actually in force or the claim cannot be verified.
    assert body["settings"]["working_width"] == settings.working_width
    assert body["settings"]["n_features"] == settings.n_features


def test_unknown_job_is_404_not_403(client: TestClient) -> None:
    """
    404 is a security property here, not a formality.

    There is no authentication: the job id IS the capability. A 403 would
    confirm that an id exists, which is exactly the fact worth withholding.
    """
    response = client.get("/api/jobs/definitely-not-a-real-id")
    assert response.status_code == 404


def test_upload_over_limit_is_rejected(client: TestClient) -> None:
    oversized = b"\0" * (settings.max_upload_bytes + 1024)
    response = client.post(
        "/api/jobs", files={"video": ("big.mp4", oversized, "video/mp4")}
    )
    assert response.status_code == 413
    # The message must name the limit, so the user knows what to do about it.
    assert "MB" in response.json()["detail"]


def test_empty_upload_is_rejected(client: TestClient) -> None:
    response = client.post("/api/jobs", files={"video": ("x.mp4", b"", "video/mp4")})
    assert response.status_code == 400


def test_accepted_upload_returns_202_and_a_poll_url(client: TestClient) -> None:
    response = client.post(
        "/api/jobs", files={"video": ("clip.mp4", b"\0" * 2048, "video/mp4")}
    )
    assert response.status_code == 202
    body = response.json()
    assert body["poll"] == f"/api/jobs/{body['job_id']}"
    # Unguessable: secrets.token_urlsafe(24) is 32 characters.
    assert len(body["job_id"]) >= 32


def test_job_ids_are_unique_and_unguessable(client: TestClient) -> None:
    ids = {
        client.post(
            "/api/jobs", files={"video": ("c.mp4", b"\0" * 1024, "video/mp4")}
        ).json()["job_id"]
        for _ in range(5)
    }
    assert len(ids) == 5


def test_undecodable_video_fails_with_a_typed_reason() -> None:
    """
    A file that is not a video must produce a diagnosed failure, never a 500
    and never a crashed background task.

    This test previously asserted that the same input SUCCEEDED, because the
    stage-0 stub never opened the file. Wiring up the real pipeline correctly
    broke it -- which is the test doing its job.
    """

    async def run() -> None:
        job = await store.create("not-a-video.mp4", 2048)
        await jobs.run_job(
            job.id,
            SpooledVideo(
                path="/dev/null", display_name="not-a-video.mp4", size_bytes=2048
            ),
            "/tmp/slam-test-dir-that-does-not-exist",
        )
        finished = await store.get(job.id)
        assert finished is not None
        assert finished.status is JobStatus.FAILED
        # Diagnosed, not an internal error: we know exactly what went wrong.
        assert finished.failure_reason is FailureReason.DECODE_FAILED
        # And the user gets a sentence, not a traceback.
        assert finished.summary()["failure_message"]

    asyncio.run(asyncio.wait_for(run(), timeout=60))


@pytest.mark.skipif(
    not (Path(__file__).resolve().parent.parent / "samples" / "synth_dolly.mp4").is_file(),
    reason="sample clips are generated, not committed; run tools/make_synthetic.py",
)
def test_worker_reconstructs_a_real_clip() -> None:
    """End to end on a real video, driving the worker directly."""
    sample = Path(__file__).resolve().parent.parent / "samples" / "synth_dolly.mp4"

    async def run() -> None:
        job = await store.create(sample.name, sample.stat().st_size)
        await jobs.run_job(
            job.id,
            SpooledVideo(
                path=str(sample), display_name=sample.name, size_bytes=sample.stat().st_size
            ),
            "/tmp/slam-test-dir-that-does-not-exist",
        )
        finished = await store.get(job.id)
        assert finished is not None, "job vanished"
        assert finished.status is JobStatus.DONE, finished.failure_detail
        assert finished.result is not None
        assert len(finished.result.poses) > 20
        assert len(finished.result.points) > 100
        # A finished job must not display 97%.
        assert finished.frames_done == finished.frames_total
        # Units are never metres, whatever else changes.
        assert finished.result.to_dict()["units"] == "arbitrary"

    asyncio.run(asyncio.wait_for(run(), timeout=180))
