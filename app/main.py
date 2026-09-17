"""
The HTTP surface.

Four endpoints and a static mount. Deliberately small: this file's job is to
accept a video, hand it to the worker, and report progress. All the interesting
work lives in vslam/, which knows nothing about HTTP.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.jobs import schedule_job
from app.store import store
from app.uploads import UploadTooLarge, cleanup_job_dir, make_job_dir, spool_video

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Monocular sparse SLAM",
    description=(
        "Upload a video, get the estimated camera trajectory and a sparse 3D "
        "point cloud, with measured per-stage timings."
    ),
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict:
    """
    Liveness plus the configuration actually in force.

    Echoing the tunables matters more here than in a typical service: the
    README quotes timings measured at specific settings, and this is how you
    confirm the deployed instance is running the settings you think it is.
    """
    return {
        "status": "ok",
        "jobs_in_memory": await store.count(),
        "settings": {
            "processed_fps": settings.processed_fps,
            "working_width": settings.working_width,
            "n_features": settings.n_features,
            "ba_window": settings.ba_window,
            "ba_every_n_keyframes": settings.ba_every_n_keyframes,
            "max_concurrent_jobs": settings.max_concurrent_jobs,
            "max_upload_mb": settings.max_upload_bytes // (1024 * 1024),
            "max_duration_seconds": settings.max_duration_seconds,
        },
    }


@app.post("/api/jobs", status_code=202)
async def create_job(video: UploadFile = File(...)) -> JSONResponse:
    """
    Accept a video and start processing it in the background.

    Returns 202 Accepted -- "I have taken this work, it is not finished" --
    which is the honest status for asynchronous work, rather than 200 ("here is
    your result") or 201 ("a resource now exists at this URL").

    This returns in milliseconds regardless of clip length. Progress comes from
    GET /api/jobs/{job_id}.
    """
    job_dir = make_job_dir()
    try:
        spooled = await spool_video(video, job_dir)
    except UploadTooLarge as exc:
        cleanup_job_dir(job_dir)
        # 413 is the specific status for this, and the message names the limit
        # so the user knows what to do rather than just that they failed.
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except Exception:
        cleanup_job_dir(job_dir)
        raise

    if spooled.size_bytes == 0:
        cleanup_job_dir(job_dir)
        raise HTTPException(status_code=400, detail="the uploaded file was empty")

    job = await store.create(spooled.display_name, spooled.size_bytes)
    schedule_job(job.id, spooled, job_dir)

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job.id,
            "status": job.status.value,
            "poll": f"/api/jobs/{job.id}",
        },
    )


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, summary: bool = False) -> dict:
    """
    Poll a job.

    `?summary=true` returns progress only. The full result carries thousands of
    3D points, and re-sending them on every one-second poll would waste most of
    the bandwidth of the exchange. The client polls the summary and fetches the
    full result exactly once, when the status turns to done.

    An unknown id returns 404. There is no authentication here -- the id itself
    is the capability, 32 bytes of randomness from secrets.token_urlsafe -- so
    the thing to avoid is confirming which ids exist. 404 for both "never
    existed" and "not yours" gives an enumerator nothing.
    """
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
    return job.summary() if summary else job.to_dict()


# The frontend. Mounted last so it cannot shadow an /api route.
#
# html=True makes StaticFiles serve index.html at the mount root, which is why
# there is no hand-written route for "/".
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
