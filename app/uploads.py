"""
Receiving the uploaded video safely.

Two concerns, both about not trusting the client. Ported from Assignment 1,
where the same two apply for the same reasons.

1. SPOOL TO DISK, WITH THE SIZE LIMIT ENFORCED *DURING* THE STREAM.
   The obvious `raw = await file.read()` pulls the whole file into memory before
   its size can be checked -- so a 2 GB upload is already resident by the time
   you could reject it. Reading in chunks and counting as we go aborts an
   oversized upload after about a megabyte, and never holds more than one chunk.

2. NEVER TRUST THE CLIENT-SUPPLIED FILENAME.
   `file.filename` is attacker-controlled. A name like "../../etc/cron.d/x"
   would, with naive path joining, write outside the upload directory. So we
   generate our own random on-disk name and keep the original purely as a label
   that never touches a filesystem path.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass

from fastapi import UploadFile

from app.config import settings

# 1 MB chunks: large enough to be efficient, small enough that the size check
# fires almost immediately on a huge upload.
CHUNK_SIZE = 1024 * 1024


class UploadTooLarge(Exception):
    """The upload exceeded max_upload_bytes."""


@dataclass
class SpooledVideo:
    """An accepted upload waiting on disk for the worker."""

    path: str          # our generated path -- never client-derived
    display_name: str  # the client's filename, for display ONLY
    size_bytes: int


def make_job_dir() -> str:
    parent = settings.upload_dir or None
    if parent:
        os.makedirs(parent, exist_ok=True)
    return tempfile.mkdtemp(prefix="slam-", dir=parent)


def cleanup_job_dir(path: str) -> None:
    """Delete a job's directory. Safe to call twice."""
    shutil.rmtree(path, ignore_errors=True)


async def spool_video(file: UploadFile, job_dir: str) -> SpooledVideo:
    """
    Stream the upload to disk, enforcing the size cap as we read.

    Raises UploadTooLarge if it exceeds the limit, removing the partial file so
    a rejected upload leaves nothing behind.
    """
    display_name = os.path.basename(file.filename or "upload.mp4")

    # Keep the original extension: OpenCV's VideoCapture picks its demuxer
    # partly from it, and a .bin suffix makes some containers fail to open.
    _, extension = os.path.splitext(display_name)
    if extension.lower() not in {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}:
        extension = ".mp4"

    destination = os.path.join(job_dir, f"{uuid.uuid4().hex}{extension}")

    size = 0
    limit = settings.max_upload_bytes
    try:
        with open(destination, "wb") as handle:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise UploadTooLarge(
                        f"{display_name} exceeds the "
                        f"{limit // (1024 * 1024)} MB limit"
                    )
                handle.write(chunk)
    except UploadTooLarge:
        try:
            os.remove(destination)
        except OSError:
            pass
        raise

    return SpooledVideo(path=destination, display_name=display_name, size_bytes=size)
