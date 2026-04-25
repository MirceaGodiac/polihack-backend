from __future__ import annotations

import secrets
import subprocess
import sys
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel

from ..config import settings

router = APIRouter(prefix="/admin", tags=["admin"])

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

_jobs: dict[str, dict] = {}


class BatchIngestResponse(BaseModel):
    run_id: str
    message: str
    sources_queued: int


class BatchStatusResponse(BaseModel):
    run_id: str
    status: str
    output: str | None = None
    error: str | None = None


def _require_admin(x_admin_secret: str | None) -> None:
    secret = settings.admin_ingest_secret
    if not secret:
        raise HTTPException(status_code=503, detail="ADMIN_INGEST_SECRET is not configured")
    if not x_admin_secret or not secrets.compare_digest(x_admin_secret, secret):
        raise HTTPException(status_code=401, detail="Invalid admin secret")


def _run_batch_task(run_id: str, write_debug: bool) -> None:
    cmd = [sys.executable, str(_REPO_ROOT / "scripts" / "run_batch_pipeline.py")]
    if write_debug:
        cmd.append("--write-debug")

    _jobs[run_id]["status"] = "running"
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT))
        if proc.returncode == 0:
            _jobs[run_id] = {"status": "completed", "output": proc.stdout}
        else:
            _jobs[run_id] = {"status": "failed", "output": proc.stdout, "error": proc.stderr}
    except Exception as exc:
        _jobs[run_id] = {"status": "failed", "error": str(exc)}


@router.post("/ingest/batch", response_model=BatchIngestResponse)
async def trigger_batch_ingest(
    background_tasks: BackgroundTasks,
    write_debug: bool = False,
    x_admin_secret: str | None = Header(default=None),
) -> BatchIngestResponse:
    """Trigger the batch ingestion pipeline for all URL sources. Protected by X-Admin-Secret."""
    _require_admin(x_admin_secret)

    sys.path.insert(0, str(_REPO_ROOT))
    from ingestion.batch import DEFAULT_SOURCES_FILE, load_url_sources

    url_sources = load_url_sources(DEFAULT_SOURCES_FILE)

    run_id = str(uuid.uuid4())[:8]
    _jobs[run_id] = {"status": "queued"}
    background_tasks.add_task(_run_batch_task, run_id, write_debug)

    return BatchIngestResponse(
        run_id=run_id,
        message="Batch ingestion started in the background",
        sources_queued=len(url_sources),
    )


@router.get("/ingest/batch/{run_id}", response_model=BatchStatusResponse)
async def get_batch_status(
    run_id: str,
    x_admin_secret: str | None = Header(default=None),
) -> BatchStatusResponse:
    """Check the status of a batch ingestion run."""
    _require_admin(x_admin_secret)
    job = _jobs.get(run_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return BatchStatusResponse(run_id=run_id, **job)
