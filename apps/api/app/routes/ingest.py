from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from typing import Optional
import subprocess
import sys

router = APIRouter(prefix="/ingest", tags=["ingestion"])


class IngestRequest(BaseModel):
    url: HttpUrl
    law_id: str
    law_title: Optional[str] = None
    out_dir: Optional[str] = "ingestion/output/auto_ingest"


class IngestResponse(BaseModel):
    message: str
    job_id: str


def run_ingestion_task(url: str, law_id: str, out_dir: str, law_title: Optional[str] = None):
    # File-based canonical parser pipeline. This does not import into DB.
    cmd = [
        sys.executable, "scripts/run_parser_pipeline.py",
        "--url", str(url),
        "--law-id", law_id,
        "--out-dir", out_dir,
    ]
    if law_title:
        cmd.extend(["--law-title", law_title])
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully ingested {law_id}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to ingest {law_id}: {e}")

@router.post("/", response_model=IngestResponse)
async def trigger_ingestion(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Trigger a single-URL ingestion pipeline in the background.
    """
    background_tasks.add_task(
        run_ingestion_task,
        str(request.url),
        request.law_id,
        request.out_dir,
        request.law_title,
    )
    return {
        "message": "Ingestion job started in the background",
        "job_id": request.law_id,
    }
