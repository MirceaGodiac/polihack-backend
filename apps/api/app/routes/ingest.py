from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from typing import Optional
import subprocess
from pathlib import Path

router = APIRouter(prefix="/ingest", tags=["ingestion"])

class IngestRequest(BaseModel):
    url: HttpUrl
    law_id: str
    out_dir: Optional[str] = "ingestion/output/auto_ingest"

class IngestResponse(BaseModel):
    message: str
    job_id: str

def run_ingestion_task(url: str, law_id: str, out_dir: str):
    # This runs the script we just built
    cmd = [
        "python", "scripts/ingest_single.py",
        "--url", str(url),
        "--law-id", law_id,
        "--out-dir", out_dir
    ]
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
        request.out_dir
    )
    return {
        "message": "Ingestion job started in the background",
        "job_id": request.law_id
    }
