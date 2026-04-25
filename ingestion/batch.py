from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

from ingestion.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCES_FILE = REPO_ROOT / "ingestion" / "sources" / "demo_sources.yaml"

logger = logging.getLogger("ingestion.batch")


def load_url_sources(sources_file: Path) -> list[dict[str, Any]]:
    logger.info("Reading sources from: %s", sources_file.resolve())
    with sources_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    all_sources = (data or {}).get("sources", [])
    url_sources = [s for s in all_sources if s.get("url")]
    logger.info("Found %d source(s), %d with a URL.", len(all_sources), len(url_sources))
    return url_sources


def run_batch(sources_file: Path = DEFAULT_SOURCES_FILE, write_debug: bool = False) -> list[dict]:
    url_sources = load_url_sources(sources_file)
    if not url_sources:
        logger.warning("No sources with a URL found in sources file.")
        return []

    logger.info("Processing %d source(s)...", len(url_sources))
    results: list[dict] = []

    for source in url_sources:
        law_id: str | None = source.get("law_id")
        url: str = source["url"]
        out_dir = REPO_ROOT / "ingestion" / "output" / (law_id or "unknown").replace(".", "_")

        logger.info("→ %s (%s)", law_id, url)
        started = time.monotonic()
        try:
            result = run_pipeline(
                url=url,
                out_dir=out_dir,
                law_id=law_id,
                law_title=source.get("law_title"),
                status=source.get("status", "unknown"),
                write_debug=write_debug,
            )
            elapsed = time.monotonic() - started
            entry: dict = {
                "law_id": result.law_id,
                "law_title": result.law_title,
                "units": result.intermediate_units_count,
                "import_ready": result.import_blocking_passed,
                "elapsed_sec": round(elapsed, 2),
                "status": "ok",
            }
            logger.info(
                "  ✓ %s: %d units, import_ready=%s (%.2fs)",
                result.law_id,
                result.intermediate_units_count,
                result.import_blocking_passed,
                elapsed,
            )
        except Exception as exc:
            elapsed = time.monotonic() - started
            entry = {
                "law_id": law_id,
                "url": url,
                "status": "error",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
                "elapsed_sec": round(elapsed, 2),
            }
            logger.error(
                "  ✗ %s (%s): %s: %s (%.2fs)",
                law_id,
                url,
                type(exc).__name__,
                exc,
                elapsed,
                exc_info=True,
            )

        results.append(entry)

    ok = sum(1 for r in results if r["status"] == "ok")
    logger.info("Done: %d/%d succeeded.", ok, len(results))
    return results
