from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

from ingestion.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCES_FILE = REPO_ROOT / "ingestion" / "sources" / "demo_sources.yaml"


def load_url_sources(sources_file: Path) -> list[dict[str, Any]]:
    with sources_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [s for s in (data or {}).get("sources", []) if s.get("url")]


def run_batch(sources_file: Path = DEFAULT_SOURCES_FILE, write_debug: bool = False) -> list[dict]:
    url_sources = load_url_sources(sources_file)
    if not url_sources:
        print("[batch-pipeline] No sources with a URL found in sources file.")
        return []

    print(f"[batch-pipeline] Processing {len(url_sources)} source(s)...")
    results: list[dict] = []

    for source in url_sources:
        law_id: str | None = source.get("law_id")
        url: str = source["url"]
        out_dir = REPO_ROOT / "ingestion" / "output" / (law_id or "unknown").replace(".", "_")

        print(f"[batch-pipeline] → {law_id} ({url})")
        try:
            result = run_pipeline(
                url=url,
                out_dir=out_dir,
                law_id=law_id,
                law_title=source.get("law_title"),
                status=source.get("status", "unknown"),
                write_debug=write_debug,
            )
            entry: dict = {
                "law_id": result.law_id,
                "law_title": result.law_title,
                "units": result.intermediate_units_count,
                "import_ready": result.import_blocking_passed,
                "status": "ok",
            }
            print(f"[batch-pipeline]   ✓ {result.law_id}: {result.intermediate_units_count} units, import_ready={result.import_blocking_passed}")
        except Exception as exc:
            entry = {"law_id": law_id, "status": "error", "error": str(exc)}
            print(f"[batch-pipeline]   ✗ {law_id}: {exc}", file=sys.stderr)

        results.append(entry)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"[batch-pipeline] Done: {ok}/{len(results)} succeeded.")
    return results
