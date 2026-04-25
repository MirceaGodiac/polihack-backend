"""
manifest.py
-----------
Generates corpus_manifest.json, the DB-importer entry-point for a batch run.

The manifest records:
  - batch_id         : deterministic slug from timestamp + corpus name
  - generated_at     : ISO-8601 UTC timestamp
  - sources          : list of source dicts from the YAML config
  - output_dir       : the output directory path (relative to repo root)
  - files            : relative paths to all generated output artefacts
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_manifest(
    batch_id: str,
    sources: list[dict[str, Any]],
    out_dir: str | Path,
) -> dict[str, Any]:
    """
    Construct the manifest dict.

    Args:
        batch_id  : stable identifier for this run (e.g. "demo_corpus_v1").
        sources   : list of source dicts loaded from YAML.
        out_dir   : the output directory (Path or string).
    """
    out_dir = Path(out_dir)

    manifest = {
        "batch_id": batch_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0.0",
        "sources": [
            {
                "law_id":       s.get("law_id"),
                "law_title":    s.get("law_title"),
                "legal_domain": s.get("legal_domain"),
            }
            for s in sources
        ],
        "output_dir": str(out_dir),
        "files": {
            "legal_units":        str(out_dir / "legal_units.json"),
            "legal_edges":        str(out_dir / "legal_edges.json"),
            "validation_report":  str(out_dir / "validation_report.json"),
            "corpus_manifest":    str(out_dir / "corpus_manifest.json"),
        },
    }
    return manifest


def save_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    """Serialize the manifest to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
