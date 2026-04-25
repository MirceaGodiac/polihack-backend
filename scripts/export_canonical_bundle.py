from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ingestion.exporters import export_canonical_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a canonical parser bundle from structural intermediate units"
    )
    parser.add_argument("--input", required=True, help="Path to intermediate legal_units.json")
    parser.add_argument("--out-dir", required=True, help="Output directory for canonical bundle")
    parser.add_argument("--law-id", required=True, help="Canonical law id, e.g. ro.codul_muncii")
    parser.add_argument("--law-title", required=True, help="Law title, e.g. Codul muncii")
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--status", default="unknown")
    parser.add_argument("--parser-version", default="0.1.0")
    parser.add_argument("--generated-at", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8") as f:
        legacy_units = json.load(f)

    generated_at = args.generated_at or datetime.now(timezone.utc).isoformat()
    metadata = {
        "law_id": args.law_id,
        "law_title": args.law_title,
        "source_id": args.source_id,
        "source_url": args.source_url,
        "status": args.status,
    }
    paths = export_canonical_bundle(
        legacy_units,
        metadata,
        args.out_dir,
        parser_version=args.parser_version,
        generated_at=generated_at,
        input_files=[str(input_path)],
    )

    for artifact, path in paths.items():
        print(f"{artifact}: {path}")

    validation_report_path = paths["validation_report"]
    validation_report = json.loads(validation_report_path.read_text(encoding="utf-8"))
    if validation_report.get("import_blocking_passed") is False:
        print(
            "import_blocking_passed=false; canonical bundle is not import-ready",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
