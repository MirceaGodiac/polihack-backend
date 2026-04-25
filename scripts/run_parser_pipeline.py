from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ingestion.pipeline import DEFAULT_OUTPUT_DIR, run_pipeline


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the canonical LexAI parser pipeline for an HTML URL",
    )
    parser.add_argument("url", nargs="?", help="Source HTML URL")
    parser.add_argument("--url", dest="url_option", default=None, help="Source HTML URL")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for the canonical bundle",
    )
    parser.add_argument("--law-id", default=None, help="Optional canonical law id")
    parser.add_argument("--law-title", default=None, help="Optional legal act title")
    parser.add_argument("--source-id", default=None, help="Optional upstream source id")
    parser.add_argument(
        "--status",
        default="unknown",
        choices=["active", "historical", "repealed", "unknown"],
    )
    parser.add_argument("--parser-version", default="0.1.0")
    parser.add_argument("--generated-at", default=None)
    parser.add_argument(
        "--write-debug",
        action="store_true",
        help="Write cleaned lines, cleaner report, and structural intermediate units next to the bundle",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    url = args.url_option or args.url
    if not url:
        parser.error("an HTML URL is required")
    try:
        result = run_pipeline(
            url=url,
            out_dir=args.out_dir,
            law_id=args.law_id,
            law_title=args.law_title,
            source_id=args.source_id,
            status=args.status,
            parser_version=args.parser_version,
            generated_at=args.generated_at,
            write_debug=args.write_debug,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[parser-pipeline] {exc}", file=sys.stderr)
        sys.exit(2)

    print(f"law_id: {result.law_id}")
    print(f"law_title: {result.law_title}")
    print(f"intermediate_units_count: {result.intermediate_units_count}")
    if result.cleaner_warnings:
        print(f"cleaner_warnings: {', '.join(result.cleaner_warnings)}")
    for artifact, path in result.artifact_paths.items():
        print(f"{artifact}: {path}")

    if not result.import_blocking_passed:
        print(
            "import_blocking_passed=false; canonical bundle is not import-ready",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
