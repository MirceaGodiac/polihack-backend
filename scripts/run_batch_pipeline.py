from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ingestion.batch import DEFAULT_SOURCES_FILE, run_batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the parser pipeline for all URL sources in the sources YAML.",
    )
    parser.add_argument(
        "--sources-file",
        default=str(DEFAULT_SOURCES_FILE),
        help="Path to sources YAML file (default: ingestion/sources/demo_sources.yaml)",
    )
    parser.add_argument(
        "--write-debug",
        action="store_true",
        help="Write cleaned lines, cleaner report, and intermediate units alongside each bundle",
    )
    args = parser.parse_args()

    results = run_batch(Path(args.sources_file), write_debug=args.write_debug)
    failed = [r for r in results if r.get("status") != "ok"]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
