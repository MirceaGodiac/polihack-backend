from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ingestion.imports import DEFAULT_EMBEDDING_DIM, build_import_plan


EXIT_VALID = 0
EXIT_INVALID_BUNDLE = 1
EXIT_UNSUPPORTED = 2
EXIT_INTERNAL_ERROR = 5


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a LexAI DB import plan from canonical ingestion artifacts.",
    )
    parser.add_argument("--source-dir", required=True, help="Canonical bundle output directory")
    parser.add_argument("--with-embeddings", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument(
        "--mode",
        choices=["validate_only", "dry_run", "apply"],
        default="validate_only",
    )
    parser.add_argument("--output", default=None, help="Optional path to write import_plan.json")
    parser.add_argument("--fail-on-orphan-edges", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode == "apply":
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "apply mode is reserved for H08 Phase D2/D3",
                },
                ensure_ascii=False,
                indent=2 if args.pretty else None,
            ),
            file=sys.stderr,
        )
        return EXIT_UNSUPPORTED

    try:
        plan = build_import_plan(
            args.source_dir,
            with_embeddings=args.with_embeddings,
            mode=args.mode,
            embedding_dim=args.embedding_dim,
            fail_on_orphan_edges=args.fail_on_orphan_edges,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_UNSUPPORTED
    except Exception as exc:
        print(f"unexpected import planner error: {exc}", file=sys.stderr)
        return EXIT_INTERNAL_ERROR

    payload = plan.model_dump()
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2 if args.pretty else None,
        separators=None if args.pretty else (",", ":"),
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return EXIT_VALID if plan.safe_for_db_import else EXIT_INVALID_BUNDLE


if __name__ == "__main__":
    sys.exit(main())
