from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/validate_embeddings_output.py` from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ingestion.embeddings import (  # noqa: E402
    EmbeddingOutputValidationSummary,
    validate_embeddings_output,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate chunk-based embeddings_output.jsonl before vector import"
    )
    parser.add_argument("--output", required=True, help="Path to embeddings_output.jsonl")
    parser.add_argument("--expected-model", default=None)
    parser.add_argument("--expected-dim", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--require-unique-record-ids", action="store_true")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        summary = validate_embeddings_output(
            Path(args.output),
            expected_model=args.expected_model,
            expected_dim=args.expected_dim,
            require_unique_record_ids=args.require_unique_record_ids,
            strict=False,
        )
    except (OSError, ValueError) as exc:
        print(f"[validate-embeddings-output] {exc}", file=sys.stderr)
        sys.exit(1)

    if args.as_json:
        print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))
    else:
        print(_format_human_report(summary))

    if args.strict and summary.errors:
        sys.exit(1)


def _format_human_report(summary: EmbeddingOutputValidationSummary) -> str:
    lines = [
        "Embeddings output validation",
        f"read_count: {summary.read_count}",
        f"valid_count: {summary.valid_count}",
        f"invalid_count: {summary.invalid_count}",
        f"duplicate_resume_key_count: {summary.duplicate_resume_key_count}",
        f"duplicate_record_id_count: {summary.duplicate_record_id_count}",
        f"empty_metadata_count: {summary.empty_metadata_count}",
        f"model_names: {', '.join(summary.model_names) if summary.model_names else '-'}",
        f"embedding_dims: {', '.join(str(dim) for dim in summary.embedding_dims) if summary.embedding_dims else '-'}",
        f"law_ids: {', '.join(summary.law_ids) if summary.law_ids else '-'}",
    ]
    if summary.warnings:
        lines.append("warnings:")
        lines.extend(f"- {_truncate_message(message)}" for message in summary.warnings[:5])
        if len(summary.warnings) > 5:
            lines.append(f"- ... {len(summary.warnings) - 5} more")
    if summary.errors:
        lines.append("errors:")
        lines.extend(f"- {_truncate_message(message)}" for message in summary.errors[:5])
        if len(summary.errors) > 5:
            lines.append(f"- ... {len(summary.errors) - 5} more")
    return "\n".join(lines)


def _truncate_message(message: str, limit: int = 240) -> str:
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


if __name__ == "__main__":
    main()
