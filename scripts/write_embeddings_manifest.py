from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ingestion.embeddings import (  # noqa: E402
    EmbeddingsImportReadinessManifest,
    build_embeddings_import_manifest,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write an offline embeddings import-readiness manifest"
    )
    parser.add_argument("--input", required=True, help="Path to embeddings_input.jsonl")
    parser.add_argument("--output", required=True, help="Path to embeddings_output.jsonl")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON")
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--expected-dim", type=int, required=True)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        manifest = build_embeddings_import_manifest(
            input_path=Path(args.input),
            output_path=Path(args.output),
            expected_model=args.expected_model,
            expected_dim=args.expected_dim,
            manifest_path=Path(args.manifest),
        )
    except (OSError, ValueError) as exc:
        print(f"[write-embeddings-manifest] {exc}", file=sys.stderr)
        sys.exit(1)

    if args.as_json:
        print(json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2))
    else:
        print(_format_human_report(Path(args.manifest), manifest))


def _format_human_report(
    manifest_path: Path,
    manifest: EmbeddingsImportReadinessManifest,
) -> str:
    return "\n".join(
        [
            "Embeddings import-readiness manifest",
            f"manifest_path: {manifest_path}",
            f"ready_for_pgvector_import: {manifest.ready_for_pgvector_import}",
            f"model_name: {manifest.model_name}",
            f"embedding_dim: {manifest.embedding_dim}",
            f"matched_output_count: {manifest.matched_output_count}",
            f"law_ids: {', '.join(manifest.law_ids) if manifest.law_ids else '-'}",
        ]
    )


if __name__ == "__main__":
    main()
