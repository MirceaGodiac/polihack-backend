from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running as `python scripts/generate_embeddings.py` from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ingestion.embeddings import (  # noqa: E402
    DeterministicFakeEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    generate_embeddings,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate chunk-based embedding output JSONL from embeddings_input.jsonl"
    )
    parser.add_argument("--input", required=True, help="Path to embeddings_input.jsonl")
    parser.add_argument("--output", required=True, help="Path to embeddings_output.jsonl")
    parser.add_argument(
        "--provider",
        default="fake",
        choices=["fake", "openai-compatible"],
        help="Embedding provider to use",
    )
    parser.add_argument("--model", required=True, help="Embedding model name")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--expected-dim", type=int, default=None)
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key-env", default=None, help="Environment variable holding the API key")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    provider = _build_provider(args, parser)
    try:
        summary = generate_embeddings(
            input_path=Path(args.input),
            output_path=Path(args.output),
            provider=provider,
            model_name=args.model,
            batch_size=args.batch_size,
            expected_dim=args.expected_dim,
            resume=args.resume,
            dry_run=args.dry_run,
            limit=args.limit,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[generate-embeddings] {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


def _build_provider(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if args.provider == "fake":
        return DeterministicFakeEmbeddingProvider(
            dimension=args.expected_dim if args.expected_dim is not None else 1024
        )

    if args.provider != "openai-compatible":
        parser.error(f"unsupported provider: {args.provider}")

    if not args.base_url or not str(args.base_url).strip():
        parser.error("base URL missing: --base-url is required for openai-compatible provider")

    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env, "").strip()
        if not api_key:
            parser.error(f"env var missing or empty: {args.api_key_env}")

    if args.dry_run:
        return DeterministicFakeEmbeddingProvider(
            dimension=args.expected_dim if args.expected_dim is not None else 1024
        )

    return OpenAICompatibleEmbeddingProvider(
        base_url=args.base_url,
        api_key=api_key,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
