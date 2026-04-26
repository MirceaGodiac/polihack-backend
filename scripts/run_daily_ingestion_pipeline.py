from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ingestion.pipeline import run_pipeline
from ingestion.embeddings import (
    DeterministicFakeEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    build_embeddings_import_manifest,
    generate_embeddings,
    validate_embeddings_output,
    validate_embeddings_pair,
)


DEFAULT_SOURCES_PATH = REPO_ROOT / "config" / "legal_sources.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "ingestion" / "output"
DEFAULT_SUMMARY_DIR = REPO_ROOT / "ingestion" / "runs"
DEFAULT_SUMMARY_NAME = "run_summary.json"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:4b"
DEFAULT_EMBEDDING_DIM = 2560
DEFAULT_EMBEDDING_OUTPUT_NAME = "embeddings_output.jsonl"
EXIT_SUCCESS = 0
EXIT_SOURCE_FAILED = 1
EXIT_INVALID_CONFIG = 2
EXIT_VALIDATION_GATE_FAILED = 3
EXIT_EMBEDDINGS_FAILED = 4
EXIT_INTERNAL_ERROR = 5
VALID_STATUSES = {"active", "historical", "repealed", "unknown"}
REQUIRED_SOURCE_FIELDS = ("source_id", "law_id", "law_title", "source_url")
REQUIRED_BUNDLE_ARTIFACTS = (
    "legal_units.json",
    "legal_edges.json",
    "legal_chunks.json",
    "embeddings_input.jsonl",
    "corpus_manifest.json",
    "validation_report.json",
    "reference_candidates.json",
)


class PipelineStageError(RuntimeError):
    exit_code = EXIT_SOURCE_FAILED
    error_type = "source_failed"


class SourceProcessingError(PipelineStageError):
    exit_code = EXIT_SOURCE_FAILED
    error_type = "source_processing_failed"


class ValidationGateError(PipelineStageError):
    exit_code = EXIT_VALIDATION_GATE_FAILED
    error_type = "validation_gate_failed"


class EmbeddingsStageError(PipelineStageError):
    exit_code = EXIT_EMBEDDINGS_FAILED
    error_type = "embeddings_stage_failed"


def load_legal_sources(sources_path: str | Path) -> list[dict[str, Any]]:
    path = Path(sources_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"sources file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"sources file is not valid JSON: {path}: {exc.msg}") from exc

    if not isinstance(data, list):
        raise ValueError("sources file must contain a JSON array")

    sources: list[dict[str, Any]] = []
    for index, raw_source in enumerate(data):
        if not isinstance(raw_source, dict):
            raise ValueError(f"source[{index}] must be a JSON object")
        sources.append(_validate_source(raw_source, index=index))
    return sources


def run_daily_ingestion(
    *,
    sources_path: str | Path = DEFAULT_SOURCES_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    summary_dir: str | Path = DEFAULT_SUMMARY_DIR,
    summary_name: str = DEFAULT_SUMMARY_NAME,
    run_id: str | None = None,
    limit_sources: int | None = None,
    dry_run: bool = False,
    railway_job: bool = False,
    write_dry_run_summary: bool = False,
    write_debug: bool = False,
    max_attempts: int | None = None,
    retry_backoff_seconds: float = 5.0,
    retry_backoff_multiplier: float = 2.0,
    allow_partial_run: bool = False,
    with_embeddings: bool = False,
    embedding_provider: str = "fake",
    embedding_base_url: str | None = None,
    embedding_model: str | None = None,
    embedding_dim: int | str | None = None,
    embedding_batch_size: int = 4,
    embedding_limit: int | None = None,
    embedding_output_name: str = DEFAULT_EMBEDDING_OUTPUT_NAME,
    skip_validation_gate: bool = False,
    allow_partial_embeddings: bool = False,
    write_manifest: bool | None = None,
    sleep_func: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    resolved_run_id = run_id.strip() if run_id and run_id.strip() else generate_run_id(started)
    resolved_max_attempts = max_attempts if max_attempts is not None else (3 if railway_job else 1)
    if limit_sources is not None and limit_sources < 1:
        raise ValueError("limit_sources must be a positive integer")
    if resolved_max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds must be greater than or equal to 0")
    if retry_backoff_multiplier < 1:
        raise ValueError("retry_backoff_multiplier must be greater than or equal to 1")
    if not summary_name.strip():
        raise ValueError("summary_name must be non-empty")
    if embedding_batch_size <= 0:
        raise ValueError("embedding_batch_size must be a positive integer")
    if embedding_limit is not None and embedding_limit < 0:
        raise ValueError("embedding_limit must be greater than or equal to 0")
    if not embedding_output_name.strip():
        raise ValueError("embedding_output_name must be non-empty")
    if embedding_provider not in {"fake", "openai-compatible"}:
        raise ValueError("embedding_provider must be one of: fake, openai-compatible")

    resolved_embedding_model = _resolve_embedding_model(embedding_model)
    resolved_embedding_dim = _resolve_embedding_dim(embedding_dim)
    resolved_embedding_base_url = _resolve_embedding_base_url(embedding_base_url)
    if with_embeddings and embedding_provider == "openai-compatible" and not resolved_embedding_base_url:
        raise ValueError(
            "embedding_base_url is required for openai-compatible provider "
            "(or set EMBEDDING_BASE_URL)"
        )
    should_write_manifest = with_embeddings if write_manifest is None else write_manifest

    sources = load_legal_sources(sources_path)
    enabled_sources = [source for source in sources if source.get("enabled", True)]
    selected_sources = enabled_sources[:limit_sources] if limit_sources else enabled_sources
    root = Path(output_root)
    output_dirs = [
        str(_output_dir_for_source(source, root)) for source in selected_sources
    ]

    summary: dict[str, Any] = {
        "run_id": resolved_run_id,
        "started_at": _iso_z(started),
        "finished_at": None,
        "duration_seconds": None,
        "railway_job": railway_job,
        "dry_run": dry_run,
        "partial_success": False,
        "sources_total": len(sources),
        "sources_enabled": len(enabled_sources),
        "sources_succeeded": 0,
        "sources_failed": 0,
        "embeddings_sources_succeeded": 0,
        "embeddings_sources_failed": 0,
        "output_dirs": output_dirs,
        "sources": [],
        "errors": [],
        "warnings": [],
        "summary_path": None,
        "exit_code": None,
        "embedding_config": {
            "with_embeddings": with_embeddings,
            "provider": embedding_provider,
            "base_url": resolved_embedding_base_url,
            "model": resolved_embedding_model,
            "dim": resolved_embedding_dim,
            "batch_size": embedding_batch_size,
            "limit": embedding_limit,
            "output_name": embedding_output_name,
            "allow_partial_embeddings": allow_partial_embeddings,
            "write_manifest": should_write_manifest,
        },
        "retry_config": {
            "max_attempts": resolved_max_attempts,
            "retry_backoff_seconds": retry_backoff_seconds,
            "retry_backoff_multiplier": retry_backoff_multiplier,
            "allow_partial_run": allow_partial_run,
        },
    }

    for source in selected_sources:
        summary["sources"].append(
            _new_source_summary(source, _output_dir_for_source(source, root), resolved_run_id)
        )

    if dry_run:
        _finalize_summary(summary, started=started, allow_partial_run=allow_partial_run)
        if railway_job or write_dry_run_summary:
            _write_run_summary(
                summary,
                summary_dir=summary_dir,
                summary_name=summary_name,
            )
        return summary

    for source, source_summary in zip(selected_sources, summary["sources"]):
        out_dir = _output_dir_for_source(source, root)
        for attempt in range(1, resolved_max_attempts + 1):
            source_summary["attempts_count"] = attempt
            try:
                _process_source_once(
                    source=source,
                    source_summary=source_summary,
                    summary=summary,
                    out_dir=out_dir,
                    write_debug=write_debug,
                    skip_validation_gate=skip_validation_gate,
                    with_embeddings=with_embeddings,
                    embedding_provider=embedding_provider,
                    embedding_base_url=resolved_embedding_base_url,
                    embedding_model=resolved_embedding_model,
                    embedding_dim=resolved_embedding_dim,
                    embedding_batch_size=embedding_batch_size,
                    embedding_limit=embedding_limit,
                    embedding_output_name=embedding_output_name,
                    allow_partial_embeddings=allow_partial_embeddings,
                    write_manifest=should_write_manifest,
                )
            except SourceProcessingError as exc:
                _record_attempt_error(source_summary, attempt=attempt, exc=exc)
                if attempt < resolved_max_attempts:
                    delay = retry_backoff_seconds * (retry_backoff_multiplier ** (attempt - 1))
                    _add_warnings(
                        summary,
                        source_summary,
                        [f"retrying_source_after_attempt_{attempt}"],
                    )
                    if delay:
                        sleep_func(delay)
                    continue
                _mark_source_failed(summary, source_summary, source, exc)
            except PipelineStageError as exc:
                _record_attempt_error(source_summary, attempt=attempt, exc=exc)
                _mark_source_failed(summary, source_summary, source, exc)
            except Exception as exc:
                internal_error = PipelineStageError(_short_error_message(exc))
                internal_error.exit_code = EXIT_INTERNAL_ERROR
                internal_error.error_type = "unexpected_internal_error"
                _record_attempt_error(source_summary, attempt=attempt, exc=internal_error)
                _mark_source_failed(summary, source_summary, source, internal_error)
            else:
                source_summary["status"] = "succeeded"
            break

    summary["sources_succeeded"] = sum(
        1 for source_summary in summary["sources"] if source_summary["status"] == "succeeded"
    )
    summary["sources_failed"] = sum(
        1 for source_summary in summary["sources"] if source_summary["status"] == "failed"
    )
    if with_embeddings:
        summary["embeddings_sources_succeeded"] = sum(
            1
            for source_summary in summary["sources"]
            if source_summary["status"] == "succeeded"
            and source_summary["embeddings_generated"]
            and source_summary["embeddings_validation_passed"]
            and source_summary["pair_validation_passed"]
        )
        summary["embeddings_sources_failed"] = sum(
            1
            for source_summary in summary["sources"]
            if source_summary["status"] == "failed"
            and source_summary["artifact_check_passed"]
            and source_summary["validation_gate_passed"]
        )

    _finalize_summary(summary, started=started, allow_partial_run=allow_partial_run)
    _write_run_summary(
        summary,
        summary_dir=summary_dir,
        summary_name=summary_name,
    )
    return summary


def sanitize_law_id(law_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", law_id.strip()).strip("_")
    if not sanitized:
        raise ValueError("law_id cannot be sanitized to an output directory name")
    return sanitized


def _validate_source(source: dict[str, Any], *, index: int) -> dict[str, Any]:
    validated = dict(source)
    for field in REQUIRED_SOURCE_FIELDS:
        value = validated.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"source[{index}].{field} must be a non-empty string")
        validated[field] = value.strip()

    status = validated.get("status", "unknown")
    if not isinstance(status, str) or status not in VALID_STATUSES:
        raise ValueError(
            f"source[{index}].status must be one of: {', '.join(sorted(VALID_STATUSES))}"
        )
    validated["status"] = status

    enabled = validated.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"source[{index}].enabled must be a boolean")
    validated["enabled"] = enabled

    parsed_url = urlparse(validated["source_url"])
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        raise ValueError(f"source[{index}].source_url must be an http(s) URL")

    sanitize_law_id(validated["law_id"])
    return validated


def _output_dir_for_source(source: dict[str, Any], output_root: Path) -> Path:
    return output_root / sanitize_law_id(source["law_id"])


def _new_source_summary(source: dict[str, Any], out_dir: Path, run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "source_id": source["source_id"],
        "law_id": source["law_id"],
        "law_title": source["law_title"],
        "source_url": source["source_url"],
        "status": "planned",
        "out_dir": str(out_dir),
        "artifact_check_passed": False,
        "validation_gate_passed": False,
        "embeddings_generated": False,
        "embeddings_output_path": None,
        "embeddings_written_count": 0,
        "embeddings_validation_passed": False,
        "pair_validation_passed": False,
        "manifest_path": None,
        "attempts_count": 0,
        "attempt_errors": [],
        "exit_code": None,
        "warnings": [],
    }


def _process_source_once(
    *,
    source: dict[str, Any],
    source_summary: dict[str, Any],
    summary: dict[str, Any],
    out_dir: Path,
    write_debug: bool,
    skip_validation_gate: bool,
    with_embeddings: bool,
    embedding_provider: str,
    embedding_base_url: str | None,
    embedding_model: str,
    embedding_dim: int,
    embedding_batch_size: int,
    embedding_limit: int | None,
    embedding_output_name: str,
    allow_partial_embeddings: bool,
    write_manifest: bool,
) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        run_pipeline(
            url=source["source_url"],
            out_dir=out_dir,
            law_id=source["law_id"],
            law_title=source["law_title"],
            source_id=source["source_id"],
            status=source.get("status", "unknown"),
            write_debug=write_debug,
        )
        _check_bundle_artifacts_or_raise(out_dir)
        source_summary["artifact_check_passed"] = True
    except Exception as exc:
        raise SourceProcessingError(_short_error_message(exc)) from exc

    try:
        validation_warnings = _apply_validation_gate(
            out_dir,
            skip_validation_gate=skip_validation_gate,
        )
        source_summary["validation_gate_passed"] = True
        _add_warnings(summary, source_summary, validation_warnings)
    except Exception as exc:
        raise ValidationGateError(_short_error_message(exc)) from exc

    if not with_embeddings:
        return

    try:
        _run_embedding_stage(
            out_dir=out_dir,
            source_summary=source_summary,
            summary=summary,
            embedding_provider=embedding_provider,
            embedding_base_url=embedding_base_url,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            embedding_batch_size=embedding_batch_size,
            embedding_limit=embedding_limit,
            embedding_output_name=embedding_output_name,
            allow_partial_embeddings=allow_partial_embeddings,
            write_manifest=write_manifest,
        )
    except Exception as exc:
        raise EmbeddingsStageError(_short_error_message(exc)) from exc


def _check_bundle_artifacts_or_raise(out_dir: Path) -> None:
    missing = [
        artifact
        for artifact in REQUIRED_BUNDLE_ARTIFACTS
        if not (out_dir / artifact).is_file()
    ]
    if missing:
        raise RuntimeError("missing required bundle artifacts: " + ", ".join(missing))


def _apply_validation_gate(
    out_dir: Path,
    *,
    skip_validation_gate: bool,
) -> list[str]:
    validation_report_path = out_dir / "validation_report.json"
    try:
        validation_report = json.loads(validation_report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("validation_report.json is not valid JSON") from exc
    if not isinstance(validation_report, dict):
        raise RuntimeError("validation_report.json must contain a JSON object")

    if "import_blocking_passed" not in validation_report:
        return ["validation_report_missing_import_blocking_passed"]
    if validation_report["import_blocking_passed"] is False:
        warning = "validation_gate_skipped_import_blocking_passed_false"
        if skip_validation_gate:
            return [warning]
        raise RuntimeError("validation_report import_blocking_passed=false")
    return []


def _run_embedding_stage(
    *,
    out_dir: Path,
    source_summary: dict[str, Any],
    summary: dict[str, Any],
    embedding_provider: str,
    embedding_base_url: str | None,
    embedding_model: str,
    embedding_dim: int,
    embedding_batch_size: int,
    embedding_limit: int | None,
    embedding_output_name: str,
    allow_partial_embeddings: bool,
    write_manifest: bool,
) -> None:
    input_path = out_dir / "embeddings_input.jsonl"
    output_path = out_dir / embedding_output_name
    provider = _build_embedding_provider(
        embedding_provider,
        embedding_base_url=embedding_base_url,
        embedding_dim=embedding_dim,
    )
    embedding_summary = generate_embeddings(
        input_path=input_path,
        output_path=output_path,
        provider=provider,
        model_name=embedding_model,
        batch_size=embedding_batch_size,
        expected_dim=embedding_dim,
        resume=True,
        dry_run=False,
        limit=embedding_limit,
    )
    source_summary["embeddings_generated"] = True
    source_summary["embeddings_output_path"] = str(output_path)
    source_summary["embeddings_written_count"] = embedding_summary.written_count
    _add_warnings(summary, source_summary, embedding_summary.warnings)

    output_validation = validate_embeddings_output(
        output_path,
        expected_model=embedding_model,
        expected_dim=embedding_dim,
        strict=True,
    )
    source_summary["embeddings_validation_passed"] = True
    _add_warnings(summary, source_summary, output_validation.warnings)

    require_all_inputs = embedding_limit is None and not allow_partial_embeddings
    pair_validation = validate_embeddings_pair(
        input_path,
        output_path,
        expected_model=embedding_model,
        expected_dim=embedding_dim,
        require_all_inputs=require_all_inputs,
        strict=True,
    )
    source_summary["pair_validation_passed"] = True
    _add_warnings(summary, source_summary, pair_validation.warnings)

    if not write_manifest:
        return
    if embedding_limit is not None or allow_partial_embeddings:
        _add_warnings(
            summary,
            source_summary,
            ["manifest_skipped_for_partial_embeddings"],
        )
        return

    manifest_path = out_dir / "validated_embeddings_manifest.json"
    build_embeddings_import_manifest(
        input_path=input_path,
        output_path=output_path,
        expected_model=embedding_model,
        expected_dim=embedding_dim,
        manifest_path=manifest_path,
    )
    source_summary["manifest_path"] = str(manifest_path)


def _build_embedding_provider(
    provider_name: str,
    *,
    embedding_base_url: str | None,
    embedding_dim: int,
):
    if provider_name == "fake":
        return DeterministicFakeEmbeddingProvider(dimension=embedding_dim)
    if not embedding_base_url:
        raise ValueError("embedding_base_url is required for openai-compatible provider")
    return OpenAICompatibleEmbeddingProvider(
        base_url=embedding_base_url,
        timeout_seconds=120.0,
    )


def _mark_source_failed(
    summary: dict[str, Any],
    source_summary: dict[str, Any],
    source: dict[str, Any],
    exc: PipelineStageError,
) -> None:
    error = _short_error_message(exc)
    source_summary["status"] = "failed"
    source_summary["exit_code"] = exc.exit_code
    summary["errors"].append(
        {
            "source_id": source["source_id"],
            "law_id": source["law_id"],
            "error": error,
            "error_type": exc.error_type,
            "exit_code": exc.exit_code,
        }
    )


def _add_warnings(
    summary: dict[str, Any],
    source_summary: dict[str, Any],
    warnings: list[str],
) -> None:
    for warning in warnings:
        source_warning = f"{source_summary['source_id']}: {warning}"
        source_summary["warnings"].append(warning)
        summary["warnings"].append(source_warning)


def _record_attempt_error(
    source_summary: dict[str, Any],
    *,
    attempt: int,
    exc: PipelineStageError,
) -> None:
    source_summary["attempt_errors"].append(
        {
            "attempt": attempt,
            "error_type": exc.error_type,
            "error": _short_error_message(exc),
            "exit_code": exc.exit_code,
        }
    )


def _finalize_summary(
    summary: dict[str, Any],
    *,
    started: datetime,
    allow_partial_run: bool,
) -> None:
    finished = datetime.now(timezone.utc)
    summary["finished_at"] = _iso_z(finished)
    summary["duration_seconds"] = round((finished - started).total_seconds(), 3)
    summary["exit_code"] = _summary_exit_code(summary, allow_partial_run=allow_partial_run)


def _summary_exit_code(summary: dict[str, Any], *, allow_partial_run: bool) -> int:
    if summary["sources_failed"] == 0:
        return EXIT_SUCCESS
    if allow_partial_run and summary["sources_succeeded"] > 0:
        summary["partial_success"] = True
        warning = "partial_success=true"
        if warning not in summary["warnings"]:
            summary["warnings"].append(warning)
        return EXIT_SUCCESS

    exit_codes = {error.get("exit_code") for error in summary["errors"]}
    if EXIT_EMBEDDINGS_FAILED in exit_codes:
        return EXIT_EMBEDDINGS_FAILED
    if EXIT_VALIDATION_GATE_FAILED in exit_codes:
        return EXIT_VALIDATION_GATE_FAILED
    if EXIT_INTERNAL_ERROR in exit_codes:
        return EXIT_INTERNAL_ERROR
    return EXIT_SOURCE_FAILED


def _write_run_summary(
    summary: dict[str, Any],
    *,
    summary_dir: str | Path,
    summary_name: str,
) -> Path:
    summary_path = Path(summary_dir) / summary["run_id"] / summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def generate_run_id(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    timestamp = current.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"daily_ingestion_{timestamp}_{secrets.token_hex(3)}"


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _short_error_message(exc: BaseException, *, max_length: int = 300) -> str:
    message = " ".join(str(exc).split()) or exc.__class__.__name__
    if len(message) <= max_length:
        return message
    return message[: max_length - 3].rstrip() + "..."


def _error_summary(args: argparse.Namespace, exc: BaseException, *, exit_code: int) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    run_id = args.run_id.strip() if getattr(args, "run_id", None) else generate_run_id(started)
    summary = {
        "run_id": run_id,
        "started_at": _iso_z(started),
        "finished_at": _iso_z(started),
        "duration_seconds": 0.0,
        "railway_job": bool(getattr(args, "railway_job", False)),
        "dry_run": bool(getattr(args, "dry_run", False)),
        "partial_success": False,
        "sources_total": 0,
        "sources_enabled": 0,
        "sources_succeeded": 0,
        "sources_failed": 1,
        "embeddings_sources_succeeded": 0,
        "embeddings_sources_failed": 0,
        "output_dirs": [],
        "sources": [],
        "errors": [
            {
                "error": _short_error_message(exc),
                "error_type": "invalid_config" if exit_code == EXIT_INVALID_CONFIG else "unexpected_internal_error",
                "exit_code": exit_code,
            }
        ],
        "warnings": [],
        "summary_path": None,
        "embedding_config": {},
        "retry_config": {},
        "exit_code": exit_code,
    }
    should_write_summary = (
        bool(getattr(args, "railway_job", False))
        or not bool(getattr(args, "dry_run", False))
        or bool(getattr(args, "write_dry_run_summary", False))
    )
    if should_write_summary:
        _write_run_summary(
            summary,
            summary_dir=getattr(args, "summary_dir", DEFAULT_SUMMARY_DIR),
            summary_name=getattr(args, "summary_name", DEFAULT_SUMMARY_NAME),
        )
    return summary


def _resolve_embedding_model(value: str | None) -> str:
    model = (value or os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL).strip()
    if not model:
        raise ValueError("embedding_model must be non-empty")
    return model


def _resolve_embedding_dim(value: int | str | None) -> int:
    raw_value = value if value is not None else os.getenv("EMBEDDING_DIM", str(DEFAULT_EMBEDDING_DIM))
    try:
        dim = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding_dim must be an integer") from exc
    if dim <= 0:
        raise ValueError("embedding_dim must be a positive integer")
    return dim


def _resolve_embedding_base_url(value: str | None) -> str | None:
    base_url = (value or os.getenv("EMBEDDING_BASE_URL") or "").strip()
    return base_url or None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run daily LexAI canonical ingestion for explicit legal sources."
    )
    parser.add_argument(
        "--sources",
        default=str(DEFAULT_SOURCES_PATH),
        help="Path to legal sources JSON file",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for canonical parser bundles",
    )
    parser.add_argument(
        "--summary-dir",
        default=str(DEFAULT_SUMMARY_DIR),
        help="Directory where run summaries are written",
    )
    parser.add_argument(
        "--summary-name",
        default=DEFAULT_SUMMARY_NAME,
        help="Run summary filename inside the run_id directory",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id; generated automatically when omitted",
    )
    parser.add_argument(
        "--limit-sources",
        type=int,
        default=None,
        help="Process at most this many enabled sources",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate source config and print planned output dirs without scraping",
    )
    parser.add_argument(
        "--write-dry-run-summary",
        action="store_true",
        help="Persist run_summary.json even for dry-runs",
    )
    parser.add_argument(
        "--railway-job",
        action="store_true",
        help="Run in Railway-safe job mode with strict exit codes and persisted summary",
    )
    parser.add_argument(
        "--write-debug",
        action="store_true",
        help="Write parser debug files next to each canonical bundle",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Max attempts per source; defaults to 1 locally and 3 with --railway-job",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=5.0,
        help="Initial retry backoff in seconds",
    )
    parser.add_argument(
        "--retry-backoff-multiplier",
        type=float,
        default=2.0,
        help="Retry backoff multiplier",
    )
    parser.add_argument(
        "--allow-partial-run",
        action="store_true",
        help="Exit 0 when at least one enabled source succeeds and others fail",
    )
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Generate and validate embeddings after each parser bundle",
    )
    parser.add_argument(
        "--embedding-provider",
        default="fake",
        choices=["fake", "openai-compatible"],
        help="Embedding provider to use when --with-embeddings is set",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=None,
        help="OpenAI-compatible embedding base URL; defaults to EMBEDDING_BASE_URL",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model; defaults to EMBEDDING_MODEL or qwen3-embedding:4b",
    )
    parser.add_argument(
        "--embedding-dim",
        default=None,
        help="Expected embedding dimension; defaults to EMBEDDING_DIM or 2560",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=4,
        help="Embedding batch size",
    )
    parser.add_argument(
        "--embedding-limit",
        type=int,
        default=None,
        help="Generate embeddings for at most this many candidate input records",
    )
    parser.add_argument(
        "--embedding-output-name",
        default=DEFAULT_EMBEDDING_OUTPUT_NAME,
        help="Filename for embeddings output JSONL inside each source output dir",
    )
    parser.add_argument(
        "--skip-validation-gate",
        action="store_true",
        help="Continue when validation_report import_blocking_passed=false",
    )
    parser.add_argument(
        "--allow-partial-embeddings",
        action="store_true",
        help="Allow pair validation to pass when some inputs have no embedding output",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        default=None,
        help="Write import-readiness manifest; defaults to true with --with-embeddings",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        summary = run_daily_ingestion(
            sources_path=args.sources,
            output_root=args.output_root,
            summary_dir=args.summary_dir,
            summary_name=args.summary_name,
            run_id=args.run_id,
            limit_sources=args.limit_sources,
            dry_run=args.dry_run,
            railway_job=args.railway_job,
            write_dry_run_summary=args.write_dry_run_summary,
            write_debug=args.write_debug,
            max_attempts=args.max_attempts,
            retry_backoff_seconds=args.retry_backoff_seconds,
            retry_backoff_multiplier=args.retry_backoff_multiplier,
            allow_partial_run=args.allow_partial_run,
            with_embeddings=args.with_embeddings,
            embedding_provider=args.embedding_provider,
            embedding_base_url=args.embedding_base_url,
            embedding_model=args.embedding_model,
            embedding_dim=args.embedding_dim,
            embedding_batch_size=args.embedding_batch_size,
            embedding_limit=args.embedding_limit,
            embedding_output_name=args.embedding_output_name,
            skip_validation_gate=args.skip_validation_gate,
            allow_partial_embeddings=args.allow_partial_embeddings,
            write_manifest=args.write_manifest,
        )
    except ValueError as exc:
        summary = _error_summary(args, exc, exit_code=EXIT_INVALID_CONFIG)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        sys.exit(EXIT_INVALID_CONFIG)
    except Exception as exc:
        summary = _error_summary(args, exc, exit_code=EXIT_INTERNAL_ERROR)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        sys.exit(EXIT_INTERNAL_ERROR)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    sys.exit(int(summary["exit_code"]))


if __name__ == "__main__":
    main()
