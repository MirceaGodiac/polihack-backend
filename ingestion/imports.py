from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


ImportMode = Literal["validate_only", "dry_run", "apply"]
IMPORT_PLAN_VERSION = "h08.phase_d1.import_plan.v1"
DEFAULT_EMBEDDING_DIM = 2560
EMBEDDINGS_MANIFEST_FILENAME = "validated_embeddings_manifest.json"


class ImportArtifactPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_dir: str
    legal_units: str | None = None
    legal_edges: str | None = None
    reference_candidates: str | None = None
    legal_chunks: str | None = None
    embeddings_input: str | None = None
    embeddings_output: str | None = None
    validation_report: str | None = None
    corpus_manifest: str | None = None
    embeddings_manifest: str | None = None


class ImportPlanWarning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    artifact: str | None = None


class ImportPlanError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    artifact: str | None = None


class ImportPlanCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    legal_units: int = 0
    legal_edges: int = 0
    embedding_records: int = 0
    manifest_sources: int = 0


class ImportPlanSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_dir: str
    law_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)


class ImportBundleValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    validation_report_passed: bool = False
    embeddings_manifest_present: bool = False
    pair_validation_assumed_from_manifest: bool = False
    warnings: list[ImportPlanWarning] = Field(default_factory=list)
    errors: list[ImportPlanError] = Field(default_factory=list)


class EmbeddingImportRecordView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str
    chunk_id: str
    legal_unit_id: str
    law_id: str
    model_name: str
    embedding_dim: int
    text_hash: str
    metadata_present: bool


class ImportPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    import_run_id: str
    source_dir: str
    mode: ImportMode
    created_at: str
    artifact_paths: ImportArtifactPaths
    counts: ImportPlanCounts
    sources: list[ImportPlanSource] = Field(default_factory=list)
    validation: ImportBundleValidationResult
    idempotency: dict[str, str]
    warnings: list[ImportPlanWarning] = Field(default_factory=list)
    errors: list[ImportPlanError] = Field(default_factory=list)
    safe_for_db_import: bool
    plan_version: str = IMPORT_PLAN_VERSION


class ImportRepositoryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempted: int = 0
    inserted: int = 0
    updated: int = 0
    skipped: int = 0


class ImportRepository(Protocol):
    def upsert_legal_units(
        self,
        records: Iterable[Mapping[str, Any]],
    ) -> ImportRepositoryResult:
        ...

    def upsert_legal_edges(
        self,
        records: Iterable[Mapping[str, Any]],
    ) -> ImportRepositoryResult:
        ...

    def upsert_embeddings(
        self,
        records: Iterable[Mapping[str, Any]],
    ) -> ImportRepositoryResult:
        ...

    def record_import_run(self, plan: ImportPlan) -> ImportRepositoryResult:
        ...

    def finalize_import_run(self, plan: ImportPlan) -> ImportRepositoryResult:
        ...


class NullImportRepository:
    """D1 repository skeleton. It never writes to DB."""

    def upsert_legal_units(
        self,
        records: Iterable[Mapping[str, Any]],
    ) -> ImportRepositoryResult:
        return ImportRepositoryResult(attempted=_iter_count(records), skipped=0)

    def upsert_legal_edges(
        self,
        records: Iterable[Mapping[str, Any]],
    ) -> ImportRepositoryResult:
        return ImportRepositoryResult(attempted=_iter_count(records), skipped=0)

    def upsert_embeddings(
        self,
        records: Iterable[Mapping[str, Any]],
    ) -> ImportRepositoryResult:
        return ImportRepositoryResult(attempted=_iter_count(records), skipped=0)

    def record_import_run(self, plan: ImportPlan) -> ImportRepositoryResult:
        return ImportRepositoryResult(attempted=1, skipped=0)

    def finalize_import_run(self, plan: ImportPlan) -> ImportRepositoryResult:
        return ImportRepositoryResult(attempted=1, skipped=0)


DryRunImportRepository = NullImportRepository


def build_import_plan(
    source_dir: str | Path,
    *,
    with_embeddings: bool = False,
    mode: ImportMode = "validate_only",
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    fail_on_orphan_edges: bool = False,
    import_run_id: str | None = None,
) -> ImportPlan:
    if mode not in {"validate_only", "dry_run", "apply"}:
        raise ValueError("mode must be validate_only, dry_run, or apply")
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be greater than 0")

    artifacts = discover_import_artifacts(source_dir)
    validation = validate_import_bundle(
        source_dir,
        with_embeddings=with_embeddings,
        embedding_dim=embedding_dim,
        fail_on_orphan_edges=fail_on_orphan_edges,
        artifact_paths=artifacts,
    )
    counts, sources = load_safe_counts(
        artifacts,
        with_embeddings=with_embeddings,
        embedding_dim=embedding_dim,
    )
    safe_for_db_import = validation.passed and mode in {"validate_only", "dry_run"}

    return ImportPlan(
        import_run_id=import_run_id or f"import_plan_{uuid.uuid4().hex[:12]}",
        source_dir=str(Path(source_dir)),
        mode=mode,
        created_at=_utc_now_iso_z(),
        artifact_paths=artifacts,
        counts=counts,
        sources=sources,
        validation=validation,
        idempotency={
            "legal_units_key": "id",
            "legal_edges_key": "id",
            "embeddings_key": "record_id + model_name + text_hash",
        },
        warnings=validation.warnings,
        errors=validation.errors,
        safe_for_db_import=safe_for_db_import,
    )


def validate_import_bundle(
    source_dir: str | Path,
    *,
    with_embeddings: bool = False,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    fail_on_orphan_edges: bool = False,
    artifact_paths: ImportArtifactPaths | None = None,
) -> ImportBundleValidationResult:
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be greater than 0")
    artifacts = artifact_paths or discover_import_artifacts(source_dir)
    warnings: list[ImportPlanWarning] = []
    errors: list[ImportPlanError] = []
    source_path = Path(source_dir)

    if not source_path.exists() or not source_path.is_dir():
        errors.append(_error("source_dir_missing", "source_dir does not exist", str(source_path)))
        return _validation_result(
            validation_report_passed=False,
            embeddings_manifest_present=False,
            pair_validation_assumed_from_manifest=False,
            warnings=warnings,
            errors=errors,
        )

    for field_name in ("legal_units", "legal_edges", "validation_report", "corpus_manifest"):
        if getattr(artifacts, field_name) is None:
            errors.append(
                _error(
                    "missing_required_artifact",
                    f"missing required artifact: {field_name}",
                    field_name,
                )
            )

    for field_name in ("reference_candidates", "legal_chunks"):
        if getattr(artifacts, field_name) is None:
            warnings.append(
                _warning(
                    "missing_optional_artifact",
                    f"missing optional artifact: {field_name}",
                    field_name,
                )
            )

    validation_report_passed = _validate_report(artifacts, errors, warnings)
    legal_units = _read_json_list(artifacts.legal_units, "legal_units", errors)
    legal_edges = _read_json_list(artifacts.legal_edges, "legal_edges", errors)
    unit_ids = _validate_unique_ids(
        legal_units,
        "legal_units",
        errors,
        required=True,
    )
    _validate_unique_ids(legal_edges, "legal_edges", errors, required=True)
    _validate_orphan_edges(
        legal_edges,
        unit_ids,
        warnings,
        errors,
        fail_on_orphan_edges=fail_on_orphan_edges,
    )

    embeddings_manifest_present = artifacts.embeddings_manifest is not None
    pair_validation_assumed = False
    if with_embeddings:
        for field_name in ("embeddings_input", "embeddings_output", "embeddings_manifest"):
            if getattr(artifacts, field_name) is None:
                errors.append(
                    _error(
                        "missing_required_embedding_artifact",
                        f"missing required embedding artifact: {field_name}",
                        field_name,
                    )
                )
        _validate_embeddings_manifest(
            artifacts,
            expected_dim=embedding_dim,
            errors=errors,
        )
        pair_validation_assumed = embeddings_manifest_present
        _validate_embedding_output(
            artifacts.embeddings_output,
            expected_dim=embedding_dim,
            errors=errors,
        )

    return _validation_result(
        validation_report_passed=validation_report_passed,
        embeddings_manifest_present=embeddings_manifest_present,
        pair_validation_assumed_from_manifest=pair_validation_assumed,
        warnings=warnings,
        errors=errors,
    )


def load_safe_counts(
    artifact_paths: ImportArtifactPaths,
    *,
    with_embeddings: bool = False,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> tuple[ImportPlanCounts, list[ImportPlanSource]]:
    legal_units = _read_json_list_no_errors(artifact_paths.legal_units)
    legal_edges = _read_json_list_no_errors(artifact_paths.legal_edges)
    embedding_records = (
        _count_embedding_records(artifact_paths.embeddings_output, expected_dim=embedding_dim)
        if with_embeddings
        else 0
    )
    manifest_sources, source_ids, source_urls = _source_metadata_from_manifest(
        artifact_paths.corpus_manifest
    )
    law_ids = sorted(
        {
            str(unit.get("law_id") or "").strip()
            for unit in legal_units
            if isinstance(unit, dict) and str(unit.get("law_id") or "").strip()
        }
    )
    if not law_ids:
        law_ids = _law_ids_from_embedding_manifest(artifact_paths.embeddings_manifest)

    counts = ImportPlanCounts(
        legal_units=len(legal_units),
        legal_edges=len(legal_edges),
        embedding_records=embedding_records,
        manifest_sources=manifest_sources,
    )
    sources = [
        ImportPlanSource(
            source_dir=artifact_paths.source_dir,
            law_ids=law_ids,
            source_ids=source_ids,
            source_urls=source_urls,
        )
    ]
    return counts, sources


def discover_import_artifacts(source_dir: str | Path) -> ImportArtifactPaths:
    source_path = Path(source_dir)

    def existing(filename: str) -> str | None:
        path = source_path / filename
        return str(path) if path.is_file() else None

    return ImportArtifactPaths(
        source_dir=str(source_path),
        legal_units=existing("legal_units.json"),
        legal_edges=existing("legal_edges.json"),
        reference_candidates=existing("reference_candidates.json"),
        legal_chunks=existing("legal_chunks.json"),
        embeddings_input=existing("embeddings_input.jsonl"),
        embeddings_output=existing("embeddings_output.jsonl"),
        validation_report=existing("validation_report.json"),
        corpus_manifest=existing("corpus_manifest.json"),
        embeddings_manifest=existing(EMBEDDINGS_MANIFEST_FILENAME),
    )


def _validate_report(
    artifacts: ImportArtifactPaths,
    errors: list[ImportPlanError],
    warnings: list[ImportPlanWarning],
) -> bool:
    if artifacts.validation_report is None:
        return False
    payload = _read_json_object(artifacts.validation_report, "validation_report", errors)
    if payload is None:
        return False
    if "import_blocking_passed" not in payload:
        warnings.append(
            _warning(
                "validation_report_missing_import_blocking_passed",
                "validation_report missing import_blocking_passed",
                "validation_report",
            )
        )
        return False
    if payload.get("import_blocking_passed") is not True:
        errors.append(
            _error(
                "validation_report_failed",
                "validation_report import_blocking_passed is not true",
                "validation_report",
            )
        )
        return False
    return True


def _validate_unique_ids(
    records: list[dict[str, Any]],
    artifact: str,
    errors: list[ImportPlanError],
    *,
    required: bool,
) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    missing_count = 0
    for record in records:
        value = record.get("id")
        if not isinstance(value, str) or not value.strip():
            missing_count += 1
            continue
        record_id = value.strip()
        if record_id in seen:
            duplicates.add(record_id)
        seen.add(record_id)
    if missing_count and required:
        errors.append(
            _error(
                f"{artifact}_missing_id",
                f"{artifact} contains {missing_count} record(s) without non-empty id",
                artifact,
            )
        )
    if duplicates:
        errors.append(
            _error(
                f"{artifact}_duplicate_id",
                f"{artifact} contains duplicate id(s): {', '.join(sorted(duplicates)[:5])}",
                artifact,
            )
        )
    return seen


def _validate_orphan_edges(
    legal_edges: list[dict[str, Any]],
    unit_ids: set[str],
    warnings: list[ImportPlanWarning],
    errors: list[ImportPlanError],
    *,
    fail_on_orphan_edges: bool,
) -> None:
    orphan_count = 0
    for edge in legal_edges:
        source_id = edge.get("source_id")
        target_id = edge.get("target_id")
        if source_id not in unit_ids or target_id not in unit_ids:
            orphan_count += 1
    if not orphan_count:
        return
    message = f"legal_edges contains {orphan_count} orphan edge(s)"
    if fail_on_orphan_edges:
        errors.append(_error("orphan_legal_edges", message, "legal_edges"))
    else:
        warnings.append(_warning("orphan_legal_edges", message, "legal_edges"))


def _validate_embeddings_manifest(
    artifacts: ImportArtifactPaths,
    *,
    expected_dim: int,
    errors: list[ImportPlanError],
) -> None:
    if artifacts.embeddings_manifest is None:
        return
    payload = _read_json_object(artifacts.embeddings_manifest, "embeddings_manifest", errors)
    if payload is None:
        return
    if payload.get("ready_for_pgvector_import") is not True:
        errors.append(
            _error(
                "embeddings_manifest_not_ready",
                "embeddings manifest ready_for_pgvector_import is not true",
                "embeddings_manifest",
            )
        )
    manifest_dim = payload.get("embedding_dim")
    if manifest_dim != expected_dim:
        errors.append(
            _error(
                "embeddings_manifest_dim_mismatch",
                f"embeddings manifest dimension mismatch: expected {expected_dim}, got {manifest_dim}",
                "embeddings_manifest",
            )
        )


def _validate_embedding_output(
    path: str | None,
    *,
    expected_dim: int,
    errors: list[ImportPlanError],
) -> None:
    if path is None:
        return
    seen_keys: set[tuple[str, str, str]] = set()
    for line_number, value in _iter_jsonl_objects(path, "embeddings_output", errors):
        if value is None:
            continue
        record_id = _required_string(value, "record_id")
        model_name = _required_string(value, "model_name")
        text_hash = _required_string(value, "text_hash")
        if not record_id or not model_name or not text_hash:
            errors.append(
                _error(
                    "embedding_output_missing_identity",
                    f"embeddings_output line {line_number} missing identity fields",
                    "embeddings_output",
                )
            )
        else:
            key = (record_id, model_name, text_hash)
            if key in seen_keys:
                errors.append(
                    _error(
                        "duplicate_embedding_identity",
                        f"embeddings_output line {line_number} duplicate embedding identity",
                        "embeddings_output",
                    )
                )
            seen_keys.add(key)

        embedding = value.get("embedding")
        if not isinstance(embedding, list):
            errors.append(
                _error(
                    "embedding_vector_invalid",
                    f"embeddings_output line {line_number} embedding must be a list",
                    "embeddings_output",
                )
            )
            continue
        actual_dim = len(embedding)
        if actual_dim != expected_dim:
            errors.append(
                _error(
                    "embedding_dim_mismatch",
                    f"embeddings_output line {line_number} embedding dimension mismatch: expected {expected_dim}, got {actual_dim}",
                    "embeddings_output",
                )
            )


def _count_embedding_records(path: str | None, *, expected_dim: int) -> int:
    if path is None:
        return 0
    count = 0
    for _, value in _iter_jsonl_objects(path, "embeddings_output", []):
        if value is None:
            continue
        embedding = value.get("embedding")
        if isinstance(embedding, list) and len(embedding) == expected_dim:
            count += 1
    return count


def _source_metadata_from_manifest(path: str | None) -> tuple[int, list[str], list[str]]:
    if path is None:
        return 0, [], []
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0, [], []
    if not isinstance(payload, dict):
        return 0, [], []
    sources = payload.get("sources")
    if not isinstance(sources, list):
        sources = []
    source_ids = sorted(
        {
            str(source.get("source_id") or "").strip()
            for source in sources
            if isinstance(source, dict) and str(source.get("source_id") or "").strip()
        }
    )
    source_urls = sorted(
        {
            str(source.get("source_url") or "").strip()
            for source in sources
            if isinstance(source, dict) and str(source.get("source_url") or "").strip()
        }
    )
    return len(sources), source_ids, source_urls


def _law_ids_from_embedding_manifest(path: str | None) -> list[str]:
    if path is None:
        return []
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    law_ids = payload.get("law_ids")
    if not isinstance(law_ids, list):
        return []
    return sorted({str(law_id) for law_id in law_ids if str(law_id).strip()})


def _read_json_list(
    path: str | None,
    artifact: str,
    errors: list[ImportPlanError],
) -> list[dict[str, Any]]:
    if path is None:
        return []
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        errors.append(_error("invalid_json", f"{artifact} is not valid JSON", artifact))
        return []
    except OSError as exc:
        errors.append(_error("artifact_read_failed", f"{artifact} read failed: {exc}", artifact))
        return []
    if not isinstance(payload, list):
        errors.append(_error("invalid_json_shape", f"{artifact} must contain a JSON array", artifact))
        return []
    records: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if isinstance(item, dict):
            records.append(item)
        else:
            errors.append(
                _error(
                    "invalid_json_shape",
                    f"{artifact}[{index}] must be a JSON object",
                    artifact,
                )
            )
    return records


def _read_json_list_no_errors(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _read_json_object(
    path: str,
    artifact: str,
    errors: list[ImportPlanError],
) -> dict[str, Any] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        errors.append(_error("invalid_json", f"{artifact} is not valid JSON", artifact))
        return None
    except OSError as exc:
        errors.append(_error("artifact_read_failed", f"{artifact} read failed: {exc}", artifact))
        return None
    if not isinstance(payload, dict):
        errors.append(_error("invalid_json_shape", f"{artifact} must contain a JSON object", artifact))
        return None
    return payload


def _iter_jsonl_objects(
    path: str,
    artifact: str,
    errors: list[ImportPlanError],
) -> Iterable[tuple[int, dict[str, Any] | None]]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(_error("artifact_read_failed", f"{artifact} read failed: {exc}", artifact))
        return
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            errors.append(
                _error(
                    "invalid_jsonl",
                    f"{artifact} line {line_number} is not valid JSON",
                    artifact,
                )
            )
            yield line_number, None
            continue
        if not isinstance(value, dict):
            errors.append(
                _error(
                    "invalid_jsonl_shape",
                    f"{artifact} line {line_number} must contain a JSON object",
                    artifact,
                )
            )
            yield line_number, None
            continue
        yield line_number, value


def _required_string(value: Mapping[str, Any], field: str) -> str | None:
    raw_value = value.get(field)
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    return raw_value.strip()


def _validation_result(
    *,
    validation_report_passed: bool,
    embeddings_manifest_present: bool,
    pair_validation_assumed_from_manifest: bool,
    warnings: list[ImportPlanWarning],
    errors: list[ImportPlanError],
) -> ImportBundleValidationResult:
    return ImportBundleValidationResult(
        passed=not errors,
        validation_report_passed=validation_report_passed,
        embeddings_manifest_present=embeddings_manifest_present,
        pair_validation_assumed_from_manifest=pair_validation_assumed_from_manifest,
        warnings=warnings,
        errors=errors,
    )


def _warning(code: str, message: str, artifact: str | None = None) -> ImportPlanWarning:
    return ImportPlanWarning(code=code, message=message, artifact=artifact)


def _error(code: str, message: str, artifact: str | None = None) -> ImportPlanError:
    return ImportPlanError(code=code, message=message, artifact=artifact)


def _iter_count(records: Iterable[Mapping[str, Any]]) -> int:
    return sum(1 for _ in records)


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
