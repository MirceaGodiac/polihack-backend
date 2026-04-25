from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ingestion.chunks import stable_text_hash
from ingestion.contracts import EmbeddingInputRecord


ResumeKey = tuple[str, str, str]
T = TypeVar("T")
EMBEDDINGS_IMPORT_READINESS_VALIDATOR_VERSION = "h07.phase5.import_readiness.v1"
EMBEDDING_OUTPUT_REQUIRED_FIELDS = (
    "record_id",
    "chunk_id",
    "legal_unit_id",
    "law_id",
    "model_name",
    "embedding_dim",
    "text_hash",
    "embedding",
    "metadata",
)
EMBEDDING_OUTPUT_REQUIRED_STRING_FIELDS = (
    "record_id",
    "chunk_id",
    "legal_unit_id",
    "law_id",
    "model_name",
    "text_hash",
)


class EmbeddingOutputRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str
    chunk_id: str
    legal_unit_id: str
    law_id: str
    model_name: str
    embedding_dim: int = Field(gt=0)
    text_hash: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("embedding", mode="before")
    @classmethod
    def validate_embedding_values(cls, value: Any) -> list[float]:
        return validate_embedding_vector(value)

    @model_validator(mode="after")
    def validate_embedding_dim_matches_vector(self) -> "EmbeddingOutputRecord":
        validate_embedding_vector(self.embedding, expected_dim=self.embedding_dim)
        return self


class EmbeddingJobSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    read_count: int = 0
    written_count: int = 0
    skipped_empty_text_count: int = 0
    skipped_resume_count: int = 0
    failed_count: int = 0
    embedding_dim: int | None = None
    warnings: list[str] = Field(default_factory=list)


class EmbeddingOutputValidationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    read_count: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    duplicate_resume_key_count: int = 0
    duplicate_record_id_count: int = 0
    model_names: list[str] = Field(default_factory=list)
    embedding_dims: list[int] = Field(default_factory=list)
    law_ids: list[str] = Field(default_factory=list)
    empty_metadata_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class EmbeddingInputOutputValidationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_read_count: int = 0
    output_read_count: int = 0
    embeddable_input_count: int = 0
    matched_output_count: int = 0
    missing_output_count: int = 0
    orphan_output_count: int = 0
    identity_mismatch_count: int = 0
    duplicate_input_record_id_count: int = 0
    duplicate_output_resume_key_count: int = 0
    unexpected_output_for_empty_input_count: int = 0
    model_names: list[str] = Field(default_factory=list)
    embedding_dims: list[int] = Field(default_factory=list)
    law_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class EmbeddingsImportReadinessManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_path: str
    output_path: str
    model_name: str
    embedding_dim: int
    input_read_count: int
    output_read_count: int
    embeddable_input_count: int
    matched_output_count: int
    missing_output_count: int
    orphan_output_count: int
    law_ids: list[str] = Field(default_factory=list)
    model_names: list[str] = Field(default_factory=list)
    embedding_dims: list[int] = Field(default_factory=list)
    ready_for_pgvector_import: bool
    validated_at: str
    validator_version: str
    warnings: list[str] = Field(default_factory=list)


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str], model_name: str) -> list[list[float]]:
        ...


class DeterministicFakeEmbeddingProvider:
    def __init__(self, dimension: int = 4):
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")
        self.dimension = dimension
        self.received_texts: list[str] = []

    def embed_texts(self, texts: list[str], model_name: str) -> list[list[float]]:
        self.received_texts.extend(texts)
        return [
            _deterministic_vector(text, model_name=model_name, dimension=self.dimension)
            for text in texts
        ]


class OpenAICompatibleEmbeddingProvider:
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
    RETRYABLE_EXCEPTIONS = (httpx.TimeoutException, httpx.NetworkError)

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep_func: Callable[[float], None] = time.sleep,
    ):
        normalized_base_url = str(base_url or "").strip().rstrip("/")
        if not normalized_base_url:
            raise ValueError("base URL missing: base_url must be non-empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        if max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to 0")

        self.base_url = normalized_base_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.embedding_url = f"{self.base_url}/embeddings"
        self._transport = transport
        self._sleep = sleep_func

    def embed_texts(self, texts: list[str], model_name: str) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": model_name, "input": texts}

        with httpx.Client(
            timeout=self.timeout_seconds,
            transport=self._transport,
        ) as client:
            response = self._post_with_retries(client, headers=headers, payload=payload)
        return self._parse_embedding_response(response, expected_count=len(texts))

    def _post_with_retries(
        self,
        client: httpx.Client,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> httpx.Response:
        last_retryable_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = client.post(self.embedding_url, headers=headers, json=payload)
            except self.RETRYABLE_EXCEPTIONS as exc:
                last_retryable_error = exc
                if attempt < self.max_retries:
                    self._sleep(_retry_delay_seconds(attempt))
                    continue
                raise RuntimeError(
                    "embedding provider request failed after retries: "
                    f"{exc.__class__.__name__}"
                ) from exc

            if response.status_code in self.RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                self._sleep(_retry_delay_seconds(attempt))
                continue
            if response.status_code < 200 or response.status_code >= 300:
                raise RuntimeError(
                    "embedding provider HTTP status "
                    f"{response.status_code}: {_response_reason(response)}"
                )
            return response

        if last_retryable_error is not None:
            raise RuntimeError(
                "embedding provider request failed after retries: "
                f"{last_retryable_error.__class__.__name__}"
            ) from last_retryable_error
        raise RuntimeError("embedding provider request failed after retries")

    def _parse_embedding_response(
        self,
        response: httpx.Response,
        *,
        expected_count: int,
    ) -> list[list[float]]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError("malformed response: invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("malformed response: root must be a JSON object")

        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("malformed response: data must be a list")

        embeddings = _parse_openai_embedding_data(data)
        if len(embeddings) != expected_count:
            raise ValueError(
                "embedding count mismatch: "
                f"expected {expected_count}, got {len(embeddings)}"
            )
        return embeddings


@dataclass(frozen=True)
class EmbeddingJobCandidate:
    record: EmbeddingInputRecord
    embedding_text_length: int


def load_embedding_input_jsonl(path: str | Path) -> list[EmbeddingInputRecord]:
    resolved_path = Path(path)
    records: list[EmbeddingInputRecord] = []
    for line_number, line in enumerate(
        resolved_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{resolved_path}:{line_number} invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{resolved_path}:{line_number} must contain a JSON object")
        try:
            records.append(_validate_embedding_input_record(value))
        except ValidationError as exc:
            raise ValueError(
                f"{resolved_path}:{line_number} invalid EmbeddingInputRecord"
            ) from exc
    return records


def load_embedding_output_jsonl(path: str | Path) -> list[EmbeddingOutputRecord]:
    resolved_path = Path(path)
    if not resolved_path.exists():
        return []

    records: list[EmbeddingOutputRecord] = []
    for line_number, line in enumerate(
        resolved_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{resolved_path}:{line_number} invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{resolved_path}:{line_number} must contain a JSON object")
        try:
            records.append(EmbeddingOutputRecord.model_validate(value))
        except ValidationError as exc:
            raise ValueError(
                f"{resolved_path}:{line_number} invalid EmbeddingOutputRecord"
            ) from exc
    return records


def validate_embeddings_output(
    output_path: str | Path,
    *,
    expected_model: str | None = None,
    expected_dim: int | None = None,
    require_unique_resume_keys: bool = True,
    require_unique_record_ids: bool = False,
    strict: bool = True,
) -> EmbeddingOutputValidationSummary:
    if expected_model is not None and not expected_model.strip():
        raise ValueError("expected_model must be non-empty when provided")
    if expected_dim is not None and expected_dim <= 0:
        raise ValueError("expected_dim must be greater than 0")

    resolved_path = Path(output_path)
    errors: list[str] = []
    warnings: list[str] = []
    model_names: set[str] = set()
    embedding_dims: set[int] = set()
    law_ids: set[str] = set()
    seen_resume_keys: set[ResumeKey] = set()
    seen_record_ids: set[str] = set()
    read_count = 0
    valid_count = 0
    invalid_count = 0
    duplicate_resume_key_count = 0
    duplicate_record_id_count = 0
    empty_metadata_count = 0

    for line_number, line in enumerate(
        resolved_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        read_count += 1
        record_errors: list[str] = []
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            invalid_count += 1
            errors.append(f"line {line_number}: invalid JSON")
            continue
        if not isinstance(value, dict):
            invalid_count += 1
            errors.append(f"line {line_number}: record must be a JSON object")
            continue

        record_label = _output_record_label(value, line_number)
        for field in EMBEDDING_OUTPUT_REQUIRED_FIELDS:
            if field not in value:
                record_errors.append(f"{record_label}: missing required field {field}")

        valid_strings: dict[str, str] = {}
        for field in EMBEDDING_OUTPUT_REQUIRED_STRING_FIELDS:
            if field not in value:
                continue
            field_value = value[field]
            if not isinstance(field_value, str) or not field_value.strip():
                record_errors.append(
                    f"{record_label}: field {field} must be a non-empty string"
                )
            else:
                valid_strings[field] = field_value

        embedding_dim = _valid_output_embedding_dim(value, record_label, record_errors)
        if embedding_dim is not None:
            embedding_dims.add(embedding_dim)
            if expected_dim is not None and embedding_dim != expected_dim:
                record_errors.append(
                    f"{record_label}: expected_dim mismatch: expected {expected_dim}, "
                    f"got {embedding_dim}"
                )

        model_name = valid_strings.get("model_name")
        if model_name is not None:
            model_names.add(model_name)
            if expected_model is not None and model_name != expected_model:
                record_errors.append(
                    f"{record_label}: expected_model mismatch: expected {expected_model}, "
                    f"got {model_name}"
                )
        law_id = valid_strings.get("law_id")
        if law_id is not None:
            law_ids.add(law_id)

        if "embedding" in value:
            try:
                validate_embedding_vector(value["embedding"], expected_dim=embedding_dim)
            except ValueError as exc:
                record_errors.append(f"{record_label}: invalid embedding vector: {exc}")

        if "metadata" in value:
            if not isinstance(value["metadata"], dict):
                record_errors.append(f"{record_label}: metadata must be a JSON object")
            elif not value["metadata"]:
                empty_metadata_count += 1

        record_id = valid_strings.get("record_id")
        text_hash = valid_strings.get("text_hash")
        if record_id is not None:
            if record_id in seen_record_ids:
                duplicate_record_id_count += 1
                duplicate_message = f"{record_label}: duplicate record_id {record_id}"
                if require_unique_record_ids:
                    record_errors.append(duplicate_message)
                else:
                    warnings.append(duplicate_message)
            seen_record_ids.add(record_id)

        if record_id is not None and text_hash is not None and model_name is not None:
            resume_key = (record_id, text_hash, model_name)
            if resume_key in seen_resume_keys:
                duplicate_resume_key_count += 1
                if require_unique_resume_keys:
                    record_errors.append(f"{record_label}: duplicate resume key")
            seen_resume_keys.add(resume_key)

        if record_errors:
            invalid_count += 1
            errors.extend(record_errors)
        else:
            valid_count += 1

    summary = EmbeddingOutputValidationSummary(
        read_count=read_count,
        valid_count=valid_count,
        invalid_count=invalid_count,
        duplicate_resume_key_count=duplicate_resume_key_count,
        duplicate_record_id_count=duplicate_record_id_count,
        model_names=sorted(model_names),
        embedding_dims=sorted(embedding_dims),
        law_ids=sorted(law_ids),
        empty_metadata_count=empty_metadata_count,
        warnings=warnings,
        errors=errors,
    )
    if strict and errors:
        preview = "; ".join(errors[:5])
        remaining = len(errors) - min(len(errors), 5)
        suffix = f"; ... {remaining} more" if remaining else ""
        raise ValueError(
            "embedding output validation failed: "
            f"invalid_count={invalid_count}, error_count={len(errors)}; "
            f"{preview}{suffix}"
        )
    return summary


def validate_embeddings_pair(
    input_path: str | Path,
    output_path: str | Path,
    *,
    expected_model: str | None = None,
    expected_dim: int | None = None,
    require_all_inputs: bool = True,
    strict: bool = True,
) -> EmbeddingInputOutputValidationSummary:
    if expected_model is not None and not expected_model.strip():
        raise ValueError("expected_model must be non-empty when provided")
    if expected_dim is not None and expected_dim <= 0:
        raise ValueError("expected_dim must be greater than 0")

    errors: list[str] = []
    warnings: list[str] = []
    output_summary = validate_embeddings_output(
        output_path,
        expected_model=expected_model,
        expected_dim=expected_dim,
        require_unique_resume_keys=True,
        strict=False,
    )
    errors.extend(f"output: {message}" for message in output_summary.errors)
    warnings.extend(f"output: {message}" for message in output_summary.warnings)

    input_records = load_embedding_input_jsonl(input_path)
    input_by_record_id: dict[str, EmbeddingInputRecord] = {}
    embeddable_record_ids: set[str] = set()
    non_embeddable_record_ids: set[str] = set()
    duplicate_input_record_id_count = 0
    for record in input_records:
        if record.record_id in input_by_record_id:
            duplicate_input_record_id_count += 1
            errors.append(f"input record_id={record.record_id}: duplicate input record_id")
            continue
        input_by_record_id[record.record_id] = record
        if record.embedding_text.strip():
            embeddable_record_ids.add(record.record_id)
        else:
            non_embeddable_record_ids.add(record.record_id)

    matched_output_count = 0
    orphan_output_count = 0
    identity_mismatch_count = 0
    unexpected_output_for_empty_input_count = 0
    matched_by_record_id: dict[str, set[str]] = {}
    output_records: list[EmbeddingOutputRecord] = []
    if not output_summary.errors:
        output_records = load_embedding_output_jsonl(output_path)

    for output in output_records:
        input_record = input_by_record_id.get(output.record_id)
        output_label = (
            f"output record_id={output.record_id} "
            f"model_name={output.model_name} text_hash={output.text_hash}"
        )
        if input_record is None:
            orphan_output_count += 1
            errors.append(f"{output_label}: orphan output record_id")
            continue
        if output.record_id in non_embeddable_record_ids:
            unexpected_output_for_empty_input_count += 1
            errors.append(f"{output_label}: unexpected output for non-embeddable input")
            continue

        mismatched_fields = [
            field
            for field in ("chunk_id", "legal_unit_id", "law_id", "text_hash")
            if getattr(output, field) != getattr(input_record, field)
        ]
        if mismatched_fields:
            identity_mismatch_count += 1
            errors.append(
                f"{output_label}: identity mismatch fields={','.join(mismatched_fields)}"
            )
            continue

        matched_output_count += 1
        matched_by_record_id.setdefault(output.record_id, set()).add(output.model_name)

    missing_output_count = 0
    for record_id in sorted(embeddable_record_ids):
        matched_models = matched_by_record_id.get(record_id, set())
        is_missing = expected_model not in matched_models if expected_model else not matched_models
        if is_missing:
            missing_output_count += 1
            message = f"input record_id={record_id}: missing embedding output"
            if require_all_inputs:
                errors.append(message)
            else:
                warnings.append(message)

    summary = EmbeddingInputOutputValidationSummary(
        input_read_count=len(input_records),
        output_read_count=output_summary.read_count,
        embeddable_input_count=len(embeddable_record_ids),
        matched_output_count=matched_output_count,
        missing_output_count=missing_output_count,
        orphan_output_count=orphan_output_count,
        identity_mismatch_count=identity_mismatch_count,
        duplicate_input_record_id_count=duplicate_input_record_id_count,
        duplicate_output_resume_key_count=output_summary.duplicate_resume_key_count,
        unexpected_output_for_empty_input_count=unexpected_output_for_empty_input_count,
        model_names=output_summary.model_names,
        embedding_dims=output_summary.embedding_dims,
        law_ids=output_summary.law_ids,
        warnings=warnings,
        errors=errors,
    )
    if strict and summary.errors:
        preview = "; ".join(summary.errors[:5])
        remaining = len(summary.errors) - min(len(summary.errors), 5)
        suffix = f"; ... {remaining} more" if remaining else ""
        raise ValueError(
            "embedding pair validation failed: "
            f"error_count={len(summary.errors)}; {preview}{suffix}"
        )
    return summary


def build_embeddings_import_manifest(
    *,
    input_path: str | Path,
    output_path: str | Path,
    expected_model: str,
    expected_dim: int,
    manifest_path: str | Path | None = None,
) -> EmbeddingsImportReadinessManifest:
    if not expected_model.strip():
        raise ValueError("expected_model must be non-empty")
    if expected_dim <= 0:
        raise ValueError("expected_dim must be greater than 0")

    summary = validate_embeddings_pair(
        input_path,
        output_path,
        expected_model=expected_model,
        expected_dim=expected_dim,
        require_all_inputs=True,
        strict=True,
    )
    manifest = EmbeddingsImportReadinessManifest(
        input_path=str(Path(input_path)),
        output_path=str(Path(output_path)),
        model_name=expected_model,
        embedding_dim=expected_dim,
        input_read_count=summary.input_read_count,
        output_read_count=summary.output_read_count,
        embeddable_input_count=summary.embeddable_input_count,
        matched_output_count=summary.matched_output_count,
        missing_output_count=summary.missing_output_count,
        orphan_output_count=summary.orphan_output_count,
        law_ids=summary.law_ids,
        model_names=summary.model_names,
        embedding_dims=summary.embedding_dims,
        ready_for_pgvector_import=True,
        validated_at=_utc_now_iso_z(),
        validator_version=EMBEDDINGS_IMPORT_READINESS_VALIDATOR_VERSION,
        warnings=summary.warnings,
    )
    if manifest_path is not None:
        resolved_manifest_path = Path(manifest_path)
        resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_manifest_path.write_text(
            json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return manifest


def write_embedding_output_jsonl(
    records: list[EmbeddingOutputRecord],
    path: str | Path,
    *,
    append: bool = False,
) -> None:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with resolved_path.open(mode, encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(
                json.dumps(
                    record.model_dump(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def read_existing_resume_keys(path: str | Path) -> set[ResumeKey]:
    return {
        output_resume_key(record)
        for record in load_embedding_output_jsonl(path)
    }


def validate_embedding_vector(
    vector: Any,
    *,
    expected_dim: int | None = None,
) -> list[float]:
    if expected_dim is not None and expected_dim <= 0:
        raise ValueError("expected_dim must be greater than 0")
    if not isinstance(vector, list):
        raise ValueError("embedding vector must be a list")
    if not vector:
        raise ValueError("embedding vector must be non-empty")
    if expected_dim is not None and len(vector) != expected_dim:
        raise ValueError(
            f"embedding vector dimension mismatch: expected {expected_dim}, got {len(vector)}"
        )

    values: list[float] = []
    for index, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"embedding vector item {index} must be an int or float")
        float_value = float(value)
        if not math.isfinite(float_value):
            raise ValueError(f"embedding vector item {index} must be finite")
        values.append(float_value)
    return values


def iter_batches(items: list[T], batch_size: int) -> Iterator[list[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def input_resume_key(record: EmbeddingInputRecord, model_name: str) -> ResumeKey:
    return (record.record_id, record.text_hash, model_name)


def output_resume_key(record: EmbeddingOutputRecord) -> ResumeKey:
    return (record.record_id, record.text_hash, record.model_name)


def generate_embeddings(
    *,
    input_path: str | Path,
    output_path: str | Path,
    provider: EmbeddingProvider,
    model_name: str,
    batch_size: int = 100,
    expected_dim: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
) -> EmbeddingJobSummary:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if expected_dim is not None and expected_dim <= 0:
        raise ValueError("expected_dim must be greater than 0")
    if limit is not None and limit < 0:
        raise ValueError("limit must be greater than or equal to 0")
    if not model_name.strip():
        raise ValueError("model_name must be non-empty")

    input_records = load_embedding_input_jsonl(input_path)
    warnings: list[str] = []
    existing_keys = read_existing_resume_keys(output_path) if resume else set()

    candidates: list[EmbeddingJobCandidate] = []
    skipped_empty_text_count = 0
    skipped_resume_count = 0
    for record in input_records:
        embedding_text_length = len(record.embedding_text)
        log_context = _record_log_context(record, embedding_text_length)

        if not record.embedding_text.strip():
            skipped_empty_text_count += 1
            warnings.append(f"skipped empty embedding_text {log_context}")
            continue

        if stable_text_hash(record.embedding_text) != record.text_hash:
            warnings.append(f"text_hash mismatch {log_context}")

        if resume and input_resume_key(record, model_name) in existing_keys:
            skipped_resume_count += 1
            continue

        candidates.append(
            EmbeddingJobCandidate(
                record=record,
                embedding_text_length=embedding_text_length,
            )
        )

    if limit is not None:
        candidates = candidates[:limit]

    summary = EmbeddingJobSummary(
        read_count=len(input_records),
        written_count=0,
        skipped_empty_text_count=skipped_empty_text_count,
        skipped_resume_count=skipped_resume_count,
        failed_count=0,
        embedding_dim=expected_dim,
        warnings=warnings,
    )
    if dry_run:
        return summary

    output_records: list[EmbeddingOutputRecord] = []
    resolved_dim = expected_dim
    for batch in iter_batches(candidates, batch_size):
        texts = [candidate.record.embedding_text for candidate in batch]
        vectors = provider.embed_texts(texts, model_name)
        if len(vectors) != len(batch):
            raise RuntimeError(
                "embedding provider returned a different number of vectors "
                f"than requested: expected {len(batch)}, got {len(vectors)}"
            )

        for candidate, vector in zip(batch, vectors):
            if resolved_dim is None:
                validated_vector = validate_embedding_vector(vector)
                resolved_dim = len(validated_vector)
            else:
                validated_vector = validate_embedding_vector(
                    vector,
                    expected_dim=resolved_dim,
                )
            record = candidate.record
            output_records.append(
                EmbeddingOutputRecord(
                    record_id=record.record_id,
                    chunk_id=record.chunk_id,
                    legal_unit_id=record.legal_unit_id,
                    law_id=record.law_id,
                    model_name=model_name,
                    embedding_dim=resolved_dim,
                    text_hash=record.text_hash,
                    embedding=validated_vector,
                    metadata=dict(record.metadata),
                )
            )

    append = resume and Path(output_path).exists()
    write_embedding_output_jsonl(output_records, output_path, append=append)

    return summary.model_copy(
        update={
            "written_count": len(output_records),
            "embedding_dim": resolved_dim,
        }
    )


def _validate_embedding_input_record(value: dict[str, Any]) -> EmbeddingInputRecord:
    embedding_text = value.get("embedding_text")
    if embedding_text == "":
        validation_value = dict(value)
        validation_value["embedding_text"] = " "
        record = EmbeddingInputRecord.model_validate(validation_value)
        return record.model_copy(update={"embedding_text": ""})
    return EmbeddingInputRecord.model_validate(value)


def _output_record_label(value: dict[str, Any], line_number: int) -> str:
    record_id = value.get("record_id")
    if isinstance(record_id, str) and record_id.strip():
        return f"line {line_number} record_id={record_id}"
    return f"line {line_number}"


def _valid_output_embedding_dim(
    value: dict[str, Any],
    record_label: str,
    errors: list[str],
) -> int | None:
    if "embedding_dim" not in value:
        return None
    embedding_dim = value["embedding_dim"]
    if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int):
        errors.append(f"{record_label}: embedding_dim must be a positive integer")
        return None
    if embedding_dim <= 0:
        errors.append(f"{record_label}: embedding_dim must be a positive integer")
        return None
    return embedding_dim


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _record_log_context(record: EmbeddingInputRecord, embedding_text_length: int) -> str:
    return (
        f"record_id={record.record_id} "
        f"chunk_id={record.chunk_id} "
        f"legal_unit_id={record.legal_unit_id} "
        f"text_hash={record.text_hash} "
        f"embedding_text_length={embedding_text_length}"
    )


def _deterministic_vector(
    text: str,
    *,
    model_name: str,
    dimension: int,
) -> list[float]:
    seed = f"{model_name}\0{text}".encode("utf-8")
    values: list[float] = []
    for index in range(dimension):
        digest = hashlib.sha256(seed + index.to_bytes(4, "big")).digest()
        integer_value = int.from_bytes(digest[:8], "big")
        values.append(integer_value / float(2**64 - 1))
    return values


def _parse_openai_embedding_data(data: list[Any]) -> list[list[float]]:
    has_any_index = any(isinstance(item, dict) and "index" in item for item in data)
    has_all_index = all(isinstance(item, dict) and "index" in item for item in data)
    if has_any_index and not has_all_index:
        raise ValueError(
            "malformed response: index must be present for every embedding item or absent for all"
        )

    items = list(data)
    if has_all_index:
        indexes: list[int] = []
        for item in items:
            if isinstance(item.get("index"), bool) or not isinstance(item.get("index"), int):
                raise ValueError("malformed response: index must be an integer")
            indexes.append(item["index"])
        expected_indexes = set(range(len(items)))
        if len(set(indexes)) != len(indexes) or set(indexes) != expected_indexes:
            raise ValueError(
                "malformed response: indexes must be unique and contiguous from 0"
            )
        items = sorted(items, key=lambda item: item["index"])

    embeddings: list[list[float]] = []
    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("malformed response: data items must be JSON objects")
        if "embedding" not in item:
            raise ValueError("malformed response: data item missing embedding")
        try:
            embeddings.append(validate_embedding_vector(item["embedding"]))
        except ValueError as exc:
            raise ValueError(
                f"invalid vector in embedding response at data item {item_index}: {exc}"
            ) from exc
    return embeddings


def _retry_delay_seconds(attempt: int) -> float:
    return 0.5 * (2**attempt)


def _response_reason(response: httpx.Response) -> str:
    return response.reason_phrase or "HTTP error"
