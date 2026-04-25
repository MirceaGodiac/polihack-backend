from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ingestion.chunks import stable_text_hash
from ingestion.contracts import EmbeddingInputRecord


ResumeKey = tuple[str, str, str]
T = TypeVar("T")


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
