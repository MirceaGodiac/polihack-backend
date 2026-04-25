"""
Canonical ingestion contracts for parser output readiness.

LegalUnit is the citable legal unit and source of legal truth. LegalChunk is a
retrieval/vector record derived from one or more LegalUnit records. In v1, each
LegalChunk is 1:1 with a LegalUnit, and evidence/citations must keep using the
LegalUnit id rather than the chunk id.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator


LegalStatus = Literal["active", "historical", "repealed", "unknown"]

ParsedLegalEdgeType = Literal[
    "contains",
    "references",
    "amends",
    "repeals",
    "derogates_from",
    "exception_to",
    "defines",
    "sanctions",
    "creates_obligation",
    "creates_right",
    "creates_prohibition",
    "procedure_step",
    "same_topic_as",
    "semantically_related",
    "historical_version_of",
]

ReferenceResolutionStatus = Literal[
    "resolved_high_confidence",
    "resolved_medium_confidence",
    "candidate_ambiguous",
    "external_unresolved",
    "unresolved_ambiguous",
    "unresolved_needs_context",
    "candidate_only",
    "unresolved",
]

LEGAL_UNIT_FIELDS = {
    "id",
    "canonical_id",
    "source_id",
    "law_id",
    "law_title",
    "act_type",
    "act_number",
    "publication_date",
    "effective_date",
    "version_start",
    "version_end",
    "status",
    "hierarchy_path",
    "article_number",
    "paragraph_number",
    "letter_number",
    "point_number",
    "raw_text",
    "normalized_text",
    "legal_domain",
    "legal_concepts",
    "source_url",
    "parent_id",
    "children_ids",
    "outgoing_reference_ids",
    "incoming_reference_ids",
}


class IngestionContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParserActMetadata(IngestionContract):
    law_id: str
    law_title: str
    legal_domain: str
    act_type: str | None = None
    act_number: str | None = None
    source_id: str | None = None
    source_url: str | None = None
    publication_date: date | None = None
    effective_date: date | None = None
    version_start: date | None = None
    version_end: date | None = None
    status: LegalStatus = "unknown"
    parser_version: str = "0.1.0"


class ParsedLegalUnit(IngestionContract):
    id: str
    canonical_id: str | None = None
    source_id: str | None = None
    law_id: str
    law_title: str
    act_type: str | None = None
    act_number: str | None = None
    publication_date: date | None = None
    effective_date: date | None = None
    version_start: date | None = None
    version_end: date | None = None
    status: LegalStatus
    hierarchy_path: list[str]
    article_number: str | None = None
    paragraph_number: str | None = None
    letter_number: str | None = None
    point_number: str | None = None
    raw_text: str
    normalized_text: str | None = None
    legal_domain: str
    legal_concepts: list[str] = Field(default_factory=list)
    source_url: str | None = None
    parent_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    outgoing_reference_ids: list[str] = Field(default_factory=list)
    incoming_reference_ids: list[str] = Field(default_factory=list)
    parser_warnings: list[str] = Field(default_factory=list)

    def to_legal_unit_dict(self, *, include_parser_warnings: bool = False) -> dict[str, Any]:
        """Return the LegalUnit shape used by retrieval/evidence."""
        fields = set(LEGAL_UNIT_FIELDS)
        if include_parser_warnings:
            fields.add("parser_warnings")
        return self.model_dump(include=fields)


class ParsedLegalEdge(IngestionContract):
    id: str
    source_id: str
    target_id: str
    type: ParsedLegalEdgeType
    weight: float = Field(default=1.0, ge=0.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LegalChunk(IngestionContract):
    """Vector/retrieval record derived from LegalUnit, not a citation target."""

    chunk_id: str
    legal_unit_id: str
    legal_unit_ids: list[str]
    chunk_type: Literal["legal_unit", "multi_unit"] = "legal_unit"
    chunk_version: str = "v1"
    law_id: str
    law_title: str
    legal_domain: str
    hierarchy_path: list[str]
    article_number: str | None = None
    paragraph_number: str | None = None
    letter_number: str | None = None
    point_number: str | None = None
    raw_text: str
    normalized_text: str | None = None
    embedding_text: str = Field(min_length=1)
    source_url: str | None = None
    source_id: str | None = None
    text_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_legal_unit(
        cls,
        legal_unit: ParsedLegalUnit | BaseModel | Mapping[str, Any],
        *,
        chunk_version: str = "v1",
        metadata: dict[str, Any] | None = None,
    ) -> "LegalChunk":
        unit = _legal_unit_mapping(legal_unit)
        legal_unit_id = unit["id"]
        embedding_text = _build_embedding_text(unit)

        return cls(
            chunk_id=f"chunk.{legal_unit_id}.{chunk_version}",
            legal_unit_id=legal_unit_id,
            legal_unit_ids=[legal_unit_id],
            chunk_version=chunk_version,
            law_id=unit["law_id"],
            law_title=unit["law_title"],
            legal_domain=unit["legal_domain"],
            hierarchy_path=list(unit.get("hierarchy_path", [])),
            article_number=unit.get("article_number"),
            paragraph_number=unit.get("paragraph_number"),
            letter_number=unit.get("letter_number"),
            point_number=unit.get("point_number"),
            raw_text=unit["raw_text"],
            normalized_text=unit.get("normalized_text"),
            embedding_text=embedding_text,
            source_url=unit.get("source_url"),
            source_id=unit.get("source_id"),
            text_hash=_stable_text_hash(embedding_text),
            metadata=metadata or {},
        )


class ReferenceCandidate(IngestionContract):
    source_unit_id: str
    raw_reference: str
    reference_type: str
    target_law_hint: str | None = None
    target_article: str | None = None
    target_paragraph: str | None = None
    target_letter: str | None = None
    target_point: str | None = None
    target_thesis: str | None = None
    resolved_target_id: str | None = None
    resolution_status: ReferenceResolutionStatus = "candidate_only"
    resolution_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    resolver_notes: list[str] = Field(default_factory=list)


class CorpusManifest(IngestionContract):
    batch_id: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = "1.0.0"
    sources: list[ParserActMetadata] = Field(default_factory=list)
    output_dir: str
    files: dict[str, str] = Field(default_factory=dict)


class ValidationReport(IngestionContract):
    schema_version: str = "1.0"
    parser_version: str
    corpus_quality: float = Field(ge=0.0, le=1.0)
    units_count: int = Field(ge=0)
    edges_count: int = Field(ge=0)
    chunks_count: int = Field(default=0, ge=0)
    reference_candidates_count: int = Field(default=0, ge=0)
    quality_metrics: dict[str, float]
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @field_validator("quality_metrics")
    @classmethod
    def validate_quality_metrics(cls, metrics: dict[str, float]) -> dict[str, float]:
        for name, value in metrics.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"quality_metrics.{name} must be in [0, 1]")
        return metrics


class EmbeddingInputRecord(IngestionContract):
    chunk_id: str
    legal_unit_id: str
    embedding_text: str = Field(min_length=1)
    text_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_chunk(
        cls,
        chunk: LegalChunk | BaseModel | Mapping[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "EmbeddingInputRecord":
        chunk_data = _model_mapping(chunk)
        return cls(
            chunk_id=chunk_data["chunk_id"],
            legal_unit_id=chunk_data["legal_unit_id"],
            embedding_text=chunk_data["embedding_text"],
            text_hash=chunk_data["text_hash"],
            metadata=metadata if metadata is not None else chunk_data.get("metadata", {}),
        )


def _legal_unit_mapping(legal_unit: ParsedLegalUnit | BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(legal_unit, ParsedLegalUnit):
        return legal_unit.to_legal_unit_dict()
    return _model_mapping(legal_unit)


def _model_mapping(value: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump()
    return dict(value)


def _build_embedding_text(unit: Mapping[str, Any]) -> str:
    hierarchy = " > ".join(unit.get("hierarchy_path") or [])
    body = unit.get("normalized_text") or unit.get("raw_text") or ""
    parts = [
        f"Domain: {unit.get('legal_domain', '')}",
        f"Law: {unit.get('law_title', '')}",
    ]
    if hierarchy:
        parts.append(f"Path: {hierarchy}")
    parts.append("")
    parts.append(body)
    return "\n".join(parts).strip()


def _stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "CorpusManifest",
    "EmbeddingInputRecord",
    "LegalChunk",
    "ParserActMetadata",
    "ParsedLegalEdge",
    "ParsedLegalUnit",
    "ReferenceCandidate",
    "ValidationReport",
]
