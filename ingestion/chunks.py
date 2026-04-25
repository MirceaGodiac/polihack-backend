from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import BaseModel

from ingestion.contracts import EmbeddingInputRecord, LegalChunk
from ingestion.normalizer import normalize_legal_text


CONTEXT_GENERATION_METHOD = "deterministic_v1"
INTERPRETATION_MARKERS = (
    "angajatorul nu poate scadea salariul",
    "angajatorul nu poate sa scada salariul",
    "scada salariul fara act aditional",
    "concluzie juridica",
    "raspuns juridic",
)


@dataclass(frozen=True)
class RetrievalContextResult:
    context: str
    context_sources: list[str]
    context_confidence: float


def build_retrieval_context(
    unit: Mapping[str, Any] | BaseModel,
    all_units: list[Mapping[str, Any] | BaseModel],
    reference_candidates: list[Mapping[str, Any] | BaseModel] | None = None,
) -> RetrievalContextResult:
    unit_data = _mapping(unit)
    units_by_id = {_mapping(candidate)["id"]: _mapping(candidate) for candidate in all_units}
    references = [
        _mapping(candidate)
        for candidate in reference_candidates or []
        if _mapping(candidate).get("source_unit_id") == unit_data.get("id")
    ]

    context_parts: list[str] = []
    sources: list[str] = []

    law_title = str(unit_data.get("law_title") or "").strip()
    legal_domain = str(unit_data.get("legal_domain") or "").strip()
    if law_title or legal_domain:
        context_parts.append(
            f"Unitate din {law_title or 'lege necunoscuta'}, domeniul {legal_domain or 'unknown'}."
        )
        if law_title:
            sources.append("law_title")
        if legal_domain:
            sources.append("legal_domain")

    location = _unit_location(unit_data)
    if location:
        context_parts.append(f"Localizare: {location}.")
        sources.extend(_location_sources(unit_data))

    hierarchy = " > ".join(str(part) for part in unit_data.get("hierarchy_path") or [])
    if hierarchy:
        context_parts.append(f"Context ierarhic: {hierarchy}.")
        sources.append("hierarchy_path")

    parent = units_by_id.get(unit_data.get("parent_id"))
    parent_location = _unit_location(parent or {})
    if parent_location:
        context_parts.append(f"Unitate parinte: {parent_location}.")
        sources.append("parent_unit_metadata")

    legal_concepts = [str(concept) for concept in unit_data.get("legal_concepts") or []]
    if legal_concepts:
        context_parts.append(f"Concepte existente: {', '.join(legal_concepts)}.")
        sources.append("legal_concepts")

    source_url = unit_data.get("source_url")
    if source_url:
        context_parts.append(f"Sursa: {source_url}.")
        sources.append("source_url")

    raw_references = _raw_unresolved_references(references)
    if raw_references:
        context_parts.append(
            "Referinte extrase nerezolvate: " + ", ".join(raw_references) + "."
        )
        sources.append("reference_candidates_unresolved")

    if not context_parts:
        context_parts.append("Context determinist indisponibil complet pentru unitate.")

    confidence = 0.6
    if law_title and legal_domain and hierarchy:
        confidence = 0.9
    if raw_references:
        confidence = min(confidence, 0.75)

    return RetrievalContextResult(
        context="\n".join(context_parts),
        context_sources=sorted(set(sources)),
        context_confidence=confidence,
    )


def build_legal_chunks(
    legal_units: list[Mapping[str, Any] | BaseModel],
    reference_candidates: list[Mapping[str, Any] | BaseModel] | None = None,
) -> list[dict[str, Any]]:
    unit_data = [_mapping(unit) for unit in legal_units]
    chunks: list[dict[str, Any]] = []
    for unit in unit_data:
        text = str(unit.get("raw_text") or "")
        context = build_retrieval_context(unit, unit_data, reference_candidates)
        retrieval_text = f"{context.context}\n\n{text}".strip()
        chunk = LegalChunk(
            chunk_id=f"chunk.{unit['id']}.0",
            legal_unit_id=unit["id"],
            legal_unit_ids=[unit["id"]],
            chunk_version="v1",
            law_id=unit["law_id"],
            law_title=unit["law_title"],
            legal_domain=unit["legal_domain"],
            hierarchy_path=list(unit.get("hierarchy_path") or []),
            article_number=unit.get("article_number"),
            paragraph_number=unit.get("paragraph_number"),
            letter_number=unit.get("letter_number"),
            point_number=unit.get("point_number"),
            text=text,
            raw_text=text,
            normalized_text=normalize_legal_text(retrieval_text),
            retrieval_context=context.context,
            retrieval_text=retrieval_text,
            context_sources=context.context_sources,
            context_generation_method=CONTEXT_GENERATION_METHOD,
            context_confidence=context.context_confidence,
            embedding_text=retrieval_text,
            source_url=unit.get("source_url"),
            source_id=unit.get("source_id"),
            text_hash=stable_text_hash(retrieval_text),
            metadata={
                "retrieval_context_is_citable": False,
                "text_source": "LegalUnit.raw_text",
            },
        )
        chunks.append(chunk.model_dump())
    return sorted(chunks, key=lambda chunk: chunk["chunk_id"])


def build_embedding_input_records(
    legal_chunks: list[Mapping[str, Any] | BaseModel],
    *,
    model_hint: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for chunk in legal_chunks:
        chunk_data = _mapping(chunk)
        metadata = {
            "context_generation_method": chunk_data.get("context_generation_method"),
            "context_sources": list(chunk_data.get("context_sources") or []),
            "retrieval_text_is_citable": False,
        }
        if model_hint is not None:
            metadata["model_hint"] = model_hint
        record = EmbeddingInputRecord(
            record_id=f"embedding.{chunk_data['chunk_id']}",
            chunk_id=chunk_data["chunk_id"],
            legal_unit_id=chunk_data["legal_unit_id"],
            law_id=chunk_data["law_id"],
            text=chunk_data["retrieval_text"],
            embedding_text=chunk_data["retrieval_text"],
            text_hash=stable_text_hash(chunk_data["retrieval_text"]),
            model_hint=model_hint,
            metadata=metadata,
        )
        records.append(record.model_dump())
    return sorted(records, key=lambda record: record["record_id"])


def stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def contains_hardcoded_interpretation(text: str | None) -> bool:
    normalized = _normalize_marker_text(text or "")
    return any(_normalize_marker_text(marker) in normalized for marker in INTERPRETATION_MARKERS)


def _mapping(value: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump()
    return dict(value)


def _unit_location(unit: Mapping[str, Any]) -> str:
    pieces = []
    if unit.get("article_number"):
        pieces.append(f"Art. {unit['article_number']}")
    if unit.get("paragraph_number"):
        pieces.append(f"Alin. ({unit['paragraph_number']})")
    if unit.get("letter_number"):
        pieces.append(f"Lit. {unit['letter_number']})")
    if unit.get("point_number"):
        pieces.append(f"Pct. {unit['point_number']}")
    return ", ".join(pieces)


def _location_sources(unit: Mapping[str, Any]) -> list[str]:
    sources = []
    for key in ("article_number", "paragraph_number", "letter_number", "point_number"):
        if unit.get(key):
            sources.append(key)
    return sources


def _raw_unresolved_references(references: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for reference in references:
        raw_reference = str(reference.get("raw_reference") or "").strip()
        if not raw_reference or raw_reference in seen:
            continue
        if reference.get("resolved_target_id"):
            continue
        seen.add(raw_reference)
        values.append(raw_reference)
    return values


def _normalize_marker_text(text: str) -> str:
    replacements = str.maketrans(
        {
            "ă": "a",
            "â": "a",
            "î": "i",
            "ș": "s",
            "ş": "s",
            "ț": "t",
            "ţ": "t",
            "Ă": "a",
            "Â": "a",
            "Î": "i",
            "Ș": "s",
            "Ş": "s",
            "Ț": "t",
            "Ţ": "t",
        }
    )
    return " ".join(text.translate(replacements).casefold().split())
