from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel

from ingestion.chunks import (
    CONTEXT_GENERATION_METHOD,
    build_embedding_input_records,
    build_legal_chunks,
    contains_hardcoded_interpretation,
    stable_text_hash,
)
from ingestion.contracts import ParserActMetadata, ParsedLegalUnit
from ingestion.html_cleaner import navigation_residue_count, text_cleanliness_score
from ingestion.legal_domains import get_registered_legal_domain
from ingestion.legal_ids import (
    make_canonical_id,
    make_law_id,
    make_parent_unit_id,
    make_unit_id,
)
from ingestion.normalizer import (
    contains_romanian_mojibake,
    normalize_legal_text,
    repair_romanian_mojibake,
)
from ingestion.reference_extractor import extract_references_from_units


CITABLE_LEVELS = {"articol", "alineat", "litera", "punct"}
DEFAULT_PARSER_VERSION = "0.1.0"
DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
CANONICAL_BUNDLE_FILENAMES = {
    "legal_units": "legal_units.json",
    "legal_edges": "legal_edges.json",
    "legal_chunks": "legal_chunks.json",
    "embeddings_input": "embeddings_input.jsonl",
    "corpus_manifest": "corpus_manifest.json",
    "validation_report": "validation_report.json",
    "reference_candidates": "reference_candidates.json",
}
RESOLVED_REFERENCE_STATUSES = {
    "resolved_high_confidence",
    "resolved_medium_confidence",
}
ALLOWED_REFERENCE_STATUSES = {
    "resolved_high_confidence",
    "resolved_medium_confidence",
    "candidate_ambiguous",
    "external_unresolved",
    "unresolved_ambiguous",
    "unresolved_needs_context",
    "candidate_only",
    "unresolved",
}
DEFAULT_VALIDATION_CONFIG: dict[str, Any] = {
    "raw_text_non_empty_min_for_citable_units": 1.0,
    "hierarchy_integrity_min": 1.0,
    "edge_endpoint_integrity_min": 1.0,
    "stable_id_rate_min": 1.0,
    "duplicate_free_score_min": 1.0,
    "reference_edge_min_confidence": 0.70,
    "text_cleanliness_blocking_min": 0.80,
    "source_url_demo_threshold": 0.80,
    "chunk_coverage_min": 1.0,
    "retrieval_text_non_empty_min": 1.0,
    "chunk_text_fidelity_min": 1.0,
    "embedding_input_hash_integrity_min": 1.0,
    "demo_law_id": "ro.codul_muncii",
    "demo_required_unit_ids": [
        "ro.codul_muncii",
        "ro.codul_muncii.art_17",
        "ro.codul_muncii.art_17.alin_3",
        "ro.codul_muncii.art_17.alin_3.lit_k",
        "ro.codul_muncii.art_41",
        "ro.codul_muncii.art_41.alin_1",
    ],
    "demo_any_of_unit_groups": [
        ["ro.codul_muncii.art_41.alin_3", "ro.codul_muncii.art_41.alin_4"],
    ],
}


def parsed_legal_unit_to_legal_unit_dict(
    unit: ParsedLegalUnit,
    *,
    include_parser_warnings: bool = True,
) -> dict[str, Any]:
    """Export the canonical ingestion LegalUnit dict for JSON/import consumers."""
    return unit.to_legal_unit_dict(include_parser_warnings=include_parser_warnings)


def build_parsed_legal_unit(
    *,
    law_title: str,
    raw_text: str,
    hierarchy_path: list[tuple[str, str]] | None = None,
    law_id: str | None = None,
    source_id: str | None = None,
    source_url: str | None = None,
    status: str | None = None,
    act_type: str | None = None,
    act_number: str | None = None,
    publication_date: date | None = None,
    effective_date: date | None = None,
    version_start: date | None = None,
    version_end: date | None = None,
    legal_concepts: list[str] | None = None,
    parser_warnings: list[str] | None = None,
) -> ParsedLegalUnit:
    path = hierarchy_path or []
    resolved_law_id = law_id or make_law_id(law_title)
    resolved_domain = get_registered_legal_domain(resolved_law_id) or "unknown"
    warnings = list(parser_warnings or [])

    if resolved_domain == "unknown":
        warnings.append("legal_domain_unknown")
    if source_id is None:
        warnings.append("source_id_unknown")
    if source_url is None:
        warnings.append("source_url_unknown")
    if status is None or status == "unknown":
        warnings.append("status_unknown")
    if publication_date is None:
        warnings.append("publication_date_unknown")
    if effective_date is None:
        warnings.append("effective_date_unknown")
    if version_start is None:
        warnings.append("version_start_unknown")
    if version_end is None:
        warnings.append("version_end_unknown")

    parsed_act_type, parsed_act_number = _parse_act_identity(law_title)
    resolved_act_type = act_type if act_type is not None else parsed_act_type
    resolved_act_number = act_number if act_number is not None else parsed_act_number

    repaired_raw_text = repair_romanian_mojibake(raw_text) or ""

    return ParsedLegalUnit(
        id=make_unit_id(resolved_law_id, path),
        canonical_id=make_canonical_id(resolved_law_id, path),
        source_id=source_id,
        law_id=resolved_law_id,
        law_title=law_title,
        act_type=resolved_act_type,
        act_number=resolved_act_number,
        publication_date=publication_date,
        effective_date=effective_date,
        version_start=version_start,
        version_end=version_end,
        status=status or "unknown",
        hierarchy_path=render_hierarchy_path(path),
        article_number=_level_value(path, "articol"),
        paragraph_number=_level_value(path, "alineat"),
        letter_number=_level_value(path, "litera"),
        point_number=_level_value(path, "punct"),
        raw_text=repaired_raw_text,
        normalized_text=normalize_legal_text(repaired_raw_text),
        legal_domain=resolved_domain,
        legal_concepts=legal_concepts or [],
        source_url=source_url,
        parent_id=make_parent_unit_id(resolved_law_id, path),
        children_ids=[],
        outgoing_reference_ids=[],
        incoming_reference_ids=[],
        parser_warnings=warnings,
    )


def build_legal_unit_dict(
    *,
    law_title: str,
    raw_text: str,
    hierarchy_path: list[tuple[str, str]] | None = None,
    law_id: str | None = None,
    source_id: str | None = None,
    source_url: str | None = None,
    status: str | None = None,
    parser_warnings: list[str] | None = None,
) -> dict[str, Any]:
    parsed = build_parsed_legal_unit(
        law_title=law_title,
        raw_text=raw_text,
        hierarchy_path=hierarchy_path,
        law_id=law_id,
        source_id=source_id,
        source_url=source_url,
        status=status,
        parser_warnings=parser_warnings,
    )
    return parsed_legal_unit_to_legal_unit_dict(parsed)


def legacy_unit_to_parsed_legal_unit(
    legacy_unit: Mapping[str, Any],
    act_metadata: ParserActMetadata | BaseModel | Mapping[str, Any],
) -> ParsedLegalUnit:
    metadata = _metadata_dict(act_metadata)
    law_id = metadata.get("law_id") or legacy_unit.get("corpus_id")
    law_title = metadata.get("law_title") or law_id or "unknown"
    source_url = metadata.get("source_url")
    source_id = metadata.get("source_id")
    status = metadata.get("status")
    raw_text = str(legacy_unit.get("raw_text") or legacy_unit.get("text") or "")
    path = citable_path_from_unit_id(str(legacy_unit.get("id", "")), law_id)

    return build_parsed_legal_unit(
        law_title=law_title,
        law_id=law_id,
        raw_text=raw_text,
        hierarchy_path=path,
        source_id=source_id,
        source_url=source_url,
        status=status,
        act_type=metadata.get("act_type"),
        act_number=metadata.get("act_number"),
        publication_date=metadata.get("publication_date"),
        effective_date=metadata.get("effective_date"),
        version_start=metadata.get("version_start"),
        version_end=metadata.get("version_end"),
        parser_warnings=["legacy_unit_converted"],
    )


def legacy_unit_to_legal_unit_dict(
    legacy_unit: Mapping[str, Any],
    act_metadata: ParserActMetadata | BaseModel | Mapping[str, Any],
) -> dict[str, Any]:
    parsed = legacy_unit_to_parsed_legal_unit(legacy_unit, act_metadata)
    return parsed_legal_unit_to_legal_unit_dict(parsed)


def build_canonical_bundle(
    legacy_units: list[Mapping[str, Any]],
    act_metadata: ParserActMetadata | BaseModel | Mapping[str, Any],
    *,
    parser_version: str = DEFAULT_PARSER_VERSION,
    generated_at: str = DEFAULT_GENERATED_AT,
    input_files: list[str] | None = None,
    source_descriptors: list[dict[str, Any]] | None = None,
    reference_candidates: list[Mapping[str, Any]] | None = None,
    additional_warnings: list[str] | None = None,
) -> dict[str, Any]:
    metadata = _metadata_dict(act_metadata)
    legal_units = _canonical_units_from_legacy(legacy_units, metadata)
    legal_edges = _build_contains_edges(legal_units, parser_version=parser_version)
    extracted_reference_candidates = (
        list(reference_candidates)
        if reference_candidates is not None
        else extract_references_from_units(legal_units)
    )
    exported_reference_candidates = _export_reference_candidates(extracted_reference_candidates)
    legal_chunks = build_legal_chunks(legal_units, exported_reference_candidates)
    embeddings_input = build_embedding_input_records(legal_chunks)
    validation_report = build_canonical_validation_report(
        legal_units,
        legal_edges,
        exported_reference_candidates,
        legal_chunks,
        embeddings_input,
        parser_version=parser_version,
        additional_warnings=additional_warnings,
    )
    content_hash = _hash_json(
        {
            "legal_units": legal_units,
            "legal_edges": legal_edges,
            "reference_candidates": exported_reference_candidates,
            "legal_chunks": legal_chunks,
            "embeddings_input": embeddings_input,
            "validation_report": validation_report,
        }
    )
    corpus_manifest = build_canonical_corpus_manifest(
        metadata,
        legal_units,
        legal_edges,
        exported_reference_candidates,
        legal_chunks,
        embeddings_input,
        validation_report,
        parser_version=parser_version,
        generated_at=generated_at,
        input_files=input_files or [],
        source_descriptors=source_descriptors,
        content_hash=content_hash,
    )

    return {
        "legal_units": legal_units,
        "legal_edges": legal_edges,
        "legal_chunks": legal_chunks,
        "embeddings_input": embeddings_input,
        "corpus_manifest": corpus_manifest,
        "validation_report": validation_report,
        "reference_candidates": exported_reference_candidates,
    }


def export_canonical_bundle(
    legacy_units: list[Mapping[str, Any]],
    act_metadata: ParserActMetadata | BaseModel | Mapping[str, Any],
    out_dir: str | Path,
    *,
    parser_version: str = DEFAULT_PARSER_VERSION,
    generated_at: str = DEFAULT_GENERATED_AT,
    input_files: list[str] | None = None,
    source_descriptors: list[dict[str, Any]] | None = None,
    reference_candidates: list[Mapping[str, Any]] | None = None,
    additional_warnings: list[str] | None = None,
) -> dict[str, Path]:
    bundle = build_canonical_bundle(
        legacy_units,
        act_metadata,
        parser_version=parser_version,
        generated_at=generated_at,
        input_files=input_files,
        source_descriptors=source_descriptors,
        reference_candidates=reference_candidates,
        additional_warnings=additional_warnings,
    )

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        artifact: output_dir / filename
        for artifact, filename in CANONICAL_BUNDLE_FILENAMES.items()
    }
    for artifact, path in paths.items():
        if artifact == "embeddings_input":
            _write_jsonl(bundle[artifact], path)
        else:
            _write_json(bundle[artifact], path)
    return paths


def build_canonical_corpus_manifest(
    act_metadata: Mapping[str, Any],
    legal_units: list[dict[str, Any]],
    legal_edges: list[dict[str, Any]],
    reference_candidates: list[dict[str, Any]],
    legal_chunks: list[dict[str, Any]],
    embeddings_input: list[dict[str, Any]],
    validation_report: dict[str, Any],
    *,
    parser_version: str,
    generated_at: str,
    input_files: list[str],
    source_descriptors: list[dict[str, Any]] | None,
    content_hash: str,
) -> dict[str, Any]:
    sources = source_descriptors or [
        {
            "law_id": act_metadata.get("law_id"),
            "law_title": act_metadata.get("law_title"),
            "source_id": act_metadata.get("source_id"),
            "source_url": act_metadata.get("source_url"),
        }
    ]
    warnings = sorted(set(validation_report.get("warnings", [])))
    return {
        "schema_version": "1.0.0",
        "parser_version": parser_version,
        "generated_at": generated_at,
        "source_count": len(sources),
        "units_count": len(legal_units),
        "edges_count": len(legal_edges),
        "chunks_count": len(legal_chunks),
        "embeddings_input_count": len(embeddings_input),
        "reference_candidates_count": len(reference_candidates),
        "contextual_retrieval_enabled": True,
        "context_generation_method": CONTEXT_GENERATION_METHOD,
        "input_files": sorted(input_files),
        "sources": sources,
        "files": dict(CANONICAL_BUNDLE_FILENAMES),
        "content_hash": content_hash,
        "bundle_hash": _hash_json(
            {
                "schema_version": "1.0.0",
                "parser_version": parser_version,
                "units_count": len(legal_units),
                "edges_count": len(legal_edges),
                "chunks_count": len(legal_chunks),
                "embeddings_input_count": len(embeddings_input),
                "reference_candidates_count": len(reference_candidates),
                "contextual_retrieval_enabled": True,
                "context_generation_method": CONTEXT_GENERATION_METHOD,
                "content_hash": content_hash,
            }
        ),
        "warnings": warnings,
    }


def build_canonical_validation_report(
    legal_units: list[dict[str, Any]],
    legal_edges: list[dict[str, Any]],
    reference_candidates: list[dict[str, Any]],
    legal_chunks: list[dict[str, Any]] | None = None,
    embeddings_input: list[dict[str, Any]] | None = None,
    *,
    parser_version: str,
    additional_warnings: list[str] | None = None,
    validation_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = _validation_config(validation_config)
    legal_chunks = legal_chunks or []
    embeddings_input = embeddings_input or []
    chunk_validation_enabled = bool(legal_chunks or embeddings_input)
    units_count = len(legal_units)
    edges_count = len(legal_edges)
    chunks_count = len(legal_chunks)
    embeddings_input_count = len(embeddings_input)
    unit_ids = [unit["id"] for unit in legal_units]
    unique_unit_ids = set(unit_ids)
    duplicate_ids = sorted(
        unit_id for unit_id in unique_unit_ids if unit_ids.count(unit_id) > 1
    )
    duplicate_free_score = 1.0 if not duplicate_ids else 0.0
    edge_endpoint_integrity = _safe_rate(
        sum(1 for edge in legal_edges if _edge_has_valid_endpoints(edge, unique_unit_ids)),
        edges_count,
    )
    hierarchy_integrity = _hierarchy_integrity(
        legal_units,
        legal_edges,
        unique_unit_ids,
    )
    source_url_coverage = _safe_rate(
        sum(1 for unit in legal_units if unit.get("source_url")),
        units_count,
    )
    raw_text_non_empty_rate = _safe_rate(
        sum(1 for unit in legal_units if str(unit.get("raw_text") or "").strip()),
        units_count,
    )
    raw_text_mojibake_free_rate = _safe_rate(
        sum(
            1
            for unit in legal_units
            if not contains_romanian_mojibake(str(unit.get("raw_text") or ""))
        ),
        units_count,
    )
    stable_id_rate = _safe_rate(
        sum(1 for unit in legal_units if _is_stable_legal_unit_id(str(unit.get("id", "")))),
        units_count,
    )
    unit_completeness = _safe_rate(
        sum(1 for unit in legal_units if _has_minimum_legal_unit_fields(unit)),
        units_count,
    )
    legal_domain_coverage = _safe_rate(
        sum(
            1
            for unit in legal_units
            if unit.get("legal_domain") not in (None, "", "unknown")
        ),
        units_count,
    )
    legal_concepts_coverage = _safe_rate(
        sum(1 for unit in legal_units if unit.get("legal_concepts")),
        units_count,
    )
    parser_warnings_rate = _safe_rate(
        sum(1 for unit in legal_units if unit.get("parser_warnings")),
        units_count,
    )
    text_cleanliness = text_cleanliness_score(
        [str(unit.get("raw_text") or "") for unit in legal_units]
    )
    reference_resolution_rate = (
        _safe_rate(
            sum(
                1
                for candidate in reference_candidates
                if candidate.get("resolution_status")
                in {"resolved_high_confidence", "resolved_medium_confidence"}
            ),
            len(reference_candidates),
        )
        if reference_candidates
        else 0.0
    )
    reference_candidates_present_rate = 1.0 if reference_candidates else 0.0
    chunk_coverage_rate = _chunk_coverage_rate(
        legal_units,
        legal_chunks,
        enabled=chunk_validation_enabled,
    )
    retrieval_text_non_empty_rate = _safe_rate(
        sum(1 for chunk in legal_chunks if str(chunk.get("retrieval_text") or "").strip()),
        chunks_count,
    )
    context_coverage_rate = _safe_rate(
        sum(1 for chunk in legal_chunks if str(chunk.get("retrieval_context") or "").strip()),
        chunks_count,
    )
    chunk_text_fidelity_rate = _chunk_text_fidelity_rate(legal_units, legal_chunks)
    embedding_input_hash_integrity = _embedding_hash_integrity(
        legal_chunks,
        embeddings_input,
    )
    quality_metrics = {
        "unit_completeness": unit_completeness,
        "raw_text_non_empty_rate": raw_text_non_empty_rate,
        "raw_text_mojibake_free_rate": raw_text_mojibake_free_rate,
        "stable_id_rate": stable_id_rate,
        "hierarchy_integrity": hierarchy_integrity,
        "edge_endpoint_integrity": edge_endpoint_integrity,
        "duplicate_free_score": duplicate_free_score,
        "source_url_coverage": source_url_coverage,
        "text_cleanliness": text_cleanliness,
        "reference_candidates_present_rate": reference_candidates_present_rate,
        "reference_resolution_rate": reference_resolution_rate,
        "legal_domain_coverage": legal_domain_coverage,
        "legal_concepts_coverage": legal_concepts_coverage,
        "parser_warnings_rate": parser_warnings_rate,
        "chunk_coverage_rate": chunk_coverage_rate,
        "retrieval_text_non_empty_rate": retrieval_text_non_empty_rate,
        "context_coverage_rate": context_coverage_rate,
        "chunk_text_fidelity_rate": chunk_text_fidelity_rate,
        "embedding_input_hash_integrity": embedding_input_hash_integrity,
    }
    demo_path_passed = _demo_path_passed(
        legal_units,
        legal_edges,
        validation_config=config,
    )
    blocking_errors = _canonical_blocking_errors(
        legal_units=legal_units,
        legal_edges=legal_edges,
        reference_candidates=reference_candidates,
        legal_chunks=legal_chunks,
        embeddings_input=embeddings_input,
        unit_ids=unit_ids,
        duplicate_ids=duplicate_ids,
        quality_metrics=quality_metrics,
        validation_config=config,
        chunk_validation_enabled=chunk_validation_enabled,
    )
    warnings = _canonical_bundle_warnings(
        legal_units,
        reference_candidates,
        legal_chunks=legal_chunks,
        embeddings_input=embeddings_input,
        quality_metrics=quality_metrics,
        demo_path_passed=demo_path_passed,
        additional_warnings=additional_warnings,
    )
    corpus_quality = _corpus_quality(quality_metrics)
    import_blocking_passed = not blocking_errors
    validation_notes = _validation_notes(
        quality_metrics=quality_metrics,
        reference_candidates=reference_candidates,
        demo_path_passed=demo_path_passed,
        warnings=warnings,
    )
    return {
        "schema_version": "1.0",
        "parser_version": parser_version,
        "corpus_quality": corpus_quality,
        "import_blocking_passed": import_blocking_passed,
        "demo_path_passed": demo_path_passed,
        "units_count": units_count,
        "edges_count": edges_count,
        "chunks_count": chunks_count,
        "embeddings_input_count": embeddings_input_count,
        "reference_candidates_count": len(reference_candidates),
        "blocking_errors": blocking_errors,
        "quality_metrics": quality_metrics,
        "validation_config": config,
        "validation_notes": validation_notes,
        "warnings": warnings,
        "errors": blocking_errors,
    }


def render_hierarchy_path(hierarchy_path: list[tuple[str, str]]) -> list[str]:
    rendered: list[str] = []
    for level_type, value in hierarchy_path:
        level = level_type.lower()
        if level == "articol":
            rendered.append(f"Art. {_strip_article_label(value)}")
        elif level == "alineat":
            rendered.append(f"Alin. ({_strip_wrapping_punctuation(value)})")
        elif level == "litera":
            rendered.append(f"Lit. {_strip_wrapping_punctuation(value).lower()})")
        elif level == "punct":
            rendered.append(f"Pct. {_strip_wrapping_punctuation(value)}")
        else:
            rendered.append(str(value))
    return rendered


def citable_path_from_unit_id(unit_id: str, law_id: str | None = None) -> list[tuple[str, str]]:
    if law_id and unit_id.startswith(f"{law_id}."):
        raw_segments = unit_id.removeprefix(f"{law_id}.").split(".")
    else:
        raw_segments = unit_id.split(".")[2:]

    path: list[tuple[str, str]] = []
    for segment in raw_segments:
        if segment.startswith("art_"):
            path.append(("articol", segment.removeprefix("art_")))
        elif segment.startswith("alin_"):
            path.append(("alineat", segment.removeprefix("alin_")))
        elif segment.startswith("lit_"):
            path.append(("litera", segment.removeprefix("lit_")))
        elif segment.startswith("pct_"):
            path.append(("punct", segment.removeprefix("pct_")))
    return path


def _canonical_units_from_legacy(
    legacy_units: list[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    root_unit = _build_act_root_unit(metadata)
    units_by_id: dict[str, dict[str, Any]] = {root_unit["id"]: root_unit}

    for legacy_unit in legacy_units:
        unit = legacy_unit_to_legal_unit_dict(legacy_unit, metadata)
        unit["hierarchy_path"] = _bundle_hierarchy_path(unit)
        units_by_id[unit["id"]] = unit

    return [units_by_id[unit_id] for unit_id in sorted(units_by_id)]


def _build_act_root_unit(metadata: Mapping[str, Any]) -> dict[str, Any]:
    law_title = str(metadata.get("law_title") or "unknown")
    law_id = metadata.get("law_id") or make_law_id(law_title)
    parsed = build_parsed_legal_unit(
        law_title=law_title,
        law_id=law_id,
        raw_text=law_title,
        hierarchy_path=[],
        source_id=metadata.get("source_id"),
        source_url=metadata.get("source_url"),
        status=metadata.get("status"),
        act_type=metadata.get("act_type"),
        act_number=metadata.get("act_number"),
        publication_date=metadata.get("publication_date"),
        effective_date=metadata.get("effective_date"),
        version_start=metadata.get("version_start"),
        version_end=metadata.get("version_end"),
        parser_warnings=["act_root_unit_from_metadata"],
    )
    unit = parsed_legal_unit_to_legal_unit_dict(parsed)
    unit["hierarchy_path"] = _bundle_hierarchy_path(unit)
    return unit


def _bundle_hierarchy_path(unit: Mapping[str, Any]) -> list[str]:
    domain_label = _domain_label(str(unit.get("legal_domain") or "unknown"))
    prefix = ["Legislatia Romaniei", domain_label, str(unit.get("law_title") or unit.get("law_id"))]
    existing = list(unit.get("hierarchy_path") or [])
    if existing[: len(prefix)] == prefix:
        return existing
    return prefix + existing


def _domain_label(domain: str) -> str:
    if domain == "unknown":
        return "Unknown"
    return domain.replace("_", " ").title()


def _build_contains_edges(
    legal_units: list[dict[str, Any]],
    *,
    parser_version: str,
) -> list[dict[str, Any]]:
    unit_ids = {unit["id"] for unit in legal_units}
    edges: list[dict[str, Any]] = []
    for unit in legal_units:
        parent_id = unit.get("parent_id")
        if not parent_id or parent_id not in unit_ids:
            continue
        edge_id = f"edge.contains.{parent_id}.{unit['id']}"
        edges.append(
            {
                "id": edge_id,
                "source_id": parent_id,
                "target_id": unit["id"],
                "type": "contains",
                "weight": 1.0,
                "confidence": 1.0,
                "metadata": {
                    "source": "canonical_bundle_export",
                    "parser_version": parser_version,
                },
            }
        )
    return sorted(edges, key=lambda edge: edge["id"])


def _export_reference_candidates(
    reference_candidates: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    # P3 deliberately does not create reference edges. Legacy candidates are not
    # promoted unless a later phase provides a reliable canonical resolver.
    if not reference_candidates:
        return []
    exported: list[dict[str, Any]] = []
    for candidate in reference_candidates:
        exported.append(
            {
                "source_unit_id": candidate.get("source_unit_id", ""),
                "raw_reference": candidate.get("raw_reference", ""),
                "reference_type": candidate.get("reference_type", "unknown"),
                "target_law_hint": candidate.get("target_law_hint"),
                "target_article": candidate.get("target_article"),
                "target_paragraph": candidate.get("target_paragraph"),
                "target_letter": candidate.get("target_letter"),
                "target_point": candidate.get("target_point"),
                "target_thesis": candidate.get("target_thesis"),
                "resolved_target_id": candidate.get("resolved_target_id") or candidate.get("target_id"),
                "resolution_status": candidate.get("resolution_status") or candidate.get("status") or "candidate_only",
                "resolution_confidence": candidate.get("resolution_confidence") or candidate.get("confidence") or 0.0,
                "resolver_notes": candidate.get("resolver_notes") or [],
            }
        )
    return sorted(
        exported,
        key=lambda item: (
            item["source_unit_id"],
            item["raw_reference"],
            item["reference_type"],
            item.get("target_article") or "",
            item.get("target_paragraph") or "",
            item.get("target_letter") or "",
            item.get("target_point") or "",
            item.get("target_thesis") or "",
        ),
    )


def _validation_config(
    validation_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    config = dict(DEFAULT_VALIDATION_CONFIG)
    config.update(validation_config or {})
    return config


def _hierarchy_integrity(
    legal_units: list[dict[str, Any]],
    legal_edges: list[dict[str, Any]],
    unit_ids: set[str],
) -> float:
    expected_links = [
        (unit.get("parent_id"), unit.get("id"))
        for unit in legal_units
        if unit.get("parent_id")
    ]
    valid_contains_edges = {
        (edge.get("source_id"), edge.get("target_id"))
        for edge in legal_edges
        if edge.get("type") == "contains"
        and edge.get("source_id") in unit_ids
        and edge.get("target_id") in unit_ids
    }
    valid_links = sum(
        1
        for parent_id, unit_id in expected_links
        if parent_id in unit_ids and (parent_id, unit_id) in valid_contains_edges
    )
    return _safe_rate(valid_links, len(expected_links))


def _edge_has_valid_endpoints(edge: Mapping[str, Any], unit_ids: set[str]) -> bool:
    return edge.get("source_id") in unit_ids and edge.get("target_id") in unit_ids


def _chunk_coverage_rate(
    legal_units: list[dict[str, Any]],
    legal_chunks: list[dict[str, Any]],
    *,
    enabled: bool,
) -> float:
    if not enabled:
        return 1.0
    unit_ids = {unit["id"] for unit in legal_units}
    chunk_unit_ids = {chunk.get("legal_unit_id") for chunk in legal_chunks}
    return _safe_rate(len(unit_ids & chunk_unit_ids), len(unit_ids))


def _chunk_text_fidelity_rate(
    legal_units: list[dict[str, Any]],
    legal_chunks: list[dict[str, Any]],
) -> float:
    units_by_id = {unit["id"]: unit for unit in legal_units}
    if not legal_chunks:
        return 1.0
    faithful = 0
    for chunk in legal_chunks:
        unit = units_by_id.get(chunk.get("legal_unit_id"))
        if unit and chunk.get("text") == unit.get("raw_text"):
            faithful += 1
    return _safe_rate(faithful, len(legal_chunks))


def _embedding_hash_integrity(
    legal_chunks: list[dict[str, Any]],
    embeddings_input: list[dict[str, Any]],
) -> float:
    if not embeddings_input:
        return 1.0 if not legal_chunks else 0.0
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in legal_chunks}
    valid = 0
    for record in embeddings_input:
        chunk = chunks_by_id.get(record.get("chunk_id"))
        text = str(record.get("text") or "")
        if (
            chunk
            and record.get("legal_unit_id") == chunk.get("legal_unit_id")
            and text == chunk.get("retrieval_text")
            and record.get("text_hash") == stable_text_hash(text)
        ):
            valid += 1
    return _safe_rate(valid, len(embeddings_input))


def _canonical_blocking_errors(
    *,
    legal_units: list[dict[str, Any]],
    legal_edges: list[dict[str, Any]],
    reference_candidates: list[dict[str, Any]],
    legal_chunks: list[dict[str, Any]],
    embeddings_input: list[dict[str, Any]],
    unit_ids: list[str],
    duplicate_ids: list[str],
    quality_metrics: Mapping[str, float],
    validation_config: Mapping[str, Any],
    chunk_validation_enabled: bool,
) -> list[str]:
    unit_id_set = set(unit_ids)
    chunk_ids = {chunk.get("chunk_id") for chunk in legal_chunks}
    errors: set[str] = set()

    if duplicate_ids:
        errors.add("duplicate_legal_unit_id")
    if unit_ids != sorted(unit_ids):
        errors.add("canonical_units_not_sorted_by_id")
    if quality_metrics["stable_id_rate"] < validation_config["stable_id_rate_min"]:
        errors.add("unstable_legal_unit_id")
    if _empty_citable_units(legal_units):
        errors.add("empty_raw_text_for_citable_unit")
    if _citable_units_with_mojibake(legal_units):
        errors.add("raw_text_contains_romanian_mojibake")
    if quality_metrics["hierarchy_integrity"] < validation_config["hierarchy_integrity_min"]:
        errors.add("hierarchy_integrity_below_threshold")
    if quality_metrics["edge_endpoint_integrity"] < validation_config["edge_endpoint_integrity_min"]:
        errors.add("invalid_edge_endpoint")
    if quality_metrics["text_cleanliness"] < validation_config["text_cleanliness_blocking_min"]:
        errors.add("raw_text_navigation_contamination")

    for edge in legal_edges:
        if edge.get("type") != "references":
            continue
        confidence = float(edge.get("confidence") or 0.0)
        if (
            edge.get("target_id") not in unit_id_set
            or confidence < validation_config["reference_edge_min_confidence"]
        ):
            errors.add("invalid_reference_edge")

    for candidate in reference_candidates:
        if candidate.get("source_unit_id") not in unit_id_set:
            errors.add("invalid_reference_candidate_source_unit_id")
        if not str(candidate.get("raw_reference") or "").strip():
            errors.add("invalid_reference_candidate_empty_raw_reference")
        status = candidate.get("resolution_status")
        if status not in ALLOWED_REFERENCE_STATUSES:
            errors.add("invalid_reference_candidate_resolution_status")
        resolved_target_id = candidate.get("resolved_target_id")
        if resolved_target_id and resolved_target_id not in unit_id_set:
            errors.add("invalid_reference_candidate_resolved_target_id")

    if chunk_validation_enabled:
        if quality_metrics["chunk_coverage_rate"] < validation_config["chunk_coverage_min"]:
            errors.add("chunk_coverage_below_threshold")
        if quality_metrics["retrieval_text_non_empty_rate"] < validation_config["retrieval_text_non_empty_min"]:
            errors.add("empty_retrieval_text")
        if quality_metrics["chunk_text_fidelity_rate"] < validation_config["chunk_text_fidelity_min"]:
            errors.add("chunk_text_not_faithful_to_legal_unit_raw_text")
        if quality_metrics["embedding_input_hash_integrity"] < validation_config["embedding_input_hash_integrity_min"]:
            errors.add("embedding_input_hash_mismatch")

        for chunk in legal_chunks:
            if chunk.get("legal_unit_id") not in unit_id_set:
                errors.add("invalid_chunk_legal_unit_id")
            if not str(chunk.get("text") or "").strip():
                errors.add("empty_chunk_text")
            if not str(chunk.get("retrieval_text") or "").strip():
                errors.add("empty_retrieval_text")
            if contains_hardcoded_interpretation(str(chunk.get("retrieval_context") or "")):
                errors.add("retrieval_context_contains_hardcoded_legal_interpretation")

        for record in embeddings_input:
            if record.get("chunk_id") not in chunk_ids:
                errors.add("invalid_embedding_input_chunk_id")
            if record.get("text_hash") != stable_text_hash(str(record.get("text") or "")):
                errors.add("embedding_input_hash_mismatch")

    return sorted(errors)


def _empty_citable_units(legal_units: list[dict[str, Any]]) -> list[str]:
    return [
        str(unit.get("id"))
        for unit in legal_units
        if _is_citable_unit(unit) and not str(unit.get("raw_text") or "").strip()
    ]


def _citable_units_with_mojibake(legal_units: list[dict[str, Any]]) -> list[str]:
    return [
        str(unit.get("id"))
        for unit in legal_units
        if _is_citable_unit(unit)
        and contains_romanian_mojibake(str(unit.get("raw_text") or ""))
    ]


def _is_citable_unit(unit: Mapping[str, Any]) -> bool:
    unit_id = str(unit.get("id") or "")
    return bool(
        unit.get("article_number")
        or unit.get("paragraph_number")
        or unit.get("letter_number")
        or unit.get("point_number")
        or re.search(r"\.(?:art|alin|lit|pct)_", unit_id)
    )


def _corpus_quality(quality_metrics: Mapping[str, float]) -> float:
    return round(
        0.25 * quality_metrics["unit_completeness"]
        + 0.20 * quality_metrics["hierarchy_integrity"]
        + 0.20 * quality_metrics["reference_resolution_rate"]
        + 0.15 * quality_metrics["source_url_coverage"]
        + 0.10 * quality_metrics["text_cleanliness"]
        + 0.10 * quality_metrics["duplicate_free_score"],
        4,
    )


def _demo_path_passed(
    legal_units: list[dict[str, Any]],
    legal_edges: list[dict[str, Any]],
    *,
    validation_config: Mapping[str, Any],
) -> bool | None:
    unit_ids = {unit["id"] for unit in legal_units}
    demo_law_id = str(validation_config["demo_law_id"])
    if not any(unit_id == demo_law_id or unit_id.startswith(f"{demo_law_id}.") for unit_id in unit_ids):
        return None

    required_ids = set(validation_config["demo_required_unit_ids"])
    if not required_ids.issubset(unit_ids):
        return False
    for group in validation_config["demo_any_of_unit_groups"]:
        if not any(unit_id in unit_ids for unit_id in group):
            return False

    units_by_id = {unit["id"]: unit for unit in legal_units}
    contains_edges = {
        (edge.get("source_id"), edge.get("target_id"))
        for edge in legal_edges
        if edge.get("type") == "contains"
    }
    for unit_id in required_ids:
        unit = units_by_id[unit_id]
        if not str(unit.get("raw_text") or "").strip():
            return False
        if contains_romanian_mojibake(str(unit.get("raw_text") or "")):
            return False
        if unit.get("legal_domain") != "munca":
            return False
        hierarchy_path = unit.get("hierarchy_path") or []
        if hierarchy_path[:3] != ["Legislatia Romaniei", "Munca", "Codul muncii"]:
            return False
        parent_id = unit.get("parent_id")
        if parent_id and (
            parent_id not in unit_ids or (parent_id, unit_id) not in contains_edges
        ):
            return False
    return not any(edge.get("type") == "references" for edge in legal_edges)


def _validation_notes(
    *,
    quality_metrics: Mapping[str, float],
    reference_candidates: list[dict[str, Any]],
    demo_path_passed: bool | None,
    warnings: list[str],
) -> list[str]:
    notes = ["unknown_stays_unknown_policy_applied"]
    if quality_metrics["legal_concepts_coverage"] < 1.0:
        notes.append("legal_concepts_coverage_is_informational_only")
    if reference_candidates and quality_metrics["reference_resolution_rate"] < 1.0:
        notes.append("unresolved_references_are_non_blocking_in_v1")
    if "source_url_coverage_below_demo_threshold_explained_by_local_fixture" in warnings:
        notes.append("source_url_coverage_gap_is_non_blocking_for_local_fixture")
    if quality_metrics["context_coverage_rate"] < 1.0:
        notes.append("context_coverage_gap_is_non_blocking_when_chunk_import_is_valid")
    if demo_path_passed is True:
        notes.append("codul_muncii_demo_path_validated")
    elif demo_path_passed is False:
        notes.append("codul_muncii_demo_path_failed")
    return notes


def _canonical_bundle_warnings(
    legal_units: list[dict[str, Any]],
    reference_candidates: list[dict[str, Any]],
    *,
    legal_chunks: list[dict[str, Any]],
    embeddings_input: list[dict[str, Any]],
    quality_metrics: Mapping[str, float],
    demo_path_passed: bool | None,
    additional_warnings: list[str] | None = None,
) -> list[str]:
    warnings = {"unknown_fields_left_null_by_policy"}
    warnings.update(additional_warnings or [])
    is_local_fixture = "local_fixture_demo_sample_not_official_complete_text" in warnings
    source_url_coverage = quality_metrics.get("source_url_coverage", 1.0)
    reference_resolution_rate = quality_metrics.get("reference_resolution_rate", 1.0)
    if any(unit.get("source_url") is None for unit in legal_units):
        warnings.add("source_url_unknown")
    if any(unit.get("source_id") is None for unit in legal_units):
        warnings.add("source_id_unknown")
    if any(unit.get("status") == "unknown" for unit in legal_units):
        warnings.add("status_unknown")
    if any(unit.get("publication_date") is None for unit in legal_units):
        warnings.add("publication_date_unknown")
    empty_concepts = sum(1 for unit in legal_units if not unit.get("legal_concepts"))
    if empty_concepts >= max(1, len(legal_units) // 2):
        warnings.add("legal_concepts_empty_for_most_units_by_v1_policy")
    unresolved_references = any(
        candidate.get("resolution_status")
        not in {"resolved_high_confidence", "resolved_medium_confidence"}
        for candidate in reference_candidates
    )
    if reference_candidates and unresolved_references:
        warnings.add("reference_candidates_extracted_unresolved")
        warnings.add("reference_resolution_deferred_to_later_phase")
        warnings.add("reference_resolution_rate_informational_in_v1")
    if not reference_candidates:
        warnings.add("reference_candidates_not_implemented_or_not_all_resolved")
    if any(unit.get("legal_domain") == "unknown" for unit in legal_units):
        warnings.add("legal_domain_unknown")
    if any(
        contains_romanian_mojibake(str(unit.get("raw_text") or ""))
        for unit in legal_units
    ):
        warnings.add("raw_text_contains_romanian_mojibake")
    if any(navigation_residue_count(str(unit.get("raw_text") or "")) for unit in legal_units):
        warnings.add("text_cleanliness_possible_navigation_residue")
    if is_local_fixture and source_url_coverage < 1.0:
        warnings.add("source_url_coverage_below_demo_threshold_explained_by_local_fixture")
    if reference_candidates and reference_resolution_rate < 1.0:
        warnings.add("reference_resolution_rate_informational_in_v1")
    if demo_path_passed is False:
        warnings.add("codul_muncii_demo_path_missing_critical_units")
    if legal_chunks:
        warnings.add("contextual_retrieval_context_derived_not_citable")
        warnings.add("embeddings_vectors_not_generated_in_p8")
    if legal_chunks and quality_metrics.get("context_coverage_rate", 1.0) < 1.0:
        warnings.add("context_sources_missing_for_some_chunks")
    if legal_chunks and any("reference_candidates_unresolved" in (chunk.get("context_sources") or []) for chunk in legal_chunks):
        warnings.add("reference_candidates_used_as_unresolved_context_hints")
    if embeddings_input and not legal_chunks:
        warnings.add("embeddings_input_present_without_legal_chunks")
    return sorted(warnings)


def _has_minimum_legal_unit_fields(unit: Mapping[str, Any]) -> bool:
    required = [
        "id",
        "canonical_id",
        "law_id",
        "law_title",
        "status",
        "hierarchy_path",
        "raw_text",
        "legal_domain",
        "legal_concepts",
    ]
    return all(key in unit and unit.get(key) is not None for key in required)


def _is_stable_legal_unit_id(unit_id: str) -> bool:
    if re.search(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        unit_id,
        re.IGNORECASE,
    ):
        return False
    return bool(re.fullmatch(r"ro\.[a-z0-9_]+(?:\.[a-z]+_[a-z0-9_]+)*", unit_id))


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return round(numerator / denominator, 4)


def _write_json(payload: Any, path: Path) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(records: list[Mapping[str, Any]], path: Path) -> None:
    lines = [
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for record in records
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _hash_json(payload: Any) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _metadata_dict(metadata: ParserActMetadata | BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(metadata, BaseModel):
        return metadata.model_dump()
    return dict(metadata)


def _level_value(hierarchy_path: list[tuple[str, str]], level_type: str) -> str | None:
    for current_type, value in reversed(hierarchy_path):
        if current_type.lower() == level_type:
            if level_type == "articol":
                return _strip_article_label(value)
            return _strip_wrapping_punctuation(value).lower() if level_type == "litera" else _strip_wrapping_punctuation(value)
    return None


def _strip_article_label(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"^(?:art\.?|articolul)\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def _strip_wrapping_punctuation(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"^(?:alin\.?|lit\.?|pct\.?)\s*", "", text, flags=re.IGNORECASE)
    return text.strip().strip("() .")


def _parse_act_identity(law_title: str) -> tuple[str | None, str | None]:
    text = str(law_title).strip()
    match = re.search(r"\b(legea|lege)\s+(?:nr\.?\s*)?(\d+\s*/\s*\d+)\b", text, re.IGNORECASE)
    if match:
        return "lege", re.sub(r"\s+", "", match.group(2))
    if re.search(r"\bcodul\b", text, re.IGNORECASE):
        return "cod", None
    return None, None
