import pytest
from pydantic import ValidationError

from apps.api.app.schemas.query import LegalUnit
from ingestion.contracts import (
    CorpusManifest,
    EmbeddingInputRecord,
    LegalChunk,
    ParserActMetadata,
    ParsedLegalEdge,
    ParsedLegalUnit,
    ReferenceCandidate,
    ValidationReport,
)


def make_parsed_unit() -> ParsedLegalUnit:
    return ParsedLegalUnit(
        id="ro.codul_muncii.art_1.alin_1",
        canonical_id="ro.codul_muncii.art_1.alin_1",
        source_id="legislatie.just.ro:53/2003",
        law_id="ro.codul_muncii",
        law_title="Codul muncii",
        act_type="lege",
        act_number="53/2003",
        status="active",
        hierarchy_path=["Art. 1", "(1)"],
        article_number="1",
        paragraph_number="1",
        raw_text="(1) Prezentul cod reglementeaza raporturile de munca.",
        normalized_text="Prezentul cod reglementeaza raporturile de munca.",
        legal_domain="dreptul_muncii",
        legal_concepts=["raporturi_de_munca"],
        source_url="https://legislatie.just.ro/Public/DetaliiDocument/53",
        parent_id="ro.codul_muncii.art_1",
        parser_warnings=["legacy-parser-shape"],
    )


def test_parsed_legal_unit_exports_api_legal_unit_shape():
    parsed = make_parsed_unit()

    legal_unit_dict = parsed.to_legal_unit_dict()
    validated = LegalUnit.model_validate(legal_unit_dict)

    assert "parser_warnings" not in legal_unit_dict
    assert validated.id == parsed.id
    assert validated.law_id == parsed.law_id
    assert validated.raw_text == parsed.raw_text
    assert validated.legal_concepts == ["raporturi_de_munca"]


def test_legal_chunk_from_legal_unit_is_deterministic_and_citable_bridge_is_preserved():
    parsed = make_parsed_unit()

    chunk = LegalChunk.from_legal_unit(parsed)
    repeated = LegalChunk.from_legal_unit(parsed)

    assert chunk.chunk_id == "chunk.ro.codul_muncii.art_1.alin_1.0"
    assert chunk.legal_unit_id == parsed.id
    assert chunk.legal_unit_ids == [parsed.id]
    assert chunk.text == parsed.raw_text
    assert chunk.retrieval_context
    assert chunk.retrieval_text.endswith(parsed.raw_text)
    assert chunk.embedding_text == chunk.retrieval_text
    assert chunk.embedding_text.strip()
    assert chunk.text_hash == repeated.text_hash


def test_parsed_legal_edge_accepts_only_canonical_edge_types():
    edge = ParsedLegalEdge(
        id="edge.ro.codul_muncii.art_1.contains.ro.codul_muncii.art_1.alin_1",
        source_id="ro.codul_muncii.art_1",
        target_id="ro.codul_muncii.art_1.alin_1",
        type="contains",
    )

    assert edge.type == "contains"
    with pytest.raises(ValidationError):
        ParsedLegalEdge(
            id="edge.invalid",
            source_id="ro.codul_muncii.art_1",
            target_id="ro.codul_muncii.art_2",
            type="retrieved_for_query",
        )


def test_reference_candidate_accepts_only_canonical_resolution_statuses():
    candidate = ReferenceCandidate(
        source_unit_id="ro.codul_muncii.art_5",
        raw_reference="art. 4 alin. (1)",
        reference_type="article",
        target_article="4",
        target_paragraph="1",
        resolution_status="resolved_high_confidence",
        resolution_confidence=0.95,
    )

    assert candidate.resolution_status == "resolved_high_confidence"
    with pytest.raises(ValidationError):
        ReferenceCandidate(
            source_unit_id="ro.codul_muncii.art_5",
            raw_reference="art. 4",
            reference_type="article",
            resolution_status="resolved",
        )


def test_embedding_input_record_preserves_chunk_and_legal_unit_ids():
    chunk = LegalChunk.from_legal_unit(make_parsed_unit())

    record = EmbeddingInputRecord.from_chunk(chunk)

    assert record.record_id == f"embedding.{chunk.chunk_id}"
    assert record.chunk_id == chunk.chunk_id
    assert record.legal_unit_id == chunk.legal_unit_id
    assert record.law_id == chunk.law_id
    assert record.text == chunk.retrieval_text
    assert record.embedding_text == chunk.retrieval_text
    assert record.text_hash == chunk.text_hash


def test_validation_report_metrics_are_unit_interval_values():
    report = ValidationReport(
        parser_version="0.1.0",
        corpus_quality=0.9,
        units_count=1,
        edges_count=1,
        chunks_count=1,
        reference_candidates_count=1,
        quality_metrics={"schema_validity": 1.0, "reference_resolution": 0.5},
    )

    assert report.quality_metrics["schema_validity"] == 1.0
    with pytest.raises(ValidationError):
        ValidationReport(
            parser_version="0.1.0",
            corpus_quality=0.9,
            units_count=1,
            edges_count=1,
            quality_metrics={"schema_validity": 1.01},
        )


def test_contracts_instantiate_without_db_or_network_dependencies():
    metadata = ParserActMetadata(
        law_id="ro.codul_muncii",
        law_title="Codul muncii",
        legal_domain="dreptul_muncii",
    )
    manifest = CorpusManifest(
        batch_id="demo_corpus_v1",
        sources=[metadata],
        output_dir="ingestion/output/demo_corpus_v1",
    )

    assert manifest.sources[0].law_id == "ro.codul_muncii"
    assert manifest.files == {}
