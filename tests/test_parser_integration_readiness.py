from pathlib import Path

from ingestion.bundle_loader import load_canonical_bundle


FIXTURE_DIR = Path("tests/fixtures/corpus")


def test_codul_muncii_bundle_manifest_and_validation_counts_match_artifacts():
    bundle = load_canonical_bundle(FIXTURE_DIR)
    manifest = bundle.corpus_manifest
    validation = bundle.validation_report

    assert manifest["units_count"] == len(bundle.legal_units)
    assert manifest["edges_count"] == len(bundle.legal_edges)
    assert manifest["reference_candidates_count"] == len(bundle.reference_candidates)
    assert manifest["chunks_count"] == len(bundle.legal_chunks)
    assert manifest["embeddings_input_count"] == len(bundle.embeddings_input)
    assert validation["units_count"] == len(bundle.legal_units)
    assert validation["edges_count"] == len(bundle.legal_edges)
    assert validation["reference_candidates_count"] == len(bundle.reference_candidates)
    assert validation["chunks_count"] == len(bundle.legal_chunks)
    assert validation["embeddings_input_count"] == len(bundle.embeddings_input)


def test_codul_muncii_bundle_cross_artifact_integrity():
    bundle = load_canonical_bundle(FIXTURE_DIR)
    unit_ids = {unit["id"] for unit in bundle.legal_units}
    chunk_ids = {chunk["chunk_id"] for chunk in bundle.legal_chunks}
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in bundle.legal_chunks}

    assert {edge["type"] for edge in bundle.legal_edges} == {"contains"}
    for edge in bundle.legal_edges:
        assert edge["source_id"] in unit_ids
        assert edge["target_id"] in unit_ids

    for candidate in bundle.reference_candidates:
        assert candidate["source_unit_id"] in unit_ids
        assert candidate["resolved_target_id"] is None

    for chunk in bundle.legal_chunks:
        assert chunk["legal_unit_id"] in unit_ids
        assert chunk["metadata"]["retrieval_context_is_citable"] is False
        assert chunk["metadata"]["text_source"] == "LegalUnit.raw_text"

    for record in bundle.embeddings_input:
        chunk = chunks_by_id[record["chunk_id"]]
        assert record["chunk_id"] in chunk_ids
        assert record["legal_unit_id"] == chunk["legal_unit_id"]
        assert record["text"] == chunk["retrieval_text"]
        assert record["metadata"]["retrieval_text_is_citable"] is False


def test_codul_muncii_bundle_validation_gates_are_ready():
    bundle = load_canonical_bundle(FIXTURE_DIR)
    report = bundle.validation_report

    assert report["import_blocking_passed"] is True
    assert report["demo_path_passed"] is True
    assert report["blocking_errors"] == []
    assert report["quality_metrics"]["chunk_coverage_rate"] == 1.0
    assert report["quality_metrics"]["embedding_input_hash_integrity"] == 1.0
    assert "local_fixture_demo_sample_not_official_complete_text" in report["warnings"]
    assert "reference_resolution_deferred_to_later_phase" in report["warnings"]
    assert "contextual_retrieval_context_derived_not_citable" in report["warnings"]
