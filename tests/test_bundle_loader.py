from pathlib import Path

import pytest

from ingestion.bundle_loader import (
    build_chunk_index,
    build_contains_adjacency,
    build_unit_index,
    load_canonical_bundle,
    load_embeddings_input_jsonl,
    load_legal_chunks,
    load_legal_edges,
    load_legal_units,
    load_reference_candidates,
    validate_bundle_files_present,
)


FIXTURE_DIR = Path("tests/fixtures/corpus")


def test_bundle_loader_loads_prefixed_codul_muncii_artifacts():
    bundle = load_canonical_bundle(FIXTURE_DIR)

    assert len(bundle.legal_units) == 9
    assert len(bundle.legal_edges) == 8
    assert len(bundle.reference_candidates) == 4
    assert len(bundle.legal_chunks) == 9
    assert len(bundle.embeddings_input) == 9
    assert bundle.corpus_manifest["units_count"] == 9
    assert bundle.validation_report["import_blocking_passed"] is True


def test_bundle_loader_file_level_helpers_load_each_artifact():
    assert load_legal_units(FIXTURE_DIR)[0]["id"] == "ro.codul_muncii"
    assert load_legal_edges(FIXTURE_DIR)[0]["type"] == "contains"
    assert load_reference_candidates(FIXTURE_DIR)[0]["raw_reference"]
    assert load_legal_chunks(FIXTURE_DIR)[0]["retrieval_text"]
    assert load_embeddings_input_jsonl(FIXTURE_DIR)[0]["chunk_id"]


def test_missing_required_bundle_file_is_detected(tmp_path):
    missing = validate_bundle_files_present(tmp_path)

    assert sorted(missing) == [
        "corpus_manifest",
        "embeddings_input",
        "legal_chunks",
        "legal_edges",
        "legal_units",
        "reference_candidates",
        "validation_report",
    ]
    with pytest.raises(FileNotFoundError):
        load_canonical_bundle(tmp_path)


def test_indexes_and_contains_adjacency_are_deterministic():
    bundle = load_canonical_bundle(FIXTURE_DIR)

    unit_index = build_unit_index(bundle.legal_units)
    chunk_index = build_chunk_index(bundle.legal_chunks)
    adjacency = build_contains_adjacency(bundle.legal_edges)

    assert unit_index["ro.codul_muncii.art_41"]["article_number"] == "41"
    assert chunk_index["chunk.ro.codul_muncii.art_41.alin_4.0"][
        "legal_unit_id"
    ] == "ro.codul_muncii.art_41.alin_4"
    assert "ro.codul_muncii.art_41.alin_4" in adjacency["children"][
        "ro.codul_muncii.art_41"
    ]
    assert adjacency["parents"]["ro.codul_muncii.art_17.alin_3.lit_k"] == [
        "ro.codul_muncii.art_17.alin_3"
    ]
