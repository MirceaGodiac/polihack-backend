import hashlib
import json
from pathlib import Path

from ingestion.chunks import (
    build_embedding_input_records,
    build_legal_chunks,
    contains_hardcoded_interpretation,
)
from ingestion.exporters import build_canonical_bundle, build_canonical_validation_report


FIXTURE_DIR = Path("tests/fixtures/corpus")
LEGACY_UNITS_PATH = FIXTURE_DIR / "codul_muncii_legacy_units.json"
ACT_METADATA = {
    "law_id": "ro.codul_muncii",
    "law_title": "Codul muncii",
    "status": "unknown",
}


def _bundle() -> dict:
    legacy_units = json.loads(LEGACY_UNITS_PATH.read_text(encoding="utf-8"))
    return build_canonical_bundle(
        legacy_units,
        ACT_METADATA,
        generated_at="2026-04-25T00:00:00+00:00",
        input_files=[LEGACY_UNITS_PATH.as_posix()],
        source_descriptors=[
            {
                "law_id": "ro.codul_muncii",
                "law_title": "Codul muncii",
                "source_type": "local_fixture_demo_sample",
                "source_url": None,
            }
        ],
        additional_warnings=["local_fixture_demo_sample_not_official_complete_text"],
    )


def test_each_legal_unit_gets_deterministic_legal_chunk():
    bundle = _bundle()
    legal_units = {unit["id"]: unit for unit in bundle["legal_units"]}
    chunks = bundle["legal_chunks"]
    chunks_by_unit_id = {chunk["legal_unit_id"]: chunk for chunk in chunks}

    assert set(chunks_by_unit_id) == set(legal_units)
    assert [chunk["chunk_id"] for chunk in chunks] == sorted(
        chunk["chunk_id"] for chunk in chunks
    )
    article_chunk = chunks_by_unit_id["ro.codul_muncii.art_41.alin_4"]
    assert article_chunk["chunk_id"] == "chunk.ro.codul_muncii.art_41.alin_4.0"
    assert article_chunk["text"] == legal_units["ro.codul_muncii.art_41.alin_4"]["raw_text"]
    assert article_chunk["raw_text"] == article_chunk["text"]


def test_retrieval_context_is_deterministic_structural_and_non_citable():
    bundle = _bundle()
    chunk = {
        item["legal_unit_id"]: item for item in bundle["legal_chunks"]
    }["ro.codul_muncii.art_41.alin_4"]

    assert "Unitate din Codul muncii, domeniul munca." in chunk["retrieval_context"]
    assert "Art. 41, Alin. (4)" in chunk["retrieval_context"]
    assert (
        "Context ierarhic: Legislatia Romaniei > Munca > Codul muncii > Art. 41 > Alin. (4)."
        in chunk["retrieval_context"]
    )
    assert "Referinte extrase nerezolvate: alin. (3)" in chunk["retrieval_context"]
    assert "hierarchy_path" in chunk["context_sources"]
    assert "reference_candidates_unresolved" in chunk["context_sources"]
    assert chunk["context_generation_method"] == "deterministic_v1"
    assert chunk["context_confidence"] == 0.75
    assert contains_hardcoded_interpretation(chunk["retrieval_context"]) is False
    assert "angajatorul nu poate scadea salariul" not in chunk["retrieval_context"].lower()


def test_retrieval_text_combines_context_and_raw_text_without_replacing_raw_text():
    bundle = _bundle()
    units = {unit["id"]: unit for unit in bundle["legal_units"]}
    chunk = {
        item["legal_unit_id"]: item for item in bundle["legal_chunks"]
    }["ro.codul_muncii.art_41.alin_4"]

    assert chunk["retrieval_text"].startswith(chunk["retrieval_context"])
    assert chunk["retrieval_text"].endswith(units["ro.codul_muncii.art_41.alin_4"]["raw_text"])
    assert "\n\n" in chunk["retrieval_text"]
    assert chunk["embedding_text"] == chunk["retrieval_text"]
    assert chunk["text_hash"] == hashlib.sha256(
        chunk["retrieval_text"].encode("utf-8")
    ).hexdigest()


def test_embedding_input_jsonl_records_match_chunks():
    bundle = _bundle()
    records = bundle["embeddings_input"]
    chunks = {chunk["chunk_id"]: chunk for chunk in bundle["legal_chunks"]}

    assert len(records) == len(chunks)
    assert [record["record_id"] for record in records] == sorted(
        record["record_id"] for record in records
    )
    for record in records:
        chunk = chunks[record["chunk_id"]]
        assert record["text"] == chunk["retrieval_text"]
        assert record["embedding_text"] == chunk["retrieval_text"]
        assert record["legal_unit_id"] == chunk["legal_unit_id"]
        assert record["law_id"] == chunk["law_id"]
        assert record["text_hash"] == hashlib.sha256(
            record["text"].encode("utf-8")
        ).hexdigest()


def test_build_chunk_helpers_are_deterministic():
    bundle = _bundle()
    first_chunks = build_legal_chunks(bundle["legal_units"], bundle["reference_candidates"])
    second_chunks = build_legal_chunks(bundle["legal_units"], bundle["reference_candidates"])
    first_records = build_embedding_input_records(first_chunks)
    second_records = build_embedding_input_records(second_chunks)

    assert first_chunks == second_chunks
    assert first_records == second_records


def test_validation_blocks_invalid_chunk_legal_unit_id():
    bundle = _bundle()
    chunks = [dict(chunk) for chunk in bundle["legal_chunks"]]
    chunks[0]["legal_unit_id"] = "ro.codul_muncii.art_404"

    report = build_canonical_validation_report(
        bundle["legal_units"],
        bundle["legal_edges"],
        bundle["reference_candidates"],
        chunks,
        bundle["embeddings_input"],
        parser_version="0.1.0",
    )

    assert report["import_blocking_passed"] is False
    assert "invalid_chunk_legal_unit_id" in report["blocking_errors"]


def test_validation_blocks_embedding_record_hash_mismatch():
    bundle = _bundle()
    records = [dict(record) for record in bundle["embeddings_input"]]
    records[0]["text_hash"] = "not-a-valid-hash"

    report = build_canonical_validation_report(
        bundle["legal_units"],
        bundle["legal_edges"],
        bundle["reference_candidates"],
        bundle["legal_chunks"],
        records,
        parser_version="0.1.0",
    )

    assert report["import_blocking_passed"] is False
    assert "embedding_input_hash_mismatch" in report["blocking_errors"]


def test_validation_blocks_chunk_text_that_is_not_legal_unit_raw_text():
    bundle = _bundle()
    chunks = [dict(chunk) for chunk in bundle["legal_chunks"]]
    chunks[0]["text"] = "text alterat"

    report = build_canonical_validation_report(
        bundle["legal_units"],
        bundle["legal_edges"],
        bundle["reference_candidates"],
        chunks,
        bundle["embeddings_input"],
        parser_version="0.1.0",
    )

    assert report["import_blocking_passed"] is False
    assert "chunk_text_not_faithful_to_legal_unit_raw_text" in report["blocking_errors"]


def test_validation_blocks_empty_chunk_text():
    bundle = _bundle()
    chunks = [dict(chunk) for chunk in bundle["legal_chunks"]]
    chunks[0]["text"] = ""

    report = build_canonical_validation_report(
        bundle["legal_units"],
        bundle["legal_edges"],
        bundle["reference_candidates"],
        chunks,
        bundle["embeddings_input"],
        parser_version="0.1.0",
    )

    assert report["import_blocking_passed"] is False
    assert "empty_chunk_text" in report["blocking_errors"]


def test_validation_blocks_hardcoded_legal_interpretation_in_context():
    bundle = _bundle()
    chunks = [dict(chunk) for chunk in bundle["legal_chunks"]]
    chunks[0]["retrieval_context"] = "Angajatorul nu poate scadea salariul fara act aditional."

    report = build_canonical_validation_report(
        bundle["legal_units"],
        bundle["legal_edges"],
        bundle["reference_candidates"],
        chunks,
        bundle["embeddings_input"],
        parser_version="0.1.0",
    )

    assert report["import_blocking_passed"] is False
    assert (
        "retrieval_context_contains_hardcoded_legal_interpretation"
        in report["blocking_errors"]
    )
