import json
from pathlib import Path

from apps.api.app.schemas.query import LegalUnit
from ingestion.exporters import build_canonical_bundle, export_canonical_bundle


MINI_CODUL_MUNCII_UNITS = [
    {
        "id": "ro.codul_muncii.art_41",
        "type": "articol",
        "raw_text": "Art. 41\nDreptul la munca nu poate fi ingradit.",
        "hierarchy_path": ["41"],
        "corpus_id": "ro.codul_muncii",
    },
    {
        "id": "ro.codul_muncii.art_41.alin_1",
        "type": "alineat",
        "raw_text": "(1) Dreptul la munca este garantat.",
        "hierarchy_path": ["41", "1"],
        "corpus_id": "ro.codul_muncii",
    },
    {
        "id": "ro.codul_muncii.art_17",
        "type": "articol",
        "raw_text": "Art. 17\nAnterior incheierii contractului individual de munca.",
        "hierarchy_path": ["17"],
        "corpus_id": "ro.codul_muncii",
    },
    {
        "id": "ro.codul_muncii.art_17.alin_3",
        "type": "alineat",
        "raw_text": "(3) Persoana selectata va fi informata cu privire la:",
        "hierarchy_path": ["17", "3"],
        "corpus_id": "ro.codul_muncii",
    },
    {
        "id": "ro.codul_muncii.art_17.alin_3.lit_k",
        "type": "litera",
        "raw_text": "k) durata perioadei de proba.",
        "hierarchy_path": ["17", "3", "k"],
        "corpus_id": "ro.codul_muncii",
    },
]

ACT_METADATA = {
    "law_id": "ro.codul_muncii",
    "law_title": "Codul muncii",
    "status": "unknown",
}
FIXTURE_DIR = Path("tests/fixtures/corpus/mini_codul_muncii")
REFERENCE_FIXTURE_PATH = Path("tests/fixtures/corpus/reference_extraction_legacy_units.json")


def test_canonical_bundle_generation_writes_required_files(tmp_path):
    paths = export_canonical_bundle(
        MINI_CODUL_MUNCII_UNITS,
        ACT_METADATA,
        tmp_path,
        generated_at="2026-04-25T00:00:00+00:00",
        input_files=["tests/fixtures/corpus/mini_codul_muncii/legacy_units.json"],
    )

    assert set(paths) == {
        "legal_units",
        "legal_edges",
        "legal_chunks",
        "embeddings_input",
        "corpus_manifest",
        "validation_report",
        "reference_candidates",
    }
    for path in paths.values():
        assert path.exists()


def test_legal_units_are_sorted_and_backend_compatible():
    bundle = build_canonical_bundle(
        MINI_CODUL_MUNCII_UNITS,
        ACT_METADATA,
        generated_at="2026-04-25T00:00:00+00:00",
    )

    legal_units = bundle["legal_units"]
    ids = [unit["id"] for unit in legal_units]

    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))
    assert "ro.codul_muncii.art_41" in ids
    assert "ro.codul_muncii.art_41.alin_1" in ids
    assert "ro.codul_muncii.art_17.alin_3.lit_k" in ids
    for unit in legal_units:
        LegalUnit.model_validate(unit)


def test_bundle_hierarchy_path_includes_frontend_context():
    bundle = build_canonical_bundle(MINI_CODUL_MUNCII_UNITS, ACT_METADATA)
    units = {unit["id"]: unit for unit in bundle["legal_units"]}

    assert units["ro.codul_muncii.art_41"]["hierarchy_path"] == [
        "Legislatia Romaniei",
        "Munca",
        "Codul muncii",
        "Art. 41",
    ]
    assert units["ro.codul_muncii.art_41.alin_1"]["hierarchy_path"] == [
        "Legislatia Romaniei",
        "Munca",
        "Codul muncii",
        "Art. 41",
        "Alin. (1)",
    ]
    assert units["ro.codul_muncii.art_17.alin_3.lit_k"]["hierarchy_path"] == [
        "Legislatia Romaniei",
        "Munca",
        "Codul muncii",
        "Art. 17",
        "Alin. (3)",
        "Lit. k)",
    ]


def test_legal_edges_are_deterministic_contains_edges_with_valid_endpoints():
    bundle = build_canonical_bundle(MINI_CODUL_MUNCII_UNITS, ACT_METADATA)
    legal_units = {unit["id"] for unit in bundle["legal_units"]}
    edges = bundle["legal_edges"]
    edge_pairs = {(edge["source_id"], edge["target_id"], edge["type"]) for edge in edges}

    assert ("ro.codul_muncii", "ro.codul_muncii.art_41", "contains") in edge_pairs
    assert ("ro.codul_muncii.art_41", "ro.codul_muncii.art_41.alin_1", "contains") in edge_pairs
    assert ("ro.codul_muncii", "ro.codul_muncii.art_17", "contains") in edge_pairs
    assert ("ro.codul_muncii.art_17", "ro.codul_muncii.art_17.alin_3", "contains") in edge_pairs
    assert (
        "ro.codul_muncii.art_17.alin_3",
        "ro.codul_muncii.art_17.alin_3.lit_k",
        "contains",
    ) in edge_pairs
    for edge in edges:
        assert edge["type"] == "contains"
        assert edge["source_id"] in legal_units
        assert edge["target_id"] in legal_units
        assert edge["weight"] == 1.0
        assert edge["confidence"] == 1.0
        assert edge["metadata"]["parser_version"] == "0.1.0"


def test_reference_candidates_file_exists_even_when_empty(tmp_path):
    export_canonical_bundle(MINI_CODUL_MUNCII_UNITS, ACT_METADATA, tmp_path)

    candidates = json.loads((tmp_path / "reference_candidates.json").read_text(encoding="utf-8"))

    assert candidates == []


def test_reference_candidates_are_exported_without_reference_edges():
    legacy_units = json.loads(REFERENCE_FIXTURE_PATH.read_text(encoding="utf-8"))
    bundle = build_canonical_bundle(legacy_units, ACT_METADATA)
    candidates = bundle["reference_candidates"]

    raw_refs = {candidate["raw_reference"] for candidate in candidates}
    assert "art. 41 alin. (3)" in raw_refs
    assert "art. 17 alin. (3) lit. k)" in raw_refs
    assert "Legea nr. 53/2003" in raw_refs
    assert "O.U.G. nr. 195/2002" in raw_refs
    assert "Codul civil" in raw_refs
    assert "prezentul cod" in raw_refs
    assert candidates == sorted(
        candidates,
        key=lambda item: (
            item["source_unit_id"],
            item["raw_reference"],
            item["reference_type"],
            item.get("target_article") or "",
            item.get("target_paragraph") or "",
            item.get("target_letter") or "",
        ),
    )
    assert {edge["type"] for edge in bundle["legal_edges"]} == {"contains"}
    assert bundle["validation_report"]["reference_candidates_count"] == len(candidates)
    assert bundle["validation_report"]["quality_metrics"]["reference_resolution_rate"] == 0.0
    assert "reference_candidates_extracted_unresolved" in bundle["validation_report"]["warnings"]
    assert "reference_resolution_deferred_to_later_phase" in bundle["validation_report"]["warnings"]


def test_validation_report_includes_v1_unknown_policy_warnings():
    bundle = build_canonical_bundle(MINI_CODUL_MUNCII_UNITS, ACT_METADATA)
    report = bundle["validation_report"]
    warnings = set(report["warnings"])

    assert report["units_count"] == 6
    assert report["edges_count"] == 5
    assert report["quality_metrics"]["duplicate_free_score"] == 1.0
    assert report["quality_metrics"]["hierarchy_integrity"] == 1.0
    assert report["quality_metrics"]["source_url_coverage"] == 0.0
    assert report["quality_metrics"]["text_cleanliness"] == 1.0
    assert report["quality_metrics"]["reference_resolution_rate"] == 0.0
    assert report["quality_metrics"]["chunk_coverage_rate"] == 1.0
    assert report["quality_metrics"]["retrieval_text_non_empty_rate"] == 1.0
    assert report["quality_metrics"]["embedding_input_hash_integrity"] == 1.0
    assert report["corpus_quality"] == 0.65
    assert report["import_blocking_passed"] is True
    assert report["demo_path_passed"] is False
    assert report["chunks_count"] == len(bundle["legal_chunks"])
    assert report["embeddings_input_count"] == len(bundle["embeddings_input"])
    assert report["blocking_errors"] == []
    assert "unknown_fields_left_null_by_policy" in warnings
    assert "source_url_unknown" in warnings
    assert "source_id_unknown" in warnings
    assert "status_unknown" in warnings
    assert "legal_concepts_empty_for_most_units_by_v1_policy" in warnings
    assert "reference_candidates_not_implemented_or_not_all_resolved" in warnings
    assert "contextual_retrieval_context_derived_not_citable" in warnings
    assert "embeddings_vectors_not_generated_in_p8" in warnings
    assert "codul_muncii_demo_path_missing_critical_units" in warnings


def test_repeated_bundle_generation_is_byte_stable_with_fixed_generated_at(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    export_canonical_bundle(
        MINI_CODUL_MUNCII_UNITS,
        ACT_METADATA,
        first_dir,
        generated_at="2026-04-25T00:00:00+00:00",
    )
    export_canonical_bundle(
        MINI_CODUL_MUNCII_UNITS,
        ACT_METADATA,
        second_dir,
        generated_at="2026-04-25T00:00:00+00:00",
    )

    for filename in (
        "legal_units.json",
        "legal_edges.json",
        "legal_chunks.json",
        "embeddings_input.jsonl",
        "corpus_manifest.json",
        "validation_report.json",
        "reference_candidates.json",
    ):
        assert (first_dir / filename).read_text(encoding="utf-8") == (
            second_dir / filename
        ).read_text(encoding="utf-8")


def test_expected_fixture_bundle_matches_current_exporter(tmp_path):
    export_canonical_bundle(
        MINI_CODUL_MUNCII_UNITS,
        ACT_METADATA,
        tmp_path,
        generated_at="2026-04-25T00:00:00+00:00",
        input_files=[(FIXTURE_DIR / "legacy_units.json").as_posix()],
    )

    expected_dir = FIXTURE_DIR / "expected"
    for filename in (
        "legal_units.json",
        "legal_edges.json",
        "legal_chunks.json",
        "embeddings_input.jsonl",
        "corpus_manifest.json",
        "validation_report.json",
        "reference_candidates.json",
    ):
        assert (tmp_path / filename).read_text(encoding="utf-8") == (
            expected_dir / filename
        ).read_text(encoding="utf-8")
