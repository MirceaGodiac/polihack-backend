"""
tests/ingestion/test_validation.py
----------------------------------
Tests for Phase 3: corpus validator and manifest generator.
"""

import json
import pytest
from pathlib import Path

from ingestion.chunks import build_embedding_input_records, build_legal_chunks
from ingestion.exporters import build_canonical_validation_report
from ingestion.validators import validate_corpus, build_validation_report, save_validation_report
from ingestion.manifest import build_manifest, save_manifest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CLEAN_UNITS = [
    {"id": "ro.test.titlu_i",           "type": "titlu",   "raw_text": "TITLUL I"},
    {"id": "ro.test.titlu_i.art_1",     "type": "articol", "raw_text": "Art. 1"},
    {"id": "ro.test.titlu_i.art_1.alin_1", "type": "alineat", "raw_text": "(1)"},
    {"id": "ro.test.titlu_i.art_2",     "type": "articol", "raw_text": "Art. 2"},
]

CONTAINS_EDGES = [
    {"source_id": "ro.test.titlu_i",       "target_id": "ro.test.titlu_i.art_1",        "type": "contains"},
    {"source_id": "ro.test.titlu_i.art_1", "target_id": "ro.test.titlu_i.art_1.alin_1", "type": "contains"},
    {"source_id": "ro.test.titlu_i",       "target_id": "ro.test.titlu_i.art_2",         "type": "contains"},
]

REF_CANDIDATES_MIXED = [
    {"source_unit_id": "ro.test.titlu_i.art_2", "target_article": "1", "status": "resolved_high_confidence"},
    {"source_unit_id": "ro.test.titlu_i.art_2", "target_article": "99", "status": "unresolved"},
    {"source_unit_id": "ro.test.titlu_i.art_1", "target_article": "200", "status": "external_unresolved"},
]

SOURCES = [
    {"law_id": "ro.test", "law_title": "Test Law", "legal_domain": "test_domain"},
]


def canonical_unit(
    unit_id: str,
    *,
    raw_text: str = "Text legal.",
    parent_id: str | None = None,
    legal_domain: str = "munca",
) -> dict:
    segments = unit_id.split(".")
    article_number = None
    paragraph_number = None
    letter_number = None
    for segment in segments:
        if segment.startswith("art_"):
            article_number = segment.removeprefix("art_")
        elif segment.startswith("alin_"):
            paragraph_number = segment.removeprefix("alin_")
        elif segment.startswith("lit_"):
            letter_number = segment.removeprefix("lit_")
    return {
        "id": unit_id,
        "canonical_id": unit_id.removeprefix("ro.").replace(".", ":"),
        "source_id": None,
        "law_id": ".".join(segments[:2]),
        "law_title": "Codul muncii" if unit_id.startswith("ro.codul_muncii") else "Test Law",
        "status": "unknown",
        "hierarchy_path": ["Legislatia Romaniei", "Munca", "Codul muncii"]
        if unit_id.startswith("ro.codul_muncii")
        else ["Test Law"],
        "article_number": article_number,
        "paragraph_number": paragraph_number,
        "letter_number": letter_number,
        "point_number": None,
        "raw_text": raw_text,
        "normalized_text": raw_text,
        "legal_domain": legal_domain,
        "legal_concepts": [],
        "source_url": None,
        "parent_id": parent_id,
        "children_ids": [],
        "outgoing_reference_ids": [],
        "incoming_reference_ids": [],
        "parser_warnings": ["source_url_unknown"],
    }


def canonical_report(
    units: list[dict],
    edges: list[dict],
    references: list[dict] | None = None,
    *,
    chunks: list[dict] | None = None,
    records: list[dict] | None = None,
    additional_warnings: list[str] | None = None,
) -> dict:
    references = references or []
    sorted_units = sorted(units, key=lambda unit: unit["id"])
    sorted_edges = sorted(edges, key=lambda edge: edge["id"])
    if chunks is None:
        chunks = build_legal_chunks(sorted_units, references)
    if records is None:
        records = build_embedding_input_records(chunks)
    return build_canonical_validation_report(
        sorted_units,
        sorted_edges,
        references,
        chunks,
        records,
        parser_version="0.1.0",
        additional_warnings=additional_warnings,
    )


def valid_canonical_units_and_edges() -> tuple[list[dict], list[dict]]:
    units = [
        canonical_unit("ro.test", raw_text="Test Law"),
        canonical_unit("ro.test.art_1", parent_id="ro.test", raw_text="Art. 1"),
        canonical_unit(
            "ro.test.art_1.alin_1",
            parent_id="ro.test.art_1",
            raw_text="(1) Text legal.",
        ),
    ]
    edges = [
        {
            "id": "edge.contains.ro.test.ro.test.art_1",
            "source_id": "ro.test",
            "target_id": "ro.test.art_1",
            "type": "contains",
            "weight": 1.0,
            "confidence": 1.0,
            "metadata": {},
        },
        {
            "id": "edge.contains.ro.test.art_1.ro.test.art_1.alin_1",
            "source_id": "ro.test.art_1",
            "target_id": "ro.test.art_1.alin_1",
            "type": "contains",
            "weight": 1.0,
            "confidence": 1.0,
            "metadata": {},
        },
    ]
    return units, edges


# ---------------------------------------------------------------------------
# validate_corpus tests (Phase 1 backward compat)
# ---------------------------------------------------------------------------

class TestValidateCorpus:
    def test_passes_clean_units(self):
        assert validate_corpus(CLEAN_UNITS) is True

    def test_blocks_on_duplicate_ids(self):
        duped = CLEAN_UNITS + [{"id": "ro.test.titlu_i.art_1", "type": "articol", "raw_text": "dup"}]
        with pytest.raises(ValueError, match="Duplicate ID"):
            validate_corpus(duped)

    def test_blocks_on_missing_id(self):
        bad = [{"type": "articol", "raw_text": "no id"}]
        with pytest.raises(ValueError, match="missing ID"):
            validate_corpus(bad)


# ---------------------------------------------------------------------------
# build_validation_report tests
# ---------------------------------------------------------------------------

class TestBuildValidationReport:
    def test_status_pass_on_clean_data(self):
        report = build_validation_report(CLEAN_UNITS, CONTAINS_EDGES, REF_CANDIDATES_MIXED)
        assert report["status"] == "PASS"
        assert report["duplicate_ids"] == []

    def test_status_fail_on_duplicates(self):
        duped = CLEAN_UNITS + [{"id": "ro.test.titlu_i.art_1", "type": "articol", "raw_text": "dup"}]
        report = build_validation_report(duped, CONTAINS_EDGES, [])
        assert report["status"] == "FAIL"
        assert "ro.test.titlu_i.art_1" in report["duplicate_ids"]

    def test_reference_resolution_rate_correct(self):
        # 1 resolved out of 3 candidates → rate = 1/3
        report = build_validation_report(CLEAN_UNITS, CONTAINS_EDGES, REF_CANDIDATES_MIXED)
        assert report["total_ref_candidates"] == 3
        assert report["resolved_candidates"] == 1
        assert abs(report["ReferenceResolutionRate"] - round(1/3, 4)) < 1e-6

    def test_rate_is_zero_when_no_candidates(self):
        report = build_validation_report(CLEAN_UNITS, CONTAINS_EDGES, [])
        assert report["ReferenceResolutionRate"] == 0.0

    def test_rate_is_one_when_all_resolved(self):
        all_resolved = [
            {"source_unit_id": "ro.test.titlu_i.art_1", "status": "resolved_high_confidence"},
            {"source_unit_id": "ro.test.titlu_i.art_2", "status": "resolved_medium_confidence"},
        ]
        report = build_validation_report(CLEAN_UNITS, CONTAINS_EDGES, all_resolved)
        assert report["ReferenceResolutionRate"] == 1.0

    def test_orphan_units_detected(self):
        # titlu_i has no incoming contains edge, so it is an orphan (root node)
        report = build_validation_report(CLEAN_UNITS, CONTAINS_EDGES, [])
        assert "ro.test.titlu_i" in report["orphan_units"]

    def test_total_counts_correct(self):
        report = build_validation_report(CLEAN_UNITS, CONTAINS_EDGES, REF_CANDIDATES_MIXED)
        assert report["total_units"] == len(CLEAN_UNITS)
        assert report["total_contains_edges"] == len(CONTAINS_EDGES)
        assert report["text_cleanliness"] == 1.0

    def test_warnings_present_for_orphans(self):
        report = build_validation_report(CLEAN_UNITS, CONTAINS_EDGES, [])
        assert any("orphan" in w.lower() for w in report["warnings"])

    def test_text_cleanliness_penalizes_navigation_residue(self):
        noisy_units = CLEAN_UNITS + [
            {"id": "ro.test.art_3", "type": "articol", "raw_text": "Meniu Cautare Acasa"},
        ]

        report = build_validation_report(noisy_units, CONTAINS_EDGES, [])

        assert report["text_cleanliness"] < 1.0
        assert any("navigation residue" in warning.lower() for warning in report["warnings"])


class TestCanonicalValidationReport:
    def test_duplicate_ids_block_import(self):
        units, edges = valid_canonical_units_and_edges()
        duplicated = units + [dict(units[1])]

        report = canonical_report(duplicated, edges)

        assert report["import_blocking_passed"] is False
        assert "duplicate_legal_unit_id" in report["blocking_errors"]

    def test_invalid_edge_endpoint_blocks_import(self):
        units, edges = valid_canonical_units_and_edges()
        edges[0] = {**edges[0], "target_id": "ro.test.art_404"}

        report = canonical_report(units, edges)

        assert report["import_blocking_passed"] is False
        assert "invalid_edge_endpoint" in report["blocking_errors"]

    def test_empty_raw_text_for_article_blocks_import(self):
        units, edges = valid_canonical_units_and_edges()
        chunks = build_legal_chunks(sorted(units, key=lambda unit: unit["id"]), [])
        records = build_embedding_input_records(chunks)
        units[1] = {**units[1], "raw_text": "", "normalized_text": ""}

        report = canonical_report(units, edges, chunks=chunks, records=records)

        assert report["import_blocking_passed"] is False
        assert "empty_raw_text_for_citable_unit" in report["blocking_errors"]

    def test_mojibake_in_citable_raw_text_blocks_import(self):
        units, edges = valid_canonical_units_and_edges()
        units[1] = {
            **units[1],
            "raw_text": "Contractul individual de muncÄ poate fi modificat.",
            "normalized_text": "Contractul individual de muncÄ poate fi modificat.",
        }

        report = canonical_report(units, edges)

        assert report["import_blocking_passed"] is False
        assert "raw_text_contains_romanian_mojibake" in report["blocking_errors"]
        assert "raw_text_contains_romanian_mojibake" in report["warnings"]

    def test_missing_demo_art_41_marks_demo_path_failed_without_import_block(self):
        units = [
            canonical_unit("ro.codul_muncii", raw_text="Codul muncii"),
            canonical_unit(
                "ro.codul_muncii.art_17",
                parent_id="ro.codul_muncii",
                raw_text="Art. 17",
            ),
        ]
        edges = [
            {
                "id": "edge.contains.ro.codul_muncii.ro.codul_muncii.art_17",
                "source_id": "ro.codul_muncii",
                "target_id": "ro.codul_muncii.art_17",
                "type": "contains",
                "weight": 1.0,
                "confidence": 1.0,
                "metadata": {},
            }
        ]

        report = canonical_report(units, edges)

        assert report["demo_path_passed"] is False
        assert report["import_blocking_passed"] is True
        assert "codul_muncii_demo_path_missing_critical_units" in report["warnings"]

    def test_local_fixture_source_url_gap_warns_without_blocking(self):
        units, edges = valid_canonical_units_and_edges()

        report = canonical_report(
            units,
            edges,
            additional_warnings=["local_fixture_demo_sample_not_official_complete_text"],
        )

        assert report["quality_metrics"]["source_url_coverage"] == 0.0
        assert report["import_blocking_passed"] is True
        assert (
            "source_url_coverage_below_demo_threshold_explained_by_local_fixture"
            in report["warnings"]
        )

    def test_unresolved_reference_candidate_warns_without_blocking(self):
        units, edges = valid_canonical_units_and_edges()
        references = [
            {
                "source_unit_id": "ro.test.art_1",
                "raw_reference": "art. 2",
                "reference_type": "article",
                "target_article": "2",
                "resolution_status": "candidate_only",
                "resolution_confidence": 0.0,
                "resolved_target_id": None,
            }
        ]

        report = canonical_report(units, edges, references)

        assert report["import_blocking_passed"] is True
        assert report["reference_candidates_count"] == 1
        assert report["quality_metrics"]["reference_resolution_rate"] == 0.0
        assert "reference_candidates_extracted_unresolved" in report["warnings"]
        assert "reference_resolution_deferred_to_later_phase" in report["warnings"]

    def test_invalid_resolved_target_id_blocks_import(self):
        units, edges = valid_canonical_units_and_edges()
        references = [
            {
                "source_unit_id": "ro.test.art_1",
                "raw_reference": "art. 404",
                "reference_type": "article",
                "target_article": "404",
                "resolution_status": "resolved_high_confidence",
                "resolution_confidence": 0.95,
                "resolved_target_id": "ro.test.art_404",
            }
        ]

        report = canonical_report(units, edges, references)

        assert report["import_blocking_passed"] is False
        assert "invalid_reference_candidate_resolved_target_id" in report["blocking_errors"]

    def test_corpus_quality_uses_handoff_formula_not_chunk_metrics(self):
        units, edges = valid_canonical_units_and_edges()

        report = canonical_report(units, edges)

        assert report["quality_metrics"]["source_url_coverage"] == 0.0
        assert report["quality_metrics"]["reference_resolution_rate"] == 0.0
        assert report["quality_metrics"]["chunk_coverage_rate"] == 1.0
        assert report["quality_metrics"]["embedding_input_hash_integrity"] == 1.0
        assert report["corpus_quality"] == 0.65

    def test_invalid_reference_edge_blocks_import(self):
        units, edges = valid_canonical_units_and_edges()
        edges.append(
            {
                "id": "edge.references.ro.test.art_1.ro.test.art_1.alin_1",
                "source_id": "ro.test.art_1",
                "target_id": "ro.test.art_1.alin_1",
                "type": "references",
                "weight": 1.0,
                "confidence": 0.2,
                "metadata": {},
            }
        )

        report = canonical_report(units, edges)

        assert report["import_blocking_passed"] is False
        assert "invalid_reference_edge" in report["blocking_errors"]


# ---------------------------------------------------------------------------
# save_validation_report
# ---------------------------------------------------------------------------

class TestSaveValidationReport:
    def test_writes_valid_json(self, tmp_path):
        report = build_validation_report(CLEAN_UNITS, CONTAINS_EDGES, REF_CANDIDATES_MIXED)
        out = tmp_path / "validation_report.json"
        save_validation_report(report, str(out))
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["status"] == "PASS"
        assert isinstance(loaded["ReferenceResolutionRate"], float)

    def test_rate_in_valid_range(self, tmp_path):
        report = build_validation_report(CLEAN_UNITS, CONTAINS_EDGES, REF_CANDIDATES_MIXED)
        out = tmp_path / "validation_report.json"
        save_validation_report(report, str(out))
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert 0.0 <= loaded["ReferenceResolutionRate"] <= 1.0


# ---------------------------------------------------------------------------
# manifest tests
# ---------------------------------------------------------------------------

class TestManifest:
    def test_manifest_has_required_keys(self, tmp_path):
        manifest = build_manifest("test_batch_v1", SOURCES, tmp_path)
        for key in ("batch_id", "generated_at", "schema_version", "sources", "output_dir", "files"):
            assert key in manifest

    def test_manifest_files_contain_all_artifacts(self, tmp_path):
        manifest = build_manifest("test_batch_v1", SOURCES, tmp_path)
        files = manifest["files"]
        assert "legal_units" in files
        assert "legal_edges" in files
        assert "validation_report" in files
        assert "corpus_manifest" in files

    def test_manifest_sources_match_input(self, tmp_path):
        manifest = build_manifest("test_batch_v1", SOURCES, tmp_path)
        assert manifest["sources"][0]["law_id"] == "ro.test"

    def test_save_manifest_writes_file(self, tmp_path):
        manifest = build_manifest("test_batch_v1", SOURCES, tmp_path)
        out = tmp_path / "corpus_manifest.json"
        save_manifest(manifest, out)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["batch_id"] == "test_batch_v1"
