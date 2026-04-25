"""
tests/test_validation.py
------------------------
Tests for Phase 3: corpus validator and manifest generator.
"""

import json
import pytest
from pathlib import Path

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

    def test_warnings_present_for_orphans(self):
        report = build_validation_report(CLEAN_UNITS, CONTAINS_EDGES, [])
        assert any("orphan" in w.lower() for w in report["warnings"])


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
