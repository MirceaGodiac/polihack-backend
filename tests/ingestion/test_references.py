"""
tests/ingestion/test_references.py
----------------------------------
Tests for Phase 2: reference extraction and resolution.
"""

import pytest
from ingestion.reference_extractor import (
    extract_references,
    REF_ART_RE,
    REF_ALIN_RE,
    REF_LOCAL_RE,
)
from ingestion.reference_resolver import resolve_references


# ---------------------------------------------------------------------------
# Fixtures – shared test data
# ---------------------------------------------------------------------------

MOCK_UNITS = [
    {"id": "ro.codul_muncii.art_17",        "type": "articol",  "raw_text": "Art. 17"},
    {"id": "ro.codul_muncii.art_17.alin_1", "type": "alineat",  "raw_text": "(1)"},
    {"id": "ro.codul_muncii.art_17.alin_2", "type": "alineat",  "raw_text": "(2)"},
    {"id": "ro.codul_muncii.art_41",        "type": "articol",  "raw_text": "Art. 41"},
]

SOURCE_UNIT = {
    "id":       "ro.codul_muncii.art_41",
    "type":     "articol",
    "raw_text": "conform art. 17 alin. (1) din prezenta lege",
    "corpus_id": "ro.codul_muncii",
}


# ---------------------------------------------------------------------------
# Reference extractor tests
# ---------------------------------------------------------------------------

class TestExtractor:
    def test_combined_article_and_paragraph(self):
        """The flagship acceptance criterion: art. 17 alin. (1) is extracted."""
        candidates = extract_references(SOURCE_UNIT)
        cand = next(c for c in candidates if c["raw_reference"] == "art. 17 alin. (1)")
        assert cand["target_article"]   == "17"
        assert cand["target_paragraph"] == "1"
        assert cand["source_unit_id"]   == "ro.codul_muncii.art_41"

    def test_local_law_hint_detected(self):
        """'prezenta lege' marks the candidate as same_act."""
        candidates = extract_references(SOURCE_UNIT)
        assert any(
            candidate["raw_reference"] == "prezenta lege"
            and candidate["target_law_hint"] == "same_act"
            for candidate in candidates
        )

    def test_standalone_article_no_paragraph(self):
        unit = {
            "id": "ro.codul_muncii.art_5",
            "raw_text": "Se aplică art. 17.",
        }
        candidates = extract_references(unit)
        assert len(candidates) == 1
        assert candidates[0]["target_article"]   == "17"
        assert candidates[0]["target_paragraph"] is None

    def test_multiple_references_in_one_unit(self):
        unit = {
            "id": "ro.codul_muncii.art_5",
            "raw_text": "Conform art. 17 alin. (1) şi art. 41.",
        }
        candidates = extract_references(unit)
        assert len(candidates) == 2
        articles = {c["target_article"] for c in candidates}
        assert articles == {"17", "41"}

    def test_external_reference_hint(self):
        unit = {
            "id": "ro.codul_muncii.art_5",
            "raw_text": "conform art. 1249 din Codul Civil.",
        }
        candidates = extract_references(unit)
        assert any(
            candidate["raw_reference"] == "art. 1249"
            and candidate["target_law_hint"] == "ro.codul_civil"
            for candidate in candidates
        )
        assert any(
            candidate["raw_reference"] == "Codul Civil"
            and candidate["target_law_hint"] == "ro.codul_civil"
            for candidate in candidates
        )

    def test_ref_art_re_standalone(self):
        matches = REF_ART_RE.findall("Vezi art. 100 şi art. 200^1.")
        assert matches == ["100", "200^1"]

    def test_ref_alin_re_standalone(self):
        matches = REF_ALIN_RE.findall("prevăzut la alin. (3) şi alin.(4).")
        assert matches == ["3", "4"]

    def test_ref_local_re(self):
        assert REF_LOCAL_RE.search("prevazute de prezenta lege")
        assert REF_LOCAL_RE.search("sensul prezentul cod")
        assert not REF_LOCAL_RE.search("conform altor reglementari")

    def test_no_references_in_empty_text(self):
        unit = {"id": "ro.test.art_1", "raw_text": ""}
        assert extract_references(unit) == []


# ---------------------------------------------------------------------------
# Reference resolver tests
# ---------------------------------------------------------------------------

class TestResolver:
    def _candidates(self, **overrides):
        base = {
            "source_unit_id":   "ro.codul_muncii.art_41",
            "raw_reference":    "art. 17 alin. (1)",
            "target_article":   "17",
            "target_paragraph": "1",
            "target_law_hint":  "same_act",
        }
        base.update(overrides)
        return [base]

    def test_full_resolution_high_confidence(self):
        """Flagship test: art_41 → art_17.alin_1 resolved with confidence ≥ 0.85."""
        candidates = extract_references(SOURCE_UNIT)
        resolved, edges = resolve_references(candidates, MOCK_UNITS)

        assert len(edges) == 1
        edge = edges[0]
        assert edge["source_id"] == "ro.codul_muncii.art_41"
        assert edge["target_id"] == "ro.codul_muncii.art_17.alin_1"
        assert edge["type"]      == "references"
        assert edge["confidence"] >= 0.85

    def test_resolved_status_high_confidence(self):
        candidates = extract_references(SOURCE_UNIT)
        resolved, _ = resolve_references(candidates, MOCK_UNITS)
        assert resolved[0]["status"] == "resolved_high_confidence"

    def test_missing_target_no_edge(self):
        """Reference to art. 99 which does not exist → no edge, status=unresolved."""
        cands = self._candidates(target_article="99", target_paragraph=None)
        resolved, edges = resolve_references(cands, MOCK_UNITS)
        assert edges == []
        assert resolved[0]["status"] == "unresolved"

    def test_article_only_fallback_medium_confidence(self):
        """
        Reference art. 17 alin. (99) – paragraph doesn't exist but article does.
        Should fall back to article-level with medium confidence.
        """
        cands = self._candidates(target_paragraph="99")
        resolved, edges = resolve_references(cands, MOCK_UNITS)
        assert len(edges) == 1
        assert edges[0]["target_id"] == "ro.codul_muncii.art_17"
        assert edges[0]["confidence"] >= 0.60
        assert resolved[0]["status"] == "resolved_medium_confidence"

    def test_external_reference_skipped(self):
        """External references must not generate edges."""
        cands = self._candidates(target_law_hint="external")
        resolved, edges = resolve_references(cands, MOCK_UNITS)
        assert edges == []
        assert resolved[0]["status"] == "external_unresolved"

    def test_multiple_candidates_mixed(self):
        """One resolvable + one unresolvable → only one edge."""
        unit = {
            "id":       "ro.codul_muncii.art_41",
            "raw_text": "Conform art. 17 alin. (1) şi art. 99.",
        }
        candidates = extract_references(unit)
        _, edges = resolve_references(candidates, MOCK_UNITS)
        target_ids = [e["target_id"] for e in edges]
        assert "ro.codul_muncii.art_17.alin_1" in target_ids
        # art. 99 should not appear in edges
        assert not any("art_99" in tid for tid in target_ids)
