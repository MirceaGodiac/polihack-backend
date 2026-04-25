import json
from pathlib import Path

from apps.api.app.schemas import (
    GraphExpansionResult,
    LegalUnit,
    QueryPlan,
    RankedCandidate,
    RankerFeatureBreakdown,
    RawRetrievalResponse,
    RetrievalCandidate,
)
from apps.api.app.services.evidence_pack_compiler import EvidencePackCompiler
from apps.api.app.services.legal_ranker import LegalRanker
from ingestion.exporters import build_canonical_bundle


FIXTURE_DIR = Path("tests/fixtures/corpus")
LEGAL_UNITS_PATH = FIXTURE_DIR / "codul_muncii_legal_units.json"
LEGAL_EDGES_PATH = FIXTURE_DIR / "codul_muncii_legal_edges.json"
VALIDATION_REPORT_PATH = FIXTURE_DIR / "codul_muncii_validation_report.json"
REFERENCE_CANDIDATES_PATH = FIXTURE_DIR / "codul_muncii_reference_candidates.json"
LEGACY_UNITS_PATH = FIXTURE_DIR / "codul_muncii_legacy_units.json"

ACT_METADATA = {
    "law_id": "ro.codul_muncii",
    "law_title": "Codul muncii",
    "status": "unknown",
}
SOURCE_DESCRIPTORS = [
    {
        "law_id": "ro.codul_muncii",
        "law_title": "Codul muncii",
        "source_type": "local_fixture_demo_sample",
        "source_url": None,
    }
]
ADDITIONAL_WARNINGS = [
    "local_fixture_demo_sample_not_official_complete_text",
    "publication_date_unknown",
    "effective_date_unknown",
    "version_start_unknown",
    "version_end_unknown",
]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _units_by_id() -> dict[str, dict]:
    return {unit["id"]: unit for unit in _load_json(LEGAL_UNITS_PATH)}


def _edges() -> list[dict]:
    return _load_json(LEGAL_EDGES_PATH)


def _query_plan() -> QueryPlan:
    return QueryPlan(
        question="Poate angajatorul sa-mi scada salariul fara act aditional?",
        normalized_question="poate angajatorul sa mi scada salariul fara act aditional",
        legal_domain="munca",
        domain_confidence=1.0,
        query_types=["obligation", "prohibition"],
        retrieval_filters={"legal_domain": "munca"},
    )


def _ranked_candidate(unit: dict, *, rank: int, score: float) -> RankedCandidate:
    return RankedCandidate(
        unit_id=unit["id"],
        rank=rank,
        rerank_score=score,
        retrieval_score=score,
        unit=unit,
        score_breakdown=RankerFeatureBreakdown(
            bm25_score=score,
            dense_score=max(0.0, score - 0.1),
            domain_match=1.0,
            legal_term_overlap=0.8,
            temporal_validity=0.6,
            source_reliability=0.8,
            parent_relevance=0.6 if unit.get("parent_id") else 0.0,
        ),
        why_ranked=["codul_muncii_demo_fixture"],
        source="fixture",
    )


def test_codul_muncii_fixture_is_exporter_compatible():
    legacy_units = _load_json(LEGACY_UNITS_PATH)
    bundle = build_canonical_bundle(
        legacy_units,
        ACT_METADATA,
        generated_at="2026-04-25T00:00:00+00:00",
        input_files=[LEGACY_UNITS_PATH.as_posix()],
        source_descriptors=SOURCE_DESCRIPTORS,
        additional_warnings=ADDITIONAL_WARNINGS,
    )

    assert bundle["legal_units"] == _load_json(LEGAL_UNITS_PATH)
    assert bundle["legal_edges"] == _load_json(LEGAL_EDGES_PATH)
    assert bundle["validation_report"] == _load_json(VALIDATION_REPORT_PATH)
    assert bundle["reference_candidates"] == _load_json(REFERENCE_CANDIDATES_PATH)


def test_codul_muncii_demo_units_are_backend_legal_units():
    units = _units_by_id()

    expected_ids = {
        "ro.codul_muncii",
        "ro.codul_muncii.art_17",
        "ro.codul_muncii.art_17.alin_3",
        "ro.codul_muncii.art_17.alin_3.lit_k",
        "ro.codul_muncii.art_41",
        "ro.codul_muncii.art_41.alin_1",
        "ro.codul_muncii.art_41.alin_2",
        "ro.codul_muncii.art_41.alin_3",
        "ro.codul_muncii.art_41.alin_4",
    }

    assert expected_ids.issubset(units)
    assert units["ro.codul_muncii.art_41"]["article_number"] == "41"
    assert units["ro.codul_muncii.art_17"]["article_number"] == "17"
    for paragraph in ("1", "2", "3", "4"):
        assert units[f"ro.codul_muncii.art_41.alin_{paragraph}"]["paragraph_number"] == paragraph
    assert units["ro.codul_muncii.art_17.alin_3.lit_k"]["letter_number"] == "k"

    for unit in units.values():
        LegalUnit.model_validate(unit)
        assert unit["raw_text"].strip()
        assert unit["normalized_text"].strip()
        assert unit["legal_domain"] == "munca"
        assert unit["legal_concepts"] == []


def test_codul_muncii_raw_text_is_fixture_text_without_html_navigation():
    forbidden_fragments = ("<html", "<nav", "<script", "cookie", "acasa", "meniu")

    for unit in _units_by_id().values():
        raw_text = unit["raw_text"]
        lowered = raw_text.lower()
        assert raw_text == raw_text.strip()
        assert all(fragment not in lowered for fragment in forbidden_fragments)
        assert unit["normalized_text"] != "" if raw_text else True


def test_codul_muncii_parent_ids_and_contains_edges_are_valid():
    units = _units_by_id()
    unit_ids = set(units)
    edge_pairs = {(edge["source_id"], edge["target_id"], edge["type"]) for edge in _edges()}

    assert units["ro.codul_muncii.art_41.alin_1"]["parent_id"] == "ro.codul_muncii.art_41"
    assert units["ro.codul_muncii.art_41.alin_4"]["parent_id"] == "ro.codul_muncii.art_41"
    assert (
        units["ro.codul_muncii.art_17.alin_3.lit_k"]["parent_id"]
        == "ro.codul_muncii.art_17.alin_3"
    )
    assert ("ro.codul_muncii", "ro.codul_muncii.art_41", "contains") in edge_pairs
    assert (
        "ro.codul_muncii.art_41",
        "ro.codul_muncii.art_41.alin_1",
        "contains",
    ) in edge_pairs
    assert (
        "ro.codul_muncii.art_17.alin_3",
        "ro.codul_muncii.art_17.alin_3.lit_k",
        "contains",
    ) in edge_pairs
    for edge in _edges():
        assert edge["type"] == "contains"
        assert edge["source_id"] in unit_ids
        assert edge["target_id"] in unit_ids


def test_codul_muncii_hierarchy_paths_are_frontend_ready():
    units = _units_by_id()

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


def test_validation_report_marks_demo_fixture_unknowns_and_no_reference_edges():
    report = _load_json(VALIDATION_REPORT_PATH)
    warnings = set(report["warnings"])

    assert report["units_count"] == 9
    assert report["edges_count"] == 8
    assert report["reference_candidates_count"] == len(_load_json(REFERENCE_CANDIDATES_PATH))
    assert report["quality_metrics"]["source_url_coverage"] == 0.0
    assert report["quality_metrics"]["reference_resolution_rate"] == 0.0
    assert report["import_blocking_passed"] is True
    assert report["demo_path_passed"] is True
    assert report["blocking_errors"] == []
    assert "local_fixture_demo_sample_not_official_complete_text" in warnings
    assert "source_url_unknown" in warnings
    assert "source_id_unknown" in warnings
    assert "status_unknown" in warnings
    assert "publication_date_unknown" in warnings
    assert "effective_date_unknown" in warnings
    assert "version_start_unknown" in warnings
    assert "version_end_unknown" in warnings
    assert "legal_concepts_empty_for_most_units_by_v1_policy" in warnings
    assert "reference_candidates_extracted_unresolved" in warnings
    assert "reference_resolution_deferred_to_later_phase" in warnings
    assert "reference_resolution_rate_informational_in_v1" in warnings
    assert "source_url_coverage_below_demo_threshold_explained_by_local_fixture" in warnings
    reference_candidates = _load_json(REFERENCE_CANDIDATES_PATH)
    assert any(candidate["raw_reference"] == "alin. (3)" for candidate in reference_candidates)
    assert any(candidate["raw_reference"] == "prezentul cod" for candidate in reference_candidates)
    assert all(candidate["resolved_target_id"] is None for candidate in reference_candidates)
    assert {edge["type"] for edge in _edges()} == {"contains"}


def test_evidence_pack_compiler_consumes_codul_muncii_fixture_units():
    units = _units_by_id()
    ranked_candidates = [
        _ranked_candidate(units["ro.codul_muncii.art_41.alin_4"], rank=1, score=0.96),
        _ranked_candidate(units["ro.codul_muncii.art_41"], rank=2, score=0.91),
        _ranked_candidate(units["ro.codul_muncii.art_17.alin_3.lit_k"], rank=3, score=0.88),
    ]

    result = EvidencePackCompiler(target_evidence_units=3).compile(
        ranked_candidates=ranked_candidates,
        plan=_query_plan(),
        debug=True,
    )

    evidence_by_id = {unit.id: unit for unit in result.evidence_units}
    assert "ro.codul_muncii.art_41.alin_4" in evidence_by_id
    assert "ro.codul_muncii.art_41" in evidence_by_id
    assert "ro.codul_muncii.art_17.alin_3.lit_k" in evidence_by_id
    assert (
        evidence_by_id["ro.codul_muncii.art_41.alin_4"].raw_text
        == units["ro.codul_muncii.art_41.alin_4"]["raw_text"]
    )
    assert evidence_by_id["ro.codul_muncii.art_41.alin_4"].excerpt
    assert result.debug["fallback_used"] is False


def test_legal_ranker_accepts_codul_muncii_fixture_candidates_without_interpretation():
    units = _units_by_id()
    retrieval_response = RawRetrievalResponse(
        candidates=[
            RetrievalCandidate(
                unit_id=unit_id,
                rank=index,
                retrieval_score=score,
                score_breakdown={"bm25": score, "dense": max(0.0, score - 0.1)},
                matched_terms=["angajator", "salariu", "act", "contract"],
                why_retrieved="codul_muncii_demo_fixture",
                unit=units[unit_id],
            )
            for index, (unit_id, score) in enumerate(
                [
                    ("ro.codul_muncii.art_41.alin_4", 0.94),
                    ("ro.codul_muncii.art_41.alin_1", 0.88),
                    ("ro.codul_muncii.art_17.alin_3.lit_k", 0.82),
                ],
                start=1,
            )
        ],
        retrieval_methods=["fixture"],
    )

    result = LegalRanker().rank(
        question=_query_plan().question,
        plan=_query_plan(),
        retrieval_response=retrieval_response,
        graph_expansion=GraphExpansionResult(),
        debug=True,
    )

    ranked_ids = [candidate.unit_id for candidate in result.ranked_candidates]
    assert ranked_ids
    assert "ro.codul_muncii.art_41.alin_4" in ranked_ids
    assert "ro.codul_muncii.art_17.alin_3.lit_k" in ranked_ids
    assert all(candidate.unit and candidate.unit["raw_text"] for candidate in result.ranked_candidates)
    assert result.debug["fallback_used"] is False
