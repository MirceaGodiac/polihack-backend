from apps.api.app.schemas import (
    ExpandedCandidate,
    GraphExpansionResult,
    QueryRequest,
    RawRetrievalResponse,
    RetrievalCandidate,
)
from apps.api.app.services.legal_ranker import (
    LEGAL_RANKER_NO_CANDIDATES,
    LEGAL_RANKER_V2_SCORING,
    LegalRanker,
)
from apps.api.app.services.query_frame import QueryFrame, QueryFrameBuilder
from apps.api.app.services.query_understanding import QueryUnderstanding
from tests.helpers.live_like_demo import (
    LIVE_LIKE_DEMO_QUERY,
    live_like_retrieval_candidates,
)


def build_plan(question: str):
    request = QueryRequest(
        question=question,
        jurisdiction="RO",
        date="current",
        mode="strict_citations",
        debug=True,
    )
    return QueryUnderstanding().build_plan(request)


def build_frame(question: str):
    plan = build_plan(question)
    return QueryFrameBuilder().build(question=question, plan=plan)


def candidate(
    unit_id: str,
    *,
    rank: int = 1,
    score_breakdown: dict[str, float] | None = None,
    unit: dict | None = None,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        unit_id=unit_id,
        rank=rank,
        retrieval_score=score_breakdown.get("bm25", 0.0) if score_breakdown else 0.0,
        score_breakdown=score_breakdown or {},
        unit=unit or {
            "legal_domain": "munca",
            "status": "active",
            "law_id": "ro.codul_muncii",
            "raw_text": "salariu contract act aditional",
            "type": "articol",
        },
    )


def empty_graph() -> GraphExpansionResult:
    return GraphExpansionResult()


def test_no_candidates_returns_safe_fallback():
    plan = build_plan("Poate angajatorul sa-mi scada salariul fara act aditional?")

    result = LegalRanker().rank(
        question=plan.question,
        plan=plan,
        retrieval_response=RawRetrievalResponse(candidates=[]),
        graph_expansion=empty_graph(),
        debug=True,
    )

    assert result.ranked_candidates == []
    assert result.warnings == [LEGAL_RANKER_NO_CANDIDATES]
    assert result.debug["fallback_used"] is True
    assert result.debug["input_candidate_count"] == 0
    assert result.debug["ranked_candidate_count"] == 0


def test_candidates_are_sorted_by_descending_rerank_score():
    plan = build_plan("Poate angajatorul sa-mi scada salariul fara act aditional?")
    low = candidate("ro.codul_muncii.art_10", score_breakdown={"bm25": 0.1})
    high = candidate(
        "ro.codul_muncii.art_41",
        score_breakdown={"bm25": 0.9, "dense": 0.8},
    )

    result = LegalRanker().rank(
        question=plan.question,
        plan=plan,
        retrieval_response=RawRetrievalResponse(candidates=[low, high]),
        graph_expansion=empty_graph(),
        debug=True,
    )

    assert [candidate.unit_id for candidate in result.ranked_candidates] == [
        "ro.codul_muncii.art_41",
        "ro.codul_muncii.art_10",
    ]
    assert result.ranked_candidates[0].rerank_score > result.ranked_candidates[1].rerank_score


def test_exact_citation_candidate_gets_boost_and_ranks_first():
    plan = build_plan("Ce spune art. 41 alin. (1) din Codul muncii?")
    exact = candidate(
        "ro.codul_muncii.art_41.alin_1",
        rank=2,
        score_breakdown={"bm25": 0.2},
    )
    generic = candidate(
        "ro.codul_muncii.art_99",
        rank=1,
        score_breakdown={"bm25": 0.2},
    )

    result = LegalRanker().rank(
        question=plan.question,
        plan=plan,
        retrieval_response=RawRetrievalResponse(candidates=[generic, exact]),
        graph_expansion=empty_graph(),
        debug=True,
    )

    top = result.ranked_candidates[0]
    assert top.unit_id == "ro.codul_muncii.art_41.alin_1"
    assert top.score_breakdown.exact_citation_match == 1.0
    assert "exact_citation_match" in top.why_ranked


def test_domain_match_for_labor_question():
    plan = build_plan("Poate angajatorul sa-mi scada salariul fara act aditional?")

    result = LegalRanker().rank(
        question=plan.question,
        plan=plan,
        retrieval_response=RawRetrievalResponse(
            candidates=[
                candidate(
                    "ro.codul_muncii.art_41",
                    unit={"legal_domain": "munca", "status": "active"},
                )
            ]
        ),
        graph_expansion=empty_graph(),
        debug=True,
    )

    ranked = result.ranked_candidates[0]
    assert ranked.score_breakdown.domain_match == 1.0
    assert "domain_match:munca" in ranked.why_ranked


def test_graph_proximity_influences_score():
    plan = build_plan("Poate angajatorul sa-mi scada salariul fara act aditional?")
    graph = GraphExpansionResult(
        expanded_candidates=[
            ExpandedCandidate(
                unit_id="ro.codul_muncii.art_41",
                source="graph_expansion",
                graph_distance=1,
                graph_proximity=0.9,
            ),
            ExpandedCandidate(
                unit_id="ro.codul_muncii.art_42",
                source="graph_expansion",
                graph_distance=1,
                graph_proximity=0.1,
            ),
        ]
    )

    result = LegalRanker().rank(
        question=plan.question,
        plan=plan,
        retrieval_response=RawRetrievalResponse(candidates=[]),
        graph_expansion=graph,
        debug=True,
    )

    assert result.ranked_candidates[0].unit_id == "ro.codul_muncii.art_41"
    assert result.ranked_candidates[0].score_breakdown.graph_proximity == 0.9


def test_repealed_status_gets_zero_temporal_validity():
    plan = build_plan("Poate angajatorul sa-mi scada salariul fara act aditional?")

    result = LegalRanker().rank(
        question=plan.question,
        plan=plan,
        retrieval_response=RawRetrievalResponse(
            candidates=[
                candidate(
                    "ro.codul_muncii.art_41",
                    unit={"legal_domain": "munca", "status": "repealed"},
                )
            ]
        ),
        graph_expansion=empty_graph(),
        debug=True,
    )

    assert result.ranked_candidates[0].score_breakdown.temporal_validity == 0.0


def test_scores_are_normalized_and_clamped_to_unit_interval():
    plan = build_plan("Poate angajatorul sa-mi scada salariul fara act aditional?")

    result = LegalRanker().rank(
        question=plan.question,
        plan=plan,
        retrieval_response=RawRetrievalResponse(
            candidates=[
                candidate("ro.codul_muncii.art_1", score_breakdown={"bm25": 10.0}),
                candidate("ro.codul_muncii.art_2", score_breakdown={"bm25": -2.0}),
            ]
        ),
        graph_expansion=empty_graph(),
        debug=True,
    )

    for ranked in result.ranked_candidates:
        assert 0.0 <= ranked.rerank_score <= 1.0
        for value in ranked.score_breakdown.model_dump().values():
            assert 0.0 <= value <= 1.0


def test_tie_break_is_deterministic_by_raw_rank_then_unit_id():
    plan = build_plan("Poate angajatorul sa-mi scada salariul fara act aditional?")
    first = candidate("ro.codul_muncii.art_a", rank=1, score_breakdown={"bm25": 0.5})
    second = candidate("ro.codul_muncii.art_b", rank=2, score_breakdown={"bm25": 0.5})

    result = LegalRanker().rank(
        question=plan.question,
        plan=plan,
        retrieval_response=RawRetrievalResponse(candidates=[second, first]),
        graph_expansion=empty_graph(),
        debug=True,
    )

    assert [candidate.unit_id for candidate in result.ranked_candidates] == [
        "ro.codul_muncii.art_a",
        "ro.codul_muncii.art_b",
    ]


def test_v2_demo_governing_rule_beats_salary_distractor():
    question = "Poate angajatorul sa-mi scada salariul fara act aditional?"
    plan = build_plan(question)
    query_frame = build_frame(question)
    governing_rule = candidate(
        "unit_governing_rule",
        score_breakdown={"bm25": 0.40},
        unit={
            "legal_domain": "munca",
            "status": "active",
            "raw_text": "Contractul individual de munca poate fi modificat numai prin acordul partilor.",
            "normalized_text": "contract individual munca modificat acord parti",
            "type": "alineat",
            "paragraph_number": "1",
        },
    )
    distractor = candidate(
        "unit_salary_distractor",
        score_breakdown={"bm25": 0.95},
        unit={
            "legal_domain": "munca",
            "status": "active",
            "raw_text": "remuneratie restanta pentru persoane angajate ilegal, inclusiv salariul",
            "normalized_text": "remuneratie restanta persoane angajate ilegal salariu",
            "type": "alineat",
            "paragraph_number": "1",
        },
    )

    result = LegalRanker().rank(
        question=question,
        plan=plan,
        retrieval_response=RawRetrievalResponse(candidates=[distractor, governing_rule]),
        graph_expansion=empty_graph(),
        query_frame=query_frame,
        debug=True,
    )

    assert result.debug["scoring_version"] == LEGAL_RANKER_V2_SCORING
    assert result.ranked_candidates[0].unit_id == "unit_governing_rule"
    ranked = {item.unit_id: item for item in result.ranked_candidates}
    assert ranked["unit_governing_rule"].score_breakdown.core_issue_score >= 0.70
    assert ranked["unit_salary_distractor"].score_breakdown.distractor_penalty >= 0.70
    assert "core_issue_match" in ranked["unit_governing_rule"].why_ranked
    assert "distractor_penalty" in ranked["unit_salary_distractor"].why_ranked


def test_v2_target_without_core_is_capped():
    question = "Poate angajatorul sa-mi scada salariul fara act aditional?"
    plan = build_plan(question)
    query_frame = build_frame(question)
    topical = candidate(
        "unit_salary_only",
        score_breakdown={"bm25": 1.0},
        unit={
            "legal_domain": "munca",
            "status": "active",
            "raw_text": "Angajatorul plateste salariul si tine evidenta drepturilor salariale.",
            "normalized_text": "angajator salariu drepturi salariale",
            "type": "alineat",
            "paragraph_number": "1",
        },
    )

    result = LegalRanker().rank(
        question=question,
        plan=plan,
        retrieval_response=RawRetrievalResponse(candidates=[topical]),
        graph_expansion=empty_graph(),
        query_frame=query_frame,
        debug=True,
    )

    top = result.ranked_candidates[0]
    assert top.score_breakdown.core_issue_score < 0.25
    assert top.score_breakdown.target_without_core_penalty > 0.0
    assert top.rerank_score <= 0.55
    assert "capped_low_core_issue" in top.why_ranked


def test_v2_distractor_is_capped():
    question = "Poate angajatorul sa-mi scada salariul fara act aditional?"
    plan = build_plan(question)
    query_frame = build_frame(question)
    distractor = candidate(
        "unit_distractor",
        score_breakdown={"bm25": 1.0},
        unit={
            "legal_domain": "munca",
            "status": "active",
            "raw_text": "remuneratie restanta pentru persoane angajate ilegal, inclusiv salariul",
            "normalized_text": "remuneratie restanta persoane angajate ilegal salariu",
            "type": "alineat",
            "paragraph_number": "1",
        },
    )

    result = LegalRanker().rank(
        question=question,
        plan=plan,
        retrieval_response=RawRetrievalResponse(candidates=[distractor]),
        graph_expansion=empty_graph(),
        query_frame=query_frame,
        debug=True,
    )

    top = result.ranked_candidates[0]
    assert top.score_breakdown.distractor_penalty >= 0.70
    assert top.score_breakdown.core_issue_score < 0.25
    assert top.rerank_score <= 0.35
    assert "capped_distractor" in top.why_ranked


def test_v2_exact_citation_override_remains_respected():
    question = "Ce spune art. 41 alin. (1) din Codul muncii?"
    plan = build_plan(question)
    query_frame = build_frame(question)
    exact = candidate(
        "ro.codul_muncii.art_41.alin_1",
        rank=2,
        score_breakdown={"bm25": 0.1},
        unit={
            "legal_domain": "munca",
            "status": "active",
            "raw_text": "Text generic fara problema de fond.",
            "type": "alineat",
            "paragraph_number": "1",
        },
    )
    generic = candidate(
        "ro.codul_muncii.art_99",
        rank=1,
        score_breakdown={"bm25": 1.0},
    )

    result = LegalRanker().rank(
        question=question,
        plan=plan,
        retrieval_response=RawRetrievalResponse(candidates=[generic, exact]),
        graph_expansion=empty_graph(),
        query_frame=query_frame,
        debug=True,
    )

    assert result.ranked_candidates[0].unit_id == "ro.codul_muncii.art_41.alin_1"
    assert result.ranked_candidates[0].score_breakdown.exact_citation_match == 1.0
    assert "boosted_exact_citation" in result.ranked_candidates[0].why_ranked


def test_v2_out_of_domain_candidate_is_capped():
    question = "Poate angajatorul sa-mi scada salariul fara act aditional?"
    plan = build_plan(question)
    query_frame = build_frame(question)
    out_of_domain = candidate(
        "unit_civil_contract",
        score_breakdown={"bm25": 1.0},
        unit={
            "legal_domain": "civil",
            "status": "active",
            "raw_text": "Contractul individual de munca poate fi modificat numai prin acordul partilor.",
            "normalized_text": "contract individual munca modificat acord parti",
            "type": "alineat",
            "paragraph_number": "1",
        },
    )

    result = LegalRanker().rank(
        question=question,
        plan=plan,
        retrieval_response=RawRetrievalResponse(candidates=[out_of_domain]),
        graph_expansion=empty_graph(),
        query_frame=query_frame,
        debug=True,
    )

    top = result.ranked_candidates[0]
    assert top.score_breakdown.domain_match == 0.0
    assert top.rerank_score <= 0.35
    assert "capped_domain_mismatch" in top.why_ranked


def test_v2_query_frame_missing_or_low_confidence_uses_compatible_v1_scoring():
    question = "Poate angajatorul sa-mi scada salariul fara act aditional?"
    plan = build_plan(question)
    low = candidate("unit_low", rank=2, score_breakdown={"bm25": 0.1})
    high = candidate("unit_high", rank=1, score_breakdown={"bm25": 0.9})
    low_confidence_frame = QueryFrame(confidence=0.2, intents=["labor_contract_modification"])

    no_frame_result = LegalRanker().rank(
        question=question,
        plan=plan,
        retrieval_response=RawRetrievalResponse(candidates=[low, high]),
        graph_expansion=empty_graph(),
        query_frame=None,
        debug=True,
    )
    low_confidence_result = LegalRanker().rank(
        question=question,
        plan=plan,
        retrieval_response=RawRetrievalResponse(candidates=[low, high]),
        graph_expansion=empty_graph(),
        query_frame=low_confidence_frame,
        debug=True,
    )

    assert no_frame_result.ranked_candidates[0].unit_id == "unit_high"
    assert low_confidence_result.ranked_candidates[0].unit_id == "unit_high"
    assert no_frame_result.debug["scoring_version"] != LEGAL_RANKER_V2_SCORING
    assert low_confidence_result.debug["scoring_version"] != LEGAL_RANKER_V2_SCORING


def test_v2_support_role_hint_for_governing_rule():
    question = "Poate angajatorul sa-mi scada salariul fara act aditional?"
    plan = build_plan(question)
    query_frame = build_frame(question)
    governing_rule = candidate(
        "unit_governing_rule",
        score_breakdown={"bm25": 0.4},
        unit={
            "legal_domain": "munca",
            "status": "active",
            "raw_text": "Contractul individual de munca poate fi modificat numai prin acordul partilor.",
            "normalized_text": "contract individual munca modificat acord parti",
            "type": "alineat",
            "paragraph_number": "1",
        },
    )

    result = LegalRanker().rank(
        question=question,
        plan=plan,
        retrieval_response=RawRetrievalResponse(candidates=[governing_rule]),
        graph_expansion=empty_graph(),
        query_frame=query_frame,
        debug=True,
    )

    top = result.ranked_candidates[0]
    assert top.score_breakdown.support_role_hint_score == 1.0
    assert "support_role_hint:direct_basis" in top.why_ranked


def test_v2_live_like_contract_modification_ranks_central_rule_above_topical_distractors():
    question = LIVE_LIKE_DEMO_QUERY
    plan = build_plan(question)
    query_frame = build_frame(question)

    result = LegalRanker().rank(
        question=question,
        plan=plan,
        retrieval_response=RawRetrievalResponse(
            candidates=live_like_retrieval_candidates()
        ),
        graph_expansion=empty_graph(),
        query_frame=query_frame,
        debug=True,
    )

    top3_ids = [candidate.unit_id for candidate in result.ranked_candidates[:3]]
    assert "ro.codul_muncii.art_41.alin_1" in top3_ids
    assert "ro.codul_muncii.art_41.alin_3" in top3_ids

    ranked = {candidate.unit_id: candidate for candidate in result.ranked_candidates}
    assert ranked["ro.codul_muncii.art_41.alin_1"].score_breakdown.core_issue_score >= 0.70
    assert ranked["ro.codul_muncii.art_41.alin_3"].score_breakdown.core_issue_score >= 0.70

    for unit_id in {
        "ro.codul_muncii.art_16.alin_1",
        "ro.codul_muncii.art_196.alin_2",
        "ro.codul_muncii.art_42.alin_1",
        "ro.codul_muncii.art_254.alin_3",
    }:
        assert ranked[unit_id].score_breakdown.core_issue_score < 0.70
        assert "support_role_hint:direct_basis" not in ranked[unit_id].why_ranked
