from apps.api.app.schemas import (
    ExpandedCandidate,
    GraphExpansionResult,
    QueryRequest,
    RawRetrievalResponse,
    RetrievalCandidate,
)
from apps.api.app.services.legal_ranker import (
    LEGAL_RANKER_NO_CANDIDATES,
    LegalRanker,
)
from apps.api.app.services.query_understanding import QueryUnderstanding


def build_plan(question: str):
    request = QueryRequest(
        question=question,
        jurisdiction="RO",
        date="current",
        mode="strict_citations",
        debug=True,
    )
    return QueryUnderstanding().build_plan(request)


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
