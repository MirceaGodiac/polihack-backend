import pytest

from apps.api.app.schemas import QueryRequest
from apps.api.app.services.evidence_pack_compiler import EvidencePackCompiler
from apps.api.app.services.graph_expansion_policy import GraphExpansionPolicy
from apps.api.app.services.legal_ranker import LegalRanker
from apps.api.app.services.query_orchestrator import QueryOrchestrator
from apps.api.app.services.query_understanding import QueryUnderstanding
from tests.helpers.fixture_handoff03 import (
    DEMO_QUERY,
    DEMO_QUERY_WITH_DIACRITICS,
    FixtureGraphClient,
    FixtureRawRetriever,
    load_codul_muncii_units,
)


def build_plan(question: str = DEMO_QUERY):
    request = QueryRequest(
        question=question,
        jurisdiction="RO",
        date="current",
        mode="strict_citations",
        debug=True,
    )
    return QueryUnderstanding().build_plan(request)


@pytest.mark.anyio
async def test_fixture_raw_retriever_loads_canonical_legal_units():
    units = load_codul_muncii_units()
    retriever = FixtureRawRetriever(units=units)

    response = await retriever.retrieve(build_plan(), debug=True)

    assert response.retrieval_methods == ["fixture_codul_muncii"]
    assert response.debug["candidate_count"] == len(units)
    unit_ids = [candidate.unit_id for candidate in response.candidates]
    assert unit_ids[:4] == [
        "ro.codul_muncii.art_41.alin_4",
        "ro.codul_muncii.art_41",
        "ro.codul_muncii.art_41.alin_3",
        "ro.codul_muncii.art_17.alin_3.lit_k",
    ]
    for candidate in response.candidates:
        assert candidate.unit == units[candidate.unit_id]
        assert candidate.unit["raw_text"]
        assert candidate.score_breakdown["bm25_score"] > 0
        assert candidate.score_breakdown["dense_score"] > 0
        assert "domain_match" in candidate.score_breakdown
        assert candidate.why_retrieved == "codul_muncii_fixture_retriever"


@pytest.mark.anyio
async def test_graph_expansion_policy_uses_canonical_contains_edges():
    raw_retrieval = await FixtureRawRetriever().retrieve(build_plan(), debug=True)

    graph = await GraphExpansionPolicy(
        neighbors_client=FixtureGraphClient(),
    ).expand(
        plan=build_plan(),
        retrieval_response=raw_retrieval,
        debug=True,
    )

    edge_pairs = {(edge.source, edge.target, edge.type) for edge in graph.graph_edges}
    assert (
        "legal_unit:ro.codul_muncii.art_41",
        "legal_unit:ro.codul_muncii.art_41.alin_3",
        "contains",
    ) in edge_pairs
    assert (
        "legal_unit:ro.codul_muncii.art_41",
        "legal_unit:ro.codul_muncii.art_41.alin_4",
        "contains",
    ) in edge_pairs
    assert (
        "legal_unit:ro.codul_muncii.art_17.alin_3",
        "legal_unit:ro.codul_muncii.art_17.alin_3.lit_k",
        "contains",
    ) in edge_pairs
    assert all(edge.type != "references" for edge in graph.graph_edges)
    assert graph.warnings == []
    assert graph.debug["fallback_used"] is False
    assert graph.debug["graph_edge_count"] >= 3


@pytest.mark.anyio
async def test_legal_ranker_accepts_real_parser_fixture_units():
    plan = build_plan()
    raw_retrieval = await FixtureRawRetriever().retrieve(plan, debug=True)
    graph = await GraphExpansionPolicy(
        neighbors_client=FixtureGraphClient(),
    ).expand(
        plan=plan,
        retrieval_response=raw_retrieval,
        debug=True,
    )

    ranked = LegalRanker().rank(
        question=plan.question,
        plan=plan,
        retrieval_response=raw_retrieval,
        graph_expansion=graph,
        debug=True,
    )

    ranked_ids = [candidate.unit_id for candidate in ranked.ranked_candidates]
    assert "ro.codul_muncii.art_41" in ranked_ids
    assert "ro.codul_muncii.art_41.alin_4" in ranked_ids
    assert ranked.ranked_candidates[0].unit_id in {
        "ro.codul_muncii.art_41",
        "ro.codul_muncii.art_41.alin_4",
    }
    assert ranked.debug["input_candidate_count"] >= 4
    assert ranked.debug["ranked_candidate_count"] >= 4


@pytest.mark.anyio
async def test_evidence_pack_compiler_produces_flat_evidence_from_parser_fixtures():
    plan = build_plan()
    raw_retrieval = await FixtureRawRetriever().retrieve(plan, debug=True)
    graph = await GraphExpansionPolicy(
        neighbors_client=FixtureGraphClient(),
    ).expand(
        plan=plan,
        retrieval_response=raw_retrieval,
        debug=True,
    )
    ranked = LegalRanker().rank(
        question=plan.question,
        plan=plan,
        retrieval_response=raw_retrieval,
        graph_expansion=graph,
        debug=True,
    )

    evidence_pack = EvidencePackCompiler(
        target_evidence_units=4,
        max_evidence_units=4,
    ).compile(
        ranked_candidates=ranked.ranked_candidates,
        graph_expansion=graph,
        plan=plan,
        debug=True,
    )

    evidence_ids = [unit.id for unit in evidence_pack.evidence_units]
    assert "ro.codul_muncii.art_41" in evidence_ids
    assert "ro.codul_muncii.art_41.alin_4" in evidence_ids
    assert "ro.codul_muncii.art_17.alin_3.lit_k" in evidence_ids
    for evidence in evidence_pack.evidence_units:
        assert evidence.raw_text
        assert evidence.excerpt == evidence.raw_text
        assert evidence.retrieval_score is not None
        assert evidence.rerank_score is not None
        assert evidence.support_role
        assert "selected_by_mmr" in evidence.why_selected
        assert evidence.score_breakdown
    evidence_payload = evidence_pack.evidence_units[0].model_dump(mode="json")
    assert "legal_unit" not in evidence_payload
    assert evidence_pack.debug["selected_evidence_count"] == 4


@pytest.mark.anyio
async def test_query_orchestrator_returns_real_evidence_units_for_demo_fixture():
    orchestrator = QueryOrchestrator(
        raw_retriever_client=FixtureRawRetriever(),
        graph_expansion_policy=GraphExpansionPolicy(
            neighbors_client=FixtureGraphClient(),
        ),
        evidence_pack_compiler=EvidencePackCompiler(
            target_evidence_units=4,
            max_evidence_units=4,
        ),
    )
    request = QueryRequest(
        question=DEMO_QUERY_WITH_DIACRITICS,
        jurisdiction="RO",
        date="current",
        mode="strict_citations",
        debug=True,
    )

    response = await orchestrator.run(request)

    evidence_ids = [unit.id for unit in response.evidence_units]
    assert len(response.evidence_units) == 4
    assert "ro.codul_muncii.art_41" in evidence_ids
    assert "ro.codul_muncii.art_41.alin_4" in evidence_ids
    assert "ro.codul_muncii.art_17.alin_3.lit_k" in evidence_ids
    assert response.citations == []
    assert response.answer.confidence == 0.0
    assert response.verifier.verifier_passed is False
    assert response.debug.retrieval["candidate_count"] >= 4
    assert response.debug.evidence_pack["selected_evidence_count"] == 4
    assert response.debug.evidence_units_count == 4
    assert response.debug.legal_ranker["ranked_candidate_count"] >= 4
    assert response.debug.graph_expansion["fallback_used"] is False
    assert "mock_no_legal_conclusion" in " ".join(response.warnings)
