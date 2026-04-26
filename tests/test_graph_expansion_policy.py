import pytest

from apps.api.app.schemas import QueryRequest, RawRetrievalResponse, RetrievalCandidate
from apps.api.app.services.graph_expansion_policy import (
    EDGE_TYPE_WEIGHTS,
    GRAPH_EXPANSION_EMPTY_OR_UNAVAILABLE,
    GRAPH_EXPANSION_NO_SEED_CANDIDATES,
    GraphExpansionPolicy,
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


@pytest.mark.anyio
async def test_no_seed_candidates_returns_safe_noop_fallback():
    plan = build_plan("Ce spune art. 41 alin. (1) din Codul muncii?")

    result = await GraphExpansionPolicy().expand(
        plan=plan,
        retrieval_response=RawRetrievalResponse(candidates=[]),
        debug=True,
    )

    assert result.seed_candidates == []
    assert result.expanded_candidates == []
    assert result.graph_nodes == []
    assert result.graph_edges == []
    assert result.warnings == [
        GRAPH_EXPANSION_NO_SEED_CANDIDATES,
        GRAPH_EXPANSION_EMPTY_OR_UNAVAILABLE,
    ]
    assert result.debug["fallback_used"] is True
    assert result.debug["reason"] == "graph expansion has no seed candidates"
    assert result.debug["seed_candidate_count"] == 0
    assert result.debug["expanded_candidate_count"] == 0


@pytest.mark.anyio
async def test_seed_candidate_without_neighbors_client_returns_seed_only_fallback():
    plan = build_plan("Ce spune art. 41 alin. (1) din Codul muncii?")
    candidate = RetrievalCandidate(
        unit_id="ro.codul_muncii.art_41.alin_1",
        rank=1,
        retrieval_score=0.91,
        score_breakdown={"bm25": 0.7},
        unit={
            "title": "Codul muncii art. 41 alin. 1",
            "legal_domain": "muncă",
            "status": "active",
            "importance": 0.9,
        },
    )

    result = await GraphExpansionPolicy(neighbors_client=None).expand(
        plan=plan,
        retrieval_response=RawRetrievalResponse(candidates=[candidate]),
        debug=True,
    )

    assert result.warnings == []
    assert len(result.seed_candidates) == 1
    assert len(result.expanded_candidates) == 1
    expanded = result.expanded_candidates[0]
    assert expanded.unit_id == "ro.codul_muncii.art_41.alin_1"
    assert expanded.source == "seed"
    assert expanded.graph_distance == 0
    assert expanded.graph_proximity == 1.0
    assert expanded.retrieval_candidate == candidate
    assert result.graph_nodes[0].type == "article"
    assert result.graph_nodes[0].legal_unit_id == "ro.codul_muncii.art_41.alin_1"
    assert result.graph_edges == []
    assert result.debug["fallback_used"] is True
    assert result.debug["graph_expansion_backend"] == "fallback"
    assert result.debug["graph_expansion_fallback_used"] is True
    assert result.debug["expanded_from_raw_candidates"] is True
    assert result.debug["reason"] == "graph neighbors endpoint is not configured"
    assert result.debug["expanded_candidate_count"] == 1


@pytest.mark.anyio
async def test_empty_neighbors_returns_seed_fallback_without_graph_warning():
    class EmptyNeighborsClient:
        is_configured = True

        def neighbors_for(self, unit_id, *, allowed_edge_types, max_depth):
            return []

    plan = build_plan("Ce spune art. 41 alin. (1) din Codul muncii?")
    candidate = RetrievalCandidate(
        unit_id="ro.codul_muncii.art_41.alin_1",
        rank=1,
        retrieval_score=0.91,
        score_breakdown={"bm25": 0.7},
        unit={
            "title": "Codul muncii art. 41 alin. 1",
            "legal_domain": "munca",
            "status": "active",
            "importance": 0.9,
        },
    )

    result = await GraphExpansionPolicy(
        neighbors_client=EmptyNeighborsClient()
    ).expand(
        plan=plan,
        retrieval_response=RawRetrievalResponse(candidates=[candidate]),
        debug=True,
    )

    assert result.warnings == []
    assert [expanded.unit_id for expanded in result.expanded_candidates] == [
        "ro.codul_muncii.art_41.alin_1"
    ]
    assert result.debug["graph_expansion_backend"] == "fallback"
    assert result.debug["expanded_from_raw_candidates"] is True


def test_policy_config_defaults_and_edge_weights_are_exposed():
    plan = build_plan("Poate angajatorul să-mi scadă salariul fără act adițional?")
    policy = GraphExpansionPolicy()

    config = policy.policy_for_plan(plan)

    assert config["max_depth"] == 2
    assert config["max_expanded_nodes"] == 80
    assert config["lambda_decay"] == 0.7
    assert config["edge_type_weights"] == EDGE_TYPE_WEIGHTS
    assert config["edge_type_weights"]["exception_to"] == 0.95
    assert "references" in config["allowed_edge_types"]
    assert "creates_obligation" in config["allowed_edge_types"]


def test_exact_citation_policy_excludes_semantically_related_edges():
    plan = build_plan("Ce spune art. 41 alin. (1) din Codul muncii?")

    allowed_edge_types = GraphExpansionPolicy().allowed_edge_types(plan)

    assert plan.exact_citations
    assert "semantically_related" not in allowed_edge_types


def test_labor_permission_question_prioritizes_expected_edge_types():
    plan = build_plan("Poate angajatorul să-mi scadă salariul fără act adițional?")

    priority_edge_types = GraphExpansionPolicy().priority_edge_types(plan)

    assert "exception_to" in priority_edge_types
    assert "sanctions" in priority_edge_types
    assert "creates_obligation" in priority_edge_types
    assert "creates_prohibition" in priority_edge_types
