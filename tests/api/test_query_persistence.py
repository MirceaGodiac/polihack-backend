import pytest
from fastapi.testclient import TestClient

import apps.api.app.routes.query as query_route
from apps.api.app.main import app
from apps.api.app.services.evidence_pack_compiler import EvidencePackCompiler
from apps.api.app.services.graph_expansion_policy import GraphExpansionPolicy
from apps.api.app.services.query_orchestrator import QueryOrchestrator
from tests.helpers.fixture_handoff03 import FixtureGraphClient, FixtureRawRetriever


DEMO_QUERY = (
    "Poate angajatorul s\u0103-mi scad\u0103 salariul "
    "f\u0103r\u0103 act adi\u021bional?"
)
DEMO_PAYLOAD = {
    "question": DEMO_QUERY,
    "jurisdiction": "RO",
    "date": "current",
    "mode": "strict_citations",
}


@pytest.fixture(autouse=True)
def clear_query_response_store():
    query_route.query_response_store.clear()
    yield
    query_route.query_response_store.clear()


@pytest.fixture
def client_with_demo_orchestrator(monkeypatch):
    monkeypatch.setattr(query_route, "orchestrator", demo_orchestrator())
    with TestClient(app) as client:
        yield client


def demo_orchestrator() -> QueryOrchestrator:
    return QueryOrchestrator(
        raw_retriever_client=FixtureRawRetriever(),
        graph_expansion_policy=GraphExpansionPolicy(
            neighbors_client=FixtureGraphClient(),
        ),
        evidence_pack_compiler=EvidencePackCompiler(
            target_evidence_units=4,
            max_evidence_units=4,
        ),
    )


def post_demo_query(client: TestClient, *, debug: bool = True):
    return client.post("/api/query", json={**DEMO_PAYLOAD, "debug": debug})


def test_post_query_graph_contains_query_node(client_with_demo_orchestrator):
    response = post_demo_query(client_with_demo_orchestrator, debug=True)

    assert response.status_code == 200
    payload = response.json()
    query_nodes = [
        node for node in payload["graph"]["nodes"] if node["type"] == "query"
    ]

    assert len(query_nodes) == 1
    query_node = query_nodes[0]
    assert query_node["id"] == f"query:{payload['query_id']}"
    assert query_node["label"] == DEMO_QUERY
    assert query_node["metadata"]["query_id"] == payload["query_id"]
    assert query_node["metadata"]["legal_domain"] == "munca"
    assert "labor_contract_modification" in query_node["metadata"]["intents"]
    assert query_node["metadata"]["meta_intents"]
    assert query_node["metadata"]["confidence"] >= 0.7


def test_post_query_graph_contains_retrieved_for_query_edges(
    client_with_demo_orchestrator,
):
    response = post_demo_query(client_with_demo_orchestrator, debug=True)

    payload = response.json()
    query_node_id = f"query:{payload['query_id']}"
    node_ids = {node["id"] for node in payload["graph"]["nodes"]}
    retrieved_edges = [
        edge
        for edge in payload["graph"]["edges"]
        if edge["type"] == "retrieved_for_query"
    ]

    assert retrieved_edges
    for edge in retrieved_edges:
        assert edge["source"] == query_node_id
        assert edge["target"] in node_ids
        assert edge["target"].startswith("legal_unit:")
        assert "evidence_id" in edge["metadata"]
        assert "support_role" in edge["metadata"]
        assert "rank" in edge["metadata"]


def test_post_query_graph_contains_cited_in_answer_edges(
    client_with_demo_orchestrator,
):
    response = post_demo_query(client_with_demo_orchestrator, debug=True)

    payload = response.json()
    cited_unit_ids = {citation["legal_unit_id"] for citation in payload["citations"]}
    cited_edges = [
        edge for edge in payload["graph"]["edges"] if edge["type"] == "cited_in_answer"
    ]
    cited_edge_targets = {
        edge["target"].removeprefix("legal_unit:") for edge in cited_edges
    }

    assert {
        "ro.codul_muncii.art_41.alin_1",
        "ro.codul_muncii.art_41.alin_3",
    }.issubset(cited_unit_ids)
    assert cited_unit_ids.issubset(cited_edge_targets)
    for edge in cited_edges:
        assert edge["metadata"]["citation_id"]
        assert edge["metadata"]["verified"] is True


def test_post_query_graph_enriches_legal_unit_node_metadata(
    client_with_demo_orchestrator,
):
    response = post_demo_query(client_with_demo_orchestrator, debug=True)

    payload = response.json()
    nodes_by_unit_id = {
        node["legal_unit_id"]: node
        for node in payload["graph"]["nodes"]
        if node.get("legal_unit_id")
    }
    cited_node = nodes_by_unit_id["ro.codul_muncii.art_41.alin_1"]
    metadata = cited_node["metadata"]

    assert metadata["is_cited"] is True
    assert metadata["support_role"] == "direct_basis"
    assert metadata["retrieval_score"] is not None
    assert metadata["rerank_score"] is not None
    assert metadata["rank"] >= 1
    assert "source_url" in metadata
    assert metadata["article_number"] == "41"
    assert metadata["paragraph_number"] == "1"
    assert metadata["letter_number"] is None


def test_post_query_graph_contains_supports_claim_edges_when_mapped(
    client_with_demo_orchestrator,
):
    response = post_demo_query(client_with_demo_orchestrator, debug=True)

    payload = response.json()
    mapped_claims = [
        claim
        for claim in payload["verifier"]["claim_results"]
        if claim["supporting_unit_ids"] or claim["citation_ids"]
    ]
    claim_nodes = [
        node for node in payload["graph"]["nodes"] if node["type"] == "cited_claim"
    ]
    support_edges = [
        edge for edge in payload["graph"]["edges"] if edge["type"] == "supports_claim"
    ]

    if not mapped_claims:
        assert not claim_nodes
        assert not support_edges
        return

    assert claim_nodes
    assert support_edges
    assert {node["metadata"]["claim_id"] for node in claim_nodes}.issubset(
        {claim["claim_id"] for claim in mapped_claims}
    )
    assert {edge["metadata"]["claim_id"] for edge in support_edges}.issubset(
        {claim["claim_id"] for claim in mapped_claims}
    )


def test_post_saves_response(client_with_demo_orchestrator):
    post_response = post_demo_query(client_with_demo_orchestrator, debug=True)

    assert post_response.status_code == 200
    post_payload = post_response.json()
    get_response = client_with_demo_orchestrator.get(
        f"/api/query/{post_payload['query_id']}"
    )

    assert get_response.status_code == 200
    get_payload = get_response.json()
    assert get_payload["query_id"] == post_payload["query_id"]
    assert get_payload["answer"] == post_payload["answer"]
    assert len(get_payload["citations"]) == len(post_payload["citations"])


def test_get_graph_works(client_with_demo_orchestrator):
    post_response = post_demo_query(client_with_demo_orchestrator, debug=True)
    query_id = post_response.json()["query_id"]

    graph_response = client_with_demo_orchestrator.get(f"/api/query/{query_id}/graph")

    assert graph_response.status_code == 200
    payload = graph_response.json()
    assert payload["query_id"] == query_id
    assert payload["question"] == DEMO_QUERY
    assert payload["graph"]["nodes"]
    assert payload["graph"]["edges"]
    assert payload["cited_unit_ids"]
    assert payload["highlighted_node_ids"]
    assert payload["highlighted_edge_ids"]
    assert "verifier_summary" in payload
    assert payload["verifier_summary"]["citations_checked"] > 0
    assert payload["verifier_summary"]["verifier_passed"] is True


def test_demo_graph_highlights_cited_units(client_with_demo_orchestrator):
    post_response = post_demo_query(client_with_demo_orchestrator, debug=True)
    query_id = post_response.json()["query_id"]

    graph_payload = client_with_demo_orchestrator.get(
        f"/api/query/{query_id}/graph"
    ).json()

    assert {
        "ro.codul_muncii.art_41.alin_1",
        "ro.codul_muncii.art_41.alin_3",
    }.issubset(set(graph_payload["cited_unit_ids"]))
    assert graph_payload["highlighted_node_ids"]
    assert graph_payload["highlighted_edge_ids"]

    graph_edges_by_id = {
        edge["id"]: edge["type"] for edge in graph_payload["graph"]["edges"]
    }
    expected_highlighted_edge_ids = {
        edge_id
        for edge_id, edge_type in graph_edges_by_id.items()
        if edge_type in {"cited_in_answer", "supports_claim"}
    }
    assert expected_highlighted_edge_ids.issubset(
        set(graph_payload["highlighted_edge_ids"])
    )
    assert {
        "cited_in_answer",
        "supports_claim",
    }.intersection(
        {graph_edges_by_id[edge_id] for edge_id in graph_payload["highlighted_edge_ids"]}
    )


def test_query_graph_has_no_duplicate_nodes_or_edges(client_with_demo_orchestrator):
    response = post_demo_query(client_with_demo_orchestrator, debug=True)

    payload = response.json()
    node_ids = [node["id"] for node in payload["graph"]["nodes"]]
    edge_ids = [edge["id"] for edge in payload["graph"]["edges"]]

    assert len(node_ids) == len(set(node_ids))
    assert len(edge_ids) == len(set(edge_ids))


def test_demo_answer_regression_unchanged(client_with_demo_orchestrator):
    response = post_demo_query(client_with_demo_orchestrator, debug=True)

    payload = response.json()
    citation_unit_ids = {citation["legal_unit_id"] for citation in payload["citations"]}
    assert "art. 41" in payload["answer"]["short_answer"]
    assert "art. 264" not in payload["answer"]["short_answer"]
    assert "ro.codul_muncii.art_264.lit_a" not in citation_unit_ids


def test_unknown_query_returns_404():
    with TestClient(app) as client:
        query_response = client.get("/api/query/does-not-exist")
        graph_response = client.get("/api/query/does-not-exist/graph")

    assert query_response.status_code == 404
    assert graph_response.status_code == 404
    assert query_response.json()["error_code"] == "query_not_found"
    assert graph_response.json()["error_code"] == "query_not_found"


def test_debug_false_still_persisted(client_with_demo_orchestrator):
    post_response = post_demo_query(client_with_demo_orchestrator, debug=False)

    assert post_response.status_code == 200
    post_payload = post_response.json()
    get_response = client_with_demo_orchestrator.get(
        f"/api/query/{post_payload['query_id']}"
    )

    assert get_response.status_code == 200
    get_payload = get_response.json()
    assert get_payload["debug"] is None
    assert get_payload["query_id"] == post_payload["query_id"]
    assert get_payload["answer"] == post_payload["answer"]
    assert get_payload["citations"]
    assert get_payload["graph"]["nodes"]
