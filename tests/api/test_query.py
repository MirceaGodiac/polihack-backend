import pytest
from fastapi.testclient import TestClient

from apps.api.app.main import app
from apps.api.app.schemas import QueryRequest, RawRetrievalResponse, RetrievalCandidate
from apps.api.app.services.query_orchestrator import QueryOrchestrator


VALID_QUERY = {
    "question": "Poate angajatorul sa-mi scada salariul fara act aditional?",
    "jurisdiction": "RO",
    "date": "current",
    "mode": "strict_citations",
}


def post_query(payload: dict) -> object:
    with TestClient(app) as client:
        return client.post("/api/query", json=payload)


def test_post_api_query_returns_unverified_fallback_without_retrieval():
    response = post_query({**VALID_QUERY, "debug": False})

    assert response.status_code == 200
    payload = response.json()
    assert payload["query_id"]
    assert payload["question"] == VALID_QUERY["question"]
    assert payload["answer"]["confidence"] == 0.0
    assert payload["answer"]["not_legal_advice"] is True
    assert payload["answer"]["refusal_reason"] == "insufficient_evidence"
    assert payload["citations"] == []
    assert payload["evidence_units"] == []
    assert payload["verifier"]["verifier_passed"] is False
    assert payload["verifier"]["repair_applied"] is True
    assert payload["verifier"]["refusal_reason"] == "insufficient_evidence"
    assert payload["verifier"]["citations_checked"] == 0
    assert "verifier_insufficient_evidence" in payload["verifier"]["warnings"]
    assert payload["graph"]["nodes"] == []
    assert payload["graph"]["edges"] == []
    assert "evidence_pack_no_ranked_candidates" in payload["warnings"]
    assert "generation_insufficient_evidence" in payload["warnings"]
    assert "verifier_insufficient_evidence" in payload["warnings"]
    assert "answer_refused_insufficient_evidence" in payload["warnings"]


def test_post_api_query_uses_snake_case_fields():
    response = post_query({**VALID_QUERY, "debug": False})

    payload = response.json()
    assert "query_id" in payload
    assert "short_answer" in payload["answer"]
    assert "groundedness_score" in payload["verifier"]
    assert "evidence_units" in payload
    assert "nodes" in payload["graph"]


def test_post_api_query_debug_true_includes_debug_payload():
    response = post_query({**VALID_QUERY, "debug": True})

    assert response.status_code == 200
    debug = response.json()["debug"]
    assert debug["orchestrator"] == "QueryOrchestrator"
    assert debug["evidence_service"] == "MockEvidenceService"
    assert debug["retrieval_mode"] == "fallback_unconfigured"
    assert debug["evidence_units_count"] == 0
    assert debug["query_understanding"]["legal_domain"] == "muncă"
    assert debug["query_understanding"]["domain_confidence"] >= 0.70
    assert "prohibition" in debug["query_understanding"]["query_types"]
    assert "obligation" in debug["query_understanding"]["query_types"]
    assert debug["query_understanding"]["exact_citations"] == []
    assert debug["query_understanding"]["retrieval_filters"]["status"] == "active"
    retrieval = debug["retrieval"]
    assert retrieval["fallback_used"] is True
    assert retrieval["request_payload"]["retrieval_filters"]["legal_domain"] == "muncă"
    assert retrieval["request_payload"]["exact_citations"] == []
    assert retrieval["response_summary"]["candidate_count"] == 0
    assert "raw_retrieval_not_configured" in retrieval["response_summary"]["warnings"]
    graph_expansion = debug["graph_expansion"]
    assert graph_expansion["fallback_used"] is True
    assert graph_expansion["seed_candidate_count"] == 0
    assert graph_expansion["expanded_candidate_count"] == 0
    assert "graph_expansion_no_seed_candidates" in graph_expansion["warnings"]
    priorities = graph_expansion["policy"]["priority_edge_types"]
    assert "exception_to" in priorities
    assert "sanctions" in priorities
    assert "creates_obligation" in priorities
    assert "creates_prohibition" in priorities
    legal_ranker = debug["legal_ranker"]
    assert legal_ranker["fallback_used"] is True
    assert legal_ranker["input_candidate_count"] == 0
    assert legal_ranker["ranked_candidate_count"] == 0
    assert legal_ranker["ranked_candidates"] == []
    assert "legal_ranker_no_candidates" in legal_ranker["warnings"]
    evidence_pack = debug["evidence_pack"]
    assert evidence_pack["fallback_used"] is True
    assert evidence_pack["input_ranked_candidate_count"] == 0
    assert evidence_pack["selected_evidence_count"] == 0
    assert "evidence_pack_no_ranked_candidates" in evidence_pack["warnings"]
    generation = debug["generation"]
    assert generation["generation_mode"] == "deterministic_extractive_v1_insufficient_evidence"
    assert generation["evidence_unit_count_used"] == 0
    assert generation["citation_unit_ids"] == []
    assert "generation_insufficient_evidence" in generation["warnings"]
    verifier = debug["verifier"]
    assert verifier["claim_extraction"]["claims_total"] == 0
    assert verifier["citation_checks"] == []
    assert "verifier_insufficient_evidence" in verifier["warnings"]
    answer_repair = debug["answer_repair"]
    assert answer_repair["repair_action"] == "refused_insufficient_evidence"
    assert answer_repair["refusal_reason"] == "insufficient_evidence"


def test_post_api_query_debug_false_returns_null_debug():
    response = post_query({**VALID_QUERY, "debug": False})

    assert response.status_code == 200
    payload = response.json()
    assert payload["debug"] is None
    assert "raw_retrieval_not_configured" in payload["warnings"]
    assert "graph_expansion_no_seed_candidates" in payload["warnings"]
    assert "legal_ranker_no_candidates" in payload["warnings"]
    assert "evidence_pack_no_ranked_candidates" in payload["warnings"]
    assert "generation_insufficient_evidence" in payload["warnings"]
    assert "verifier_insufficient_evidence" in payload["warnings"]
    assert "answer_refused_insufficient_evidence" in payload["warnings"]


def test_post_api_query_debug_true_includes_exact_citations():
    response = post_query(
        {
            **VALID_QUERY,
            "question": "Ce spune art. 41 alin. (1) din Codul muncii?",
            "debug": True,
        }
    )

    assert response.status_code == 200
    payload = response.json()
    citation = payload["debug"]["query_understanding"]["exact_citations"][0]
    retrieval = payload["debug"]["retrieval"]
    assert payload["debug"]["query_understanding"]["legal_domain"] == "muncă"
    assert citation["article"] == "41"
    assert citation["paragraph"] == "1"
    assert citation["law_id_hint"] == "ro.codul_muncii"
    request_payload = retrieval["request_payload"]
    assert request_payload["exact_citations"][0]["article"] == "41"
    assert request_payload["exact_citations"][0]["paragraph"] == "1"
    assert request_payload["exact_citations"][0]["act_hint"] == "Codul muncii"
    filters = request_payload["retrieval_filters"]["exact_citation_filters"][0]
    assert filters["article_number"] == "41"
    assert filters["paragraph_number"] == "1"
    assert filters["law_id"] == "ro.codul_muncii"
    assert payload["answer"]["confidence"] == 0.0
    assert payload["answer"]["refusal_reason"] == "insufficient_evidence"
    assert payload["verifier"]["verifier_passed"] is False
    assert payload["verifier"]["repair_applied"] is True
    assert payload["verifier"]["refusal_reason"] == "insufficient_evidence"
    assert "verifier_insufficient_evidence" in payload["verifier"]["warnings"]
    assert "raw_retrieval_not_configured" in payload["warnings"]
    assert "graph_expansion_no_seed_candidates" in payload["warnings"]
    assert "legal_ranker_no_candidates" in payload["warnings"]
    assert "evidence_pack_no_ranked_candidates" in payload["warnings"]
    assert payload["debug"]["graph_expansion"]["fallback_used"] is True
    assert payload["debug"]["legal_ranker"]["fallback_used"] is True
    assert payload["debug"]["evidence_pack"]["fallback_used"] is True


class FakeRawRetrieverClient:
    async def retrieve(self, plan, *, top_k: int = 50, debug: bool = False):
        return RawRetrievalResponse(
            candidates=[
                RetrievalCandidate(
                    unit_id="ro.codul_muncii.art_41",
                    rank=1,
                    retrieval_score=0.78,
                    score_breakdown={"bm25": 0.9, "dense": 0.7},
                    why_retrieved="fixture_candidate",
                    unit={
                        "id": "ro.codul_muncii.art_41",
                        "law_id": "ro.codul_muncii",
                        "law_title": "Codul muncii",
                        "status": "active",
                        "hierarchy_path": ["Codul muncii", "art. 41"],
                        "article_number": "41",
                        "raw_text": "Contractul individual de munca poate fi modificat prin acordul partilor.",
                        "legal_domain": "munca",
                        "legal_concepts": ["contract", "salariu"],
                        "source_url": "https://legislatie.just.ro/test",
                        "type": "articol",
                    },
                )
            ],
            retrieval_methods=["fixture"],
            warnings=[],
            debug={
                "fallback_used": False,
                "request_payload": {
                    "question": plan.question,
                    "retrieval_filters": plan.retrieval_filters,
                    "exact_citations": [],
                    "top_k": top_k,
                    "debug": debug,
                },
            }
            if debug
            else None,
        )


@pytest.mark.anyio
async def test_query_orchestrator_populates_evidence_units_with_fake_candidates():
    orchestrator = QueryOrchestrator(raw_retriever_client=FakeRawRetrieverClient())
    response = await orchestrator.run(QueryRequest(**VALID_QUERY, debug=True))

    assert len(response.evidence_units) == 1
    evidence = response.evidence_units[0]
    assert evidence.id == "ro.codul_muncii.art_41"
    assert evidence.law_title == "Codul muncii"
    assert evidence.raw_text == "Contractul individual de munca poate fi modificat prin acordul partilor."
    assert evidence.excerpt == "Contractul individual de munca poate fi modificat prin acordul partilor."
    assert evidence.retrieval_score == 0.78
    assert evidence.rerank_score is not None
    assert evidence.support_role in {"direct_basis", "condition"}
    assert "selected_by_mmr" in evidence.why_selected
    assert evidence.score_breakdown
    evidence_payload = response.model_dump(mode="json")["evidence_units"][0]
    assert "legal_unit" not in evidence_payload
    assert evidence_payload["id"] == "ro.codul_muncii.art_41"
    assert evidence_payload["law_title"] == "Codul muncii"
    assert evidence_payload["retrieval_score"] == 0.78
    assert evidence_payload["support_role"] in {"direct_basis", "condition"}
    assert isinstance(evidence_payload["why_selected"], list)
    assert evidence_payload["score_breakdown"]
    assert response.graph.nodes
    assert response.debug.evidence_pack["fallback_used"] is False
    assert response.debug.retrieval_mode == "raw_retriever_client:FakeRawRetrieverClient"
    assert response.debug.evidence_pack["selected_evidence_count"] == 1
    assert response.debug.generation["evidence_unit_count_used"] == 1
    assert response.citations == []
    assert response.answer.refusal_reason == "unsupported_claims"
    assert response.answer.confidence == 0.0
    assert response.verifier.citations_checked == 1
    assert response.verifier.claims_total > 0
    assert response.verifier.verifier_passed is False
    assert response.verifier.repair_applied is True
    assert "answer_refused_unsupported_claims" in response.warnings
    assert response.debug.verifier["claim_extraction"]["claims_total"] > 0
    assert response.debug.answer_repair["repair_action"] == "refused_unsupported_claims"
    assert response.debug.answer_repair["removed_citation_ids"] == ["citation:1"]


def test_handoff04_graph_endpoints_are_not_registered():
    paths = {route.path for route in app.routes}

    assert "/api/retrieve/raw" not in paths
    assert "/api/legal-units/{id}/neighbors" not in paths
    assert "/api/explore/root" not in paths
    assert "/api/explore/node/{id}/children" not in paths
    assert "/api/query/{id}/graph" not in paths


def test_post_api_query_rejects_short_question():
    response = post_query({**VALID_QUERY, "question": "Scurt"})

    assert response.status_code == 422


def test_post_api_query_rejects_non_ro_jurisdiction():
    response = post_query({**VALID_QUERY, "jurisdiction": "US"})

    assert response.status_code == 422


def test_post_api_query_rejects_unknown_mode():
    response = post_query({**VALID_QUERY, "mode": "balanced"})

    assert response.status_code == 422


def test_post_api_query_real_verifier_failure_is_explicit():
    response = post_query({**VALID_QUERY, "debug": False})

    payload = response.json()
    warnings = " ".join(payload["warnings"] + payload["verifier"]["warnings"])
    assert "verifier_insufficient_evidence" in warnings
    assert "answer_refused_insufficient_evidence" in warnings
    assert "citation_verifier_failed" not in warnings
    assert "CitationVerifier has not run yet" not in warnings
    assert "generation_unverified_citation_verifier_not_run" not in warnings
    assert payload["verifier"]["verifier_passed"] is False
    assert payload["verifier"]["repair_applied"] is True
