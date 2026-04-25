from fastapi.testclient import TestClient

from apps.api.app.main import app


VALID_QUERY = {
    "question": "Poate angajatorul sa-mi scada salariul fara act aditional?",
    "jurisdiction": "RO",
    "date": "current",
    "mode": "strict_citations",
}


def post_query(payload: dict) -> object:
    with TestClient(app) as client:
        return client.post("/api/query", json=payload)


def test_post_api_query_returns_mock_evidence_pack():
    response = post_query({**VALID_QUERY, "debug": False})

    assert response.status_code == 200
    payload = response.json()
    assert payload["query_id"]
    assert payload["question"] == VALID_QUERY["question"]
    assert payload["answer"]["confidence"] == 0.0
    assert payload["answer"]["not_legal_advice"] is True
    assert payload["answer"]["refusal_reason"] == "mock_evidence_pack_not_verified"
    assert len(payload["citations"]) >= 2
    assert len(payload["evidence_units"]) >= 3
    assert payload["verifier"]["verifier_passed"] is False
    assert payload["graph"]["nodes"]
    assert payload["graph"]["edges"]
    assert payload["warnings"]


def test_post_api_query_uses_snake_case_fields():
    response = post_query({**VALID_QUERY, "debug": False})

    payload = response.json()
    assert "query_id" in payload
    assert "short_answer" in payload["answer"]
    assert "id" in payload["evidence_units"][0]["legal_unit"]
    assert "groundedness_score" in payload["verifier"]
    assert "source" in payload["graph"]["edges"][0]


def test_post_api_query_debug_true_includes_debug_payload():
    response = post_query({**VALID_QUERY, "debug": True})

    assert response.status_code == 200
    debug = response.json()["debug"]
    assert debug["orchestrator"] == "QueryOrchestrator"
    assert debug["evidence_service"] == "MockEvidenceService"
    assert debug["evidence_units_count"] >= 3
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


def test_post_api_query_debug_false_returns_null_debug():
    response = post_query({**VALID_QUERY, "debug": False})

    assert response.status_code == 200
    assert response.json()["debug"] is None


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
    assert payload["answer"]["refusal_reason"] == "mock_evidence_pack_not_verified"
    assert payload["verifier"]["verifier_passed"] is False
    assert "raw_retrieval_not_configured" in payload["warnings"]


def test_post_api_query_rejects_short_question():
    response = post_query({**VALID_QUERY, "question": "Scurt"})

    assert response.status_code == 422


def test_post_api_query_rejects_non_ro_jurisdiction():
    response = post_query({**VALID_QUERY, "jurisdiction": "US"})

    assert response.status_code == 422


def test_post_api_query_rejects_unknown_mode():
    response = post_query({**VALID_QUERY, "mode": "balanced"})

    assert response.status_code == 422


def test_post_api_query_mock_warning_and_verifier_failure_are_explicit():
    response = post_query({**VALID_QUERY, "debug": False})

    payload = response.json()
    warnings = " ".join(payload["warnings"] + payload["verifier"]["warnings"])
    assert "mock" in warnings.lower()
    assert "unverified" in warnings.lower()
    assert payload["verifier"]["verifier_passed"] is False
