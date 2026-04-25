import pytest

from apps.api.app.schemas import QueryRequest
from apps.api.app.services.raw_retriever_client import (
    RAW_RETRIEVAL_NOT_CONFIGURED,
    RAW_RETRIEVAL_UNAVAILABLE,
    RawRetrieverClient,
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
async def test_no_url_returns_safe_fallback_without_exception():
    plan = build_plan("Poate angajatorul sa-mi scada salariul fara act aditional?")
    response = await RawRetrieverClient(base_url=None).retrieve(
        plan,
        top_k=50,
        debug=True,
    )

    assert response.candidates == []
    assert response.retrieval_methods == []
    assert response.warnings == [RAW_RETRIEVAL_NOT_CONFIGURED]
    assert response.debug["fallback_used"] is True
    assert response.debug["reason"] == "raw retrieval endpoint is not configured"


@pytest.mark.anyio
async def test_request_payload_includes_required_fields():
    plan = build_plan("Poate angajatorul sa-mi scada salariul fara act aditional?")
    response = await RawRetrieverClient(base_url=None).retrieve(
        plan,
        top_k=25,
        debug=True,
    )

    payload = response.debug["request_payload"]
    assert payload["question"] == plan.question
    assert payload["retrieval_filters"]["legal_domain"] == "muncă"
    assert payload["exact_citations"] == []
    assert payload["top_k"] == 25
    assert payload["debug"] is True


@pytest.mark.anyio
async def test_exact_citation_payload_includes_phase3_filters():
    plan = build_plan("Ce spune art. 41 alin. (1) din Codul muncii?")
    response = await RawRetrieverClient(base_url=None).retrieve(plan, debug=True)

    payload = response.debug["request_payload"]
    citation = payload["exact_citations"][0]
    filters = payload["retrieval_filters"]["exact_citation_filters"][0]
    assert citation["article"] == "41"
    assert citation["paragraph"] == "1"
    assert citation["act_hint"] == "Codul muncii"
    assert filters["law_id"] == "ro.codul_muncii"
    assert filters["article_number"] == "41"
    assert filters["paragraph_number"] == "1"


@pytest.mark.anyio
async def test_debug_false_omits_internal_debug_payload():
    plan = build_plan("Poate angajatorul sa-mi scada salariul fara act aditional?")
    response = await RawRetrieverClient(base_url=None).retrieve(plan, debug=False)

    assert response.debug is None
    assert response.warnings == [RAW_RETRIEVAL_NOT_CONFIGURED]


@pytest.mark.anyio
async def test_configured_endpoint_failure_returns_unavailable_fallback(monkeypatch):
    async def fail_send_payload(_request_payload):
        raise RuntimeError("endpoint down")

    plan = build_plan("Ce spune art. 41 din Codul muncii?")
    client = RawRetrieverClient(base_url="http://retrieval.local")
    monkeypatch.setattr(client, "_send_payload", fail_send_payload)

    response = await client.retrieve(plan, debug=True)

    assert response.candidates == []
    assert response.retrieval_methods == []
    assert response.warnings == [RAW_RETRIEVAL_UNAVAILABLE]
    assert response.debug["fallback_used"] is True
    assert response.debug["reason"] == "raw retrieval endpoint request failed"
