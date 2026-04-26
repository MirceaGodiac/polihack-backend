import json

import httpx
import pytest
from fastapi.testclient import TestClient

import apps.api.app.routes.query as query_route
from apps.api.app.config import settings
from apps.api.app.main import app
from apps.api.app.schemas import QueryRequest, RawRetrievalRequest
from apps.api.app.services.llm_query_decomposer import (
    LLMQueryDecomposer,
    LLMQueryDecomposerResult,
    LLMQueryDecomposition,
    merge_query_frames,
)
from apps.api.app.services.query_frame import QueryFrame, QueryFrameBuilder
from apps.api.app.services.query_orchestrator import QueryOrchestrator
from apps.api.app.services.query_understanding import QueryUnderstanding
from apps.api.app.services.raw_retriever import EmptyRawRetrievalStore, RawRetriever
from apps.api.app.services.raw_retriever_client import RawRetrieverClient


DEMO_QUESTION = "Poate angajatorul sa-mi scada salariul fara act aditional?"
VALID_LLM_PAYLOAD = {
    "domain": "munca",
    "intents": ["labor_contract_modification"],
    "meta_intents": ["modification", "permission"],
    "targets": ["salary"],
    "actors": ["employer", "employee"],
    "qualifiers": ["without_addendum", "without_agreement"],
    "retrieval_queries": [
        "modificarea contractului individual de munca numai prin acordul partilor",
        "modificarea contractului individual de munca salariul",
        "salariul element al contractului individual de munca",
        "angajator reducere salariu fara acordul partilor",
    ],
    "required_evidence_concepts": [
        "agreement_of_parties",
        "contract_modification",
        "salary_as_contract_element",
    ],
    "confidence": 0.88,
}


@pytest.mark.anyio
async def test_feature_flag_false_does_not_call_llm(monkeypatch):
    monkeypatch.setattr(settings, "enable_llm_query_decomposer", False)
    decomposer = ExplodingDecomposer()
    orchestrator = QueryOrchestrator(
        raw_retriever_client=RawRetrieverClient(base_url=None, use_internal=False),
        query_decomposer=decomposer,
    )

    response = await orchestrator.run(_query_request(debug=True))

    assert decomposer.called == 0
    assert response.debug.query_frame["decomposition_source"] == "deterministic"
    assert response.debug.query_decomposer["enabled"] is False
    assert response.debug.query_decomposer["attempted"] is False


@pytest.mark.anyio
async def test_valid_llm_output_merges_retrieval_queries(monkeypatch):
    monkeypatch.setattr(settings, "enable_llm_query_decomposer", True)
    decomposer = _mock_decomposer(VALID_LLM_PAYLOAD)

    result = await decomposer.decompose(
        question=DEMO_QUESTION,
        deterministic_query_frame=_deterministic_frame(),
        known_intents=["labor_contract_modification"],
        allowed_domains=["munca"],
    )
    merged = merge_query_frames(_deterministic_frame(), result.decomposition)

    assert result.debug["attempted"] is True
    assert result.debug["succeeded"] is True
    assert merged.decomposition_source == "merged"
    assert merged.llm_confidence == 0.88
    assert "labor_contract_modification" in merged.intents
    assert VALID_LLM_PAYLOAD["retrieval_queries"][0] in merged.retrieval_queries
    assert "salary_as_contract_element" in merged.required_evidence_concepts


@pytest.mark.anyio
async def test_invalid_json_falls_back_to_deterministic(monkeypatch):
    monkeypatch.setattr(settings, "enable_llm_query_decomposer", True)

    async def handler(_request):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "not json"}}]},
        )

    result = await _decomposer_with_transport(handler).decompose(
        question=DEMO_QUESTION,
        deterministic_query_frame=_deterministic_frame(),
        known_intents=["labor_contract_modification"],
        allowed_domains=["munca"],
    )

    assert result.decomposition is None
    assert result.debug["fallback_reason"] == "invalid_json"
    assert result.debug["succeeded"] is False


@pytest.mark.anyio
async def test_timeout_falls_back_to_deterministic(monkeypatch):
    monkeypatch.setattr(settings, "enable_llm_query_decomposer", True)

    async def handler(_request):
        raise httpx.TimeoutException("timeout")

    result = await _decomposer_with_transport(handler).decompose(
        question=DEMO_QUESTION,
        deterministic_query_frame=_deterministic_frame(),
        known_intents=["labor_contract_modification"],
        allowed_domains=["munca"],
    )

    assert result.decomposition is None
    assert result.debug["fallback_reason"] == "timeout"
    assert result.debug["succeeded"] is False


@pytest.mark.anyio
async def test_forbidden_keys_are_rejected(monkeypatch):
    monkeypatch.setattr(settings, "enable_llm_query_decomposer", True)
    forbidden_payload = {
        **VALID_LLM_PAYLOAD,
        "article_number": "41",
        "unit_id": "ro.codul_muncii.art_41",
        "citations": [],
    }

    result = await _mock_decomposer(forbidden_payload).decompose(
        question=DEMO_QUESTION,
        deterministic_query_frame=_deterministic_frame(),
        known_intents=["labor_contract_modification"],
        allowed_domains=["munca"],
    )

    assert result.decomposition is None
    assert result.debug["fallback_reason"] == "forbidden_keys"
    assert {"article_number", "unit_id", "citations"}.issubset(
        set(result.debug["forbidden_keys_detected"])
    )


def test_low_confidence_merge_keeps_deterministic_frame():
    deterministic = _deterministic_frame()
    llm = LLMQueryDecomposition.model_validate(
        {
            **VALID_LLM_PAYLOAD,
            "retrieval_queries": ["modificarea contractului individual de munca"],
            "confidence": 0.42,
        }
    )

    merged = merge_query_frames(deterministic, llm)

    assert merged.decomposition_source == "deterministic"
    assert merged.intents == deterministic.intents
    assert merged.retrieval_queries == []
    assert merged.llm_confidence == 0.42


@pytest.mark.anyio
async def test_raw_retriever_debug_includes_llm_retrieval_queries():
    frame = _deterministic_frame().model_copy(
        update={
            "retrieval_queries": [
                "modificarea contractului individual de munca salariul",
                "angajator reducere salariu fara acordul partilor",
            ],
            "required_evidence_concepts": ["agreement_of_parties"],
            "decomposition_source": "merged",
            "llm_confidence": 0.88,
        }
    )
    response = await RawRetriever(EmptyRawRetrievalStore()).retrieve(
        RawRetrievalRequest(
            question=DEMO_QUESTION,
            filters={"legal_domain": "munca", "status": "active"},
            query_frame=frame.model_dump(mode="json"),
            exact_citations=[],
            top_k=5,
            debug=True,
        )
    )

    assert "modificarea contractului individual de munca salariul" in response.debug[
        "expanded_terms"
    ]
    assert "angajator reducere salariu fara acordul partilor" in response.debug[
        "decomposition_retrieval_queries"
    ]
    assert response.debug["decomposition_expanded_terms"] == response.debug[
        "decomposition_retrieval_queries"
    ]
    assert response.debug["decomposition_source"] == "merged"


def test_api_query_still_passes_with_deterministic_fallback(monkeypatch):
    monkeypatch.setattr(settings, "enable_llm_query_decomposer", False)
    original_orchestrator = query_route.orchestrator
    query_route.orchestrator = QueryOrchestrator(
        raw_retriever_client=RawRetrieverClient(base_url=None, use_internal=False)
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/query",
                json=_query_request(debug=True).model_dump(mode="json"),
            )
    finally:
        query_route.orchestrator = original_orchestrator

    payload = response.json()
    assert response.status_code == 200
    assert payload["debug"]["query_frame"]["decomposition_source"] == "deterministic"
    assert payload["debug"]["query_decomposer"]["enabled"] is False
    assert payload["answer"]["refusal_reason"] == "insufficient_evidence"


class ExplodingDecomposer:
    enabled = True
    model = "fake"

    def __init__(self) -> None:
        self.called = 0

    async def decompose(self, **_kwargs):
        self.called += 1
        raise AssertionError("LLM decomposer should not be called")


def _query_request(*, debug: bool) -> QueryRequest:
    return QueryRequest(
        question=DEMO_QUESTION,
        jurisdiction="RO",
        date="current",
        mode="strict_citations",
        debug=debug,
    )


def _deterministic_frame() -> QueryFrame:
    request = _query_request(debug=True)
    plan = QueryUnderstanding().build_plan(request)
    return QueryFrameBuilder().build(question=request.question, plan=plan)


def _mock_decomposer(payload: dict) -> LLMQueryDecomposer:
    async def handler(request):
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "test-model"
        assert body["messages"][0]["role"] == "system"
        user_payload = json.loads(body["messages"][1]["content"])
        assert user_payload["question"] == DEMO_QUESTION
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(payload)}}]},
        )

    return _decomposer_with_transport(handler)


def _decomposer_with_transport(handler) -> LLMQueryDecomposer:
    return LLMQueryDecomposer(
        enabled=True,
        base_url="http://llm.local/v1",
        api_key="test-key",
        model="test-model",
        timeout_seconds=0.01,
        transport=httpx.MockTransport(handler),
    )
