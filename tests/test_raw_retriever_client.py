import pytest

from apps.api.app.schemas import QueryRequest, RawRetrievalResponse, RetrievalCandidate
from apps.api.app.services.raw_retriever_client import (
    RAW_RETRIEVAL_NOT_CONFIGURED,
    RAW_RETRIEVAL_UNAVAILABLE,
    RawRetrieverClient,
)
from apps.api.app.services.query_frame import QueryFrameBuilder
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


def build_frame(question: str):
    plan = build_plan(question)
    return QueryFrameBuilder().build(question=question, plan=plan)


@pytest.mark.anyio
async def test_no_url_returns_safe_fallback_without_exception():
    plan = build_plan("Poate angajatorul sa-mi scada salariul fara act aditional?")
    response = await RawRetrieverClient(base_url=None, use_internal=False).retrieve(
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
    response = await RawRetrieverClient(base_url=None, use_internal=False).retrieve(
        plan,
        top_k=25,
        debug=True,
    )

    payload = response.debug["request_payload"]
    assert payload["question"] == plan.question
    assert payload["filters"]["legal_domain"] == "munca"
    assert payload["retrieval_filters"]["legal_domain"] == "munca"
    assert payload["exact_citations"] == []
    assert payload["top_k"] == 25
    assert payload["debug"] is True


@pytest.mark.anyio
async def test_request_payload_includes_query_frame_when_provided():
    question = "Poate angajatorul sa-mi scada salariul fara act aditional?"
    plan = build_plan(question)
    query_frame = build_frame(question)

    response = await RawRetrieverClient(base_url=None, use_internal=False).retrieve(
        plan,
        query_frame=query_frame,
        top_k=25,
        debug=True,
    )

    payload = response.debug["request_payload"]
    assert payload["query_frame"]["domain"] == "munca"
    assert "labor_contract_modification" in payload["query_frame"]["intents"]
    assert "salary" in payload["query_frame"]["targets"]


@pytest.mark.anyio
async def test_exact_citation_payload_includes_phase3_filters():
    plan = build_plan("Ce spune art. 41 alin. (1) din Codul muncii?")
    response = await RawRetrieverClient(base_url=None, use_internal=False).retrieve(
        plan,
        debug=True,
    )

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
    response = await RawRetrieverClient(base_url=None, use_internal=False).retrieve(
        plan,
        debug=False,
    )

    assert response.debug is None
    assert response.warnings == [RAW_RETRIEVAL_NOT_CONFIGURED]


@pytest.mark.anyio
async def test_default_no_url_uses_internal_retriever_and_reports_db_unavailable(
    monkeypatch,
):
    import apps.api.app.db.session as db_module

    monkeypatch.setattr(db_module.settings, "database_url", None)
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_sessionmaker", None)

    plan = build_plan("Poate angajatorul sa-mi scada salariul fara act aditional?")
    response = await RawRetrieverClient(base_url=None).retrieve(plan, debug=True)

    assert response.candidates == []
    assert response.retrieval_methods == []
    assert response.warnings == [RAW_RETRIEVAL_UNAVAILABLE]
    assert response.debug["backend"] == "internal"
    assert response.debug["fallback_used"] is True
    assert response.debug["reason"] == "internal raw retrieval failed"


@pytest.mark.anyio
async def test_internal_retriever_factory_returns_real_candidates():
    plan = build_plan("Poate angajatorul sa-mi scada salariul fara act aditional?")
    client = RawRetrieverClient(
        base_url=None,
        internal_retriever_factory=lambda: FakeInternalRetriever(),
    )

    response = await client.retrieve(plan, top_k=40, debug=True)

    assert response.candidates
    assert response.candidates[0].unit_id == "ro.codul_muncii.art_41.alin_1"
    assert response.candidates[0].unit["raw_text"]
    assert response.candidates[0].score_breakdown["bm25"] == 0.9
    assert response.debug["backend"] == "internal"
    assert response.debug["fallback_used"] is False
    assert response.debug["request_payload"]["filters"] == {
        "legal_domain": "munca",
        "status": "active",
        "date_context": "current",
    }
    assert response.debug["response_summary"]["candidate_count"] == 1


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


class FakeInternalRetriever:
    async def retrieve(self, request):
        assert request.filters["legal_domain"] == "munca"
        assert request.filters["status"] == "active"
        assert request.top_k == 40
        return RawRetrievalResponse(
            candidates=[
                RetrievalCandidate(
                    unit_id="ro.codul_muncii.art_41.alin_1",
                    rank=1,
                    retrieval_score=0.91,
                    score_breakdown={
                        "bm25": 0.9,
                        "dense": 0.0,
                        "domain_match": 1.0,
                        "metadata_validity": 1.0,
                        "exact_citation_boost": 0.0,
                        "rrf": 0.01,
                    },
                    matched_terms=["salariu", "act aditional"],
                    why_retrieved="lexical match",
                    unit={
                        "id": "ro.codul_muncii.art_41.alin_1",
                        "law_id": "ro.codul_muncii",
                        "law_title": "Codul muncii",
                        "status": "active",
                        "hierarchy_path": ["Codul muncii", "Art. 41", "Alin. (1)"],
                        "article_number": "41",
                        "paragraph_number": "1",
                        "raw_text": "(1) Contractul individual de munca poate fi modificat numai prin acordul partilor.",
                        "normalized_text": "contract individual munca modificat act aditional salariu",
                        "legal_domain": "munca",
                        "legal_concepts": ["contract", "salariu"],
                    },
                )
            ],
            retrieval_methods=["internal_fixture"],
            warnings=["dense_retrieval_skipped_no_query_embedding"],
            debug={
                "candidate_count": 1,
                "warnings": ["dense_retrieval_skipped_no_query_embedding"],
            },
        )
