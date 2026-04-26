import pytest

from apps.api.app.schemas import QueryRequest, RawRetrievalResponse, RetrievalCandidate
from apps.api.app.services.evidence_pack_compiler import (
    EvidencePackCompiler,
    EVIDENCE_PACK_MISSING_UNIT_RAW_TEXT,
)
from apps.api.app.services.graph_expansion_policy import GraphExpansionPolicy
from apps.api.app.services.query_orchestrator import QueryOrchestrator
from tests.helpers.fixture_handoff03 import (
    DEMO_QUERY_WITH_DIACRITICS,
    FixtureGraphClient,
    FixtureRawRetriever,
)


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


class NoRawTextRetriever:
    async def retrieve(self, plan, *, top_k: int = 50, debug: bool = False):
        return RawRetrievalResponse(
            candidates=[
                RetrievalCandidate(
                    unit_id="ro.codul_muncii.art_no_raw",
                    rank=1,
                    retrieval_score=0.9,
                    score_breakdown={"bm25": 0.9, "dense": 0.8},
                    why_retrieved="unit_without_raw_text",
                    unit={
                        "id": "ro.codul_muncii.art_no_raw",
                        "law_id": "ro.codul_muncii",
                        "law_title": "Codul muncii",
                        "status": "active",
                        "hierarchy_path": ["Codul muncii", "art. no raw"],
                        "article_number": "41",
                        "normalized_text": "NU TREBUIE CITAT",
                        "text": "NU TREBUIE CITAT",
                        "legal_domain": "munca",
                        "type": "articol",
                    },
                )
            ],
            retrieval_methods=["fixture_missing_raw_text"],
            warnings=[],
            debug={
                "fallback_used": False,
                "candidate_count": 1,
                "request_payload": {
                    "question": plan.question,
                    "top_k": top_k,
                    "debug": debug,
                },
            }
            if debug
            else None,
        )


@pytest.mark.anyio
async def test_query_orchestrator_generates_grounded_draft_with_debug():
    response = await demo_orchestrator().run(
        QueryRequest(
            question=DEMO_QUERY_WITH_DIACRITICS,
            jurisdiction="RO",
            date="current",
            mode="strict_citations",
            debug=True,
        )
    )

    evidence_ids = {unit.id for unit in response.evidence_units}
    citation_ids = {citation.legal_unit_id for citation in response.citations}
    assert response.answer.short_answer
    assert response.answer.confidence == 0.0
    assert response.answer.refusal_reason is None
    assert response.debug.retrieval_mode == "raw_retriever_client:FixtureRawRetriever"
    assert response.evidence_units
    assert response.citations
    assert citation_ids <= evidence_ids
    assert "ro.codul_muncii.art_41.alin_1" in citation_ids
    assert "ro.codul_muncii.art_41.alin_3" in citation_ids
    assert "art. 41" in response.answer.short_answer
    assert "art. 264" not in response.answer.short_answer
    assert all("art_264" not in citation_id for citation_id in citation_ids)
    normalized_answer = response.answer.short_answer.casefold()
    assert "acordul partilor" in normalized_answer
    assert "remuneratie restanta" not in normalized_answer
    assert "persoane angajate ilegal" not in normalized_answer
    for citation in response.citations:
        evidence = next(
            unit
            for unit in response.evidence_units
            if unit.id == citation.legal_unit_id
        )
        assert citation.quote in evidence.raw_text
        assert citation.verified is True
    assert response.verifier.citations_checked == len(response.citations)
    assert response.verifier.claims_total > 0
    assert response.verifier.verifier_passed is True
    assert response.verifier.repair_applied is False
    warnings = " ".join(response.warnings + response.verifier.warnings)
    assert "answer_refused" not in warnings
    assert "answer_repaired" not in warnings
    assert "answer_tempered" not in warnings
    assert "CitationVerifier has not run yet" not in warnings
    assert "generation_unverified_citation_verifier_not_run" not in warnings
    assert "nu a fost verificat final de CitationVerifier" not in response.answer.short_answer
    assert "CitationVerifier V1" not in response.answer.short_answer
    assert response.debug.generation["generation_mode"] == "deterministic_extractive_v1"
    assert response.debug.generation["evidence_unit_count_used"] == len(response.citations)
    assert set(response.debug.generation["citation_unit_ids"]) == citation_ids
    assert response.debug.verifier["claim_extraction"]["claims_total"] > 0
    assert response.debug.answer_repair["repair_action"] == "none"
    assert response.debug.answer_repair["warnings_added"] == []


@pytest.mark.anyio
async def test_query_orchestrator_refuses_when_retriever_returns_unit_without_raw_text():
    response = await QueryOrchestrator(
        raw_retriever_client=NoRawTextRetriever(),
    ).run(
        QueryRequest(
            question=DEMO_QUERY_WITH_DIACRITICS,
            jurisdiction="RO",
            date="current",
            mode="strict_citations",
            debug=True,
        )
    )

    assert response.evidence_units == []
    assert response.citations == []
    assert response.answer.refusal_reason == "insufficient_evidence"
    assert response.verifier.verifier_passed is False
    assert response.verifier.repair_applied is True
    assert EVIDENCE_PACK_MISSING_UNIT_RAW_TEXT in response.warnings
    assert "generation_insufficient_evidence" in response.warnings
    assert "NU TREBUIE CITAT" not in response.answer.short_answer
    assert response.debug.retrieval_mode == "raw_retriever_client:NoRawTextRetriever"
    assert response.debug.evidence_pack["selected_evidence_count"] == 0
    assert response.debug.generation["generation_mode"] == "deterministic_extractive_v1_insufficient_evidence"
    assert response.debug.answer_repair["repair_action"] == "refused_insufficient_evidence"


@pytest.mark.anyio
async def test_query_orchestrator_omits_generation_debug_when_debug_false():
    response = await demo_orchestrator().run(
        QueryRequest(
            question=DEMO_QUERY_WITH_DIACRITICS,
            jurisdiction="RO",
            date="current",
            mode="strict_citations",
            debug=False,
        )
    )

    assert response.debug is None
    assert response.citations
    assert response.answer.confidence == 0.0
    assert response.verifier.citations_checked == len(response.citations)
    assert response.verifier.claims_total > 0
    assert response.verifier.repair_applied is False
