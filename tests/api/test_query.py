import pytest
from fastapi.testclient import TestClient

import apps.api.app.routes.query as query_route
from apps.api.app.main import app
from apps.api.app.schemas import (
    EvidenceUnit,
    ExpandedCandidate,
    GraphExpansionResult,
    LegalRankerResult,
    QueryRequest,
    RawRetrievalResponse,
    RankedCandidate,
    RankerFeatureBreakdown,
    RetrievalCandidate,
)
from apps.api.app.services.query_orchestrator import QueryOrchestrator
from apps.api.app.services.generation_adapter import GenerationAdapter
from apps.api.app.services.raw_retriever_client import RawRetrieverClient
from apps.api.app.services.evidence_pack_compiler import EvidencePackCompiler


VALID_QUERY = {
    "question": "Poate angajatorul sa-mi scada salariul fara act aditional?",
    "jurisdiction": "RO",
    "date": "current",
    "mode": "strict_citations",
}


def fallback_orchestrator() -> QueryOrchestrator:
    return QueryOrchestrator(
        raw_retriever_client=RawRetrieverClient(base_url=None, use_internal=False)
    )


def post_query(payload: dict, orchestrator: QueryOrchestrator | None = None) -> object:
    original_orchestrator = query_route.orchestrator
    query_route.orchestrator = orchestrator or fallback_orchestrator()
    try:
        with TestClient(app) as client:
            return client.post("/api/query", json=payload)
    finally:
        query_route.orchestrator = original_orchestrator


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
    assert [node["type"] for node in payload["graph"]["nodes"]] == ["query"]
    assert payload["graph"]["nodes"][0]["metadata"]["query_id"] == payload["query_id"]
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
    assert debug["query_understanding"]["legal_domain"] == "munca"
    assert debug["query_understanding"]["domain_confidence"] >= 0.70
    assert "prohibition" in debug["query_understanding"]["query_types"]
    assert "obligation" in debug["query_understanding"]["query_types"]
    assert debug["query_understanding"]["exact_citations"] == []
    assert debug["query_understanding"]["retrieval_filters"]["status"] == "active"
    assert debug["query_frame"]["domain"] == "munca"
    assert "labor_contract_modification" in debug["query_frame"]["intents"]
    assert "salary" in debug["query_frame"]["targets"]
    assert {
        "without_addendum",
        "without_agreement",
    }.intersection(debug["query_frame"]["qualifiers"])
    assert debug["query_frame"]["requires_clarification"] is False
    retrieval = debug["retrieval"]
    assert retrieval["fallback_used"] is True
    assert retrieval["request_payload"]["filters"]["legal_domain"] == "munca"
    assert retrieval["request_payload"]["retrieval_filters"]["legal_domain"] == "munca"
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
    assert payload["debug"]["query_understanding"]["legal_domain"] == "munca"
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


def test_post_api_query_demo_uses_raw_retriever_client_internal_candidates():
    orchestrator = QueryOrchestrator(
        raw_retriever_client=RawRetrieverClient(
            base_url=None,
            internal_retriever_factory=lambda: Art41InternalRetriever(),
        )
    )

    response = post_query({**VALID_QUERY, "debug": True}, orchestrator=orchestrator)

    assert response.status_code == 200
    payload = response.json()
    evidence_ids = {unit["id"] for unit in payload["evidence_units"]}
    assert evidence_ids.intersection(
        {
            "ro.codul_muncii.art_41",
            "ro.codul_muncii.art_41.alin_1",
            "ro.codul_muncii.art_41.alin_2",
            "ro.codul_muncii.art_41.alin_3",
            "ro.codul_muncii.art_41.alin_3.lit_e",
        }
    )
    assert {
        "ro.codul_muncii.art_41.alin_1",
        "ro.codul_muncii.art_41.alin_3",
    }.issubset(evidence_ids)
    assert len(payload["evidence_units"]) <= 8
    evidence_by_id = {unit["id"]: unit for unit in payload["evidence_units"]}
    assert evidence_by_id["ro.codul_muncii.art_41.alin_1"]["support_role"] == "direct_basis"
    assert evidence_by_id["ro.codul_muncii.art_41.alin_3"]["support_role"] in {
        "condition",
        "direct_basis",
    }
    direct_basis_ids = {
        unit_id
        for unit_id, unit in evidence_by_id.items()
        if unit["support_role"] == "direct_basis"
    }
    assert direct_basis_ids <= {
        "ro.codul_muncii.art_41.alin_1",
        "ro.codul_muncii.art_41.alin_3",
    }
    for forbidden_direct_basis_id in {
        "ro.codul_muncii.art_17.alin_3.lit_b",
        "ro.codul_muncii.art_35.alin_1",
        "ro.codul_muncii.art_166",
        "ro.codul_muncii.art_260.alin_1.lit_a",
    }:
        if forbidden_direct_basis_id in evidence_by_id:
            assert evidence_by_id[forbidden_direct_basis_id]["support_role"] != "direct_basis"
    assert payload["evidence_units"]
    first = payload["evidence_units"][0]
    assert first["raw_text"]
    assert first["law_id"] == "ro.codul_muncii"
    assert first["law_title"] == "Codul muncii"
    assert first["article_number"] == "41"
    assert "retrieval_score" in first
    assert "rerank_score" in first
    assert "why_selected" in first
    assert {
        "bm25_score",
        "dense_score",
        "domain_match",
        "temporal_validity",
    }.issubset(first["score_breakdown"])
    assert "raw_retrieval_not_configured" not in payload["warnings"]
    assert "raw_retrieval_unavailable" not in payload["warnings"]
    assert "legal_ranker_no_candidates" not in payload["warnings"]
    assert "evidence_pack_no_ranked_candidates" not in payload["warnings"]
    assert "answer_refused_insufficient_evidence" not in payload["warnings"]
    assert "art. 41" in payload["answer"]["short_answer"]
    assert "art. 264" not in payload["answer"]["short_answer"]
    normalized_answer = payload["answer"]["short_answer"].casefold()
    assert "muncă" in normalized_answer
    assert "regulă" in normalized_answer
    assert "părților" in normalized_answer
    assert "excepțiile" in normalized_answer
    assert "munca " not in normalized_answer
    assert "partilor" not in normalized_answer
    assert "exceptiile" not in normalized_answer
    assert "remuneratie restanta" not in normalized_answer
    assert "persoane angajate ilegal" not in normalized_answer
    citation_unit_ids = {
        citation["legal_unit_id"] for citation in payload["citations"]
    }
    assert citation_unit_ids.issubset(evidence_ids)
    assert {
        "ro.codul_muncii.art_41.alin_1",
        "ro.codul_muncii.art_41.alin_3",
    }.issubset(citation_unit_ids)
    assert all(citation["verified"] is True for citation in payload["citations"])
    assert all(unit_id.startswith("ro.codul_muncii.art_41") for unit_id in citation_unit_ids)
    if "ro.codul_muncii.art_264.lit_a" in evidence_by_id:
        assert evidence_by_id["ro.codul_muncii.art_264.lit_a"]["support_role"] != "direct_basis"
    assert "ro.codul_muncii.art_264.lit_a" not in citation_unit_ids
    assert "ro.codul_muncii.art_17.alin_3.lit_b" not in citation_unit_ids
    assert "ro.codul_muncii.art_35.alin_1" not in citation_unit_ids
    assert "ro.codul_muncii.art_166" not in citation_unit_ids
    assert "ro.codul_muncii.art_260.alin_1.lit_a" not in citation_unit_ids
    assert "muncă" in evidence_by_id["ro.codul_muncii.art_41.alin_1"]["raw_text"]
    assert "părților" in evidence_by_id["ro.codul_muncii.art_41.alin_1"]["raw_text"]
    for unit_id in (
        "ro.codul_muncii.art_41.alin_1",
        "ro.codul_muncii.art_41.alin_3",
    ):
        raw_text = evidence_by_id[unit_id]["raw_text"]
        assert "muncÄ" not in raw_text
        assert "pÄ" not in raw_text
        assert "È" not in raw_text
    coverage = payload["debug"]["evidence_pack"]["requirement_coverage"]
    assert coverage["intent_id"] == "labor_contract_modification"
    assert coverage["required_requirements_total"] == 2
    assert coverage["required_requirements_covered"] == 2
    assert coverage["coverage_passed"] is True
    assert coverage["missing_required_requirements"] == []
    with TestClient(app) as client:
        graph_response = client.get(f"/api/query/{payload['query_id']}/graph")
    assert graph_response.status_code == 200
    graph_payload = graph_response.json()
    assert graph_payload["highlighted_edge_ids"]
    assert {
        "ro.codul_muncii.art_41.alin_1",
        "ro.codul_muncii.art_41.alin_3",
    }.issubset(set(graph_payload["cited_unit_ids"]))
    assert "cited_in_answer" in {
        edge["type"] for edge in graph_payload["graph"]["edges"]
    }
    retrieval_debug = payload["debug"]["retrieval"]
    assert retrieval_debug["backend"] == "internal"
    assert retrieval_debug["response_summary"]["candidate_count"] >= 6
    assert retrieval_debug["request_payload"]["filters"] == {
        "legal_domain": "munca",
        "status": "active",
        "date_context": "current",
    }
    assert payload["debug"]["query_frame"]["domain"] == "munca"
    assert "labor_contract_modification" in payload["debug"]["query_frame"]["intents"]
    assert "salary" in payload["debug"]["query_frame"]["targets"]
    assert {
        "without_addendum",
        "without_agreement",
    }.intersection(payload["debug"]["query_frame"]["qualifiers"])
    assert payload["debug"]["evidence_pack"]["selected_evidence_count"] > 0


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
                        "raw_text": "Contractul individual de muncă poate fi modificat prin acordul părților.",
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


class Art41InternalRetriever:
    async def retrieve(self, request):
        assert request.filters["legal_domain"] == "munca"
        assert request.filters["status"] == "active"
        assert request.top_k == 50
        candidates = []
        for index, unit in enumerate(_art41_units(), start=1):
            candidates.append(
                RetrievalCandidate(
                    unit_id=unit["id"],
                    rank=index,
                    retrieval_score=round(0.95 - index * 0.03, 6),
                    score_breakdown={
                        "bm25": round(1.0 - index * 0.05, 6),
                        "dense": 0.0,
                        "domain_match": 1.0,
                        "metadata_validity": 1.0,
                        "exact_citation_boost": 0.0,
                        "rrf": round(0.05 - index * 0.002, 6),
                    },
                    matched_terms=["salariu", "contract", "act aditional"],
                    why_retrieved="lexical match",
                    unit=unit,
                )
            )
        return RawRetrievalResponse(
            candidates=candidates,
            retrieval_methods=["internal_fixture"],
            warnings=["dense_retrieval_skipped_no_query_embedding"],
            debug={
                "candidate_count": len(candidates),
                "warnings": ["dense_retrieval_skipped_no_query_embedding"],
            },
        )


def _art41_units():
    base = {
        "canonical_id": None,
        "source_id": "fixture",
        "law_id": "ro.codul_muncii",
        "law_title": "Codul muncii",
        "act_type": "code",
        "act_number": None,
        "publication_date": None,
        "effective_date": None,
        "version_start": None,
        "version_end": None,
        "status": "active",
        "legal_domain": "munca",
        "legal_concepts": ["contract", "salariu"],
        "source_url": "https://legislatie.just.ro/test",
        "parser_warnings": [],
        "created_at": None,
        "updated_at": None,
    }
    return [
        {
            **base,
            "id": "ro.codul_muncii.art_264.lit_a",
            "hierarchy_path": ["Codul muncii", "Art. 264", "Lit. a)"],
            "article_number": "264",
            "paragraph_number": None,
            "letter_number": "a",
            "point_number": None,
            "parent_id": "ro.codul_muncii.art_264",
            "raw_text": "a) remuneratie restanta pentru persoane angajate ilegal, inclusiv salariul si alte drepturi salariale;",
            "normalized_text": "remuneratie restanta persoane angajate ilegal salariu drepturi salariale",
            "type": "alineat",
        },
        {
            **base,
            "id": "ro.codul_muncii.art_17.alin_3.lit_b",
            "hierarchy_path": ["Codul muncii", "Art. 17", "Alin. (3)", "Lit. b)"],
            "article_number": "17",
            "paragraph_number": "3",
            "letter_number": "b",
            "point_number": None,
            "parent_id": "ro.codul_muncii.art_17.alin_3",
            "raw_text": (
                "b) locul muncii sau, in lipsa unui loc de munca fix, "
                "posibilitatea ca salariatul sa munceasca in diverse locuri;"
            ),
            "normalized_text": "locul muncii loc munca fix salariat munceasca diverse locuri",
            "type": "litera",
        },
        {
            **base,
            "id": "ro.codul_muncii.art_35.alin_1",
            "hierarchy_path": ["Codul muncii", "Art. 35", "Alin. (1)"],
            "article_number": "35",
            "paragraph_number": "1",
            "letter_number": None,
            "point_number": None,
            "parent_id": "ro.codul_muncii.art_35",
            "raw_text": (
                "Orice salariat are dreptul de a munci la angajatori diferiti "
                "sau la acelasi angajator, in baza unor contracte individuale de munca."
            ),
            "normalized_text": "salariat dreptul munci angajatori diferiti contracte individuale munca",
            "type": "alineat",
        },
        {
            **base,
            "id": "ro.codul_muncii.art_166",
            "hierarchy_path": ["Codul muncii", "Art. 166"],
            "article_number": "166",
            "paragraph_number": None,
            "letter_number": None,
            "point_number": None,
            "parent_id": None,
            "raw_text": "Salariul se plateste in bani cel putin o data pe luna.",
            "normalized_text": "salariul plateste bani luna",
            "type": "articol",
        },
        {
            **base,
            "id": "ro.codul_muncii.art_260.alin_1.lit_a",
            "hierarchy_path": ["Codul muncii", "Art. 260", "Alin. (1)", "Lit. a)"],
            "article_number": "260",
            "paragraph_number": "1",
            "letter_number": "a",
            "point_number": None,
            "parent_id": "ro.codul_muncii.art_260.alin_1",
            "raw_text": (
                "a) primirea la munca a uneia sau a mai multor persoane fara "
                "incheierea unui contract individual de munca constituie contraventie."
            ),
            "normalized_text": "primirea munca persoane fara incheiere contract individual munca constituie contraventie",
            "type": "litera",
        },
        {
            **base,
            "id": "ro.codul_muncii.art_41.alin_1",
            "hierarchy_path": ["Codul muncii", "Art. 41", "Alin. (1)"],
            "article_number": "41",
            "paragraph_number": "1",
            "letter_number": None,
            "point_number": None,
            "parent_id": "ro.codul_muncii.art_41",
            "raw_text": "(1) Contractul individual de muncă poate fi modificat numai prin acordul părților.",
            "normalized_text": "contract individual munca modificat acord parti act aditional salariu",
            "type": "alineat",
        },
        {
            **base,
            "id": "ro.codul_muncii.art_41",
            "hierarchy_path": ["Codul muncii", "Art. 41"],
            "article_number": "41",
            "paragraph_number": None,
            "letter_number": None,
            "point_number": None,
            "parent_id": None,
            "raw_text": "Articolul 41\nContractul individual de muncă poate fi modificat numai prin acordul părților.",
            "normalized_text": "contract individual munca modificat acord parti",
            "type": "articol",
        },
        {
            **base,
            "id": "ro.codul_muncii.art_41.alin_2",
            "hierarchy_path": ["Codul muncii", "Art. 41", "Alin. (2)"],
            "article_number": "41",
            "paragraph_number": "2",
            "letter_number": None,
            "point_number": None,
            "parent_id": "ro.codul_muncii.art_41",
            "raw_text": "(2) Modificarea contractului individual de muncă se referă la elementele contractuale.",
            "normalized_text": "modificare contract individual munca elemente contractuale",
            "type": "alineat",
        },
        {
            **base,
            "id": "ro.codul_muncii.art_41.alin_3",
            "hierarchy_path": ["Codul muncii", "Art. 41", "Alin. (3)"],
            "article_number": "41",
            "paragraph_number": "3",
            "letter_number": None,
            "point_number": None,
            "parent_id": "ro.codul_muncii.art_41",
            "raw_text": "(3) Modificarea contractului individual de muncă poate privi durata, locul muncii, felul muncii, condițiile de muncă, salariul și timpul de muncă.",
            "normalized_text": "modificare contract individual munca durata loc fel conditii salariu timp",
            "type": "alineat",
        },
        {
            **base,
            "id": "ro.codul_muncii.art_41.alin_3.lit_e",
            "hierarchy_path": ["Codul muncii", "Art. 41", "Alin. (3)", "Lit. e)"],
            "article_number": "41",
            "paragraph_number": "3",
            "letter_number": "e",
            "point_number": None,
            "parent_id": "ro.codul_muncii.art_41.alin_3",
            "raw_text": "e) salariul;",
            "normalized_text": "salariul",
            "type": "litera",
        },
    ]


class AgreementOnlyRetriever:
    async def retrieve(self, request):
        units = {unit["id"]: unit for unit in _art41_units()}
        agreement = dict(units["ro.codul_muncii.art_41.alin_1"])
        agreement["normalized_text"] = (
            "contract individual munca poate fi modificat numai prin acordul partilor"
        )
        ordered = [
            agreement,
            units["ro.codul_muncii.art_264.lit_a"],
        ]
        candidates = []
        for index, unit in enumerate(ordered, start=1):
            candidates.append(
                RetrievalCandidate(
                    unit_id=unit["id"],
                    rank=index,
                    retrieval_score=round(0.94 - index * 0.05, 6),
                    score_breakdown={
                        "bm25": round(0.98 - index * 0.05, 6),
                        "dense": 0.0,
                        "domain_match": 1.0,
                        "metadata_validity": 1.0,
                    },
                    matched_terms=["salariu", "contract", "act aditional"],
                    why_retrieved="agreement_only_fixture",
                    unit=unit,
                )
            )
        return RawRetrievalResponse(
            candidates=candidates,
            retrieval_methods=["agreement_only_fixture"],
            warnings=[],
            debug={"candidate_count": len(candidates)},
        )


class BackfillExpandedGraphPolicy:
    async def expand(self, *, plan, retrieval_response, debug=False):
        units = {unit["id"]: unit for unit in _art41_units()}
        scope = dict(units["ro.codul_muncii.art_41.alin_3"])
        scope["raw_text"] = (
            "(3) Modificarea contractului individual de munca se refera la "
            "oricare dintre urmatoarele elemente:"
        )
        scope["normalized_text"] = (
            "modificarea contractului individual de munca se refera la "
            "oricare dintre urmatoarele elemente"
        )
        salary_child = units["ro.codul_muncii.art_41.alin_3.lit_e"]
        expanded = [
            ExpandedCandidate(
                unit_id=scope["id"],
                source="graph_expansion",
                graph_distance=1,
                graph_proximity=0.92,
                expansion_edge_type="contains_child",
                expansion_reason="fixture_graph:contains_child",
                retrieval_candidate=RetrievalCandidate(
                    unit_id=scope["id"],
                    rank=3,
                    retrieval_score=0.52,
                    score_breakdown={
                        "bm25": 0.52,
                        "dense": 0.0,
                        "domain_match": 1.0,
                        "graph_proximity": 0.45,
                    },
                    matched_terms=["modificare", "contract", "elemente"],
                    why_retrieved="graph_expansion_scope_fixture",
                    unit=scope,
                ),
                score_breakdown={"graph_proximity": 0.45},
            ),
            ExpandedCandidate(
                unit_id=salary_child["id"],
                source="graph_expansion",
                graph_distance=1,
                graph_proximity=0.62,
                expansion_edge_type="contains_child",
                expansion_reason="fixture_graph:contains_child",
                retrieval_candidate=RetrievalCandidate(
                    unit_id=salary_child["id"],
                    rank=4,
                    retrieval_score=0.48,
                    score_breakdown={
                        "bm25": 0.48,
                        "dense": 0.0,
                        "domain_match": 1.0,
                        "graph_proximity": 0.62,
                    },
                    matched_terms=["salariu"],
                    why_retrieved="graph_expansion_salary_child_fixture",
                    unit=salary_child,
                ),
                score_breakdown={"graph_proximity": 0.62},
            ),
        ]
        return GraphExpansionResult(
            seed_candidates=[],
            expanded_candidates=expanded,
            warnings=[],
            debug={
                "fallback_used": False,
                "expanded_candidate_count": len(expanded),
            }
            if debug
            else None,
        )


class MissingRawTextGraphPolicy:
    async def expand(self, *, plan, retrieval_response, debug=False):
        units = {unit["id"]: unit for unit in _art41_units()}
        broken_scope = dict(units["ro.codul_muncii.art_41.alin_3"])
        broken_scope["raw_text"] = ""
        broken_scope["normalized_text"] = ""
        expanded = [
            ExpandedCandidate(
                unit_id=broken_scope["id"],
                source="graph_expansion",
                graph_distance=1,
                graph_proximity=0.91,
                expansion_edge_type="contains_child",
                expansion_reason="fixture_graph:contains_child",
                retrieval_candidate=RetrievalCandidate(
                    unit_id=broken_scope["id"],
                    rank=3,
                    retrieval_score=0.72,
                    score_breakdown={
                        "bm25": 0.72,
                        "dense": 0.0,
                        "domain_match": 1.0,
                        "graph_proximity": 0.91,
                    },
                    matched_terms=["modificare", "contract", "salariu"],
                    why_retrieved="graph_expansion_broken_scope_fixture",
                    unit=broken_scope,
                ),
                score_breakdown={"graph_proximity": 0.91},
            )
        ]
        return GraphExpansionResult(
            seed_candidates=[],
            expanded_candidates=expanded,
            warnings=[],
            debug={
                "fallback_used": False,
                "expanded_candidate_count": len(expanded),
            }
            if debug
            else None,
        )


def _backfill_ranked_candidate(
    unit: dict,
    *,
    rank: int,
    rerank_score: float,
    retrieval_score: float,
    why_ranked: list[str] | None = None,
    source: str = "fixture_backfill_ranker",
) -> RankedCandidate:
    return RankedCandidate(
        unit_id=unit["id"],
        rank=rank,
        rerank_score=rerank_score,
        retrieval_score=retrieval_score,
        unit=unit,
        score_breakdown=RankerFeatureBreakdown(
            bm25_score=rerank_score,
            dense_score=0.0,
            domain_match=1.0,
            temporal_validity=1.0,
        ),
        why_ranked=why_ranked or [source],
        source=source,
    )


class BackfillLegalRanker:
    def rank(
        self,
        *,
        question,
        plan,
        retrieval_response,
        graph_expansion,
        query_frame=None,
        debug=False,
    ):
        raw_units = {candidate.unit_id: candidate.unit for candidate in retrieval_response.candidates}
        expanded_units = {
            candidate.unit_id: candidate.retrieval_candidate.unit
            for candidate in graph_expansion.expanded_candidates
            if candidate.retrieval_candidate is not None and candidate.retrieval_candidate.unit
        }
        ranked_candidates = [
            _backfill_ranked_candidate(
                raw_units["ro.codul_muncii.art_41.alin_1"],
                rank=1,
                rerank_score=0.93,
                retrieval_score=0.93,
                why_ranked=["agreement_rule_seed"],
            ),
            _backfill_ranked_candidate(
                raw_units["ro.codul_muncii.art_264.lit_a"],
                rank=2,
                rerank_score=0.71,
                retrieval_score=0.71,
                why_ranked=["salary_distractor"],
            ),
            _backfill_ranked_candidate(
                expanded_units["ro.codul_muncii.art_41.alin_3"],
                rank=3,
                rerank_score=0.62,
                retrieval_score=0.62,
                why_ranked=["scope_available_in_backfill_pool"],
            ),
            _backfill_ranked_candidate(
                expanded_units["ro.codul_muncii.art_41.alin_3.lit_e"],
                rank=4,
                rerank_score=0.54,
                retrieval_score=0.54,
                why_ranked=["salary_child_available_in_backfill_pool"],
            ),
        ]
        return LegalRankerResult(
            ranked_candidates=ranked_candidates,
            warnings=[],
            debug={
                "fallback_used": False,
                "ranked_candidate_count": len(ranked_candidates),
                "ranked_candidates": [
                    candidate.model_dump(mode="json") for candidate in ranked_candidates
                ],
            }
            if debug
            else None,
        )


class NoScopeBackfillLegalRanker:
    def rank(
        self,
        *,
        question,
        plan,
        retrieval_response,
        graph_expansion,
        query_frame=None,
        debug=False,
    ):
        raw_units = {candidate.unit_id: candidate.unit for candidate in retrieval_response.candidates}
        ranked_candidates = [
            _backfill_ranked_candidate(
                raw_units["ro.codul_muncii.art_41.alin_1"],
                rank=1,
                rerank_score=0.93,
                retrieval_score=0.93,
                why_ranked=["agreement_rule_seed"],
            ),
            _backfill_ranked_candidate(
                raw_units["ro.codul_muncii.art_264.lit_a"],
                rank=2,
                rerank_score=0.71,
                retrieval_score=0.71,
                why_ranked=["salary_distractor"],
            ),
        ]
        return LegalRankerResult(
            ranked_candidates=ranked_candidates,
            warnings=[],
            debug={
                "fallback_used": False,
                "ranked_candidate_count": len(ranked_candidates),
                "ranked_candidates": [
                    candidate.model_dump(mode="json") for candidate in ranked_candidates
                ],
            }
            if debug
            else None,
        )


def _evidence_unit(
    unit_id: str,
    *,
    raw_text: str,
    support_role: str,
    rank: int,
    rerank_score: float,
    retrieval_score: float,
) -> EvidenceUnit:
    return EvidenceUnit(
        id=unit_id,
        law_id="ro.codul_muncii",
        law_title="Codul muncii",
        status="active",
        hierarchy_path=["Codul muncii", unit_id],
        article_number="41",
        paragraph_number=None,
        letter_number=None,
        point_number=None,
        raw_text=raw_text,
        normalized_text=raw_text,
        legal_domain="munca",
        legal_concepts=["contract", "salariu"],
        source_url="https://legislatie.just.ro/test",
        parent_id=None,
        evidence_id=f"evidence:{unit_id}",
        excerpt=raw_text,
        rank=rank,
        relevance_score=rerank_score,
        retrieval_method="fixture",
        retrieval_score=retrieval_score,
        rerank_score=rerank_score,
        support_role=support_role,
        why_selected=["fixture"],
        score_breakdown={},
    )


@pytest.mark.anyio
async def test_query_orchestrator_populates_evidence_units_with_fake_candidates():
    orchestrator = QueryOrchestrator(raw_retriever_client=FakeRawRetrieverClient())
    response = await orchestrator.run(
        QueryRequest(
            **{
                **VALID_QUERY,
                "question": "Ce spune art. 41 din Codul muncii?",
            },
            debug=True,
        )
    )

    assert len(response.evidence_units) == 1
    evidence = response.evidence_units[0]
    assert evidence.id == "ro.codul_muncii.art_41"
    assert evidence.law_title == "Codul muncii"
    assert evidence.raw_text == "Contractul individual de muncă poate fi modificat prin acordul părților."
    assert evidence.excerpt == "Contractul individual de muncă poate fi modificat prin acordul părților."
    assert evidence.retrieval_score == 0.78
    assert evidence.rerank_score is not None
    assert evidence.support_role in {"direct_basis", "condition"}
    assert {
        "selected_by_mmr",
        "priority_direct_legal_basis:agreement_rule",
    }.intersection(evidence.why_selected)
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
    assert len(response.citations) == 1
    assert response.citations[0].legal_unit_id == "ro.codul_muncii.art_41"
    assert response.answer.refusal_reason is None
    assert response.answer.confidence == 0.0
    assert response.verifier.citations_checked == 1
    assert response.verifier.claims_total > 0
    assert response.verifier.verifier_passed is True
    assert response.verifier.repair_applied is False
    assert "answer_repaired_unsupported_claims_removed" not in response.warnings
    assert "answer_refused_unsupported_claims" not in response.warnings
    assert response.debug.verifier["claim_extraction"]["claims_total"] > 0
    assert response.debug.answer_repair["repair_action"] == "none"
    assert response.debug.answer_repair["removed_citation_ids"] == []


def test_post_api_query_backfills_missing_salary_scope_before_generation():
    orchestrator = QueryOrchestrator(
        raw_retriever_client=RawRetrieverClient(
            base_url=None,
            internal_retriever_factory=lambda: AgreementOnlyRetriever(),
        ),
        graph_expansion_policy=BackfillExpandedGraphPolicy(),
        legal_ranker=BackfillLegalRanker(),
        evidence_pack_compiler=EvidencePackCompiler(
            candidate_pool_size=1,
            target_evidence_units=1,
            max_evidence_units=3,
        ),
    )

    response = post_query({**VALID_QUERY, "debug": True}, orchestrator=orchestrator)

    assert response.status_code == 200
    payload = response.json()
    evidence_ids = {unit["id"] for unit in payload["evidence_units"]}
    citation_ids = {citation["legal_unit_id"] for citation in payload["citations"]}
    assert "ro.codul_muncii.art_41.alin_1" in evidence_ids
    assert "ro.codul_muncii.art_41.alin_3" in evidence_ids
    assert {
        "ro.codul_muncii.art_41.alin_1",
        "ro.codul_muncii.art_41.alin_3",
    }.issubset(citation_ids)
    assert "ro.codul_muncii.art_264.lit_a" not in citation_ids
    assert "art. 264" not in payload["answer"]["short_answer"]
    backfill = payload["debug"]["requirement_backfill"]
    assert backfill["enabled"] is True
    assert backfill["coverage_before"]["coverage_passed"] is False
    assert backfill["coverage_before"]["missing_required_requirements"] == [
        "contract_modification_salary_scope"
    ]
    assert "ro.codul_muncii.art_41.alin_3" in backfill["added_unit_ids"]
    assert backfill["coverage_after"]["coverage_passed"] is True


def test_post_api_query_keeps_refusal_when_backfill_has_no_real_scope_text():
    orchestrator = QueryOrchestrator(
        raw_retriever_client=RawRetrieverClient(
            base_url=None,
            internal_retriever_factory=lambda: AgreementOnlyRetriever(),
        ),
        graph_expansion_policy=MissingRawTextGraphPolicy(),
        legal_ranker=NoScopeBackfillLegalRanker(),
        evidence_pack_compiler=EvidencePackCompiler(
            candidate_pool_size=1,
            target_evidence_units=1,
            max_evidence_units=3,
        ),
    )

    response = post_query({**VALID_QUERY, "debug": True}, orchestrator=orchestrator)

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]["refusal_reason"] in {
        "insufficient_evidence",
        "no_verifiable_legal_claims",
    }
    assert payload["citations"] == []
    assert "ro.codul_muncii.art_41.alin_3" not in {
        unit["id"] for unit in payload["evidence_units"]
    }
    backfill = payload["debug"]["requirement_backfill"]
    assert backfill["enabled"] is True
    assert backfill["coverage_after"]["coverage_passed"] is False
    assert backfill["coverage_after"]["missing_required_requirements"] == [
        "contract_modification_salary_scope"
    ]
    assert "requirement_backfill_unresolved:contract_modification_salary_scope" in payload["warnings"]
    assert "requirement_backfill_unresolved:contract_modification_salary_scope" in backfill["warnings"]


def test_generation_adapter_scores_focused_contract_modification_over_distractor():
    adapter = GenerationAdapter()
    distractor = _evidence_unit(
        "ro.codul_muncii.art_264.lit_a",
        raw_text="orice remuneratie restanta datorata persoanelor angajate ilegal, inclusiv salariul;",
        support_role="context",
        rank=1,
        rerank_score=0.95,
        retrieval_score=0.95,
    )
    agreement_rule = _evidence_unit(
        "ro.codul_muncii.art_41.alin_1",
        raw_text="Contractul individual de munca poate fi modificat numai prin acordul partilor.",
        support_role="direct_basis",
        rank=2,
        rerank_score=0.80,
        retrieval_score=0.80,
    )
    modification_scope = _evidence_unit(
        "ro.codul_muncii.art_41.alin_3",
        raw_text="Modificarea contractului individual de munca se refera la elementele contractului, inclusiv salariul.",
        support_role="direct_basis",
        rank=3,
        rerank_score=0.75,
        retrieval_score=0.75,
    )

    distractor_score = adapter._score_labor_contract_modification_answer_evidence(distractor)
    agreement_score = adapter._score_labor_contract_modification_answer_evidence(agreement_rule)
    scope_score = adapter._score_labor_contract_modification_answer_evidence(modification_scope)
    assert agreement_score["answer_score"] > distractor_score["answer_score"]
    assert scope_score["answer_score"] > distractor_score["answer_score"]
    assert agreement_score["core_issue"] >= 0.70
    assert scope_score["core_issue"] >= 0.70
    assert distractor_score["distractor"] > 0

    focused = adapter._select_focused_answer_evidence(
        VALID_QUERY["question"],
        [distractor, agreement_rule, modification_scope],
    )
    focused_ids = {unit.id for unit in focused}
    assert agreement_rule.id in focused_ids
    assert modification_scope.id in focused_ids
    assert distractor.id not in focused_ids


def test_query_graph_endpoint_is_registered_without_explore_endpoints():
    paths = {route.path for route in app.routes}

    assert "/api/retrieve/raw" in paths
    assert "/api/query/{query_id}" in paths
    assert "/api/query/{query_id}/graph" in paths
    assert "/api/legal-units/{id}/neighbors" not in paths
    assert "/api/explore/root" not in paths
    assert "/api/explore/node/{id}/children" not in paths


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
