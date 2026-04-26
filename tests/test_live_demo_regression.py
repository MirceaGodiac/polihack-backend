import pytest

from apps.api.app.schemas import GraphExpansionResult, QueryRequest, RawRetrievalResponse
from apps.api.app.services.evidence_pack_compiler import EvidencePackCompiler
from apps.api.app.services.graph_expansion_policy import GraphExpansionPolicy
from apps.api.app.services.legal_issue_frame import CandidateRoleClassifier
from apps.api.app.services.legal_ranker import LegalRanker
from apps.api.app.services.query_frame import QueryFrameBuilder
from apps.api.app.services.query_orchestrator import QueryOrchestrator
from apps.api.app.services.query_understanding import QueryUnderstanding
from tests.helpers.live_like_demo import (
    LIVE_LIKE_DEMO_QUERY,
    LiveLikeRawRetriever,
    live_like_retrieval_candidates,
)


FORBIDDEN_LIVE_LIKE_DISTRACTORS = {
    "ro.codul_muncii.art_16.alin_1",
    "ro.codul_muncii.art_196.alin_2",
    "ro.codul_muncii.art_42.alin_1",
    "ro.codul_muncii.art_254.alin_3",
}

REQUIRED_ART_41_UNITS = {
    "ro.codul_muncii.art_41.alin_1",
    "ro.codul_muncii.art_41.alin_3",
}


def _demo_plan_and_frame():
    request = QueryRequest(
        question=LIVE_LIKE_DEMO_QUERY,
        jurisdiction="RO",
        date="current",
        mode="strict_citations",
        debug=True,
    )
    plan = QueryUnderstanding().build_plan(request)
    query_frame = QueryFrameBuilder().build(
        question=request.question,
        plan=plan,
    )
    return plan, query_frame


def test_candidate_role_classifier_live_like_demo_roles():
    _, query_frame = _demo_plan_and_frame()
    classifier = CandidateRoleClassifier()
    decisions = {
        candidate.unit_id: classifier.classify(
            query_frame=query_frame,
            unit_id=candidate.unit_id,
            unit=candidate.unit or {},
        )
        for candidate in live_like_retrieval_candidates()
    }

    agreement = decisions["ro.codul_muncii.art_41.alin_1"]
    assert agreement.support_role == "direct_basis"
    assert "contract_modification_agreement_rule" in agreement.matched_requirement_ids
    assert "requirement_match:contract_modification_agreement_rule" in agreement.why_role

    salary_scope = decisions["ro.codul_muncii.art_41.alin_3"]
    assert salary_scope.support_role == "direct_basis"
    assert "contract_modification_salary_scope" in salary_scope.matched_requirement_ids

    salary_target = decisions["ro.codul_muncii.art_41.alin_3.lit_e"]
    assert salary_target.support_role == "condition"
    assert "salary_target_element" in salary_target.matched_requirement_ids

    assert decisions["ro.codul_muncii.art_196.alin_2"].support_role == "context"
    assert decisions["ro.codul_muncii.art_16.alin_1"].support_role == "context"
    assert decisions["ro.codul_muncii.art_42.alin_1"].support_role in {
        "exception",
        "context",
    }
    assert decisions["ro.codul_muncii.art_254.alin_3"].support_role == "context"
    for unit_id in FORBIDDEN_LIVE_LIKE_DISTRACTORS:
        assert decisions[unit_id].support_role != "direct_basis"


def test_evidence_pack_live_like_uses_requirements_for_support_roles():
    plan, query_frame = _demo_plan_and_frame()
    retrieval_response = RawRetrievalResponse(
        candidates=live_like_retrieval_candidates(),
        retrieval_methods=["live_like_regression_fixture"],
    )
    ranked = LegalRanker().rank(
        question=LIVE_LIKE_DEMO_QUERY,
        plan=plan,
        retrieval_response=retrieval_response,
        graph_expansion=GraphExpansionResult(),
        query_frame=query_frame,
        debug=True,
    )
    compiled = EvidencePackCompiler(
        target_evidence_units=7,
        max_evidence_units=7,
    ).compile(
        ranked_candidates=ranked.ranked_candidates,
        graph_expansion=GraphExpansionResult(),
        plan=plan,
        query_frame=query_frame,
        debug=True,
    )

    evidence_by_id = {unit.id: unit for unit in compiled.evidence_units}
    assert REQUIRED_ART_41_UNITS.issubset(evidence_by_id)
    assert evidence_by_id["ro.codul_muncii.art_41.alin_1"].support_role == "direct_basis"
    assert evidence_by_id["ro.codul_muncii.art_41.alin_3"].support_role == "direct_basis"

    direct_basis_ids = {
        unit.id for unit in compiled.evidence_units if unit.support_role == "direct_basis"
    }
    assert direct_basis_ids <= {
        "ro.codul_muncii.art_41.alin_1",
        "ro.codul_muncii.art_41.alin_3",
    }
    assert not direct_basis_ids.intersection(FORBIDDEN_LIVE_LIKE_DISTRACTORS)
    for unit_id in REQUIRED_ART_41_UNITS:
        why_selected = evidence_by_id[unit_id].why_selected
        assert any(reason.startswith("requirement_match:") for reason in why_selected)
        assert "role_classifier:direct_basis" in why_selected


@pytest.mark.anyio
async def test_live_like_demo_flow_cites_art_41_not_topical_distractors():
    response = await QueryOrchestrator(
        raw_retriever_client=LiveLikeRawRetriever(),
        graph_expansion_policy=GraphExpansionPolicy(),
        evidence_pack_compiler=EvidencePackCompiler(
            target_evidence_units=7,
            max_evidence_units=7,
        ),
    ).run(
        QueryRequest(
            question=LIVE_LIKE_DEMO_QUERY,
            jurisdiction="RO",
            date="current",
            mode="strict_citations",
            debug=True,
        )
    )

    top3_ranked_ids = [
        candidate["unit_id"]
        for candidate in response.debug.legal_ranker["ranked_candidates"][:3]
    ]
    assert "ro.codul_muncii.art_41.alin_1" in top3_ranked_ids
    assert "ro.codul_muncii.art_41.alin_3" in top3_ranked_ids

    citation_unit_ids = {citation.legal_unit_id for citation in response.citations}
    assert REQUIRED_ART_41_UNITS.issubset(citation_unit_ids)
    assert all(unit_id.startswith("ro.codul_muncii.art_41") for unit_id in citation_unit_ids)
    assert not citation_unit_ids.intersection(FORBIDDEN_LIVE_LIKE_DISTRACTORS)
    assert (
        response.debug.generation["generation_mode"]
        == "deterministic_template_v1_labor_contract_modification"
    )
    answer = response.answer.short_answer.casefold()
    assert "art. 41" in answer
    assert "art. 196" not in answer
    assert "art. 16" not in answer
    assert "art. 42" not in answer
    assert "art. 254" not in answer
