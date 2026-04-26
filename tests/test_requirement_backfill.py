import pytest

from apps.api.app.schemas import GraphExpansionResult, QueryRequest, RankedCandidate, RankerFeatureBreakdown
from apps.api.app.services.evidence_pack_compiler import EvidencePackCompiler
from apps.api.app.services.query_frame import QueryFrameBuilder
from apps.api.app.services.query_understanding import QueryUnderstanding
from apps.api.app.services.requirement_backfill import RequirementBackfillService
from tests.helpers.live_like_demo import LIVE_LIKE_DEMO_QUERY


def _plan_and_frame():
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


def _unit(
    unit_id: str,
    raw_text: str,
    *,
    article_number: str,
    paragraph_number: str | None = None,
    letter_number: str | None = None,
    parent_id: str | None = None,
    unit_type: str = "alineat",
) -> dict:
    return {
        "id": unit_id,
        "law_id": "ro.codul_muncii",
        "law_title": "Codul muncii",
        "status": "active",
        "hierarchy_path": ["Codul muncii", f"Art. {article_number}"],
        "article_number": article_number,
        "paragraph_number": paragraph_number,
        "letter_number": letter_number,
        "parent_id": parent_id,
        "raw_text": raw_text,
        "normalized_text": raw_text.casefold(),
        "legal_domain": "munca",
        "legal_concepts": ["contract", "salariu"],
        "source_url": "https://legislatie.just.ro/test",
        "type": unit_type,
    }


def _ranked(
    unit_id: str,
    raw_text: str,
    *,
    rank: int,
    rerank_score: float,
    retrieval_score: float,
    article_number: str,
    paragraph_number: str | None = None,
    letter_number: str | None = None,
    parent_id: str | None = None,
) -> RankedCandidate:
    return RankedCandidate(
        unit_id=unit_id,
        rank=rank,
        rerank_score=rerank_score,
        retrieval_score=retrieval_score,
        unit=_unit(
            unit_id,
            raw_text,
            article_number=article_number,
            paragraph_number=paragraph_number,
            letter_number=letter_number,
            parent_id=parent_id,
            unit_type="litera" if letter_number else "alineat",
        ),
        score_breakdown=RankerFeatureBreakdown(
            bm25_score=rerank_score,
            dense_score=rerank_score,
            domain_match=1.0,
            temporal_validity=1.0,
        ),
        why_ranked=["fixture_ranked_candidate"],
        source="fixture",
    )


@pytest.mark.anyio
async def test_requirement_backfill_selects_salary_scope_over_salary_distractor():
    plan, query_frame = _plan_and_frame()
    ranked_candidates = [
        _ranked(
            "ro.codul_muncii.art_41.alin_1",
            "Contractul individual de munca poate fi modificat numai prin acordul partilor.",
            rank=1,
            rerank_score=0.92,
            retrieval_score=0.92,
            article_number="41",
            paragraph_number="1",
        ),
        _ranked(
            "ro.codul_muncii.art_264.lit_a",
            "Remuneratie restanta pentru persoane angajate ilegal, inclusiv salariul.",
            rank=2,
            rerank_score=0.88,
            retrieval_score=0.88,
            article_number="264",
            letter_number="a",
            parent_id="ro.codul_muncii.art_264",
        ),
        _ranked(
            "ro.codul_muncii.art_41.alin_3",
            (
                "Modificarea contractului individual de munca poate privi "
                "durata contractului, locul muncii, felul muncii si salariul."
            ),
            rank=3,
            rerank_score=0.63,
            retrieval_score=0.63,
            article_number="41",
            paragraph_number="3",
        ),
    ]
    compiler = EvidencePackCompiler(
        candidate_pool_size=1,
        target_evidence_units=1,
        max_evidence_units=2,
    )
    initial = compiler.compile(
        ranked_candidates=ranked_candidates,
        graph_expansion=GraphExpansionResult(),
        plan=plan,
        query_frame=query_frame,
        debug=True,
    )

    assert [unit.id for unit in initial.evidence_units] == [
        "ro.codul_muncii.art_41.alin_1"
    ]
    assert initial.debug["requirement_coverage"]["coverage_passed"] is False

    result = await RequirementBackfillService(
        evidence_pack_compiler=compiler
    ).backfill(
        plan=plan,
        query_frame=query_frame,
        ranked_candidates=ranked_candidates,
        evidence_result=initial,
        graph_expansion=GraphExpansionResult(),
        debug=True,
    )

    evidence_ids = [unit.id for unit in result.evidence_result.evidence_units]
    assert "ro.codul_muncii.art_41.alin_3" in result.added_unit_ids
    assert "ro.codul_muncii.art_41.alin_3" in evidence_ids
    assert "ro.codul_muncii.art_264.lit_a" not in result.added_unit_ids
    assert result.debug["coverage_before"]["coverage_passed"] is False
    assert result.debug["coverage_after"]["coverage_passed"] is True
    assert any(
        row["unit_id"] == "ro.codul_muncii.art_41.alin_3" and row["selected"] is True
        for row in result.debug["scored_candidates"]
    )
    assert any(
        row["unit_id"] == "ro.codul_muncii.art_264.lit_a" and row["selected"] is False
        for row in result.debug["scored_candidates"]
    )
