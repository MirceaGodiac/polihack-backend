from apps.api.app.schemas import EvidenceUnit
from apps.api.app.services.generation_adapter import (
    GENERATION_INSUFFICIENT_EVIDENCE,
    GENERATION_LIMITED_EVIDENCE,
    GENERATION_UNVERIFIED_WARNING,
    GenerationAdapter,
)
from tests.helpers.fixture_handoff03 import DEMO_QUERY, load_codul_muncii_units


def evidence_unit(
    unit_id: str,
    raw_text: str,
    *,
    rank: int = 1,
    support_role: str = "direct_basis",
    law_title: str = "Codul muncii",
    article_number: str | None = "41",
    paragraph_number: str | None = None,
    letter_number: str | None = None,
) -> EvidenceUnit:
    return EvidenceUnit(
        id=unit_id,
        law_id="ro.codul_muncii",
        law_title=law_title,
        status="active",
        hierarchy_path=["Codul muncii", unit_id],
        article_number=article_number,
        paragraph_number=paragraph_number,
        letter_number=letter_number,
        raw_text=raw_text,
        normalized_text="retrieval-only generated context should not appear",
        legal_domain="munca",
        legal_concepts=[],
        source_url="https://legislatie.just.ro/test",
        evidence_id=f"evidence:{unit_id}",
        excerpt="retrieval-only excerpt should not appear",
        rank=rank,
        relevance_score=0.8,
        retrieval_method="fixture",
        retrieval_score=0.7,
        rerank_score=0.8,
        support_role=support_role,
    )


def evidence_from_fixture(
    unit: dict,
    *,
    rank: int,
    support_role: str,
) -> EvidenceUnit:
    return EvidenceUnit(
        **{
            key: unit.get(key)
            for key in (
                "id",
                "canonical_id",
                "source_id",
                "law_id",
                "law_title",
                "act_type",
                "act_number",
                "publication_date",
                "effective_date",
                "version_start",
                "version_end",
                "status",
                "hierarchy_path",
                "article_number",
                "paragraph_number",
                "letter_number",
                "point_number",
                "raw_text",
                "normalized_text",
                "legal_domain",
                "legal_concepts",
                "source_url",
                "parent_id",
                "children_ids",
                "outgoing_reference_ids",
                "incoming_reference_ids",
            )
        },
        evidence_id=f"evidence:{unit['id']}",
        excerpt=unit["raw_text"],
        rank=rank,
        relevance_score=0.9,
        retrieval_method="fixture_codul_muncii",
        retrieval_score=0.9,
        rerank_score=0.9,
        support_role=support_role,
        why_selected=["selected_by_test"],
    )


def test_empty_evidence_pack_returns_insufficient_evidence_warning():
    draft = GenerationAdapter().generate(
        question="Poate angajatorul sa-mi scada salariul fara act aditional?",
        evidence_units=[],
    )

    assert GENERATION_INSUFFICIENT_EVIDENCE in draft.warnings
    assert GENERATION_UNVERIFIED_WARNING in draft.warnings
    assert draft.citations == []
    assert draft.confidence == 0.0
    assert "Codul muncii" not in draft.short_answer
    assert "art. 41" not in draft.short_answer


def test_generation_uses_only_raw_text_for_citation_snippets():
    raw_direct = (
        "(4) Orice modificare a salariului impune incheierea unui act "
        "aditional la contract."
    )
    raw_condition = (
        "(3) Modificarea contractului individual de munca se refera la salariul."
    )
    evidence = [
        evidence_unit(
            "ro.codul_muncii.art_41.alin_4",
            raw_direct,
            rank=1,
            support_role="direct_basis",
            paragraph_number="4",
        ),
        evidence_unit(
            "ro.codul_muncii.art_41.alin_3",
            raw_condition,
            rank=2,
            support_role="condition",
            paragraph_number="3",
        ),
    ]

    draft = GenerationAdapter().generate(
        question="Poate angajatorul sa-mi scada salariul fara act aditional?",
        evidence_units=evidence,
    )

    evidence_by_id = {unit.id: unit for unit in evidence}
    assert draft.confidence == 0.0
    assert {citation.unit_id for citation in draft.citations} == set(evidence_by_id)
    for citation in draft.citations:
        assert citation.snippet in evidence_by_id[citation.unit_id].raw_text
    rendered = f"{draft.short_answer}\n{draft.detailed_answer}"
    assert "retrieval-only" not in rendered
    assert GENERATION_UNVERIFIED_WARNING in draft.warnings


def test_limited_evidence_stays_unverified_and_warns():
    draft = GenerationAdapter().generate(
        question="Ce spune textul recuperat?",
        evidence_units=[
            evidence_unit(
                "ro.codul_muncii.art_41",
                "Contractul individual de munca poate fi modificat numai prin acordul partilor.",
            )
        ],
    )

    assert draft.confidence == 0.0
    assert len(draft.citations) == 1
    assert GENERATION_LIMITED_EVIDENCE in draft.warnings
    assert GENERATION_UNVERIFIED_WARNING in draft.warnings


def test_demo_fixture_answer_cites_only_existing_codul_muncii_units():
    units = load_codul_muncii_units()
    evidence = [
        evidence_from_fixture(
            units["ro.codul_muncii.art_41.alin_4"],
            rank=1,
            support_role="direct_basis",
        ),
        evidence_from_fixture(
            units["ro.codul_muncii.art_41"],
            rank=2,
            support_role="direct_basis",
        ),
        evidence_from_fixture(
            units["ro.codul_muncii.art_41.alin_3"],
            rank=3,
            support_role="condition",
        ),
        evidence_from_fixture(
            units["ro.codul_muncii.art_17.alin_3.lit_k"],
            rank=4,
            support_role="context",
        ),
    ]

    draft = GenerationAdapter().generate(question=DEMO_QUERY, evidence_units=evidence)

    evidence_ids = {unit.id for unit in evidence}
    citation_ids = {citation.unit_id for citation in draft.citations}
    assert {
        "ro.codul_muncii.art_41",
        "ro.codul_muncii.art_41.alin_3",
    }.issubset(citation_ids)
    assert "ro.codul_muncii.art_41.alin_4" not in citation_ids
    assert "ro.codul_muncii.art_17.alin_3.lit_k" not in citation_ids
    assert citation_ids <= evidence_ids
    assert DEMO_QUERY not in draft.short_answer
    assert "acordul partilor" in f"{draft.short_answer}\n{draft.detailed_answer}"
    for citation in draft.citations:
        assert citation.snippet in units[citation.unit_id]["raw_text"]
    assert draft.confidence == 0.0
    assert GENERATION_UNVERIFIED_WARNING in draft.warnings
