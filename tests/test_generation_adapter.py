from apps.api.app.schemas import EvidenceUnit
from apps.api.app.services.generation_adapter import (
    GENERATION_INSUFFICIENT_EVIDENCE,
    GENERATION_LIMITED_EVIDENCE,
    GENERATION_MODE_TEMPLATE_LABOR_CONTRACT_MODIFICATION,
    GENERATION_MODE_TEMPLATE_V1,
    GENERATION_NO_DIRECT_BASIS,
    GENERATION_UNVERIFIED_WARNING,
    GenerationAdapter,
)
from apps.api.app.services.query_frame import QueryFrame
from tests.helpers.live_like_demo import LIVE_LIKE_DEMO_QUERY
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
    parent_id: str | None = None,
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
        parent_id=parent_id,
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
        question="Ce spune textul recuperat despre salariu?",
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
    rendered = f"{draft.short_answer}\n{draft.detailed_answer}"
    assert "părților" in rendered
    assert "partilor" not in rendered
    for citation in draft.citations:
        assert citation.snippet in units[citation.unit_id]["raw_text"]
    assert draft.confidence == 0.0
    assert GENERATION_UNVERIFIED_WARNING in draft.warnings


def test_demo_labor_contract_modification_template_keeps_focused_citations():
    evidence = [
        evidence_unit(
            "ro.codul_muncii.art_41.alin_1",
            "Contractul individual de munca poate fi modificat numai prin acordul partilor.",
            rank=2,
            support_role="direct_basis",
            paragraph_number="1",
        ),
        evidence_unit(
            "ro.codul_muncii.art_41.alin_3",
            "Modificarea contractului individual de munca se refera la elementele contractului, inclusiv salariul.",
            rank=3,
            support_role="direct_basis",
            paragraph_number="3",
        ),
        evidence_unit(
            "ro.codul_muncii.art_41.alin_3.lit_e",
            "e) salariul.",
            rank=4,
            support_role="direct_basis",
            paragraph_number="3",
            letter_number="e",
        ),
        evidence_unit(
            "ro.codul_muncii.art_264.lit_a",
            "orice remuneratie restanta datorata persoanelor angajate ilegal, inclusiv salariul;",
            rank=1,
            support_role="context",
            article_number="264",
            letter_number="a",
        ),
    ]
    query_frame = QueryFrame(
        domain="munca",
        intents=["labor_contract_modification"],
        meta_intents=["modification", "permission"],
        targets=["salary"],
        confidence=0.9,
    )

    draft = GenerationAdapter().generate(
        question=DEMO_QUERY,
        evidence_units=evidence,
        query_frame=query_frame,
    )

    citation_ids = [citation.unit_id for citation in draft.citations]
    assert draft.generation_mode == GENERATION_MODE_TEMPLATE_LABOR_CONTRACT_MODIFICATION
    assert "ro.codul_muncii.art_41.alin_1" in citation_ids
    assert "ro.codul_muncii.art_41.alin_3" in citation_ids
    assert "ro.codul_muncii.art_41.alin_3.lit_e" in citation_ids
    assert "ro.codul_muncii.art_264.lit_a" not in citation_ids
    assert set(draft.used_evidence_unit_ids) == set(citation_ids)
    assert "Codul muncii, art. 41 alin. (1)" in draft.short_answer
    assert "evidence:ro.codul_muncii.art_41.alin_1" in draft.short_answer
    assert "muncă" in draft.short_answer
    assert "regulă" in draft.short_answer
    assert "părților" in draft.short_answer
    assert "Excepțiile" in draft.short_answer
    assert "partilor" not in draft.short_answer
    assert "Exceptiile" not in draft.short_answer


def test_labor_contract_modification_template_accepts_condition_salary_scope():
    evidence = [
        evidence_unit(
            "ro.codul_muncii.art_41.alin_1",
            "Contractul individual de munca poate fi modificat numai prin acordul partilor.",
            rank=1,
            support_role="direct_basis",
            paragraph_number="1",
        ),
        evidence_unit(
            "ro.codul_muncii.art_41.alin_3",
            (
                "Modificarea contractului individual de munca poate privi "
                "durata contractului, locul muncii, felul muncii si salariul."
            ),
            rank=2,
            support_role="condition",
            paragraph_number="3",
        ),
    ]
    query_frame = QueryFrame(
        domain="munca",
        intents=["labor_contract_modification"],
        meta_intents=["modification", "permission"],
        targets=["salary"],
        confidence=0.9,
    )

    draft = GenerationAdapter().generate(
        question=LIVE_LIKE_DEMO_QUERY,
        evidence_units=evidence,
        query_frame=query_frame,
    )

    citation_ids = [citation.unit_id for citation in draft.citations]
    assert draft.generation_mode == GENERATION_MODE_TEMPLATE_LABOR_CONTRACT_MODIFICATION
    assert citation_ids == [
        "ro.codul_muncii.art_41.alin_1",
        "ro.codul_muncii.art_41.alin_3",
    ]
    assert draft.used_evidence_unit_ids == citation_ids
    assert "muncă" in draft.short_answer
    assert "părților" in draft.short_answer


def test_labor_contract_modification_template_uses_parent_scope_with_salary_child():
    evidence = [
        evidence_unit(
            "ro.codul_muncii.art_41.alin_1",
            "Contractul individual de munca poate fi modificat numai prin acordul partilor.",
            rank=1,
            support_role="direct_basis",
            paragraph_number="1",
        ),
        evidence_unit(
            "ro.codul_muncii.art_41.alin_3",
            (
                "Modificarea contractului individual de munca se refera la "
                "oricare dintre urmatoarele elemente:"
            ),
            rank=2,
            support_role="condition",
            paragraph_number="3",
        ),
        evidence_unit(
            "ro.codul_muncii.art_41.alin_3.lit_e",
            "e) salariul.",
            rank=3,
            support_role="condition",
            paragraph_number="3",
            letter_number="e",
            parent_id="ro.codul_muncii.art_41.alin_3",
        ),
    ]
    query_frame = QueryFrame(
        domain="munca",
        intents=["labor_contract_modification"],
        meta_intents=["modification", "permission"],
        targets=["salary"],
        confidence=0.9,
    )

    draft = GenerationAdapter().generate(
        question=LIVE_LIKE_DEMO_QUERY,
        evidence_units=evidence,
        query_frame=query_frame,
    )

    citation_ids = [citation.unit_id for citation in draft.citations]
    assert draft.generation_mode == GENERATION_MODE_TEMPLATE_LABOR_CONTRACT_MODIFICATION
    assert "ro.codul_muncii.art_41.alin_1" in citation_ids
    assert "ro.codul_muncii.art_41.alin_3" in citation_ids
    assert "ro.codul_muncii.art_41.alin_3.lit_e" in citation_ids
    assert set(draft.used_evidence_unit_ids) == set(citation_ids)


def test_generic_obligation_template_uses_only_cited_direct_basis():
    evidence = [
        evidence_unit(
            "fixture.obligation",
            "Angajatorul are obligatia de a informa salariatul in scris.",
            support_role="direct_basis",
            article_number="10",
        )
    ]
    query_frame = QueryFrame(
        domain="munca",
        meta_intents=["obligation"],
        confidence=0.9,
    )

    draft = GenerationAdapter().generate(
        question="Ce obligatie are angajatorul?",
        evidence_units=evidence,
        query_frame=query_frame,
    )

    assert draft.generation_mode == GENERATION_MODE_TEMPLATE_V1
    assert draft.meta_intent_used == "obligation"
    assert draft.template_id == "meta_intent:obligation"
    assert GENERATION_INSUFFICIENT_EVIDENCE not in draft.warnings
    assert [citation.unit_id for citation in draft.citations] == ["fixture.obligation"]
    assert draft.used_evidence_unit_ids == ["fixture.obligation"]
    assert "Codul muncii, art. 10" in draft.short_answer


def test_generic_procedure_template_stays_prudent_about_missing_steps():
    evidence = [
        evidence_unit(
            "fixture.procedure",
            "Cererea se depune in termenul prevazut si se solutioneaza potrivit procedurii.",
            support_role="procedure",
            article_number="20",
        )
    ]
    query_frame = QueryFrame(
        domain="contraventional",
        meta_intents=["procedure"],
        confidence=0.9,
    )

    draft = GenerationAdapter().generate(
        question="Cum contest?",
        evidence_units=evidence,
        query_frame=query_frame,
    )

    assert draft.generation_mode == GENERATION_MODE_TEMPLATE_V1
    assert draft.meta_intent_used == "procedure"
    assert "Nu completez pași care nu apar în EvidencePack" in draft.short_answer
    assert [citation.unit_id for citation in draft.citations] == ["fixture.procedure"]


def test_sanction_template_does_not_use_unrelated_context():
    evidence = [
        evidence_unit(
            "fixture.sanction",
            "Fapta se sanctioneaza cu amenda contraventionala.",
            support_role="sanction",
            article_number="30",
        ),
        evidence_unit(
            "fixture.context",
            "Context general despre procedura administrativa.",
            support_role="context",
            article_number="31",
        ),
    ]
    query_frame = QueryFrame(
        domain="contraventional",
        meta_intents=["sanction"],
        confidence=0.9,
    )

    draft = GenerationAdapter().generate(
        question="Ce sanctiune se aplica?",
        evidence_units=evidence,
        query_frame=query_frame,
    )

    assert draft.generation_mode == GENERATION_MODE_TEMPLATE_V1
    assert [citation.unit_id for citation in draft.citations] == ["fixture.sanction"]
    assert "fixture.context" not in draft.used_evidence_unit_ids


def test_exception_evidence_is_used_only_when_query_frame_asks_exception():
    evidence = [
        evidence_unit(
            "fixture.basis",
            "Regula generala se aplica in conditiile textului citat.",
            support_role="direct_basis",
            article_number="40",
        ),
        evidence_unit(
            "fixture.exception",
            "Prin exceptie, regula nu se aplica in situatiile speciale indicate.",
            support_role="exception",
            article_number="41",
        ),
    ]

    no_exception = GenerationAdapter().generate(
        question="Care este regula?",
        evidence_units=evidence,
        query_frame=QueryFrame(meta_intents=["obligation"], confidence=0.9),
    )
    asks_exception = GenerationAdapter().generate(
        question="Exista exceptii?",
        evidence_units=evidence,
        query_frame=QueryFrame(meta_intents=["exception"], confidence=0.9),
    )

    assert [citation.unit_id for citation in no_exception.citations] == ["fixture.basis"]
    assert "fixture.exception" not in no_exception.short_answer
    assert "fixture.exception" in {citation.unit_id for citation in asks_exception.citations}


def test_template_with_only_context_warns_no_direct_basis():
    evidence = [
        evidence_unit(
            "fixture.context_only",
            "Context recuperat despre termenul folosit in intrebare.",
            support_role="context",
            article_number="50",
        )
    ]
    query_frame = QueryFrame(meta_intents=["obligation"], confidence=0.9)

    draft = GenerationAdapter().generate(
        question="Ce obligatie exista?",
        evidence_units=evidence,
        query_frame=query_frame,
    )

    assert draft.generation_mode == GENERATION_MODE_TEMPLATE_V1
    assert GENERATION_NO_DIRECT_BASIS in draft.warnings
    assert [citation.unit_id for citation in draft.citations] == ["fixture.context_only"]


def test_labor_contract_modification_with_only_topical_distractors_refuses_extractively():
    query_frame = QueryFrame(
        domain="munca",
        intents=["labor_contract_modification"],
        meta_intents=["modification", "permission"],
        targets=["salary"],
        confidence=0.9,
    )
    evidence = [
        evidence_unit(
            "ro.codul_muncii.art_196.alin_2",
            (
                "Modalitatea concreta de formare profesionala, drepturile si "
                "obligatiile partilor, durata formarii profesionale si alte "
                "aspecte fac obiectul unor acte aditionale."
            ),
            article_number="196",
            paragraph_number="2",
        ),
        evidence_unit(
            "ro.codul_muncii.art_254.alin_3",
            (
                "Recuperarea contravalorii pagubei se poate realiza prin "
                "acordul partilor, potrivit notei de constatare."
            ),
            article_number="254",
            paragraph_number="3",
        ),
    ]

    draft = GenerationAdapter().generate(
        question=LIVE_LIKE_DEMO_QUERY,
        evidence_units=evidence,
        query_frame=query_frame,
    )

    assert draft.generation_mode == "deterministic_extractive_v1_insufficient_evidence"
    assert GENERATION_INSUFFICIENT_EVIDENCE in draft.warnings
    assert draft.citations == []
    assert "art. 196" not in draft.short_answer
