from apps.api.app.schemas import AnswerPayload, Citation, ClaimResult, EvidenceUnit, VerifierStatus
from apps.api.app.services.answer_repair import (
    ANSWER_REFUSED_INSUFFICIENT_EVIDENCE,
    ANSWER_REFUSED_NO_VERIFIABLE_LEGAL_CLAIMS,
    ANSWER_REFUSED_UNSUPPORTED_CLAIMS,
    ANSWER_REPAIRED_UNSUPPORTED_CLAIMS_REMOVED,
    ANSWER_TEMPERED_WEAK_SUPPORT,
    INSUFFICIENT_EVIDENCE_ANSWER,
    INVALID_CITATIONS_REMOVED,
    REFUSAL_INSUFFICIENT_EVIDENCE,
    REFUSAL_NO_VERIFIABLE_LEGAL_CLAIMS,
    REFUSAL_UNSUPPORTED_CLAIMS,
    WEAK_SUPPORT_TEMPERING,
    AnswerRepair,
)


def evidence_unit(unit_id: str = "ro.codul_muncii.art_41") -> EvidenceUnit:
    raw_text = "Contractul individual de munca poate fi modificat numai prin acordul partilor."
    return EvidenceUnit(
        id=unit_id,
        law_id="ro.codul_muncii",
        law_title="Codul muncii",
        status="active",
        hierarchy_path=["Codul muncii", "Art. 41"],
        article_number="41",
        raw_text=raw_text,
        normalized_text=None,
        legal_domain="munca",
        legal_concepts=[],
        source_url=None,
        evidence_id=f"evidence:{unit_id}",
        excerpt=raw_text,
        rank=1,
        relevance_score=0.9,
        retrieval_method="test",
        retrieval_score=0.9,
        rerank_score=0.9,
        support_role="direct_basis",
    )


def citation(
    *,
    citation_id: str = "citation:1",
    unit_id: str = "ro.codul_muncii.art_41",
    verified: bool = True,
) -> Citation:
    return Citation(
        citation_id=citation_id,
        evidence_id=f"evidence:{unit_id}",
        legal_unit_id=unit_id,
        label="Codul muncii, art. 41",
        quote="Contractul individual de munca poate fi modificat numai prin acordul partilor.",
        verified=verified,
    )


def answer(
    short_answer: str = "Contractul individual de munca poate fi modificat numai prin acordul partilor.",
) -> AnswerPayload:
    return AnswerPayload(
        short_answer=short_answer,
        detailed_answer=None,
        confidence=0.0,
        not_legal_advice=True,
    )


def claim(
    *,
    claim_id: str = "claim:1",
    text: str = "Contractul individual de munca poate fi modificat numai prin acordul partilor.",
    status: str = "supported",
    citation_ids: list[str] | None = None,
) -> ClaimResult:
    support_score = {
        "strongly_supported": 0.85,
        "supported": 0.7,
        "weakly_supported": 0.5,
        "unsupported": 0.2,
    }[status]
    return ClaimResult(
        claim_id=claim_id,
        claim_text=text,
        status=status,
        citation_ids=citation_ids or ["citation:1"],
        confidence=support_score,
        support_score=support_score,
        supporting_unit_ids=["ro.codul_muncii.art_41"] if status != "unsupported" else [],
    )


def verifier(
    claims: list[ClaimResult],
    *,
    passed: bool | None = None,
    warnings: list[str] | None = None,
) -> VerifierStatus:
    supported = [
        item
        for item in claims
        if item.status in {"strongly_supported", "supported"}
    ]
    weak = [item for item in claims if item.status == "weakly_supported"]
    unsupported = [item for item in claims if item.status == "unsupported"]
    return VerifierStatus(
        groundedness_score=len(supported) / len(claims) if claims else 0.0,
        claims_total=len(claims),
        claims_supported=len(supported),
        claims_weakly_supported=len(weak),
        claims_unsupported=len(unsupported),
        citations_checked=1,
        verifier_passed=(
            passed
            if passed is not None
            else bool(claims) and not weak and not unsupported
        ),
        claim_results=claims,
        warnings=warnings or [],
    )


def test_answer_repair_empty_evidence_refuses():
    result = AnswerRepair().repair(
        answer=answer(),
        citations=[],
        evidence_units=[],
        verifier=verifier([], passed=False),
        debug=True,
    )

    assert result.answer.short_answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.answer.refusal_reason == REFUSAL_INSUFFICIENT_EVIDENCE
    assert result.citations == []
    assert result.repair_applied is True
    assert result.verifier.repair_applied is True
    assert result.verifier.refusal_reason == REFUSAL_INSUFFICIENT_EVIDENCE
    assert ANSWER_REFUSED_INSUFFICIENT_EVIDENCE in result.warnings
    assert result.debug["repair_action"] == "refused_insufficient_evidence"


def test_answer_repair_no_legal_claims_refuses():
    result = AnswerRepair().repair(
        answer=answer("Acesta este doar text operational."),
        citations=[],
        evidence_units=[evidence_unit()],
        verifier=verifier([], passed=False),
    )

    assert result.answer.refusal_reason == REFUSAL_NO_VERIFIABLE_LEGAL_CLAIMS
    assert result.verifier.verifier_passed is False
    assert result.repair_applied is True
    assert ANSWER_REFUSED_NO_VERIFIABLE_LEGAL_CLAIMS in result.warnings


def test_answer_repair_removes_unsupported_claim():
    supported = claim(claim_id="claim:1")
    unsupported = claim(
        claim_id="claim:2",
        text="Angajatorul poate concedia salariatul fara preaviz.",
        status="unsupported",
        citation_ids=["citation:1"],
    )

    result = AnswerRepair().repair(
        answer=answer(
            "Contractul individual de munca poate fi modificat numai prin acordul partilor. "
            "Angajatorul poate concedia salariatul fara preaviz."
        ),
        citations=[citation()],
        evidence_units=[evidence_unit()],
        verifier=verifier([supported, unsupported], passed=False),
        debug=True,
    )

    rendered = f"{result.answer.short_answer}\n{result.answer.detailed_answer}"
    assert supported.claim_text in rendered
    assert unsupported.claim_text not in rendered
    assert result.repair_applied is True
    assert result.verifier.verifier_passed is False
    assert ANSWER_REPAIRED_UNSUPPORTED_CLAIMS_REMOVED in result.warnings
    assert result.debug["removed_claims"] == [unsupported.claim_text]


def test_answer_repair_refuses_when_all_claims_unsupported():
    unsupported = claim(
        text="Angajatorul poate concedia salariatul fara preaviz.",
        status="unsupported",
    )

    result = AnswerRepair().repair(
        answer=answer(unsupported.claim_text),
        citations=[citation()],
        evidence_units=[evidence_unit()],
        verifier=verifier([unsupported], passed=False),
    )

    assert result.answer.refusal_reason == REFUSAL_UNSUPPORTED_CLAIMS
    assert result.citations == []
    assert result.verifier.verifier_passed is False
    assert ANSWER_REFUSED_UNSUPPORTED_CLAIMS in result.warnings


def test_answer_repair_tempers_weak_support():
    weak = claim(status="weakly_supported")

    result = AnswerRepair().repair(
        answer=answer(),
        citations=[citation()],
        evidence_units=[evidence_unit()],
        verifier=verifier([weak], passed=False),
    )

    assert result.answer.short_answer.startswith(WEAK_SUPPORT_TEMPERING)
    assert result.repair_applied is True
    assert result.verifier.verifier_passed is False
    assert ANSWER_TEMPERED_WEAK_SUPPORT in result.warnings


def test_answer_repair_leaves_fully_supported_answer_unchanged():
    original = answer()
    result = AnswerRepair().repair(
        answer=original,
        citations=[citation()],
        evidence_units=[evidence_unit()],
        verifier=verifier([claim()], passed=True),
    )

    assert result.answer.short_answer == original.short_answer
    assert result.repair_applied is False
    assert result.verifier.verifier_passed is True
    assert ANSWER_TEMPERED_WEAK_SUPPORT not in result.warnings
    assert ANSWER_REPAIRED_UNSUPPORTED_CLAIMS_REMOVED not in result.warnings


def test_answer_repair_keeps_valid_cited_labor_contract_modification_answer():
    original = answer(
        "Contractul individual de munca poate fi modificat numai prin acordul "
        "partilor [Codul muncii, art. 41 alin. (1); "
        "evidence:ro.codul_muncii.art_41.alin_1]. "
        "Modificarea contractului individual de munca poate privi elementele "
        "contractului, inclusiv salariul [Codul muncii, art. 41 alin. (3); "
        "evidence:ro.codul_muncii.art_41.alin_3]."
    )
    first_claim = claim(
        claim_id="claim:1",
        text="Contractul individual de munca poate fi modificat numai prin acordul partilor.",
        citation_ids=["citation:1"],
    )
    second_claim = claim(
        claim_id="claim:2",
        text=(
            "Modificarea contractului individual de munca poate privi "
            "elementele contractului, inclusiv salariul."
        ),
        citation_ids=["citation:2"],
    )

    result = AnswerRepair().repair(
        answer=original,
        citations=[
            citation(
                citation_id="citation:1",
                unit_id="ro.codul_muncii.art_41.alin_1",
            ),
            citation(
                citation_id="citation:2",
                unit_id="ro.codul_muncii.art_41.alin_3",
            ),
        ],
        evidence_units=[
            evidence_unit("ro.codul_muncii.art_41.alin_1"),
            evidence_unit("ro.codul_muncii.art_41.alin_3"),
        ],
        verifier=verifier([first_claim, second_claim], passed=True),
        debug=True,
    )

    assert result.answer.short_answer == original.short_answer
    assert [item.legal_unit_id for item in result.citations] == [
        "ro.codul_muncii.art_41.alin_1",
        "ro.codul_muncii.art_41.alin_3",
    ]
    assert result.repair_applied is False
    assert result.debug["repair_action"] == "none"


def test_answer_repair_removes_invalid_citations():
    valid = citation(citation_id="citation:1", verified=True)
    invalid = citation(
        citation_id="citation:bad",
        unit_id="ro.codul_muncii.art_missing",
        verified=False,
    )
    result = AnswerRepair().repair(
        answer=answer(),
        citations=[valid, invalid],
        evidence_units=[evidence_unit()],
        verifier=verifier([claim(citation_ids=["citation:1"])], passed=False),
        debug=True,
    )

    assert [item.citation_id for item in result.citations] == ["citation:1"]
    assert result.repair_applied is True
    assert result.verifier.verifier_passed is False
    assert INVALID_CITATIONS_REMOVED in result.warnings
    assert result.debug["removed_citation_ids"] == ["citation:bad"]
