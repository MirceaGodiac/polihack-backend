from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..schemas import AnswerPayload, Citation, ClaimResult, EvidenceUnit, VerifierStatus

ANSWER_REFUSED_INSUFFICIENT_EVIDENCE = "answer_refused_insufficient_evidence"
ANSWER_REFUSED_NO_VERIFIABLE_LEGAL_CLAIMS = (
    "answer_refused_no_verifiable_legal_claims"
)
ANSWER_REPAIRED_UNSUPPORTED_CLAIMS_REMOVED = (
    "answer_repaired_unsupported_claims_removed"
)
ANSWER_REFUSED_UNSUPPORTED_CLAIMS = "answer_refused_unsupported_claims"
ANSWER_TEMPERED_WEAK_SUPPORT = "answer_tempered_weak_support"
INVALID_CITATIONS_REMOVED = "invalid_citations_removed"

REFUSAL_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
REFUSAL_NO_VERIFIABLE_LEGAL_CLAIMS = "no_verifiable_legal_claims"
REFUSAL_UNSUPPORTED_CLAIMS = "unsupported_claims"

REPAIR_ACTION_NONE = "none"
REPAIR_ACTION_REMOVED_UNSUPPORTED = "removed_unsupported_claims"
REPAIR_ACTION_TEMPERED_WEAK_SUPPORT = "tempered_weak_support"
REPAIR_ACTION_REFUSED_INSUFFICIENT = "refused_insufficient_evidence"
REPAIR_ACTION_REFUSED_UNSUPPORTED = "refused_unsupported_claims"
REPAIR_ACTION_REFUSED_NO_CLAIMS = "refused_no_verifiable_claims"

INSUFFICIENT_EVIDENCE_ANSWER = (
    "Nu pot formula un r\u0103spuns juridic sus\u021binut pe baza "
    "corpusului disponibil."
)
INSUFFICIENT_EVIDENCE_DETAIL = (
    "EvidencePack-ul disponibil este gol sau insuficient pentru a sus\u021bine "
    "un r\u0103spuns juridic verificabil."
)
NO_VERIFIABLE_CLAIMS_ANSWER = (
    "Nu pot identifica afirma\u021bii juridice verificabile \u00een "
    "r\u0103spunsul generat."
)
NO_VERIFIABLE_CLAIMS_DETAIL = (
    "R\u0103spunsul generat nu con\u021bine claim-uri juridice pe care "
    "CitationVerifier V1 s\u0103 le poat\u0103 valida fa\u021b\u0103 de "
    "EvidencePack."
)
UNSUPPORTED_CLAIMS_REFUSAL = (
    "Nu pot publica r\u0103spunsul generat deoarece afirma\u021biile "
    "juridice nu sunt sus\u021binute de EvidencePack."
)
UNSUPPORTED_CLAIMS_DETAIL = (
    "CitationVerifier V1 a marcat toate afirma\u021biile juridice "
    "publicabile ca unsupported."
)
WEAK_SUPPORT_TEMPERING = (
    "Pe baza corpusului disponibil, suportul pentru unele afirma\u021bii "
    "este limitat."
)
REPAIRED_DETAIL_HEADER = "Afirma\u021bii p\u0103strate dup\u0103 verificare:"

SUPPORTED_STATUSES = {"strongly_supported", "supported"}
WEAK_STATUS = "weakly_supported"
UNSUPPORTED_STATUS = "unsupported"


class AnswerRepairResult(BaseModel):
    answer: AnswerPayload
    citations: list[Citation] = Field(default_factory=list)
    verifier: VerifierStatus
    warnings: list[str] = Field(default_factory=list)
    repair_applied: bool
    refusal_reason: str | None = None
    debug: dict[str, Any] | None = None


class AnswerRepair:
    def repair(
        self,
        *,
        answer: AnswerPayload,
        citations: list[Citation],
        evidence_units: list[EvidenceUnit],
        verifier: VerifierStatus,
        warnings: list[str] | None = None,
        debug: bool = False,
    ) -> AnswerRepairResult:
        existing_warnings = list(warnings or [])
        valid_citations, removed_citations = self._filter_invalid_citations(
            citations,
            evidence_units,
        )
        warnings_added: list[str] = []
        if removed_citations:
            warnings_added.append(INVALID_CITATIONS_REMOVED)

        if not evidence_units:
            return self._refusal_result(
                answer=answer,
                verifier=verifier,
                warnings=existing_warnings,
                warnings_added=[
                    *warnings_added,
                    ANSWER_REFUSED_INSUFFICIENT_EVIDENCE,
                ],
                action=REPAIR_ACTION_REFUSED_INSUFFICIENT,
                reason=REFUSAL_INSUFFICIENT_EVIDENCE,
                short_answer=INSUFFICIENT_EVIDENCE_ANSWER,
                detailed_answer=INSUFFICIENT_EVIDENCE_DETAIL,
                removed_citations=removed_citations,
                kept_claims=[],
                removed_claims=[],
                debug=debug,
            )

        if verifier.claims_total == 0:
            return self._refusal_result(
                answer=answer,
                verifier=verifier,
                warnings=existing_warnings,
                warnings_added=[
                    *warnings_added,
                    ANSWER_REFUSED_NO_VERIFIABLE_LEGAL_CLAIMS,
                ],
                action=REPAIR_ACTION_REFUSED_NO_CLAIMS,
                reason=REFUSAL_NO_VERIFIABLE_LEGAL_CLAIMS,
                short_answer=NO_VERIFIABLE_CLAIMS_ANSWER,
                detailed_answer=NO_VERIFIABLE_CLAIMS_DETAIL,
                removed_citations=removed_citations,
                kept_claims=[],
                removed_claims=verifier.claim_results,
                debug=debug,
            )

        invalidated_claims = self._claims_invalidated_by_removed_citations(
            verifier.claim_results,
            removed_citations,
        )
        unsupported_claims = [
            claim
            for claim in verifier.claim_results
            if claim.status == UNSUPPORTED_STATUS or claim in invalidated_claims
        ]
        weak_claims = [
            claim
            for claim in verifier.claim_results
            if claim.status == WEAK_STATUS
        ]
        kept_claims = self._publishable_claims(
            verifier.claim_results,
            invalidated_claims,
        )

        if unsupported_claims:
            if not kept_claims:
                return self._refusal_result(
                    answer=answer,
                    verifier=verifier,
                    warnings=existing_warnings,
                    warnings_added=[
                        *warnings_added,
                        ANSWER_REFUSED_UNSUPPORTED_CLAIMS,
                    ],
                    action=REPAIR_ACTION_REFUSED_UNSUPPORTED,
                    reason=REFUSAL_UNSUPPORTED_CLAIMS,
                    short_answer=UNSUPPORTED_CLAIMS_REFUSAL,
                    detailed_answer=UNSUPPORTED_CLAIMS_DETAIL,
                    removed_citations=removed_citations or citations,
                    kept_claims=[],
                    removed_claims=unsupported_claims,
                    debug=debug,
                )

            repaired_answer = self._answer_from_kept_claims(answer, kept_claims)
            repaired_verifier = self._updated_verifier(
                verifier,
                verifier_passed=False,
                repair_applied=True,
                refusal_reason=None,
                warnings_added=[
                    *warnings_added,
                    ANSWER_REPAIRED_UNSUPPORTED_CLAIMS_REMOVED,
                ],
            )
            final_warnings = self._dedupe(
                existing_warnings
                + warnings_added
                + [ANSWER_REPAIRED_UNSUPPORTED_CLAIMS_REMOVED]
            )
            return AnswerRepairResult(
                answer=repaired_answer,
                citations=valid_citations,
                verifier=repaired_verifier,
                warnings=final_warnings,
                repair_applied=True,
                refusal_reason=None,
                debug=self._debug_payload(
                    action=REPAIR_ACTION_REMOVED_UNSUPPORTED,
                    removed_claims=unsupported_claims,
                    kept_claims=kept_claims,
                    removed_citations=removed_citations,
                    refusal_reason=None,
                    warnings_added=[
                        *warnings_added,
                        ANSWER_REPAIRED_UNSUPPORTED_CLAIMS_REMOVED,
                    ],
                )
                if debug
                else None,
            )

        fully_supported = verifier.verifier_passed or (
            self._all_claims_supported(verifier.claim_results)
            and not verifier.warnings
        )
        if weak_claims or not fully_supported:
            tempered_answer = self._temper_answer(answer)
            repaired_verifier = self._updated_verifier(
                verifier,
                verifier_passed=False,
                repair_applied=True,
                refusal_reason=None,
                warnings_added=[*warnings_added, ANSWER_TEMPERED_WEAK_SUPPORT],
            )
            final_warnings = self._dedupe(
                existing_warnings + warnings_added + [ANSWER_TEMPERED_WEAK_SUPPORT]
            )
            return AnswerRepairResult(
                answer=tempered_answer,
                citations=valid_citations,
                verifier=repaired_verifier,
                warnings=final_warnings,
                repair_applied=True,
                refusal_reason=None,
                debug=self._debug_payload(
                    action=REPAIR_ACTION_TEMPERED_WEAK_SUPPORT,
                    removed_claims=[],
                    kept_claims=verifier.claim_results,
                    removed_citations=removed_citations,
                    refusal_reason=None,
                    warnings_added=[*warnings_added, ANSWER_TEMPERED_WEAK_SUPPORT],
                )
                if debug
                else None,
            )

        if removed_citations:
            repaired_verifier = self._updated_verifier(
                verifier,
                verifier_passed=False,
                repair_applied=True,
                refusal_reason=None,
                warnings_added=warnings_added,
            )
            final_warnings = self._dedupe(existing_warnings + warnings_added)
            return AnswerRepairResult(
                answer=answer,
                citations=valid_citations,
                verifier=repaired_verifier,
                warnings=final_warnings,
                repair_applied=True,
                refusal_reason=None,
                debug=self._debug_payload(
                    action=REPAIR_ACTION_TEMPERED_WEAK_SUPPORT,
                    removed_claims=[],
                    kept_claims=verifier.claim_results,
                    removed_citations=removed_citations,
                    refusal_reason=None,
                    warnings_added=warnings_added,
                )
                if debug
                else None,
            )

        passed_verifier = self._updated_verifier(
            verifier,
            verifier_passed=True,
            repair_applied=False,
            refusal_reason=None,
            warnings_added=[],
        )
        return AnswerRepairResult(
            answer=answer,
            citations=valid_citations,
            verifier=passed_verifier,
            warnings=self._dedupe(existing_warnings),
            repair_applied=False,
            refusal_reason=None,
            debug=self._debug_payload(
                action=REPAIR_ACTION_NONE,
                removed_claims=[],
                kept_claims=verifier.claim_results,
                removed_citations=[],
                refusal_reason=None,
                warnings_added=[],
            )
            if debug
            else None,
        )

    def _filter_invalid_citations(
        self,
        citations: list[Citation],
        evidence_units: list[EvidenceUnit],
    ) -> tuple[list[Citation], list[Citation]]:
        evidence_ids = {unit.id for unit in evidence_units}
        valid: list[Citation] = []
        removed: list[Citation] = []
        for citation in citations:
            if citation.verified and citation.legal_unit_id in evidence_ids:
                valid.append(citation)
            else:
                removed.append(citation)
        return valid, removed

    def _claims_invalidated_by_removed_citations(
        self,
        claims: list[ClaimResult],
        removed_citations: list[Citation],
    ) -> list[ClaimResult]:
        removed_ids = {citation.citation_id for citation in removed_citations}
        if not removed_ids:
            return []
        return [
            claim
            for claim in claims
            if claim.citation_ids
            and removed_ids.intersection(claim.citation_ids)
            and not set(claim.citation_ids).difference(removed_ids)
        ]

    def _publishable_claims(
        self,
        claims: list[ClaimResult],
        invalidated_claims: list[ClaimResult],
    ) -> list[ClaimResult]:
        invalidated_ids = {claim.claim_id for claim in invalidated_claims}
        return [
            claim
            for claim in claims
            if claim.status in SUPPORTED_STATUSES
            and claim.claim_id not in invalidated_ids
        ]

    def _answer_from_kept_claims(
        self,
        answer: AnswerPayload,
        kept_claims: list[ClaimResult],
    ) -> AnswerPayload:
        kept_texts = self._dedupe([claim.claim_text for claim in kept_claims])
        detailed = "\n".join(
            [REPAIRED_DETAIL_HEADER]
            + [f"- {claim_text}" for claim_text in kept_texts]
        )
        return answer.model_copy(
            update={
                "short_answer": kept_texts[0],
                "detailed_answer": detailed,
                "refusal_reason": None,
                "confidence": 0.0,
            }
        )

    def _temper_answer(self, answer: AnswerPayload) -> AnswerPayload:
        if answer.short_answer.startswith(WEAK_SUPPORT_TEMPERING):
            short_answer = answer.short_answer
        else:
            short_answer = f"{WEAK_SUPPORT_TEMPERING} {answer.short_answer}"
        return answer.model_copy(
            update={
                "short_answer": short_answer,
                "confidence": 0.0,
            }
        )

    def _refusal_result(
        self,
        *,
        answer: AnswerPayload,
        verifier: VerifierStatus,
        warnings: list[str],
        warnings_added: list[str],
        action: str,
        reason: str,
        short_answer: str,
        detailed_answer: str,
        removed_citations: list[Citation],
        kept_claims: list[ClaimResult],
        removed_claims: list[ClaimResult],
        debug: bool,
    ) -> AnswerRepairResult:
        repaired_answer = answer.model_copy(
            update={
                "short_answer": short_answer,
                "detailed_answer": detailed_answer,
                "confidence": 0.0,
                "refusal_reason": reason,
            }
        )
        repaired_verifier = self._updated_verifier(
            verifier,
            verifier_passed=False,
            repair_applied=True,
            refusal_reason=reason,
            warnings_added=warnings_added,
        )
        return AnswerRepairResult(
            answer=repaired_answer,
            citations=[],
            verifier=repaired_verifier,
            warnings=self._dedupe(warnings + warnings_added),
            repair_applied=True,
            refusal_reason=reason,
            debug=self._debug_payload(
                action=action,
                removed_claims=removed_claims,
                kept_claims=kept_claims,
                removed_citations=removed_citations,
                refusal_reason=reason,
                warnings_added=warnings_added,
            )
            if debug
            else None,
        )

    def _updated_verifier(
        self,
        verifier: VerifierStatus,
        *,
        verifier_passed: bool,
        repair_applied: bool,
        refusal_reason: str | None,
        warnings_added: list[str],
    ) -> VerifierStatus:
        return verifier.model_copy(
            update={
                "verifier_passed": verifier_passed,
                "repair_applied": repair_applied,
                "refusal_reason": refusal_reason,
                "warnings": self._dedupe(verifier.warnings + warnings_added),
            }
        )

    def _all_claims_supported(self, claims: list[ClaimResult]) -> bool:
        return bool(claims) and all(
            claim.status in SUPPORTED_STATUSES for claim in claims
        )

    def _debug_payload(
        self,
        *,
        action: str,
        removed_claims: list[ClaimResult],
        kept_claims: list[ClaimResult],
        removed_citations: list[Citation],
        refusal_reason: str | None,
        warnings_added: list[str],
    ) -> dict[str, Any]:
        return {
            "repair_action": action,
            "removed_claims": [claim.claim_text for claim in removed_claims],
            "kept_claims": [claim.claim_text for claim in kept_claims],
            "removed_citation_ids": [
                citation.citation_id for citation in removed_citations
            ],
            "removed_unit_ids": [
                citation.legal_unit_id for citation in removed_citations
            ],
            "refusal_reason": refusal_reason,
            "warnings_added": self._dedupe(warnings_added),
        }

    def _dedupe(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped
