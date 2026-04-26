from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any

from ..schemas import AnswerPayload, Citation, ClaimResult, EvidenceUnit, VerifierStatus

VERIFIER_INSUFFICIENT_EVIDENCE = "verifier_insufficient_evidence"
NO_LEGAL_CLAIMS_DETECTED = "no_legal_claims_detected"
CITATION_UNIT_MISSING = "citation_unit_missing"
CITATION_SNIPPET_NOT_FOUND = "citation_snippet_not_found_in_raw_text"
CITATION_SNIPPET_WEAK_ALIGNMENT = "citation_snippet_weak_alignment"
CITATION_MISSING_FOR_CLAIM = "citation_missing_for_claim"
CITATION_VERIFIER_FAILED = "citation_verifier_failed"
UNSUPPORTED_LEGAL_CLAIMS_DETECTED = "unsupported_legal_claims_detected"
WEAKLY_SUPPORTED_LEGAL_CLAIMS_DETECTED = "weakly_supported_legal_claims_detected"

LEGAL_TERMS = {
    "act aditional",
    "acord",
    "alin",
    "angajator",
    "articol",
    "contract",
    "drept",
    "este interzis",
    "este permis",
    "lege",
    "lit",
    "modificare",
    "munca",
    "obligatie",
    "poate",
    "salariat",
    "salariu",
    "trebuie",
}

CLAIM_TERMS = LEGAL_TERMS | {
    "are voie",
    "nu poate",
    "contractului",
    "contractul",
    "modificarea",
    "salariului",
}

DISCLAIMER_TERMS = (
    "citationverifier",
    "consultanta juridica",
    "corpus",
    "draft",
    "evidencepack",
    "extractive_first",
    "exceptiile sau situatiile speciale",
    "nu a rulat",
    "nu reprezinta",
    "raw_text",
    "status",
    "verificat final",
)

STOPWORDS = {
    "a",
    "acesta",
    "al",
    "ale",
    "ca",
    "cu",
    "de",
    "din",
    "este",
    "in",
    "la",
    "pe",
    "pentru",
    "prin",
    "sau",
    "se",
    "si",
    "un",
    "unei",
    "unui",
}


@dataclass(frozen=True)
class CitationCheck:
    citation_id: str
    unit_id: str
    confidence: float
    valid: bool
    warnings: list[str]


@dataclass(frozen=True)
class SupportScore:
    unit_id: str
    score: float
    breakdown: dict[str, float]


@dataclass(frozen=True)
class CitationVerifierResult:
    verifier: VerifierStatus
    warnings: list[str]
    debug: dict[str, Any] | None
    verified_citation_ids: set[str]


class CitationVerifier:
    def verify(
        self,
        *,
        answer: AnswerPayload,
        citations: list[Citation],
        evidence_units: list[EvidenceUnit],
        debug: bool = False,
    ) -> CitationVerifierResult:
        evidence_by_id = {unit.id: unit for unit in evidence_units}
        citation_checks = [
            self._check_citation(citation, evidence_by_id)
            for citation in citations
        ]
        warnings = self._warnings_from_citation_checks(citation_checks)

        if not evidence_units:
            warnings.append(VERIFIER_INSUFFICIENT_EVIDENCE)
            verifier = self._status(
                claim_results=[],
                citations_checked=len(citations),
                warnings=warnings,
                verifier_passed=False,
            )
            return CitationVerifierResult(
                verifier=verifier,
                warnings=verifier.warnings,
                debug=self._debug_payload(
                    claims=[],
                    claim_rows=[],
                    citation_checks=citation_checks,
                    warnings=verifier.warnings,
                )
                if debug
                else None,
                verified_citation_ids=self._verified_citation_ids(citation_checks),
            )

        claims = self.extract_legal_claims(answer)
        if not claims:
            warnings.append(NO_LEGAL_CLAIMS_DETECTED)

        claim_rows: list[dict[str, Any]] = []
        claim_results = [
            self._verify_claim(
                claim_id=f"claim:{index}",
                claim_text=claim,
                citations=citations,
                evidence_units=evidence_units,
                citation_checks=citation_checks,
                claim_rows=claim_rows,
            )
            for index, claim in enumerate(claims, start=1)
        ]
        warnings.extend(
            warning
            for result in claim_results
            for warning in result.warnings
            if warning == CITATION_MISSING_FOR_CLAIM
        )

        if any(result.status == "unsupported" for result in claim_results):
            warnings.append(UNSUPPORTED_LEGAL_CLAIMS_DETECTED)
        if any(result.status == "weakly_supported" for result in claim_results):
            warnings.append(WEAKLY_SUPPORTED_LEGAL_CLAIMS_DETECTED)

        warnings = self._dedupe(warnings)
        verifier_passed = self._passes(
            claim_results=claim_results,
            evidence_units=evidence_units,
            citation_checks=citation_checks,
            warnings=warnings,
        )
        if not verifier_passed:
            warnings.append(CITATION_VERIFIER_FAILED)
        warnings = self._dedupe(warnings)

        verifier = self._status(
            claim_results=claim_results,
            citations_checked=len(citations),
            warnings=warnings,
            verifier_passed=verifier_passed,
        )
        return CitationVerifierResult(
            verifier=verifier,
            warnings=verifier.warnings,
            debug=self._debug_payload(
                claims=claims,
                claim_rows=claim_rows,
                citation_checks=citation_checks,
                warnings=verifier.warnings,
            )
            if debug
            else None,
            verified_citation_ids=self._verified_citation_ids(citation_checks),
        )

    def extract_legal_claims(self, answer: AnswerPayload) -> list[str]:
        text_parts = [answer.short_answer]
        if answer.detailed_answer:
            text_parts.append(answer.detailed_answer)

        claims: list[str] = []
        for text in text_parts:
            for candidate in self._claim_candidates(text):
                if self._is_legal_claim(candidate):
                    claims.append(candidate)
        return self._dedupe(claims)

    def _claim_candidates(self, text: str) -> list[str]:
        candidates: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            is_bullet = bool(re.match(r"^\s*[-*]\s+", line))
            line = re.sub(r"^\s*[-*]\s+", "", line).strip()
            if is_bullet:
                candidates.append(line)
                continue
            if line.endswith(":") and len(line.split()) <= 5:
                continue
            if line.startswith(("Baza legala", "Conditii", "Context recuperat")):
                continue
            candidates.extend(self._split_sentences(line))
        return candidates

    def _split_sentences(self, line: str) -> list[str]:
        protected = (
            line.replace("art.", "art§")
            .replace("Art.", "Art§")
            .replace("alin.", "alin§")
            .replace("lit.", "lit§")
            .replace("pct.", "pct§")
        )
        parts = re.split(r"(?<=[.!?])\s+(?=[A-ZĂÂÎȘȚ0-9])", protected)
        return [
            part.replace("§", ".").strip()
            for part in parts
            if part.replace("§", ".").strip()
        ]

    def _is_legal_claim(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        if len(normalized.split()) < 4:
            return False
        if any(term in normalized for term in DISCLAIMER_TERMS):
            return False
        return any(term in normalized for term in CLAIM_TERMS)

    def _verify_claim(
        self,
        *,
        claim_id: str,
        claim_text: str,
        citations: list[Citation],
        evidence_units: list[EvidenceUnit],
        citation_checks: list[CitationCheck],
        claim_rows: list[dict[str, Any]],
    ) -> ClaimResult:
        direct_citations = self._direct_citations_for_claim(claim_text, citations)
        claim_warnings: list[str] = []
        if direct_citations:
            candidate_units = [
                unit
                for citation in direct_citations
                for unit in evidence_units
                if unit.id == citation.legal_unit_id
            ]
        else:
            candidate_units = evidence_units
            claim_warnings.append(CITATION_MISSING_FOR_CLAIM)

        support_rows = [
            self._support_score(
                claim_text,
                unit,
                citation_confidence=self._citation_confidence_for_unit(
                    unit.id,
                    direct_citations or citations,
                    citation_checks,
                ),
            )
            for unit in candidate_units
        ]
        best = max(
            support_rows,
            key=lambda row: (row.score, row.breakdown["citation_confidence"], row.unit_id),
            default=SupportScore(unit_id="", score=0.0, breakdown={}),
        )
        status = self._support_status(best.score)
        if status in {"weakly_supported", "unsupported"}:
            claim_warnings.append(f"claim_{status}")

        citation_ids = [
            citation.citation_id
            for citation in direct_citations
            if citation.legal_unit_id == best.unit_id
        ]
        if not citation_ids and direct_citations:
            citation_ids = [citation.citation_id for citation in direct_citations]

        claim_rows.append(
            {
                "claim_id": claim_id,
                "claim_text": claim_text,
                "support_status": status,
                "support_score": best.score,
                "best_supporting_unit_id": best.unit_id or None,
                "supporting_unit_ids": [best.unit_id] if best.unit_id else [],
                "citation_ids": citation_ids,
                "warnings": self._dedupe(claim_warnings),
                "score_breakdown": best.breakdown,
            }
        )
        return ClaimResult(
            claim_id=claim_id,
            claim_text=claim_text,
            status=status,
            citation_ids=citation_ids,
            confidence=best.score,
            warning=";".join(self._dedupe(claim_warnings)) or None,
            support_score=best.score,
            supporting_unit_ids=[best.unit_id] if best.unit_id else [],
            warnings=self._dedupe(claim_warnings),
            score_breakdown=best.breakdown,
        )

    def _support_score(
        self,
        claim_text: str,
        unit: EvidenceUnit,
        *,
        citation_confidence: float,
    ) -> SupportScore:
        lexical = self._lexical_overlap(claim_text, unit.raw_text)
        concept = self._concept_overlap(claim_text, unit)
        lexical_semantic = self._lexical_semantic_fallback(claim_text, unit.raw_text)
        leading_raw_text_alignment = self._leading_raw_text_alignment(
            claim_text,
            unit.raw_text,
        )
        contract_modification_alignment = self._contract_modification_alignment(
            claim_text,
            unit.raw_text,
        )
        score = round(
            0.40 * lexical
            + 0.25 * concept
            + 0.25 * lexical_semantic
            + 0.10 * citation_confidence,
            6,
        )
        if citation_confidence >= 0.8 and leading_raw_text_alignment >= 0.95:
            score = max(score, 0.8)
        if citation_confidence >= 0.8 and contract_modification_alignment >= 0.95:
            score = max(score, 0.8)
        return SupportScore(
            unit_id=unit.id,
            score=score,
            breakdown={
                "lexical_overlap": round(lexical, 6),
                "concept_overlap": round(concept, 6),
                "lexical_semantic_similarity_fallback": round(lexical_semantic, 6),
                "citation_confidence": round(citation_confidence, 6),
                "leading_raw_text_alignment": round(leading_raw_text_alignment, 6),
                "contract_modification_alignment": round(
                    contract_modification_alignment,
                    6,
                ),
            },
        )

    def _lexical_overlap(self, claim_text: str, raw_text: str) -> float:
        claim_tokens = self._tokens(claim_text)
        raw_tokens = self._tokens(raw_text)
        if not claim_tokens or not raw_tokens:
            return 0.0
        return len(claim_tokens & raw_tokens) / len(claim_tokens)

    def _concept_overlap(self, claim_text: str, unit: EvidenceUnit) -> float:
        claim_tokens = self._tokens(claim_text)
        if unit.legal_concepts:
            concept_tokens = {
                token
                for concept in unit.legal_concepts
                for token in self._tokens(str(concept))
            }
            if not concept_tokens:
                return 0.0
            return len(concept_tokens & claim_tokens) / len(concept_tokens)

        claim_terms = self._legal_terms(claim_text)
        raw_terms = self._legal_terms(unit.raw_text)
        if not claim_terms:
            return 0.0
        return len(claim_terms & raw_terms) / len(claim_terms)

    def _lexical_semantic_fallback(self, claim_text: str, raw_text: str) -> float:
        claim_tokens = self._tokens(claim_text)
        raw_tokens = self._tokens(raw_text)
        token_jaccard = self._jaccard(claim_tokens, raw_tokens)
        char_jaccard = self._jaccard(
            self._char_ngrams(self._normalize_text(claim_text)),
            self._char_ngrams(self._normalize_text(raw_text)),
        )
        return 0.65 * token_jaccard + 0.35 * char_jaccard

    def _check_citation(
        self,
        citation: Citation,
        evidence_by_id: dict[str, EvidenceUnit],
    ) -> CitationCheck:
        unit = evidence_by_id.get(citation.legal_unit_id)
        if unit is None:
            return CitationCheck(
                citation_id=citation.citation_id,
                unit_id=citation.legal_unit_id,
                confidence=0.0,
                valid=False,
                warnings=[CITATION_UNIT_MISSING],
            )

        confidence = self._snippet_confidence(citation.quote, unit.raw_text)
        warnings: list[str] = []
        if confidence < 0.8:
            warnings.append(CITATION_SNIPPET_NOT_FOUND)
        elif confidence < 1.0:
            warnings.append(CITATION_SNIPPET_WEAK_ALIGNMENT)
        return CitationCheck(
            citation_id=citation.citation_id,
            unit_id=citation.legal_unit_id,
            confidence=confidence,
            valid=confidence >= 0.8,
            warnings=warnings,
        )

    def _snippet_confidence(self, snippet: str, raw_text: str) -> float:
        normalized_snippet = self._normalize_text(snippet)
        normalized_raw = self._normalize_text(raw_text)
        if not normalized_snippet:
            return 0.0
        if normalized_snippet in normalized_raw:
            return 1.0
        snippet_tokens = self._tokens(snippet)
        raw_tokens = self._tokens(raw_text)
        if not snippet_tokens:
            return 0.0
        recall = len(snippet_tokens & raw_tokens) / len(snippet_tokens)
        if recall >= 0.75:
            return 0.8
        return 0.4

    def _leading_raw_text_alignment(self, claim_text: str, raw_text: str) -> float:
        normalized_claim = self._normalize_text(claim_text)
        normalized_raw = self._normalize_text(raw_text)
        if normalized_raw and normalized_claim.startswith(normalized_raw):
            return 1.0
        for line in raw_text.splitlines():
            normalized_line = self._normalize_text(line.strip())
            if normalized_line and normalized_claim.startswith(normalized_line):
                return 1.0
        return 0.0

    def _contract_modification_alignment(self, claim_text: str, raw_text: str) -> float:
        normalized_claim = self._normalize_text(claim_text)
        normalized_raw = self._normalize_text(raw_text)
        if self._has_contract_agreement_rule(normalized_claim) and self._has_contract_agreement_rule(normalized_raw):
            return 1.0
        if self._has_contract_modification_scope(normalized_claim) and self._has_contract_modification_scope(normalized_raw):
            return 1.0
        return 0.0

    def _has_contract_agreement_rule(self, normalized: str) -> bool:
        has_contract = (
            "contractul individual de munca" in normalized
            or "contract individual de munca" in normalized
        )
        has_modified = "modificat" in normalized or "modificare" in normalized
        has_agreement = "acordul partilor" in normalized
        return has_contract and has_modified and has_agreement

    def _has_contract_modification_scope(self, normalized: str) -> bool:
        has_scope = (
            "modificarea contractului individual de munca" in normalized
            or "modificare contract individual munca" in normalized
        )
        has_element = "element" in normalized or "salariu" in normalized or "salariul" in normalized
        return has_scope and has_element

    def _direct_citations_for_claim(
        self,
        claim_text: str,
        citations: list[Citation],
    ) -> list[Citation]:
        normalized_claim = self._normalize_text(claim_text)
        direct: list[Citation] = []
        for citation in citations:
            if self._normalize_text(citation.label) in normalized_claim:
                direct.append(citation)
                continue
            if self._normalize_text(citation.quote) in normalized_claim:
                direct.append(citation)
                continue
            quote_tokens = self._tokens(citation.quote)
            if quote_tokens and len(quote_tokens & self._tokens(claim_text)) / len(quote_tokens) >= 0.75:
                direct.append(citation)
        return direct

    def _citation_confidence_for_unit(
        self,
        unit_id: str,
        citations: list[Citation],
        citation_checks: list[CitationCheck],
    ) -> float:
        citation_ids = {
            citation.citation_id
            for citation in citations
            if citation.legal_unit_id == unit_id
        }
        confidences = [
            check.confidence
            for check in citation_checks
            if check.citation_id in citation_ids
        ]
        if not confidences:
            return 0.0
        return max(confidences)

    def _support_status(self, support_score: float) -> str:
        if support_score >= 0.75:
            return "strongly_supported"
        if support_score >= 0.60:
            return "supported"
        if support_score >= 0.45:
            return "weakly_supported"
        return "unsupported"

    def _passes(
        self,
        *,
        claim_results: list[ClaimResult],
        evidence_units: list[EvidenceUnit],
        citation_checks: list[CitationCheck],
        warnings: list[str],
    ) -> bool:
        if not evidence_units or not claim_results:
            return False
        if any(not check.valid for check in citation_checks):
            return False
        if any(result.status in {"weakly_supported", "unsupported", "not_checked"} for result in claim_results):
            return False
        if warnings:
            blocking = {
                VERIFIER_INSUFFICIENT_EVIDENCE,
                NO_LEGAL_CLAIMS_DETECTED,
                CITATION_UNIT_MISSING,
                CITATION_SNIPPET_NOT_FOUND,
                CITATION_MISSING_FOR_CLAIM,
                UNSUPPORTED_LEGAL_CLAIMS_DETECTED,
                WEAKLY_SUPPORTED_LEGAL_CLAIMS_DETECTED,
            }
            if any(warning in blocking for warning in warnings):
                return False
        return True

    def _status(
        self,
        *,
        claim_results: list[ClaimResult],
        citations_checked: int,
        warnings: list[str],
        verifier_passed: bool,
    ) -> VerifierStatus:
        supported = [
            result
            for result in claim_results
            if result.status in {"strongly_supported", "supported"}
        ]
        weak = [
            result for result in claim_results if result.status == "weakly_supported"
        ]
        unsupported = [
            result for result in claim_results if result.status == "unsupported"
        ]
        groundedness = (
            len(supported) / len(claim_results)
            if claim_results
            else 0.0
        )
        return VerifierStatus(
            groundedness_score=round(groundedness, 6),
            claims_total=len(claim_results),
            claims_supported=len(supported),
            claims_weakly_supported=len(weak),
            claims_unsupported=len(unsupported),
            citations_checked=citations_checked,
            verifier_passed=verifier_passed,
            claim_results=claim_results,
            warnings=self._dedupe(warnings),
            repair_applied=False,
            refusal_reason=None,
        )

    def _debug_payload(
        self,
        *,
        claims: list[str],
        claim_rows: list[dict[str, Any]],
        citation_checks: list[CitationCheck],
        warnings: list[str],
    ) -> dict[str, Any]:
        return {
            "claim_extraction": {
                "claims_total": len(claims),
                "claims": claims,
                "ignored_disclaimer_terms": list(DISCLAIMER_TERMS),
            },
            "claims": claim_rows,
            "citation_checks": [
                {
                    "citation_id": check.citation_id,
                    "unit_id": check.unit_id,
                    "confidence": check.confidence,
                    "valid": check.valid,
                    "warnings": check.warnings,
                }
                for check in citation_checks
            ],
            "failed_citations": [
                check.citation_id for check in citation_checks if not check.valid
            ],
            "warnings": warnings,
            "scoring_formula": {
                "lexical_overlap": 0.40,
                "concept_overlap": 0.25,
                "lexical_semantic_similarity_fallback": 0.25,
                "citation_confidence": 0.10,
            },
        }

    def _warnings_from_citation_checks(
        self,
        citation_checks: list[CitationCheck],
    ) -> list[str]:
        return self._dedupe(
            [
                warning
                for check in citation_checks
                for warning in check.warnings
            ]
        )

    def _verified_citation_ids(
        self,
        citation_checks: list[CitationCheck],
    ) -> set[str]:
        return {check.citation_id for check in citation_checks if check.valid}

    def _legal_terms(self, text: str) -> set[str]:
        normalized = self._normalize_text(text)
        return {term for term in LEGAL_TERMS if term in normalized}

    def _tokens(self, text: str) -> set[str]:
        normalized = self._normalize_text(text)
        return {
            token
            for token in re.split(r"[^a-z0-9_]+", normalized)
            if len(token) > 1 and token not in STOPWORDS
        }

    def _char_ngrams(self, text: str, size: int = 3) -> set[str]:
        compact = re.sub(r"\s+", " ", text.strip())
        if len(compact) < size:
            return {compact} if compact else set()
        return {compact[index : index + size] for index in range(len(compact) - size + 1)}

    def _jaccard(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    def _normalize_text(self, text: str) -> str:
        replacements = {
            "Ãƒâ€žÃ†â€™": "a",
            "Ãƒâ€žÃ¢â‚¬Å¡": "a",
            "ÃƒË†Ã¢â€žÂ¢": "s",
            "ÃƒË†Ã‹Å“": "s",
            "ÃƒË†Ã¢â‚¬Âº": "t",
            "ÃƒË†Ã…Â¡": "t",
            "ÃƒÆ’Ã‚Â¢": "a",
            "ÃƒÆ’Ã‚Â®": "i",
            "Ãƒâ€¦Ã…Â¸": "s",
            "Ãƒâ€¦Ã‚Â£": "t",
        }
        for broken, fixed in replacements.items():
            text = text.replace(broken, fixed)
        normalized = unicodedata.normalize("NFD", text.casefold())
        stripped = "".join(
            char for char in normalized if unicodedata.category(char) != "Mn"
        )
        return " ".join(stripped.replace("-", "_").split())

    def _dedupe(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped
