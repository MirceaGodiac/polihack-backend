from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, Field

from .query_frame import QueryFrame

EVIDENCE_ROLE = Literal[
    "direct_basis",
    "condition",
    "exception",
    "definition",
    "procedure",
    "sanction",
    "context",
]

LABOR_CONTRACT_MODIFICATION_INTENT = "labor_contract_modification"

LABOR_CONTRACT_MODIFICATION_NEGATIVE_CONTEXTS = [
    "formare profesionala",
    "durata formarii profesionale",
    "semnatura electronica",
    "delegarea",
    "detasarea",
    "recuperarea pagubei",
    "recuperarea contravalorii pagubei",
    "nota de constatare",
    "clauza de neconcurenta",
    "salariat temporar",
    "salariul minim",
    "evidenta orelor",
    "drepturile si obligatiile partilor",
]


class EvidenceRequirement(BaseModel):
    id: str
    intent_id: str
    role: EVIDENCE_ROLE
    required_for_answer: bool = False
    positive_phrases: list[str] = Field(default_factory=list)
    required_any_terms: list[str] = Field(default_factory=list)
    required_all_groups: list[list[str]] = Field(default_factory=list)
    target_terms: list[str] = Field(default_factory=list)
    qualifier_terms: list[str] = Field(default_factory=list)
    negative_contexts: list[str] = Field(default_factory=list)
    disqualifying_contexts: list[str] = Field(default_factory=list)


class EvidenceRequirementMatch(BaseModel):
    requirement_id: str
    matched: bool
    score: float
    positive_hits: list[str] = Field(default_factory=list)
    negative_hits: list[str] = Field(default_factory=list)
    missing_groups: list[list[str]] = Field(default_factory=list)


class CandidateRoleDecision(BaseModel):
    unit_id: str
    support_role: EVIDENCE_ROLE
    role_confidence: float
    requirement_matches: list[EvidenceRequirementMatch] = Field(default_factory=list)
    matched_requirement_ids: list[str] = Field(default_factory=list)
    disqualified_requirement_ids: list[str] = Field(default_factory=list)
    why_role: list[str] = Field(default_factory=list)


class LegalIssueFrame(BaseModel):
    intent_id: str
    domain: str | None = None
    meta_intents: list[str] = Field(default_factory=list)
    required_evidence: list[EvidenceRequirement] = Field(default_factory=list)


class LegalIssueFrameBuilder:
    def build(self, query_frame: QueryFrame | None) -> LegalIssueFrame | None:
        if query_frame is None:
            return None
        if LABOR_CONTRACT_MODIFICATION_INTENT not in query_frame.intents:
            return None
        return LegalIssueFrame(
            intent_id=LABOR_CONTRACT_MODIFICATION_INTENT,
            domain=query_frame.domain,
            meta_intents=query_frame.meta_intents,
            required_evidence=self._labor_contract_modification_requirements(),
        )

    def _labor_contract_modification_requirements(self) -> list[EvidenceRequirement]:
        negative_contexts = list(LABOR_CONTRACT_MODIFICATION_NEGATIVE_CONTEXTS)
        return [
            EvidenceRequirement(
                id="contract_modification_agreement_rule",
                intent_id=LABOR_CONTRACT_MODIFICATION_INTENT,
                role="direct_basis",
                required_for_answer=True,
                positive_phrases=[
                    "contractul individual de munca poate fi modificat",
                    "contract individual de munca poate fi modificat",
                    "modificat numai prin acordul partilor",
                    "numai prin acordul partilor",
                ],
                required_all_groups=[
                    ["contract", "individual", "munca"],
                    ["modificat", "modificarea", "modificare"],
                    ["acord", "partilor", "parti"],
                ],
                negative_contexts=negative_contexts,
                disqualifying_contexts=negative_contexts,
            ),
            EvidenceRequirement(
                id="contract_modification_salary_scope",
                intent_id=LABOR_CONTRACT_MODIFICATION_INTENT,
                role="direct_basis",
                required_for_answer=True,
                positive_phrases=[
                    "modificarea contractului individual de munca",
                    "elementele contractului",
                    "salariul",
                ],
                required_all_groups=[
                    ["modificarea", "modificare", "modificat"],
                    ["contract", "individual", "munca"],
                    ["salariu", "salariul", "salarizare"],
                ],
                target_terms=["salariu", "salariul", "salarizare"],
                negative_contexts=negative_contexts,
                disqualifying_contexts=negative_contexts,
            ),
            EvidenceRequirement(
                id="salary_target_element",
                intent_id=LABOR_CONTRACT_MODIFICATION_INTENT,
                role="condition",
                required_for_answer=False,
                positive_phrases=["salariul"],
                required_any_terms=["salariu", "salariul", "salarizare"],
                target_terms=["salariu", "salariul", "salarizare"],
                negative_contexts=negative_contexts,
                disqualifying_contexts=negative_contexts,
            ),
        ]


class CandidateRoleClassifier:
    def __init__(self, issue_frame_builder: LegalIssueFrameBuilder | None = None) -> None:
        self.issue_frame_builder = issue_frame_builder or LegalIssueFrameBuilder()

    def classify(
        self,
        *,
        query_frame: QueryFrame | None,
        unit_id: str,
        unit: dict[str, Any],
        ranked_score_breakdown: dict[str, Any] | None = None,
        existing_why_ranked: list[str] | None = None,
    ) -> CandidateRoleDecision:
        issue_frame = self.issue_frame_builder.build(query_frame)
        if issue_frame is None or (query_frame and query_frame.confidence < 0.35):
            return CandidateRoleDecision(
                unit_id=unit_id,
                support_role="context",
                role_confidence=0.0,
                why_role=["role_classifier:fallback_unavailable"],
            )
        if issue_frame.intent_id == LABOR_CONTRACT_MODIFICATION_INTENT:
            return self._classify_labor_contract_modification(
                issue_frame=issue_frame,
                unit_id=unit_id,
                unit=unit,
                ranked_score_breakdown=ranked_score_breakdown or {},
                existing_why_ranked=existing_why_ranked or [],
            )
        return CandidateRoleDecision(
            unit_id=unit_id,
            support_role="context",
            role_confidence=0.0,
            why_role=["role_classifier:unsupported_intent"],
        )

    def _classify_labor_contract_modification(
        self,
        *,
        issue_frame: LegalIssueFrame,
        unit_id: str,
        unit: dict[str, Any],
        ranked_score_breakdown: dict[str, Any],
        existing_why_ranked: list[str],
    ) -> CandidateRoleDecision:
        haystack = self._candidate_haystack(
            unit_id=unit_id,
            unit=unit,
            score_breakdown=ranked_score_breakdown,
            why_ranked=existing_why_ranked,
        )
        requirement_matches = [
            self._match_requirement(requirement, haystack)
            for requirement in issue_frame.required_evidence
        ]
        requirement_matches = self._apply_labor_contract_parent_scope_match(
            requirement_matches,
            haystack=haystack,
            unit=unit,
            score_breakdown=ranked_score_breakdown,
            why_ranked=existing_why_ranked,
        )
        requirement_matches = self._apply_salary_target_atomic_match(
            requirement_matches,
            unit=unit,
        )
        matches_by_id = {
            match.requirement_id: match for match in requirement_matches
        }
        matched_requirement_ids = [
            match.requirement_id for match in requirement_matches if match.matched
        ]
        disqualified_requirement_ids = [
            match.requirement_id
            for match in requirement_matches
            if match.negative_hits
        ]
        negative_hits = self._dedupe(
            [
                hit
                for match in requirement_matches
                for hit in match.negative_hits
            ]
        )

        why_role: list[str] = []
        for requirement_id in matched_requirement_ids:
            why_role.append(f"requirement_match:{requirement_id}")
        for hit in negative_hits:
            why_role.append(f"disqualified_by:{hit}")

        support_role: EVIDENCE_ROLE = "context"
        confidence = 0.35

        agreement = matches_by_id["contract_modification_agreement_rule"]
        salary_scope = matches_by_id["contract_modification_salary_scope"]
        salary_target = matches_by_id["salary_target_element"]
        salary_scope_parent_context = self._is_salary_scope_parent_context(
            haystack=haystack,
            unit=unit,
            score_breakdown=ranked_score_breakdown,
            why_ranked=existing_why_ranked,
        )

        if agreement.matched and agreement.score >= 0.70:
            support_role = "direct_basis"
            confidence = agreement.score
        elif salary_scope.matched and salary_scope.score >= 0.70:
            support_role = "condition" if salary_scope_parent_context else "direct_basis"
            confidence = salary_scope.score
        elif salary_target.matched:
            support_role = "condition"
            confidence = min(0.65, salary_target.score)
        elif self._is_delegation_exception(haystack):
            support_role = "exception"
            confidence = 0.65
        elif negative_hits:
            support_role = "context"
            confidence = 0.55

        why_role.append(f"role_classifier:{support_role}")
        return CandidateRoleDecision(
            unit_id=unit_id,
            support_role=support_role,
            role_confidence=self._clamp(confidence),
            requirement_matches=requirement_matches,
            matched_requirement_ids=matched_requirement_ids,
            disqualified_requirement_ids=self._dedupe(disqualified_requirement_ids),
            why_role=self._dedupe(why_role),
        )

    def _apply_labor_contract_parent_scope_match(
        self,
        matches: list[EvidenceRequirementMatch],
        *,
        haystack: str,
        unit: dict[str, Any],
        score_breakdown: dict[str, Any],
        why_ranked: list[str],
    ) -> list[EvidenceRequirementMatch]:
        if not self._is_salary_scope_parent_context(
            haystack=haystack,
            unit=unit,
            score_breakdown=score_breakdown,
            why_ranked=why_ranked,
        ):
            return matches

        updated: list[EvidenceRequirementMatch] = []
        for match in matches:
            if (
                match.requirement_id != "contract_modification_salary_scope"
                or match.matched
            ):
                updated.append(match)
                continue
            updated.append(
                match.model_copy(
                    update={
                        "matched": True,
                        "score": max(match.score, 0.75),
                        "positive_hits": self._dedupe(
                            [
                                *match.positive_hits,
                                "modificarea contractului individual de munca",
                                "intent_governing_rule_parent_context",
                            ]
                        ),
                        "missing_groups": [],
                    }
                )
            )
        return updated

    def _apply_salary_target_atomic_match(
        self,
        matches: list[EvidenceRequirementMatch],
        *,
        unit: dict[str, Any],
    ) -> list[EvidenceRequirementMatch]:
        updated: list[EvidenceRequirementMatch] = []
        for match in matches:
            if (
                match.requirement_id != "salary_target_element"
                or not match.matched
                or self._is_atomic_salary_target(unit)
            ):
                updated.append(match)
                continue
            updated.append(
                match.model_copy(
                    update={
                        "matched": False,
                        "score": min(match.score, 0.35),
                        "positive_hits": [],
                        "missing_groups": [["atomic_salary_child"]],
                    }
                )
            )
        return updated

    def _is_salary_scope_parent_context(
        self,
        *,
        haystack: str,
        unit: dict[str, Any],
        score_breakdown: dict[str, Any],
        why_ranked: list[str],
    ) -> bool:
        if not self._is_contract_modification_scope_parent(haystack):
            return False
        if self._has_salary_target(haystack):
            return False
        if bool(unit.get("intent_governing_rule_parent_context")):
            return True
        if self._float(score_breakdown.get("intent_governing_rule_parent")) > 0.0:
            return True
        return any(
            "intent_governing_rule_parent_context:labor_contract_modification" in reason
            for reason in why_ranked
        )

    def _is_contract_modification_scope_parent(self, haystack: str) -> bool:
        has_contract_scope = self._phrase_present(
            "modificarea contractului individual de munca",
            haystack,
        ) or self._phrase_present(
            "modificare contract individual munca",
            haystack,
        )
        has_scope_marker = any(
            self._phrase_present(marker, haystack)
            for marker in (
                "elemente",
                "elementele",
                "urmatoarele elemente",
                "poate privi",
                "se refera",
                "modificarea",
                "modificare",
            )
        )
        return has_contract_scope and has_scope_marker

    def _has_salary_target(self, haystack: str) -> bool:
        return any(
            self._phrase_present(term, haystack)
            for term in ("salariu", "salariul", "salarizare")
        )

    def _is_atomic_salary_target(self, unit: dict[str, Any]) -> bool:
        unit_type = normalize_legal_issue_text(str(unit.get("type") or ""))
        if not (
            unit.get("letter_number")
            or unit.get("point_number")
            or unit_type in {"litera", "letter", "punct", "point"}
        ):
            return False
        raw_text = str(unit.get("raw_text") or unit.get("text") or "")
        tokens = re.findall(r"[a-z0-9]+", normalize_legal_issue_text(raw_text))
        if not tokens:
            return False
        salary_tokens = {"salariu", "salariul", "salarizare"}
        if not salary_tokens.intersection(tokens):
            return False
        allowed_markers = {
            normalize_legal_issue_text(str(unit.get("letter_number") or "")),
            normalize_legal_issue_text(str(unit.get("point_number") or "")),
        }
        allowed = {*salary_tokens, *allowed_markers}
        return all(
            token in allowed or len(token) <= 2
            for token in tokens
        )

    def _float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _match_requirement(
        self,
        requirement: EvidenceRequirement,
        haystack: str,
    ) -> EvidenceRequirementMatch:
        positive_hits = [
            phrase
            for phrase in requirement.positive_phrases
            if self._phrase_present(phrase, haystack)
        ]
        if requirement.required_any_terms:
            positive_hits.extend(
                term
                for term in requirement.required_any_terms
                if self._phrase_present(term, haystack)
            )
        missing_groups = [
            group
            for group in requirement.required_all_groups
            if not any(self._phrase_present(term, haystack) for term in group)
        ]
        negative_hits = [
            context
            for context in self._dedupe(
                requirement.negative_contexts + requirement.disqualifying_contexts
            )
            if self._phrase_present(context, haystack)
        ]

        group_count = len(requirement.required_all_groups)
        group_score = 1.0
        if group_count:
            group_score = (group_count - len(missing_groups)) / group_count

        phrase_score = 0.0
        if requirement.positive_phrases:
            phrase_score = min(1.0, len(positive_hits) / len(requirement.positive_phrases))
        elif requirement.required_any_terms:
            phrase_score = 1.0 if positive_hits else 0.0

        if group_count:
            score = 0.55 * group_score + 0.45 * phrase_score
        else:
            score = phrase_score

        if positive_hits and not missing_groups:
            score = max(score, 0.80)
        if negative_hits:
            score = min(score, 0.35)

        matched = bool(positive_hits) and not missing_groups and not negative_hits
        return EvidenceRequirementMatch(
            requirement_id=requirement.id,
            matched=matched,
            score=self._clamp(score),
            positive_hits=self._dedupe(positive_hits),
            negative_hits=self._dedupe(negative_hits),
            missing_groups=missing_groups,
        )

    def _candidate_haystack(
        self,
        *,
        unit_id: str,
        unit: dict[str, Any],
        score_breakdown: dict[str, Any],
        why_ranked: list[str],
    ) -> str:
        values: list[str] = [unit_id]
        values.extend(why_ranked)
        values.append(str(score_breakdown))
        for key in (
            "raw_text",
            "normalized_text",
            "text",
            "label",
            "title",
            "legal_concepts",
            "article_number",
            "paragraph_number",
            "letter_number",
        ):
            value = unit.get(key)
            if value:
                values.append(str(value))
        return normalize_legal_issue_text(" ".join(values))

    def _is_delegation_exception(self, haystack: str) -> bool:
        return (
            self._phrase_present("locul muncii poate fi modificat unilateral", haystack)
            or self._phrase_present("delegarea", haystack)
            or self._phrase_present("detasarea", haystack)
        )

    def _phrase_present(self, phrase: str, haystack: str) -> bool:
        normalized = normalize_legal_issue_text(phrase)
        if not normalized:
            return False
        if " " in normalized or "_" in normalized:
            return normalized in haystack
        tokens = set(haystack.split())
        if normalized in tokens:
            return True
        if len(normalized) >= 6:
            return any(
                token.startswith(normalized[:6]) or normalized.startswith(token[:6])
                for token in tokens
                if len(token) >= 6
            )
        return False

    def _dedupe(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


def normalize_legal_issue_text(text: str) -> str:
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
    stripped = re.sub(r"\badi[?\ufffd]ional(a?)\b", r"aditional\1", stripped)
    return " ".join(stripped.replace(".", " ").replace("-", "_").split())
