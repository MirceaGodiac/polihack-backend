from __future__ import annotations

import re
import unicodedata

from ..schemas import (
    DraftAnswer,
    DraftCitation,
    EvidenceUnit,
    GenerationConstraints,
)

GENERATION_MODE_EXTRACTIVE_V1 = "deterministic_extractive_v1"
GENERATION_MODE_INSUFFICIENT_EVIDENCE = "deterministic_extractive_v1_insufficient_evidence"
GENERATION_INSUFFICIENT_EVIDENCE = "generation_insufficient_evidence"
GENERATION_LIMITED_EVIDENCE = "generation_limited_evidence"
GENERATION_MISSING_CITABLE_RAW_TEXT = "generation_missing_citable_raw_text"
GENERATION_MISSING_CITATION_METADATA = "generation_missing_citation_metadata"
GENERATION_NO_DIRECT_BASIS = "generation_no_direct_basis"
GENERATION_FAILED = "generation_failed"
GENERATION_UNVERIFIED_WARNING = (
    "generation_unverified_citation_verifier_not_run: draft answer generated "
    "over EvidencePack; real CitationVerifier has not run yet."
)
ANSWER_INTENT_LABOR_CONTRACT_MODIFICATION = "labor_contract_modification"

INSUFFICIENT_EVIDENCE_ANSWER = (
    "Nu pot formula un r\u0103spuns juridic sus\u021binut pe baza "
    "corpusului disponibil."
)

ROLE_PRIORITY = {
    "direct_basis": 0,
    "condition": 1,
    "exception": 2,
    "definition": 3,
    "context": 4,
    "procedure": 5,
    "sanction": 6,
}

STOPWORDS = {
    "a",
    "al",
    "ale",
    "alin",
    "art",
    "cu",
    "de",
    "din",
    "este",
    "fara",
    "in",
    "la",
    "mi",
    "nu",
    "pe",
    "poate",
    "sa",
    "sau",
    "se",
    "si",
}

LABOR_CONTRACT_MODIFICATION_TRIGGERS = (
    "act aditional",
    "fara act aditional",
    "scada salariul",
    "scade salariul",
    "modificare contract",
    "modificarea contractului",
)

LABOR_CONTRACT_REDUCTION_TERMS = ("scada", "scade", "reduca", "reducere", "diminuare")
LABOR_CONTRACT_TARGET_TERMS = ("salariu", "salariul", "salarizare")
LABOR_CONTRACT_ACTOR_TERMS = ("angajator", "angajatorul", "salariat", "salariatul")
LABOR_CONTRACT_DISTRACTOR_TERMS = (
    "remuneratie restanta",
    "remuneratia restanta",
    "persoane angajate ilegal",
    "persoana angajata ilegal",
    "neplata salariului",
    "intarzierea platii salariului",
    "intarziere la plata salariului",
    "salariul minim",
    "salariul de baza minim",
    "confidentialitatea salariului",
    "salariul este confidential",
    "registrul salariatilor",
    "registrul general de evidenta",
    "munca nedeclarata",
)


class GenerationAdapter:
    def generate(
        self,
        *,
        question: str,
        evidence_units: list[EvidenceUnit],
        constraints: GenerationConstraints | None = None,
    ) -> DraftAnswer:
        constraints = constraints or GenerationConstraints()
        selected, warnings = self._select_evidence(evidence_units, constraints)
        answer_intent = self._detect_answer_intent(question, selected)
        focused_contract_modification = (
            answer_intent == ANSWER_INTENT_LABOR_CONTRACT_MODIFICATION
        )
        if focused_contract_modification:
            focused = self._select_focused_answer_evidence(question, selected)
            if focused:
                selected = focused
        if not selected:
            return DraftAnswer(
                short_answer=INSUFFICIENT_EVIDENCE_ANSWER,
                detailed_answer=None,
                citations=[],
                used_evidence_unit_ids=[],
                generation_mode=GENERATION_MODE_INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                warnings=self._dedupe(
                    [
                        *warnings,
                        GENERATION_INSUFFICIENT_EVIDENCE,
                        GENERATION_UNVERIFIED_WARNING,
                    ]
                ),
            )

        citations = [
            self._draft_citation(
                unit,
                question=question,
                warnings=warnings,
                focused_contract_modification=focused_contract_modification,
            )
            for unit in selected
        ]
        citations = [citation for citation in citations if citation is not None]
        if not citations:
            return DraftAnswer(
                short_answer=INSUFFICIENT_EVIDENCE_ANSWER,
                detailed_answer=None,
                citations=[],
                used_evidence_unit_ids=[],
                generation_mode=GENERATION_MODE_INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                warnings=self._dedupe(
                    [
                        *warnings,
                        GENERATION_INSUFFICIENT_EVIDENCE,
                        GENERATION_MISSING_CITABLE_RAW_TEXT,
                        GENERATION_UNVERIFIED_WARNING,
                    ]
                ),
            )

        if len(citations) < 2:
            warnings.append(GENERATION_LIMITED_EVIDENCE)
        if not any(unit.support_role == "direct_basis" for unit in selected):
            warnings.append(GENERATION_NO_DIRECT_BASIS)
        warnings.append(GENERATION_UNVERIFIED_WARNING)

        return DraftAnswer(
            short_answer=self._short_answer(
                citations,
                question=question,
                selected=selected,
            ),
            detailed_answer=self._detailed_answer(selected, citations),
            citations=citations,
            used_evidence_unit_ids=[citation.unit_id for citation in citations],
            generation_mode=GENERATION_MODE_EXTRACTIVE_V1,
            confidence=0.0,
            warnings=self._dedupe(warnings),
        )

    def _select_evidence(
        self,
        evidence_units: list[EvidenceUnit],
        constraints: GenerationConstraints,
    ) -> tuple[list[EvidenceUnit], list[str]]:
        warnings: list[str] = []
        citable_units: list[EvidenceUnit] = []
        seen: set[str] = set()
        for unit in evidence_units:
            if unit.id in seen:
                continue
            seen.add(unit.id)
            if not unit.raw_text.strip():
                warnings.append(GENERATION_MISSING_CITABLE_RAW_TEXT)
                continue
            citable_units.append(unit)

        selected = sorted(
            citable_units,
            key=lambda unit: (
                ROLE_PRIORITY.get(unit.support_role, 99),
                unit.rank,
                -unit.rerank_score,
                unit.id,
            ),
        )[: constraints.max_evidence_units]
        return selected, warnings

    def _draft_citation(
        self,
        unit: EvidenceUnit,
        *,
        question: str,
        warnings: list[str],
        focused_contract_modification: bool = False,
    ) -> DraftCitation | None:
        raw_text = unit.raw_text.strip()
        if not raw_text:
            warnings.append(GENERATION_MISSING_CITABLE_RAW_TEXT)
            return None

        label = self._label(unit)
        if label == unit.id:
            warnings.append(GENERATION_MISSING_CITATION_METADATA)
        snippet = (
            self._focused_contract_modification_snippet(unit)
            if focused_contract_modification
            else self._best_snippet(raw_text, question)
        )
        if not snippet:
            warnings.append(GENERATION_MISSING_CITABLE_RAW_TEXT)
            return None
        return DraftCitation(
            unit_id=unit.id,
            label=label,
            snippet=snippet,
            source_url=unit.source_url,
            support_score=unit.rerank_score,
        )

    def _short_answer(
        self,
        citations: list[DraftCitation],
        *,
        question: str,
        selected: list[EvidenceUnit] | None = None,
    ) -> str:
        if (
            selected
            and self._detect_answer_intent(question, selected)
            == ANSWER_INTENT_LABOR_CONTRACT_MODIFICATION
            and self._contract_modification_basis_unit(selected)
            and self._salary_scope_unit(selected)
        ):
            return self._contract_modification_short_answer(selected, citations)

        question_tokens = self._tokenize(question)
        primary = max(
            citations,
            key=lambda citation: (
                len(self._tokenize(citation.snippet) & question_tokens),
                self._label_specificity(citation.label),
                citation.support_score or 0.0,
            ),
        )
        return (
            "Din unitatile legale recuperate reiese urmatorul temei: "
            f"{primary.snippet} [{primary.label}]. "
            "Acesta este un draft generat peste EvidencePack si nu a fost "
            "verificat final de CitationVerifier."
        )

    def _detect_answer_intent(
        self,
        question: str,
        evidence_units: list[EvidenceUnit],
    ) -> str | None:
        normalized_question = self._normalize_text(question)
        if any(term in normalized_question for term in LABOR_CONTRACT_MODIFICATION_TRIGGERS):
            return ANSWER_INTENT_LABOR_CONTRACT_MODIFICATION
        tokens = set(normalized_question.split())
        if {"act", "aditional"}.issubset(tokens):
            return ANSWER_INTENT_LABOR_CONTRACT_MODIFICATION
        if any(term in tokens for term in LABOR_CONTRACT_REDUCTION_TERMS) and any(
            term in tokens for term in LABOR_CONTRACT_TARGET_TERMS
        ):
            return ANSWER_INTENT_LABOR_CONTRACT_MODIFICATION
        if self._contract_modification_basis_unit(evidence_units) and self._salary_scope_unit(evidence_units):
            return ANSWER_INTENT_LABOR_CONTRACT_MODIFICATION
        return None

    def _select_focused_answer_evidence(
        self,
        question: str,
        evidence_units: list[EvidenceUnit],
    ) -> list[EvidenceUnit]:
        if self._detect_answer_intent(question, evidence_units) != ANSWER_INTENT_LABOR_CONTRACT_MODIFICATION:
            return []

        scored = [
            (unit, self._score_labor_contract_modification_answer_evidence(unit))
            for unit in evidence_units
        ]
        asks_exception = self._asks_for_exception(question)
        direct_available = any(
            score["core_issue"] >= 0.70
            and unit.support_role == "direct_basis"
            and score["distractor"] == 0.0
            for unit, score in scored
        )

        eligible: list[tuple[EvidenceUnit, dict[str, float]]] = []
        for unit, score in scored:
            if unit.support_role == "distractor":
                continue
            if unit.support_role == "context" and direct_available:
                continue
            if unit.support_role == "exception" and not asks_exception:
                continue
            if score["core_issue"] < 0.25 and score["distractor"] > 0:
                continue
            if score["core_issue"] >= 0.70:
                eligible.append((unit, score))

        eligible.sort(
            key=lambda item: (
                item[1]["answer_score"],
                item[1]["core_issue"],
                item[1]["target_object"],
                item[0].rerank_score,
                -item[0].rank,
                item[0].id,
            ),
            reverse=True,
        )

        basis = self._contract_modification_basis_unit([unit for unit, _ in eligible])
        scope = self._salary_scope_unit([unit for unit, _ in eligible])
        if basis is None or scope is None:
            return []

        focused: list[EvidenceUnit] = []
        for unit in (basis, scope):
            if unit.id not in {item.id for item in focused}:
                focused.append(unit)

        salary_element = self._salary_element_unit(evidence_units, scope_unit=scope)
        if salary_element is not None and salary_element.id not in {unit.id for unit in focused}:
            focused.append(salary_element)
        return focused

    def _score_labor_contract_modification_answer_evidence(
        self,
        unit: EvidenceUnit,
    ) -> dict[str, float]:
        haystack = self._unit_haystack(unit)
        core_issue = self._core_issue_score(haystack)
        target_object = self._term_score(haystack, LABOR_CONTRACT_TARGET_TERMS)
        actor = self._term_score(haystack, LABOR_CONTRACT_ACTOR_TERMS)
        distractor = self._term_score(haystack, LABOR_CONTRACT_DISTRACTOR_TERMS)
        rerank = self._clamp(unit.rerank_score)
        retrieval = self._clamp(unit.retrieval_score)
        answer_score = self._clamp(
            0.55 * core_issue
            + 0.20 * target_object
            + 0.10 * actor
            + 0.10 * rerank
            + 0.05 * retrieval
            - 0.35 * distractor
        )
        return {
            "core_issue": round(core_issue, 6),
            "target_object": round(target_object, 6),
            "actor": round(actor, 6),
            "existing_rerank": round(rerank, 6),
            "existing_retrieval": round(retrieval, 6),
            "distractor": round(distractor, 6),
            "answer_score": round(answer_score, 6),
        }

    def _core_issue_score(self, haystack: str) -> float:
        if (
            "contractul individual de munca poate fi modificat numai prin acordul partilor" in haystack
            or "contract individual de munca poate fi modificat numai prin acordul partilor" in haystack
        ):
            return 1.0
        if "poate fi modificat numai prin acordul partilor" in haystack:
            return 0.95
        if self._has_contract_modification_scope(haystack) and "element" in haystack:
            return 0.85
        if self._has_contract_modification_scope(haystack):
            return 0.75
        has_contract = (
            "contractul individual de munca" in haystack
            or "contract individual de munca" in haystack
        )
        has_modified = "modificat" in haystack or "modificarea" in haystack
        has_agreement = "acordul partilor" in haystack
        if has_contract and has_modified and has_agreement:
            return 0.75
        if has_contract and has_modified:
            return 0.50
        return 0.0

    def _term_score(self, haystack: str, terms: tuple[str, ...]) -> float:
        if not terms:
            return 0.0
        matches = sum(1 for term in terms if self._normalize_text(term) in haystack)
        return self._clamp(matches / len(terms))

    def _asks_for_exception(self, question: str) -> bool:
        normalized = self._normalize_text(question)
        return any(term in normalized for term in ("exceptie", "exceptii", "derogare", "derogari"))

    def _contract_modification_basis_unit(
        self,
        evidence_units: list[EvidenceUnit],
    ) -> EvidenceUnit | None:
        candidates = [
            unit
            for unit in evidence_units
            if unit.support_role == "direct_basis"
            and self._is_contract_modification_basis(self._unit_haystack(unit))
        ]
        return self._best_contract_unit(candidates)

    def _salary_scope_unit(
        self,
        evidence_units: list[EvidenceUnit],
    ) -> EvidenceUnit | None:
        candidates = [
            unit
            for unit in evidence_units
            if unit.support_role == "direct_basis"
            and self._is_salary_scope(self._unit_haystack(unit))
        ]
        return self._best_contract_unit(candidates)

    def _salary_element_unit(
        self,
        evidence_units: list[EvidenceUnit],
        *,
        scope_unit: EvidenceUnit,
    ) -> EvidenceUnit | None:
        candidates = [
            unit
            for unit in evidence_units
            if unit.id != scope_unit.id
            and self._is_salary_element(unit, scope_unit=scope_unit)
        ]
        return self._best_contract_unit(candidates)

    def _best_contract_unit(
        self,
        units: list[EvidenceUnit],
    ) -> EvidenceUnit | None:
        if not units:
            return None
        return max(
            units,
            key=lambda unit: (
                self._focused_unit_score(unit),
                self._label_specificity(self._label(unit)),
                unit.rerank_score,
                -unit.rank,
                unit.id,
            ),
        )

    def _focused_unit_score(self, unit: EvidenceUnit) -> int:
        normalized = self._unit_haystack(unit)
        score = 0
        if self._is_contract_modification_basis(normalized):
            score += 100
        if self._is_salary_scope(normalized):
            score += 80
        if "salariu" in normalized or "salariul" in normalized:
            score += 20
        if "element" in normalized:
            score += 10
        if unit.paragraph_number:
            score += 4
        if unit.letter_number:
            score += 3
        return score

    def _is_contract_modification_basis(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        has_contract = "contractul individual de munca" in normalized or "contract individual de munca" in normalized
        has_modification = "poate fi modificat" in normalized or "modificat numai prin acordul partilor" in normalized
        has_agreement = "acordul partilor" in normalized
        return has_contract and has_modification and has_agreement

    def _is_salary_scope(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        has_contract_scope = (
            "modificarea contractului individual de munca" in normalized
            or "modificare contract individual munca" in normalized
        )
        has_scope_marker = any(
            marker in normalized
            for marker in (
                "elemente",
                "elementele",
                "poate privi",
                "se refera",
                "salariul",
                "salariu",
            )
        )
        return has_contract_scope and has_scope_marker

    def _has_contract_modification_scope(self, normalized: str) -> bool:
        return (
            "modificarea contractului individual de munca" in normalized
            or "modificare contract individual munca" in normalized
        )

    def _is_salary_element(
        self,
        unit: EvidenceUnit,
        *,
        scope_unit: EvidenceUnit,
    ) -> bool:
        normalized = self._normalize_text(f"{unit.raw_text} {unit.normalized_text or ''}")
        if "salariu" not in normalized and "salariul" not in normalized:
            return False
        if self._is_irrelevant_salary_context(normalized):
            return False
        if unit.parent_id and unit.parent_id == scope_unit.id:
            return True
        raw_tokens = self._tokenize(unit.raw_text)
        return bool(raw_tokens) and raw_tokens <= {"e", "salariu", "salariul"}

    def _is_irrelevant_salary_context(self, normalized: str) -> bool:
        return any(
            term in normalized
            for term in (
                "remuneratie restanta",
                "remuneratia restanta",
                "persoane angajate ilegal",
                "persoana angajata ilegal",
                "munca nedeclarata",
                "plata salariului",
                "confidentialitatea salariului",
                "salariul minim",
                "salariul de baza minim",
            )
        )

    def _unit_haystack(self, unit: EvidenceUnit) -> str:
        return self._normalize_text(f"{unit.raw_text} {unit.normalized_text or ''}")

    def _focused_contract_modification_snippet(self, unit: EvidenceUnit) -> str:
        if self._is_contract_modification_basis(unit.raw_text):
            preferred = self._first_matching_line(
                unit.raw_text,
                ("poate fi modificat", "acordul partilor"),
            )
            return self._trim(preferred or unit.raw_text.strip(), 420)
        if self._is_salary_scope(unit.raw_text):
            preferred = self._first_matching_line(
                unit.raw_text,
                ("modificarea contractului individual de munca",),
            )
            return self._trim(preferred or unit.raw_text.strip(), 420)
        return self._trim(unit.raw_text.strip(), 420)

    def _first_matching_line(self, raw_text: str, terms: tuple[str, ...]) -> str | None:
        for line in raw_text.splitlines():
            cleaned = line.strip()
            normalized = self._normalize_text(cleaned)
            if cleaned and all(term in normalized for term in terms):
                return cleaned
        return None

    def _contract_modification_short_answer(
        self,
        selected: list[EvidenceUnit],
        citations: list[DraftCitation],
    ) -> str:
        citation_by_unit_id = {citation.unit_id: citation for citation in citations}
        basis = self._contract_modification_basis_unit(selected)
        scope = self._salary_scope_unit(selected)
        if basis is None or scope is None:
            return ""

        basis_citation = citation_by_unit_id[basis.id]
        scope_citation = citation_by_unit_id[scope.id]
        salary = self._salary_element_unit(selected, scope_unit=scope)
        salary_citation = citation_by_unit_id.get(salary.id) if salary else None
        salary_citation_text = (
            f"; {salary_citation.label}; evidence:{salary_citation.unit_id}"
            if salary_citation
            else ""
        )
        return (
            "Pe baza unitatilor recuperate din Codul muncii, contractul "
            "individual de munca poate fi modificat, ca regula, numai prin "
            "acordul partilor "
            f"[{basis_citation.label}; evidence:{basis_citation.unit_id}]. "
            "Modificarea contractului individual de munca vizeaza elementele "
            "contractului, iar salariul trebuie verificat in elementele "
            "enumerate de textul legal recuperat "
            f"[{scope_citation.label}; evidence:{scope_citation.unit_id}"
            f"{salary_citation_text}]. "
            "Exceptiile sau situatiile speciale se analizeaza separat pe "
            "baza unitatilor recuperate "
            f"[{basis_citation.label}; evidence:{basis_citation.unit_id}; "
            f"{scope_citation.label}; evidence:{scope_citation.unit_id}]."
        )

    def _detailed_answer(
        self,
        selected: list[EvidenceUnit],
        citations: list[DraftCitation],
    ) -> str:
        units_by_id = {unit.id: unit for unit in selected}
        lines = [
            "Raspuns draft, extractive-first, bazat numai pe EvidenceUnit.raw_text.",
            "",
            "Baza legala recuperata:",
        ]
        for citation in citations:
            lines.append(
                f"- {citation.snippet} [{citation.label}; evidence:{citation.unit_id}]"
            )

        condition_lines = [
            citation
            for citation in citations
            if units_by_id[citation.unit_id].support_role in {"condition", "exception"}
        ]
        if condition_lines:
            lines.extend(["", "Conditii sau limite recuperate:"])
            for citation in condition_lines:
                lines.append(
                    f"- {citation.snippet} [{citation.label}; evidence:{citation.unit_id}]"
                )

        context_lines = [
            citation
            for citation in citations
            if units_by_id[citation.unit_id].support_role in {"definition", "context"}
        ]
        if context_lines:
            lines.extend(["", "Context recuperat:"])
            for citation in context_lines:
                lines.append(
                    f"- {citation.snippet} [{citation.label}; evidence:{citation.unit_id}]"
                )

        lines.extend(
            [
                "",
                "Status: draft generat peste EvidencePack; CitationVerifier real nu a rulat inca.",
            ]
        )
        return "\n".join(lines)

    def _label(self, unit: EvidenceUnit) -> str:
        law_title = self._clean_metadata(unit.law_title)
        if not law_title:
            return unit.id

        location_parts: list[str] = []
        if unit.article_number:
            location_parts.append(f"art. {unit.article_number}")
        if unit.paragraph_number:
            location_parts.append(f"alin. ({unit.paragraph_number})")
        if unit.letter_number:
            location_parts.append(f"lit. {unit.letter_number})")
        if unit.point_number:
            location_parts.append(f"pct. {unit.point_number}")

        if location_parts:
            return f"{law_title}, {' '.join(location_parts)}"
        return law_title

    def _label_specificity(self, label: str) -> int:
        markers = ("art.", "alin.", "lit.", "pct.")
        return sum(1 for marker in markers if marker in label)

    def _best_snippet(self, raw_text: str, question: str, max_length: int = 420) -> str:
        candidates = self._snippet_candidates(raw_text)
        if not candidates:
            return self._trim(raw_text.strip(), max_length)

        question_tokens = self._tokenize(question)
        best = max(
            candidates,
            key=lambda candidate: (
                len(self._tokenize(candidate) & question_tokens),
                len(self._tokenize(candidate)),
                -len(candidate),
            ),
        )
        return self._trim(best.strip(), max_length)

    def _snippet_candidates(self, raw_text: str) -> list[str]:
        candidates: list[str] = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if re.fullmatch(r"Art\.\s*\d+[A-Za-z0-9^]*", line):
                continue
            candidates.append(line)
        return candidates

    def _trim(self, text: str, max_length: int) -> str:
        if len(text) <= max_length:
            return text
        trimmed = text[:max_length].rsplit(" ", 1)[0].strip()
        return trimmed or text[:max_length].strip()

    def _tokenize(self, text: str) -> set[str]:
        normalized = self._normalize_text(text)
        return {
            token
            for token in re.split(r"[^a-z0-9_]+", normalized)
            if len(token) > 1 and token not in STOPWORDS
        }

    def _normalize_text(self, text: str) -> str:
        replacements = {
            "Ã„Æ’": "a",
            "Ã„â€š": "a",
            "Ãˆâ„¢": "s",
            "ÃˆËœ": "s",
            "Ãˆâ€º": "t",
            "ÃˆÅ¡": "t",
            "ÃƒÂ¢": "a",
            "ÃƒÂ®": "i",
            "Ã…Å¸": "s",
            "Ã…Â£": "t",
        }
        for broken, fixed in replacements.items():
            text = text.replace(broken, fixed)
        normalized = unicodedata.normalize("NFD", text.casefold())
        stripped = "".join(
            char for char in normalized if unicodedata.category(char) != "Mn"
        )
        return " ".join(stripped.replace(".", " ").replace("-", "_").split())

    def _clean_metadata(self, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        if not cleaned or cleaned.casefold() == "unknown":
            return None
        return cleaned

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _dedupe(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped
