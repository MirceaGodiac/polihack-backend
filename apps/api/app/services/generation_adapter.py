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
            self._draft_citation(unit, question=question, warnings=warnings)
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
            short_answer=self._short_answer(citations, question=question),
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
    ) -> DraftCitation | None:
        raw_text = unit.raw_text.strip()
        if not raw_text:
            warnings.append(GENERATION_MISSING_CITABLE_RAW_TEXT)
            return None

        label = self._label(unit)
        if label == unit.id:
            warnings.append(GENERATION_MISSING_CITATION_METADATA)
        snippet = self._best_snippet(raw_text, question)
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
    ) -> str:
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
            lines.append(f"- {citation.label}: {citation.snippet}")

        condition_lines = [
            citation
            for citation in citations
            if units_by_id[citation.unit_id].support_role in {"condition", "exception"}
        ]
        if condition_lines:
            lines.extend(["", "Conditii sau limite recuperate:"])
            for citation in condition_lines:
                lines.append(f"- {citation.label}: {citation.snippet}")

        context_lines = [
            citation
            for citation in citations
            if units_by_id[citation.unit_id].support_role in {"definition", "context"}
        ]
        if context_lines:
            lines.extend(["", "Context recuperat:"])
            for citation in context_lines:
                lines.append(f"- {citation.label}: {citation.snippet}")

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

    def _dedupe(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped
