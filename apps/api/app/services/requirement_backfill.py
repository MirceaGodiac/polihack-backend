from __future__ import annotations

import re
import unicodedata
from typing import Any

from pydantic import BaseModel, Field

from ..schemas import (
    EvidencePackResult,
    ExpandedCandidate,
    GraphExpansionResult,
    QueryPlan,
    RankedCandidate,
    RankerFeatureBreakdown,
    RetrievalCandidate,
)
from .evidence_pack_compiler import EvidencePackCompiler
from .legal_issue_frame import LABOR_CONTRACT_MODIFICATION_INTENT
from .query_frame import QueryFrame

CONTRACT_MODIFICATION_AGREEMENT_RULE = "contract_modification_agreement_rule"
CONTRACT_MODIFICATION_SALARY_SCOPE = "contract_modification_salary_scope"
REQUIREMENT_BACKFILL_TAG = "requirement_backfill"
REQUIREMENT_BACKFILL_UNRESOLVED = (
    "requirement_backfill_unresolved:contract_modification_salary_scope"
)
REQUIREMENT_BACKFILL_NO_AGREEMENT_SEED = "requirement_backfill_no_agreement_seed"
REQUIREMENT_BACKFILL_NO_CANDIDATES = (
    "requirement_backfill_no_candidates:contract_modification_salary_scope"
)

SALARY_SCOPE_SIGNALS = (
    "modificarea contractului individual de munca",
    "modificarea contractului",
    "contractul individual de munca",
    "poate privi",
    "elementele contractului",
    "salariul",
    "salariu",
)

SALARY_SCOPE_DISTRACTORS = (
    "remuneratie restanta",
    "persoane angajate ilegal",
    "munca nedeclarata",
    "salariul minim",
    "confidentialitatea salariului",
    "plata salariului",
    "intarzierea platii salariului",
    "registrul salariatilor",
    "registrul general de evidenta",
)


class RequirementBackfillResult(BaseModel):
    evidence_result: EvidencePackResult
    ranked_candidates: list[RankedCandidate] = Field(default_factory=list)
    added_unit_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    debug: dict[str, Any] | None = None


class RequirementBackfillService:
    def __init__(
        self,
        *,
        evidence_pack_compiler: EvidencePackCompiler | None = None,
        score_threshold: float = 0.55,
        max_backfill_candidates: int = 3,
    ) -> None:
        self.evidence_pack_compiler = evidence_pack_compiler or EvidencePackCompiler()
        self.score_threshold = score_threshold
        self.max_backfill_candidates = max_backfill_candidates

    async def backfill(
        self,
        *,
        plan: QueryPlan,
        query_frame: QueryFrame | None,
        ranked_candidates: list[RankedCandidate],
        evidence_result: EvidencePackResult,
        graph_expansion: GraphExpansionResult | None = None,
        debug: bool = False,
    ) -> RequirementBackfillResult:
        coverage_before = self._requirement_coverage(evidence_result)
        intent_id = str(
            coverage_before.get("intent_id")
            or (query_frame.intents[0] if query_frame and query_frame.intents else "")
        )
        missing = self._missing_requirements(coverage_before)
        enabled = (
            self._supports_intent(query_frame=query_frame, intent_id=intent_id)
            and CONTRACT_MODIFICATION_SALARY_SCOPE in missing
            and not bool(coverage_before.get("coverage_passed"))
        )
        if not enabled:
            return RequirementBackfillResult(
                evidence_result=evidence_result,
                ranked_candidates=ranked_candidates,
                warnings=[],
                debug=self._debug_payload(
                    enabled=False,
                    intent_id=intent_id,
                    coverage_before=coverage_before,
                    coverage_after=coverage_before,
                    scored_candidates=[],
                    added_unit_ids=[],
                    warnings=[],
                )
                if debug
                else None,
            )

        agreement_seed = self._agreement_seed(
            evidence_result=evidence_result,
            ranked_candidates=ranked_candidates,
        )
        if agreement_seed is None:
            warnings = [REQUIREMENT_BACKFILL_NO_AGREEMENT_SEED]
            return RequirementBackfillResult(
                evidence_result=evidence_result,
                ranked_candidates=ranked_candidates,
                warnings=warnings,
                debug=self._debug_payload(
                    enabled=True,
                    intent_id=intent_id,
                    coverage_before=coverage_before,
                    coverage_after=coverage_before,
                    scored_candidates=[],
                    added_unit_ids=[],
                    warnings=warnings,
                )
                if debug
                else None,
            )

        existing_ids = {unit.id for unit in evidence_result.evidence_units}
        candidate_pool = self._candidate_pool(
            ranked_candidates=ranked_candidates,
            graph_expansion=graph_expansion,
            exclude_unit_ids=existing_ids,
        )
        scored_candidates = self._score_candidates(
            candidate_pool,
            requirement_id=CONTRACT_MODIFICATION_SALARY_SCOPE,
            agreement_seed=agreement_seed,
        )
        selected_rows = [
            row for row in scored_candidates if row["score"] >= self.score_threshold
        ][: self.max_backfill_candidates]

        warnings: list[str] = []
        if not selected_rows:
            warnings.append(REQUIREMENT_BACKFILL_NO_CANDIDATES)
            warnings.append(REQUIREMENT_BACKFILL_UNRESOLVED)
            return RequirementBackfillResult(
                evidence_result=evidence_result,
                ranked_candidates=ranked_candidates,
                warnings=self._dedupe(warnings),
                debug=self._debug_payload(
                    enabled=True,
                    intent_id=intent_id,
                    coverage_before=coverage_before,
                    coverage_after=coverage_before,
                    scored_candidates=scored_candidates,
                    added_unit_ids=[],
                    warnings=self._dedupe(warnings),
                )
                if debug
                else None,
            )

        augmented_ranked = self._augment_ranked_candidates(
            ranked_candidates=ranked_candidates,
            selected_rows=selected_rows,
            agreement_seed=agreement_seed,
        )
        final_evidence = self._recompile_with_backfill_capacity(
            ranked_candidates=augmented_ranked,
            graph_expansion=graph_expansion,
            plan=plan,
            query_frame=query_frame,
            minimum_selected=len(evidence_result.evidence_units) + len(selected_rows),
        )
        coverage_after = self._requirement_coverage(final_evidence)
        added_unit_ids = [
            unit.id
            for unit in final_evidence.evidence_units
            if unit.id not in existing_ids
        ]
        if not bool(coverage_after.get("coverage_passed")):
            warnings.append(REQUIREMENT_BACKFILL_UNRESOLVED)

        return RequirementBackfillResult(
            evidence_result=final_evidence,
            ranked_candidates=augmented_ranked,
            added_unit_ids=added_unit_ids,
            warnings=self._dedupe(warnings),
            debug=self._debug_payload(
                enabled=True,
                intent_id=intent_id,
                coverage_before=coverage_before,
                coverage_after=coverage_after,
                scored_candidates=scored_candidates,
                added_unit_ids=added_unit_ids,
                warnings=self._dedupe(warnings),
            )
            if debug
            else None,
        )

    def _recompile_with_backfill_capacity(
        self,
        *,
        ranked_candidates: list[RankedCandidate],
        graph_expansion: GraphExpansionResult | None,
        plan: QueryPlan,
        query_frame: QueryFrame | None,
        minimum_selected: int,
    ) -> EvidencePackResult:
        compiler = EvidencePackCompiler(
            mmr_lambda=self.evidence_pack_compiler.mmr_lambda,
            candidate_pool_size=max(
                self.evidence_pack_compiler.candidate_pool_size,
                len(ranked_candidates),
            ),
            target_evidence_units=max(
                self.evidence_pack_compiler.target_evidence_units,
                minimum_selected,
            ),
            max_evidence_units=max(
                self.evidence_pack_compiler.max_evidence_units,
                minimum_selected,
            ),
            role_classifier=self.evidence_pack_compiler.role_classifier,
        )
        return compiler.compile(
            ranked_candidates=ranked_candidates,
            graph_expansion=graph_expansion,
            plan=plan,
            query_frame=query_frame,
            debug=True,
        )

    def _supports_intent(
        self,
        *,
        query_frame: QueryFrame | None,
        intent_id: str,
    ) -> bool:
        if intent_id == LABOR_CONTRACT_MODIFICATION_INTENT:
            return True
        return bool(
            query_frame and LABOR_CONTRACT_MODIFICATION_INTENT in query_frame.intents
        )

    def _requirement_coverage(
        self,
        evidence_result: EvidencePackResult,
    ) -> dict[str, Any]:
        if evidence_result.debug and isinstance(evidence_result.debug, dict):
            coverage = evidence_result.debug.get("requirement_coverage")
            if isinstance(coverage, dict):
                return coverage
        return {
            "enabled": False,
            "intent_id": None,
            "covered_requirement_ids": [],
            "missing_required_requirements": [],
            "coverage_passed": True,
        }

    def _missing_requirements(self, coverage: dict[str, Any]) -> list[str]:
        missing = coverage.get("missing_required_requirements")
        if isinstance(missing, list):
            return [str(item) for item in missing]
        legacy = coverage.get("missing_required_requirement_ids")
        if isinstance(legacy, list):
            return [str(item) for item in legacy]
        return []

    def _agreement_seed(
        self,
        *,
        evidence_result: EvidencePackResult,
        ranked_candidates: list[RankedCandidate],
    ) -> dict[str, Any] | None:
        ranked_by_id = {candidate.unit_id: candidate for candidate in ranked_candidates}
        for unit in evidence_result.evidence_units:
            matched_requirement = CONTRACT_MODIFICATION_AGREEMENT_RULE in set(
                self._matched_requirement_ids(unit)
            )
            if matched_requirement or self._is_agreement_rule(unit.raw_text):
                ranked = ranked_by_id.get(unit.id)
                return {
                    "unit_id": unit.id,
                    "law_id": unit.law_id,
                    "article_number": unit.article_number,
                    "parent_id": unit.parent_id,
                    "rank": ranked.rank if ranked is not None else unit.rank,
                }
        for candidate in ranked_candidates:
            unit = candidate.unit or {}
            if self._is_agreement_rule(str(unit.get("raw_text") or "")):
                return {
                    "unit_id": candidate.unit_id,
                    "law_id": str(unit.get("law_id") or ""),
                    "article_number": self._optional_str(unit.get("article_number")),
                    "parent_id": self._optional_str(unit.get("parent_id")),
                    "rank": candidate.rank,
                }
        return None

    def _candidate_pool(
        self,
        *,
        ranked_candidates: list[RankedCandidate],
        graph_expansion: GraphExpansionResult | None,
        exclude_unit_ids: set[str],
    ) -> list[RankedCandidate]:
        pool: dict[str, RankedCandidate] = {}
        for candidate in ranked_candidates:
            if candidate.unit_id in exclude_unit_ids:
                continue
            unit = candidate.unit or {}
            if not self._candidate_text(unit):
                continue
            pool[candidate.unit_id] = candidate

        if graph_expansion is not None:
            for expanded in graph_expansion.expanded_candidates:
                if expanded.unit_id in exclude_unit_ids or expanded.unit_id in pool:
                    continue
                extra = self._ranked_from_expanded(expanded)
                if extra is None:
                    continue
                pool[extra.unit_id] = extra

        return list(pool.values())

    def _ranked_from_expanded(
        self,
        expanded: ExpandedCandidate,
    ) -> RankedCandidate | None:
        retrieval = expanded.retrieval_candidate
        if retrieval is None or not retrieval.unit:
            return None
        if not self._candidate_text(retrieval.unit):
            return None
        breakdown = dict(retrieval.score_breakdown)
        breakdown["graph_proximity"] = max(
            float(breakdown.get("graph_proximity", 0.0)),
            float(expanded.graph_proximity),
        )
        why_ranked = self._dedupe(
            [
                retrieval.why_retrieved or "",
                expanded.expansion_reason or "",
                expanded.expansion_edge_type or "",
            ]
        )
        return RankedCandidate(
            unit_id=retrieval.unit_id,
            rank=retrieval.rank,
            rerank_score=self._clamp(retrieval.retrieval_score),
            retrieval_score=retrieval.retrieval_score,
            unit=retrieval.unit,
            score_breakdown=RankerFeatureBreakdown(**breakdown),
            why_ranked=why_ranked,
            source="graph_expansion",
        )

    def _score_candidates(
        self,
        candidates: list[RankedCandidate],
        *,
        requirement_id: str,
        agreement_seed: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            unit = candidate.unit or {}
            text = self._candidate_text(unit)
            if not text:
                continue
            requirement_match = self._requirement_match(
                text=text,
                unit=unit,
                seed=agreement_seed,
            )
            same_article = self._same_article(unit=unit, seed=agreement_seed)
            same_law = self._same_law(unit=unit, seed=agreement_seed)
            phrase_proximity = self._phrase_proximity(text)
            unit_specificity = self._unit_specificity(unit)
            existing_rank_quality = self._existing_rank_quality(candidate)
            metadata_validity = self._metadata_validity(unit)
            distractor_penalty = self._distractor_penalty(text)
            score = self._clamp(
                0.30 * requirement_match
                + 0.20 * same_article
                + 0.15 * same_law
                + 0.15 * phrase_proximity
                + 0.10 * unit_specificity
                + 0.05 * existing_rank_quality
                + 0.05 * metadata_validity
                - 0.20 * distractor_penalty
            )
            rows.append(
                {
                    "unit_id": candidate.unit_id,
                    "requirement_id": requirement_id,
                    "candidate": candidate,
                    "score": round(score, 6),
                    "score_breakdown": {
                        "requirement_match": round(requirement_match, 6),
                        "same_article": round(same_article, 6),
                        "same_law": round(same_law, 6),
                        "phrase_proximity": round(phrase_proximity, 6),
                        "unit_specificity": round(unit_specificity, 6),
                        "existing_rank_quality": round(existing_rank_quality, 6),
                        "metadata_validity": round(metadata_validity, 6),
                        "distractor_penalty": round(distractor_penalty, 6),
                    },
                    "selected": False,
                }
            )
        rows.sort(
            key=lambda row: (
                row["score"],
                row["score_breakdown"]["requirement_match"],
                row["score_breakdown"]["same_article"],
                row["score_breakdown"]["unit_specificity"],
                row["unit_id"],
            ),
            reverse=True,
        )
        return rows

    def _augment_ranked_candidates(
        self,
        *,
        ranked_candidates: list[RankedCandidate],
        selected_rows: list[dict[str, Any]],
        agreement_seed: dict[str, Any],
    ) -> list[RankedCandidate]:
        selected_map = {
            row["unit_id"]: row for row in selected_rows
        }
        updated: list[RankedCandidate] = []
        for candidate in ranked_candidates:
            row = selected_map.get(candidate.unit_id)
            if row is None:
                updated.append(candidate)
                continue
            updated.append(
                self._backfill_candidate(
                    candidate,
                    row=row,
                    seed_rank=int(agreement_seed.get("rank") or 1),
                    ordinal=selected_rows.index(row) + 1,
                )
            )

        ranked_ids = {candidate.unit_id for candidate in ranked_candidates}
        seed_rank = int(agreement_seed.get("rank") or 1)
        for index, row in enumerate(selected_rows, start=1):
            if row["unit_id"] in ranked_ids:
                continue
            updated.append(
                self._backfill_candidate(
                    row["candidate"],
                    row=row,
                    seed_rank=seed_rank,
                    ordinal=index,
                )
            )
        return updated

    def _backfill_candidate(
        self,
        candidate: RankedCandidate,
        *,
        row: dict[str, Any],
        seed_rank: int,
        ordinal: int,
    ) -> RankedCandidate:
        why_ranked = self._dedupe(
            [
                *candidate.why_ranked,
                REQUIREMENT_BACKFILL_TAG,
                f"covers_missing_requirement:{row['requirement_id']}",
            ]
        )
        row["selected"] = True
        return candidate.model_copy(
            update={
                "rank": min(candidate.rank, seed_rank + ordinal),
                "rerank_score": max(candidate.rerank_score, float(row["score"])),
                "why_ranked": why_ranked,
                "source": candidate.source or "requirement_backfill",
            }
        )

    def _matched_requirement_ids(self, unit: Any) -> list[str]:
        why_selected = getattr(unit, "why_selected", [])
        matched_ids: list[str] = []
        for reason in why_selected:
            prefix = "requirement_match:"
            if isinstance(reason, str) and reason.startswith(prefix):
                matched_ids.append(reason[len(prefix) :])
        return matched_ids

    def _requirement_match(
        self,
        *,
        text: str,
        unit: dict[str, Any],
        seed: dict[str, Any],
    ) -> float:
        normalized = self._normalize_text(text)
        has_contract_scope = any(
            phrase in normalized
            for phrase in (
                "modificarea contractului individual de munca",
                "modificarea contractului",
                "contractul individual de munca",
                "contract individual de munca",
            )
        )
        has_scope_marker = any(
            phrase in normalized
            for phrase in ("poate privi", "elementele contractului", "elemente")
        )
        has_salary = any(term in normalized for term in ("salariu", "salariul"))
        if has_contract_scope and has_salary and has_scope_marker:
            return 1.0
        if has_contract_scope and has_salary:
            return 1.0
        if has_contract_scope and has_scope_marker:
            return 0.7
        if (
            has_salary
            and self._same_article(unit=unit, seed=seed) > 0.0
            and self._is_specific_child(unit)
        ):
            return 0.5
        return 0.0

    def _same_article(self, *, unit: dict[str, Any], seed: dict[str, Any]) -> float:
        law_id = str(unit.get("law_id") or "")
        article_number = self._optional_str(unit.get("article_number"))
        if (
            law_id
            and article_number
            and law_id == str(seed.get("law_id") or "")
            and article_number == self._optional_str(seed.get("article_number"))
        ):
            return 1.0
        unit_parent = self._optional_str(unit.get("parent_id"))
        seed_parent = self._optional_str(seed.get("parent_id"))
        seed_unit_id = str(seed.get("unit_id") or "")
        if unit_parent and unit_parent == seed_unit_id:
            return 0.5
        if seed_parent and str(unit.get("id") or "") == seed_parent:
            return 0.5
        return 0.0

    def _same_law(self, *, unit: dict[str, Any], seed: dict[str, Any]) -> float:
        law_id = str(unit.get("law_id") or "")
        return 1.0 if law_id and law_id == str(seed.get("law_id") or "") else 0.0

    def _phrase_proximity(self, text: str) -> float:
        normalized = self._normalize_text(text)
        hits = sum(1 for signal in SALARY_SCOPE_SIGNALS if signal in normalized)
        score = hits / max(1, len(SALARY_SCOPE_SIGNALS))
        if (
            "modificarea contractului individual de munca" in normalized
            and "salariu" in normalized
        ):
            return max(score, 0.9)
        if "poate privi" in normalized and "salariu" in normalized:
            return max(score, 0.8)
        if "elementele contractului" in normalized:
            return max(score, 0.65)
        if "salariu" in normalized:
            return max(score, 0.35)
        return self._clamp(score)

    def _unit_specificity(self, unit: dict[str, Any]) -> float:
        if unit.get("letter_number") or unit.get("point_number"):
            return 1.0
        if unit.get("paragraph_number"):
            return 0.8
        if unit.get("article_number"):
            return 0.6
        unit_type = self._normalize_text(str(unit.get("type") or unit.get("unit_type") or ""))
        if unit_type in {"legal_act", "title", "chapter", "section"}:
            return 0.3
        return 0.3

    def _existing_rank_quality(self, candidate: RankedCandidate) -> float:
        retrieval = candidate.retrieval_score or 0.0
        return self._clamp(max(candidate.rerank_score, retrieval))

    def _metadata_validity(self, unit: dict[str, Any]) -> float:
        required = (
            unit.get("law_id"),
            unit.get("law_title"),
            unit.get("article_number"),
            unit.get("raw_text"),
            unit.get("source_url"),
        )
        present = sum(1 for value in required if value not in (None, "", [], {}))
        return present / len(required)

    def _distractor_penalty(self, text: str) -> float:
        normalized = self._normalize_text(text)
        has_contract_modification = (
            "modificarea contractului" in normalized
            or "contractul individual de munca" in normalized
        )
        if any(term in normalized for term in SALARY_SCOPE_DISTRACTORS):
            return 0.0 if has_contract_modification else 1.0
        if "salariu" in normalized or "salariul" in normalized:
            return 0.0 if has_contract_modification else 0.5
        return 0.0

    def _is_agreement_rule(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        has_contract = (
            "contractul individual de munca" in normalized
            or "contract individual de munca" in normalized
        )
        has_modification = (
            "poate fi modificat" in normalized
            or "modificat numai prin acordul partilor" in normalized
            or "numai prin acordul partilor" in normalized
        )
        has_agreement = "acordul partilor" in normalized
        return has_contract and has_modification and has_agreement

    def _is_specific_child(self, unit: dict[str, Any]) -> bool:
        return bool(unit.get("letter_number") or unit.get("point_number"))

    def _candidate_text(self, unit: dict[str, Any]) -> str:
        value = unit.get("raw_text")
        return value.strip() if isinstance(value, str) and value.strip() else ""

    def _debug_payload(
        self,
        *,
        enabled: bool,
        intent_id: str,
        coverage_before: dict[str, Any],
        coverage_after: dict[str, Any],
        scored_candidates: list[dict[str, Any]],
        added_unit_ids: list[str],
        warnings: list[str],
    ) -> dict[str, Any]:
        return {
            "enabled": enabled,
            "intent_id": intent_id,
            "coverage_before": {
                "coverage_passed": coverage_before.get("coverage_passed", True),
                "missing_required_requirements": self._missing_requirements(
                    coverage_before
                ),
            },
            "candidate_count": len(scored_candidates),
            "scored_candidates": [
                {
                    "unit_id": row["unit_id"],
                    "requirement_id": row["requirement_id"],
                    "score": row["score"],
                    "score_breakdown": row["score_breakdown"],
                    "selected": row["selected"],
                }
                for row in scored_candidates
            ],
            "added_unit_ids": added_unit_ids,
            "coverage_after": {
                "coverage_passed": coverage_after.get("coverage_passed", True),
                "missing_required_requirements": self._missing_requirements(
                    coverage_after
                ),
            },
            "warnings": warnings,
        }

    def _optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _dedupe(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        for value in values:
            if value and value not in deduped:
                deduped.append(value)
        return deduped

    def _normalize_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFD", text.casefold())
        stripped = "".join(
            char for char in normalized if unicodedata.category(char) != "Mn"
        )
        return " ".join(re.findall(r"[a-z0-9_]+", stripped))

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))
