from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata
from typing import Any

from ..schemas import (
    EvidencePackResult,
    EvidenceUnit,
    GraphEdge,
    GraphExpansionResult,
    GraphNode,
    LegalUnit,
    QueryPlan,
    RankedCandidate,
)
from .legal_issue_frame import CandidateRoleClassifier, CandidateRoleDecision
from .query_frame import QueryFrame

EVIDENCE_PACK_NO_RANKED_CANDIDATES = "evidence_pack_no_ranked_candidates"
EVIDENCE_PACK_PARTIAL = "evidence_pack_partial"
EVIDENCE_PACK_MISSING_UNIT_TEXT = "evidence_pack_missing_unit_text"
EVIDENCE_PACK_MISSING_UNIT_RAW_TEXT = "evidence_pack_missing_unit_raw_text"
EVIDENCE_PACK_MISSING_UNIT_METADATA = "evidence_pack_missing_unit_metadata"

SUPPORT_ROLES = (
    "direct_basis",
    "definition",
    "condition",
    "exception",
    "sanction",
    "procedure",
    "context",
)

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
    "pe",
    "sau",
    "se",
    "si",
}

CONTRACT_MODIFICATION_TRIGGERS = (
    "act aditional",
    "modificare contract",
    "modificarea contractului",
    "scada salariul",
    "scade salariul",
    "salariul fara act aditional",
)

CONTRACT_MODIFICATION_REDUCTION_TERMS = (
    "scada",
    "scade",
    "reducere",
    "reduca",
    "diminuare",
    "diminueze",
)

CONTRACT_MODIFICATION_TARGET_TERMS = (
    "salariu",
    "salariul",
    "salarizare",
)

AGREEMENT_RULE_TERMS = (
    "contractul individual de munca poate fi modificat",
    "contract individual de munca poate fi modificat",
    "poate fi modificat numai prin acordul partilor",
    "modificat numai prin acordul partilor",
    "numai prin acordul partilor",
)

MODIFICATION_SCOPE_TERMS = (
    "modificarea contractului individual de munca",
    "modificare contract individual munca",
    "contractului individual de munca",
    "contract individual de munca",
    "contractul individual de munca",
)

SALARY_CONTEXT_TERMS = (
    "plata salariului",
    "platii salariului",
    "drepturi salariale",
    "drepturile salariale",
    "salariul de baza minim",
    "salariul minim",
    "salariul este confidential",
    "confidentialitatea salariului",
    "remuneratie restanta",
    "remuneratia restanta",
    "remuneratie",
    "remunerarea",
    "remunerat",
    "persoane angajate ilegal",
    "persoana angajata ilegal",
    "neplata salariului",
    "intarzierea platii salariului",
    "registrul salariatilor",
    "registrul general de evidenta",
    "munca nedeclarata",
    "primirea la munca",
)

DIRECT_BASIS_KINDS = ("agreement_rule", "modification_scope")

SUPPORT_ROLE_PRIORITY_SCORE = {
    "direct_basis": 1.0,
    "condition": 0.85,
    "exception": 0.65,
    "definition": 0.55,
    "procedure": 0.50,
    "sanction": 0.45,
    "context": 0.35,
}

CONTRACT_MODIFICATION_GENERIC_CONTEXT_TERMS = (
    "anterior incheierii contractului individual de munca",
    "persoana selectata",
    "informarea persoanei selectate",
    "elementele din contract",
    "locul muncii",
    "felul muncii",
    "durata contractului",
    "drepturile si obligatiile partilor",
)


@dataclass
class _EvidenceCandidate:
    ranked: RankedCandidate
    unit: dict[str, Any]
    text: str
    tokens: set[str]
    support_role: str
    why_selected: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    role_decision: CandidateRoleDecision | None = None
    mmr_score: float = 0.0


class EvidencePackCompiler:
    def __init__(
        self,
        mmr_lambda: float = 0.75,
        candidate_pool_size: int = 40,
        target_evidence_units: int = 12,
        max_evidence_units: int = 14,
        role_classifier: CandidateRoleClassifier | None = None,
    ) -> None:
        self.mmr_lambda = mmr_lambda
        self.candidate_pool_size = candidate_pool_size
        self.target_evidence_units = target_evidence_units
        self.max_evidence_units = max_evidence_units
        self.role_classifier = role_classifier or CandidateRoleClassifier()

    def compile(
        self,
        *,
        ranked_candidates: list[RankedCandidate],
        graph_expansion: GraphExpansionResult | None = None,
        plan: QueryPlan | None = None,
        query_frame: QueryFrame | None = None,
        debug: bool = False,
    ) -> EvidencePackResult:
        if not ranked_candidates:
            return self._fallback_result(debug=debug)

        warnings: list[str] = []
        unique_ranked = self._dedupe_ranked_candidates(ranked_candidates)
        candidate_pool = self._candidate_pool(
            unique_ranked,
            plan=plan,
            query_frame=query_frame,
            warnings=warnings,
        )
        if not candidate_pool:
            warnings.append(EVIDENCE_PACK_NO_RANKED_CANDIDATES)
            return self._empty_result(
                debug=debug,
                input_ranked_candidate_count=len(ranked_candidates),
                warnings=self._dedupe(warnings),
            )

        selected = self._select_with_mmr(candidate_pool, plan=plan)
        if self._is_contract_modification_query(plan):
            self._ensure_role_candidate(selected, candidate_pool, "exception")
        else:
            self._ensure_role_candidate(selected, candidate_pool, "exception")
            self._ensure_role_candidate(selected, candidate_pool, "definition")
            self._ensure_role_candidate(selected, candidate_pool, "sanction")
        self._include_parent_context(selected, candidate_pool)
        selected = self._compact_contract_modification_selection(
            selected,
            candidate_pool,
            plan=plan,
        )
        selected = self._normalize_contract_modification_roles(
            selected,
            plan=plan,
        )

        if len(selected) < min(self.target_evidence_units, 8):
            warnings.append(EVIDENCE_PACK_PARTIAL)

        evidence_units = [
            self._evidence_unit(candidate, rank=index)
            for index, candidate in enumerate(selected, start=1)
        ]
        graph_nodes, graph_edges = self._build_graph(
            selected=selected,
            graph_expansion=graph_expansion,
        )

        result = EvidencePackResult(
            evidence_units=evidence_units,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            warnings=self._dedupe(warnings),
            debug=None,
        )
        if debug:
            result.debug = self._debug_payload(
                fallback_used=False,
                input_ranked_candidate_count=len(ranked_candidates),
                candidate_pool_size=len(candidate_pool),
                selected=selected,
                plan=plan,
                query_frame=query_frame,
                warnings=result.warnings,
            )
        return result

    def _fallback_result(self, *, debug: bool) -> EvidencePackResult:
        warnings = [EVIDENCE_PACK_NO_RANKED_CANDIDATES]
        result = EvidencePackResult(warnings=warnings)
        if debug:
            result.debug = {
                "fallback_used": True,
                "input_ranked_candidate_count": 0,
                "candidate_pool_size": 0,
                "selected_evidence_count": 0,
                "lambda": self.mmr_lambda,
                "target_evidence_units": self.target_evidence_units,
                "max_evidence_units": self.max_evidence_units,
                "selected_units": [],
                "warnings": warnings,
            }
        return result

    def _empty_result(
        self,
        *,
        debug: bool,
        input_ranked_candidate_count: int,
        warnings: list[str],
    ) -> EvidencePackResult:
        result = EvidencePackResult(warnings=warnings)
        if debug:
            result.debug = {
                "fallback_used": True,
                "input_ranked_candidate_count": input_ranked_candidate_count,
                "candidate_pool_size": 0,
                "selected_evidence_count": 0,
                "lambda": self.mmr_lambda,
                "target_evidence_units": self.target_evidence_units,
                "max_evidence_units": self.max_evidence_units,
                "selected_units": [],
                "warnings": warnings,
            }
        return result

    def _dedupe_ranked_candidates(
        self,
        ranked_candidates: list[RankedCandidate],
    ) -> list[RankedCandidate]:
        best_by_unit_id: dict[str, RankedCandidate] = {}
        for candidate in ranked_candidates:
            current = best_by_unit_id.get(candidate.unit_id)
            if current is None or self._candidate_quality(candidate) > self._candidate_quality(current):
                best_by_unit_id[candidate.unit_id] = candidate
        return sorted(
            best_by_unit_id.values(),
            key=lambda candidate: (candidate.rank, -candidate.rerank_score, candidate.unit_id),
        )

    def _candidate_quality(self, candidate: RankedCandidate) -> tuple[int, float, int]:
        unit = candidate.unit or {}
        info_score = sum(1 for value in unit.values() if value not in (None, "", [], {}))
        text_score = 1 if self._candidate_text(unit) else 0
        return (text_score, candidate.rerank_score, info_score)

    def _candidate_pool(
        self,
        ranked_candidates: list[RankedCandidate],
        *,
        plan: QueryPlan | None,
        query_frame: QueryFrame | None,
        warnings: list[str],
    ) -> list[_EvidenceCandidate]:
        pool: list[_EvidenceCandidate] = []
        missing_raw_text = False
        missing_metadata = False
        for ranked in ranked_candidates[: self.candidate_pool_size]:
            unit = ranked.unit or {}
            text = self._candidate_text(unit)
            if not unit or not text:
                missing_raw_text = True
                continue
            if not self._has_required_metadata(unit):
                missing_metadata = True
            role_decision = self._role_decision(
                ranked=ranked,
                unit=unit,
                query_frame=query_frame,
            )
            support_role = (
                role_decision.support_role
                if role_decision is not None
                else self._support_role(ranked, unit, plan=plan)
            )
            why_selected = self._base_selection_reasons(ranked)
            role_warnings: list[str] = []
            if role_decision is not None:
                why_selected.extend(role_decision.why_role)
                role_warnings.extend(
                    f"role_disqualified:{requirement_id}"
                    for requirement_id in role_decision.disqualified_requirement_ids
                )
            pool.append(
                _EvidenceCandidate(
                    ranked=ranked,
                    unit=unit,
                    text=text,
                    tokens=self._tokenize(text),
                    support_role=support_role,
                    why_selected=why_selected,
                    warnings=role_warnings,
                    role_decision=role_decision,
                )
            )
        if missing_raw_text:
            warnings.append(EVIDENCE_PACK_MISSING_UNIT_RAW_TEXT)
            warnings.append(EVIDENCE_PACK_MISSING_UNIT_TEXT)
        if missing_metadata:
            warnings.append(EVIDENCE_PACK_MISSING_UNIT_METADATA)
        return pool

    def _select_with_mmr(
        self,
        candidate_pool: list[_EvidenceCandidate],
        *,
        plan: QueryPlan | None,
    ) -> list[_EvidenceCandidate]:
        target = self._selection_target(candidate_pool, plan=plan)
        selected = self._priority_direct_basis_candidates(
            candidate_pool,
            plan=plan,
            target=target,
        )
        self._include_contract_modification_salary_scope(
            selected,
            candidate_pool,
            plan=plan,
            target=target,
        )
        self._include_contract_modification_salary_target(
            selected,
            candidate_pool,
            plan=plan,
            target=target,
        )
        self._include_contract_modification_condition(
            selected,
            candidate_pool,
            plan=plan,
            target=target,
        )
        selected_ids = {candidate.ranked.unit_id for candidate in selected}
        remaining = [
            candidate
            for candidate in candidate_pool
            if candidate.ranked.unit_id not in selected_ids
        ]
        while remaining and len(selected) < target:
            best = max(
                remaining,
                key=lambda candidate: self._mmr_sort_key(candidate, selected),
            )
            best.mmr_score = self._mmr_score(best, selected)
            self._append_reason(best, "selected_by_mmr")
            selected.append(best)
            remaining.remove(best)
        return selected

    def _selection_target(
        self,
        candidate_pool: list[_EvidenceCandidate],
        *,
        plan: QueryPlan | None,
    ) -> int:
        target = min(self.target_evidence_units, len(candidate_pool))
        if not self._is_contract_modification_query(plan):
            return target
        required_covered = sum(
            1
            for requirement_id in (
                "contract_modification_agreement_rule",
                "contract_modification_salary_scope",
            )
            if any(
                self._candidate_covers_requirement(candidate, requirement_id)
                for candidate in candidate_pool
            )
        )
        target = min(target, 8)
        if len(candidate_pool) >= 6:
            target = max(target, 6)
        if required_covered:
            target = max(target, required_covered)
        return min(target, self.max_evidence_units, len(candidate_pool))

    def _include_contract_modification_condition(
        self,
        selected: list[_EvidenceCandidate],
        candidate_pool: list[_EvidenceCandidate],
        *,
        plan: QueryPlan | None,
        target: int,
    ) -> None:
        if (
            len(selected) >= target
            or not self._is_contract_modification_query(plan)
        ):
            return
        selected_ids = {candidate.ranked.unit_id for candidate in selected}
        candidates = [
            candidate
            for candidate in candidate_pool
            if candidate.ranked.unit_id not in selected_ids
            and self._is_contract_modification_condition(
                self._haystack(candidate.ranked, candidate.unit)
            )
        ]
        if not candidates:
            return
        best = max(
            candidates,
            key=lambda candidate: (
                self._unit_specificity(candidate.unit),
                candidate.ranked.rerank_score,
                -candidate.ranked.rank,
                candidate.ranked.unit_id,
            ),
        )
        best.support_role = "condition"
        best.mmr_score = self._mmr_score(best, selected)
        self._append_reason(best, "contract_modification_condition")
        self._append_reason(best, "priority_condition:act_additional")
        selected.append(best)

    def _include_contract_modification_salary_scope(
        self,
        selected: list[_EvidenceCandidate],
        candidate_pool: list[_EvidenceCandidate],
        *,
        plan: QueryPlan | None,
        target: int,
    ) -> None:
        if (
            len(selected) >= target
            or not self._is_contract_modification_query(plan)
            or any(
                self._candidate_covers_requirement(
                    candidate,
                    "contract_modification_salary_scope",
                )
                for candidate in selected
            )
        ):
            return
        selected_ids = {candidate.ranked.unit_id for candidate in selected}
        candidates = [
            candidate
            for candidate in candidate_pool
            if candidate.ranked.unit_id not in selected_ids
            and self._candidate_covers_requirement(
                candidate,
                "contract_modification_salary_scope",
            )
        ]
        if not candidates:
            return
        best = max(
            candidates,
            key=lambda candidate: (
                candidate.support_role == "direct_basis",
                candidate.ranked.rerank_score,
                candidate.ranked.retrieval_score or 0.0,
                self._unit_specificity(candidate.unit),
                -candidate.ranked.rank,
                candidate.ranked.unit_id,
            ),
        )
        if best.support_role == "context":
            best.support_role = "condition"
        best.mmr_score = self._mmr_score(best, selected)
        self._append_reason(best, "contract_modification_salary_scope")
        self._append_reason(best, "priority_requirement:contract_modification_salary_scope")
        selected.append(best)

    def _include_contract_modification_salary_target(
        self,
        selected: list[_EvidenceCandidate],
        candidate_pool: list[_EvidenceCandidate],
        *,
        plan: QueryPlan | None,
        target: int,
    ) -> None:
        if (
            len(selected) >= target
            or not self._is_contract_modification_query(plan)
        ):
            return
        selected_ids = {candidate.ranked.unit_id for candidate in selected}
        scope_parent_ids = {
            candidate.ranked.unit_id
            for candidate in selected
            if self._is_contract_modification_scope_parent(
                self._haystack(candidate.ranked, candidate.unit)
            )
        }
        if not scope_parent_ids:
            return
        candidates = [
            candidate
            for candidate in candidate_pool
            if candidate.ranked.unit_id not in selected_ids
            and self._is_salary_target_child(candidate, parent_ids=scope_parent_ids)
        ]
        if not candidates:
            return
        best = max(
            candidates,
            key=lambda candidate: (
                candidate.ranked.rerank_score,
                candidate.ranked.retrieval_score or 0.0,
                -candidate.ranked.rank,
                candidate.ranked.unit_id,
            ),
        )
        best.support_role = "condition"
        best.mmr_score = self._mmr_score(best, selected)
        self._append_reason(best, "contract_modification_salary_target_child")
        self._append_reason(best, "priority_condition:salary_target_element")
        selected.append(best)

    def _candidate_covers_requirement(
        self,
        candidate: _EvidenceCandidate,
        requirement_id: str,
    ) -> bool:
        if candidate.role_decision is not None and (
            requirement_id in candidate.role_decision.matched_requirement_ids
        ):
            return True
        haystack = self._haystack(candidate.ranked, candidate.unit)
        if requirement_id == "contract_modification_agreement_rule":
            return self._is_contract_modification_agreement_rule(haystack)
        if requirement_id == "contract_modification_salary_scope":
            return (
                self._is_contract_modification_salary_scope(haystack)
                or self._is_contract_modification_scope_parent(haystack)
                and (
                    candidate.ranked.score_breakdown.intent_governing_rule_parent > 0
                    or "intent_governing_rule_parent_context:labor_contract_modification"
                    in candidate.ranked.why_ranked
                )
            )
        return False

    def _priority_direct_basis_candidates(
        self,
        candidate_pool: list[_EvidenceCandidate],
        *,
        plan: QueryPlan | None,
        target: int,
    ) -> list[_EvidenceCandidate]:
        if target <= 0 or not self._is_contract_modification_query(plan):
            return []

        selected: list[_EvidenceCandidate] = []
        selected_ids: set[str] = set()
        for kind in DIRECT_BASIS_KINDS:
            if len(selected) >= target:
                break
            candidates = [
                candidate
                for candidate in candidate_pool
                if candidate.ranked.unit_id not in selected_ids
                and self._direct_basis_kind(candidate, plan=plan)[0] == kind
            ]
            if not candidates:
                continue
            best = max(
                candidates,
                key=lambda candidate: self._direct_basis_sort_key(
                    candidate,
                    plan=plan,
                ),
            )
            best.support_role = "direct_basis"
            best.mmr_score = self._mmr_score(best, selected)
            self._append_reason(best, "direct_contract_modification_basis")
            self._append_reason(best, f"priority_direct_legal_basis:{kind}")
            selected.append(best)
            selected_ids.add(best.ranked.unit_id)
        return selected

    def _direct_basis_sort_key(
        self,
        candidate: _EvidenceCandidate,
        *,
        plan: QueryPlan | None,
    ) -> tuple[int, int, float, int, str]:
        _, priority = self._direct_basis_kind(candidate, plan=plan)
        return (
            priority,
            self._unit_specificity(candidate.unit),
            candidate.ranked.rerank_score,
            -candidate.ranked.rank,
            candidate.ranked.unit_id,
        )

    def _mmr_sort_key(
        self,
        candidate: _EvidenceCandidate,
        selected: list[_EvidenceCandidate],
    ) -> tuple[float, float, float, int, str]:
        final_score, _ = self._evidence_final_score(
            candidate,
            selected=selected,
        )
        return (
            final_score,
            self._mmr_score(candidate, selected),
            candidate.ranked.rerank_score,
            -candidate.ranked.rank,
            candidate.ranked.unit_id,
        )

    def _mmr_score(
        self,
        candidate: _EvidenceCandidate,
        selected: list[_EvidenceCandidate],
    ) -> float:
        max_similarity = 0.0
        if selected:
            max_similarity = max(
                self._jaccard_similarity(candidate.tokens, selected_candidate.tokens)
                for selected_candidate in selected
            )
        return round(
            self.mmr_lambda * candidate.ranked.rerank_score
            - (1.0 - self.mmr_lambda) * max_similarity,
            6,
        )

    def _compact_contract_modification_selection(
        self,
        selected: list[_EvidenceCandidate],
        candidate_pool: list[_EvidenceCandidate],
        *,
        plan: QueryPlan | None,
    ) -> list[_EvidenceCandidate]:
        if not self._is_contract_modification_query(plan):
            return selected

        target = self._selection_target(candidate_pool, plan=plan)
        required_ids = {
            "contract_modification_agreement_rule",
            "contract_modification_salary_scope",
        }
        required: list[_EvidenceCandidate] = []
        optional: list[_EvidenceCandidate] = []
        for candidate in selected:
            if any(
                self._candidate_covers_requirement(candidate, requirement_id)
                for requirement_id in required_ids
            ):
                required.append(candidate)
            else:
                optional.append(candidate)

        required = self._dedupe_candidates(required)
        optional = self._dedupe_candidates(optional)
        kept: list[_EvidenceCandidate] = []
        for candidate in sorted(
            required,
            key=lambda item: self._evidence_final_sort_key(item, kept),
            reverse=True,
        ):
            if candidate.ranked.unit_id not in {item.ranked.unit_id for item in kept}:
                self._append_reason(candidate, "contract_modification_required_coverage")
                kept.append(candidate)

        salary_parent_ids = {
            candidate.ranked.unit_id
            for candidate in kept
            if self._candidate_covers_requirement(
                candidate,
                "contract_modification_salary_scope",
            )
        }
        salary_children = [
            candidate
            for candidate in optional
            if self._is_salary_target_child(candidate, parent_ids=salary_parent_ids)
        ]
        for candidate in sorted(
            salary_children,
            key=lambda item: self._evidence_final_sort_key(item, kept),
            reverse=True,
        ):
            if len(kept) >= target:
                break
            if candidate.ranked.unit_id not in {item.ranked.unit_id for item in kept}:
                self._append_reason(candidate, "contract_modification_optional_salary_target")
                kept.append(candidate)

        remaining = [
            candidate
            for candidate in optional
            if candidate.ranked.unit_id not in {item.ranked.unit_id for item in kept}
        ]
        remaining.sort(
            key=lambda item: self._evidence_final_sort_key(item, kept),
            reverse=True,
        )
        for candidate in remaining:
            if len(kept) >= target:
                break
            score, breakdown = self._evidence_final_score(candidate, selected=kept)
            if (
                candidate.support_role == "context"
                and breakdown["generic_context_penalty"] >= 0.7
                and len(kept) >= max(2, len(required))
            ):
                continue
            self._append_reason(candidate, "contract_modification_compact_selection")
            kept.append(candidate)

        if len(kept) < len(selected):
            for candidate in kept:
                self._append_reason(candidate, "contract_modification_pruned_generic_context")
        return kept

    def _normalize_contract_modification_roles(
        self,
        selected: list[_EvidenceCandidate],
        *,
        plan: QueryPlan | None,
    ) -> list[_EvidenceCandidate]:
        if not self._is_contract_modification_query(plan):
            return selected

        scope_parent_ids = {
            candidate.ranked.unit_id
            for candidate in selected
            if self._candidate_covers_requirement(
                candidate,
                "contract_modification_salary_scope",
            )
        }
        for candidate in selected:
            original_role = candidate.support_role
            normalized_role = self._normalized_contract_modification_role(
                candidate,
                selected=selected,
                scope_parent_ids=scope_parent_ids,
            )
            if normalized_role == original_role:
                continue
            candidate.support_role = normalized_role
            self._append_reason(
                candidate,
                f"role_normalized:{original_role}->{normalized_role}",
            )
        return selected

    def _normalized_contract_modification_role(
        self,
        candidate: _EvidenceCandidate,
        *,
        selected: list[_EvidenceCandidate],
        scope_parent_ids: set[str],
    ) -> str:
        if self._candidate_covers_requirement(
            candidate,
            "contract_modification_agreement_rule",
        ):
            if self._has_more_specific_requirement_candidate(
                candidate,
                selected=selected,
                requirement_id="contract_modification_agreement_rule",
            ):
                return "context"
            return "direct_basis"
        if self._candidate_covers_requirement(
            candidate,
            "contract_modification_salary_scope",
        ):
            return "condition"
        if self._is_salary_target_child(candidate, parent_ids=scope_parent_ids):
            return "condition"

        haystack = self._haystack(candidate.ranked, candidate.unit)
        if self._is_sanction_support(candidate, haystack):
            return "sanction"
        if self._is_exception_support(candidate, haystack):
            return "exception"
        if candidate.support_role in {"definition", "procedure"}:
            return candidate.support_role
        return "context"

    def _has_more_specific_requirement_candidate(
        self,
        candidate: _EvidenceCandidate,
        *,
        selected: list[_EvidenceCandidate],
        requirement_id: str,
    ) -> bool:
        candidate_specificity = self._unit_specificity(candidate.unit)
        candidate_law = str(candidate.unit.get("law_id") or "")
        candidate_article = self._optional_str(candidate.unit.get("article_number"))
        for other in selected:
            if other.ranked.unit_id == candidate.ranked.unit_id:
                continue
            if not self._candidate_covers_requirement(other, requirement_id):
                continue
            if self._unit_specificity(other.unit) <= candidate_specificity:
                continue
            other_law = str(other.unit.get("law_id") or "")
            other_article = self._optional_str(other.unit.get("article_number"))
            if candidate_law and candidate_article and (
                candidate_law == other_law and candidate_article == other_article
            ):
                return True
            if self._optional_str(other.unit.get("parent_id")) == candidate.ranked.unit_id:
                return True
        return False

    def _is_sanction_support(
        self,
        candidate: _EvidenceCandidate,
        haystack: str,
    ) -> bool:
        score = candidate.ranked.score_breakdown
        if score.is_sanction > 0.0:
            return True
        return self._contains_any(
            haystack,
            (
                "amenda",
                "amenzii",
                "contraventie",
                "contraventionala",
                "sanctiune",
                "sanction",
                "constituie contraventie",
            ),
        )

    def _is_exception_support(
        self,
        candidate: _EvidenceCandidate,
        haystack: str,
    ) -> bool:
        score = candidate.ranked.score_breakdown
        if score.is_exception > 0.0:
            return True
        return self._contains_any(
            haystack,
            (
                "exceptie",
                "exceptii",
                "cu titlu de exceptie",
                "locul muncii poate fi modificat unilateral",
                "poate fi modificat unilateral",
                "delegarea",
                "detasarea",
                "derogare",
            ),
        )

    def _dedupe_candidates(
        self,
        candidates: list[_EvidenceCandidate],
    ) -> list[_EvidenceCandidate]:
        seen: set[str] = set()
        deduped: list[_EvidenceCandidate] = []
        for candidate in candidates:
            if candidate.ranked.unit_id in seen:
                continue
            seen.add(candidate.ranked.unit_id)
            deduped.append(candidate)
        return deduped

    def _evidence_final_sort_key(
        self,
        candidate: _EvidenceCandidate,
        selected: list[_EvidenceCandidate],
    ) -> tuple[float, float, float, int, str]:
        score, _ = self._evidence_final_score(candidate, selected=selected)
        return (
            score,
            candidate.ranked.rerank_score,
            candidate.ranked.retrieval_score or 0.0,
            -candidate.ranked.rank,
            candidate.ranked.unit_id,
        )

    def _evidence_final_score(
        self,
        candidate: _EvidenceCandidate,
        *,
        selected: list[_EvidenceCandidate],
    ) -> tuple[float, dict[str, float]]:
        mmr = self._mmr_score(candidate, selected)
        required_requirement_coverage = self._required_requirement_coverage_score(candidate)
        same_article_as_core = self._same_article_as_core_score(candidate, selected)
        support_role_priority = SUPPORT_ROLE_PRIORITY_SCORE.get(
            candidate.support_role,
            0.35,
        )
        generic_context_penalty = self._generic_context_penalty(candidate, selected)
        distractor_penalty = self._evidence_distractor_penalty(candidate)
        final_score = self._clamp(
            mmr
            + 0.20 * required_requirement_coverage
            + 0.10 * same_article_as_core
            + 0.08 * support_role_priority
            - 0.15 * generic_context_penalty
            - 0.20 * distractor_penalty
        )
        return (
            round(final_score, 6),
            {
                "mmr": round(mmr, 6),
                "required_requirement_coverage": round(required_requirement_coverage, 6),
                "same_article_as_core": round(same_article_as_core, 6),
                "support_role_priority": round(support_role_priority, 6),
                "generic_context_penalty": round(generic_context_penalty, 6),
                "distractor_penalty": round(distractor_penalty, 6),
            },
        )

    def _required_requirement_coverage_score(
        self,
        candidate: _EvidenceCandidate,
    ) -> float:
        if self._candidate_covers_requirement(
            candidate,
            "contract_modification_agreement_rule",
        ):
            return 1.0
        if self._candidate_covers_requirement(
            candidate,
            "contract_modification_salary_scope",
        ):
            return 1.0
        if candidate.role_decision is not None and (
            "salary_target_element" in candidate.role_decision.matched_requirement_ids
        ):
            return 0.35
        return 0.0

    def _same_article_as_core_score(
        self,
        candidate: _EvidenceCandidate,
        selected: list[_EvidenceCandidate],
    ) -> float:
        core_units = [
            item
            for item in selected
            if self._candidate_covers_requirement(
                item,
                "contract_modification_agreement_rule",
            )
            or self._candidate_covers_requirement(
                item,
                "contract_modification_salary_scope",
            )
        ]
        if not core_units:
            return 1.0 if self._required_requirement_coverage_score(candidate) else 0.0

        candidate_law = str(candidate.unit.get("law_id") or "")
        candidate_article = self._optional_str(candidate.unit.get("article_number"))
        candidate_parent = self._optional_str(candidate.unit.get("parent_id"))
        for core in core_units:
            core_law = str(core.unit.get("law_id") or "")
            core_article = self._optional_str(core.unit.get("article_number"))
            if (
                candidate_law
                and candidate_article
                and candidate_law == core_law
                and candidate_article == core_article
            ):
                return 1.0
            if candidate_parent and candidate_parent == core.ranked.unit_id:
                return 0.5
            core_parent = self._optional_str(core.unit.get("parent_id"))
            if core_parent and core_parent == candidate.ranked.unit_id:
                return 0.5
        return 0.0

    def _generic_context_penalty(
        self,
        candidate: _EvidenceCandidate,
        selected: list[_EvidenceCandidate],
    ) -> float:
        if self._required_requirement_coverage_score(candidate) > 0:
            return 0.0
        haystack = self._haystack(candidate.ranked, candidate.unit)
        same_article = self._same_article_as_core_score(candidate, selected)
        has_core = (
            self._is_contract_modification_agreement_rule(haystack)
            or self._is_contract_modification_salary_scope(haystack)
            or self._is_contract_modification_scope_parent(haystack)
        )
        if has_core:
            return 0.0
        if candidate.support_role == "context" and same_article == 0.0:
            return 1.0
        if self._contains_any(haystack, CONTRACT_MODIFICATION_GENERIC_CONTEXT_TERMS):
            return 0.8 if same_article == 0.0 else 0.4
        if (
            self._contains_any(haystack, CONTRACT_MODIFICATION_TARGET_TERMS)
            and same_article == 0.0
        ):
            return 0.7
        if candidate.support_role == "context":
            return 0.5
        return 0.0

    def _evidence_distractor_penalty(
        self,
        candidate: _EvidenceCandidate,
    ) -> float:
        haystack = self._haystack(candidate.ranked, candidate.unit)
        score = candidate.ranked.score_breakdown
        if score.distractor_penalty >= 0.7:
            return 1.0
        if self._is_salary_context(haystack):
            return 0.0 if self._required_requirement_coverage_score(candidate) else 1.0
        if (
            self._contains_any(haystack, CONTRACT_MODIFICATION_TARGET_TERMS)
            and not self._required_requirement_coverage_score(candidate)
        ):
            return 0.5
        return 0.0

    def _ensure_role_candidate(
        self,
        selected: list[_EvidenceCandidate],
        candidate_pool: list[_EvidenceCandidate],
        role: str,
    ) -> None:
        if len(selected) >= self.max_evidence_units:
            return
        if any(candidate.support_role == role for candidate in selected):
            return
        candidates = [
            candidate
            for candidate in candidate_pool
            if candidate.support_role == role
            and candidate.ranked.unit_id not in {item.ranked.unit_id for item in selected}
        ]
        if not candidates:
            return
        best = max(
            candidates,
            key=lambda candidate: (
                candidate.ranked.rerank_score,
                -candidate.ranked.rank,
                candidate.ranked.unit_id,
            ),
        )
        best.mmr_score = self._mmr_score(best, selected)
        self._append_reason(best, f"{role}_candidate")
        self._append_reason(best, "selected_by_role_coverage")
        selected.append(best)

    def _include_parent_context(
        self,
        selected: list[_EvidenceCandidate],
        candidate_pool: list[_EvidenceCandidate],
    ) -> None:
        selected_unit_ids = {candidate.ranked.unit_id for candidate in selected}
        pool_by_unit_id = {
            candidate.ranked.unit_id: candidate
            for candidate in candidate_pool
        }
        for candidate in list(selected):
            if len(selected) >= self.max_evidence_units:
                return
            parent_id = candidate.unit.get("parent_id")
            if not isinstance(parent_id, str) or not parent_id:
                continue
            if parent_id in selected_unit_ids:
                continue
            parent = pool_by_unit_id.get(parent_id)
            if parent is None:
                continue
            if self._candidate_covers_requirement(
                parent,
                "contract_modification_salary_scope",
            ):
                if parent.support_role == "context":
                    parent.support_role = "condition"
                self._append_reason(
                    parent,
                    "parent_context_requirement:contract_modification_salary_scope",
                )
            else:
                parent.support_role = "context"
            parent.mmr_score = self._mmr_score(parent, selected)
            self._append_reason(parent, "parent_context_candidate")
            self._append_reason(parent, "selected_parent_context")
            selected.append(parent)
            selected_unit_ids.add(parent_id)

    def _evidence_unit(
        self,
        candidate: _EvidenceCandidate,
        *,
        rank: int,
    ) -> EvidenceUnit:
        ranked = candidate.ranked
        legal_unit = self._legal_unit_from_dict(
            unit=candidate.unit,
            fallback_unit_id=ranked.unit_id,
            text=candidate.text,
        )
        return EvidenceUnit(
            **legal_unit.model_dump(),
            evidence_id=f"evidence:{ranked.unit_id}",
            excerpt=candidate.text,
            rank=rank,
            relevance_score=self._clamp(ranked.rerank_score),
            retrieval_method=ranked.source or "legal_ranker_mmr",
            retrieval_score=ranked.retrieval_score or 0.0,
            rerank_score=self._clamp(ranked.rerank_score),
            mmr_score=candidate.mmr_score,
            support_role=candidate.support_role,
            why_selected=self._dedupe(candidate.why_selected),
            score_breakdown=ranked.score_breakdown.model_dump(mode="json"),
            warnings=self._dedupe(candidate.warnings),
        )

    def _legal_unit_from_dict(
        self,
        *,
        unit: dict[str, Any],
        fallback_unit_id: str,
        text: str,
    ) -> LegalUnit:
        law_id = str(unit.get("law_id") or unit.get("law") or "unknown")
        law_title = str(
            unit.get("law_title")
            or unit.get("legal_act")
            or unit.get("act_title")
            or law_id
        )
        status = str(unit.get("status") or "unknown")
        if status not in {"active", "historical", "repealed", "unknown"}:
            status = "unknown"
        hierarchy_path = unit.get("hierarchy_path")
        if not isinstance(hierarchy_path, list) or not hierarchy_path:
            hierarchy_path = [law_title, fallback_unit_id]
        return LegalUnit(
            id=str(unit.get("id") or unit.get("legal_unit_id") or fallback_unit_id),
            canonical_id=unit.get("canonical_id"),
            source_id=unit.get("source_id"),
            law_id=law_id,
            law_title=law_title,
            act_type=unit.get("act_type"),
            act_number=unit.get("act_number"),
            publication_date=unit.get("publication_date"),
            effective_date=unit.get("effective_date"),
            version_start=unit.get("version_start"),
            version_end=unit.get("version_end"),
            status=status,
            hierarchy_path=[str(item) for item in hierarchy_path],
            article_number=self._optional_str(unit.get("article_number")),
            paragraph_number=self._optional_str(unit.get("paragraph_number")),
            letter_number=self._optional_str(unit.get("letter_number")),
            point_number=self._optional_str(unit.get("point_number")),
            raw_text=text,
            normalized_text=self._optional_str(unit.get("normalized_text")),
            legal_domain=str(unit.get("legal_domain") or unit.get("domain") or "unknown"),
            legal_concepts=[
                str(concept)
                for concept in unit.get("legal_concepts", [])
                if isinstance(unit.get("legal_concepts"), list)
            ],
            source_url=unit.get("source_url"),
            parent_id=self._optional_str(unit.get("parent_id")),
            children_ids=self._str_list(unit.get("children_ids")),
            outgoing_reference_ids=self._str_list(unit.get("outgoing_reference_ids")),
            incoming_reference_ids=self._str_list(unit.get("incoming_reference_ids")),
        )

    def _build_graph(
        self,
        *,
        selected: list[_EvidenceCandidate],
        graph_expansion: GraphExpansionResult | None,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        nodes_by_id: dict[str, GraphNode] = {}
        edges_by_id: dict[str, GraphEdge] = {}

        if graph_expansion:
            for node in graph_expansion.graph_nodes:
                nodes_by_id[node.id] = node
            for edge in graph_expansion.graph_edges:
                edges_by_id[edge.id] = edge

        for candidate in selected:
            node = self._graph_node(candidate)
            nodes_by_id.setdefault(node.id, node)

        selected_ids = {candidate.ranked.unit_id for candidate in selected}
        for candidate in selected:
            parent_id = candidate.unit.get("parent_id")
            if not isinstance(parent_id, str) or parent_id not in selected_ids:
                continue
            edge_id = f"edge:parent:{parent_id}:{candidate.ranked.unit_id}"
            edges_by_id.setdefault(
                edge_id,
                GraphEdge(
                    id=edge_id,
                    source=f"legal_unit:{parent_id}",
                    target=f"legal_unit:{candidate.ranked.unit_id}",
                    type="contains",
                    weight=1.0,
                    confidence=1.0,
                    explanation="Parent-child relation derived from input parent_id.",
                    metadata={"derived_from_input_parent_id": True},
                ),
            )

        return list(nodes_by_id.values()), list(edges_by_id.values())

    def _graph_node(self, candidate: _EvidenceCandidate) -> GraphNode:
        unit = candidate.unit
        unit_id = candidate.ranked.unit_id
        label = (
            unit.get("label")
            or unit.get("title")
            or unit.get("law_title")
            or unit.get("id")
            or unit_id
        )
        return GraphNode(
            id=f"legal_unit:{unit_id}",
            type=self._graph_node_type(unit),
            label=str(label),
            legal_unit_id=unit_id,
            domain=self._optional_str(unit.get("legal_domain") or unit.get("domain")),
            status=self._optional_str(unit.get("status")),
            importance=candidate.ranked.rerank_score,
            metadata={
                **self._scalar_metadata(unit),
                "support_role": candidate.support_role,
                "rerank_score": candidate.ranked.rerank_score,
            },
        )

    def _support_role(
        self,
        ranked: RankedCandidate,
        unit: dict[str, Any],
        *,
        plan: QueryPlan | None,
    ) -> str:
        score = ranked.score_breakdown
        haystack = self._haystack(ranked, unit)
        contract_modification_query = self._is_contract_modification_query(plan)
        if contract_modification_query:
            if score.distractor_penalty >= 0.5:
                return "context"
            if score.core_issue_score >= 0.70 and score.distractor_penalty < 0.5:
                return "direct_basis"
            if self._contains_any(
                haystack,
                (
                    "exceptie",
                    "exceptii",
                    "cu titlu de exceptie",
                    "locul muncii poate fi modificat unilateral",
                    "unilateral",
                    "delegarea",
                    "detasarea",
                ),
            ):
                return "exception"
            return "context"
        if "support_role_hint:context" in ranked.why_ranked:
            return "context"
        if "support_role_hint:direct_basis" in ranked.why_ranked:
            return "direct_basis"
        if "support_role_hint:definition" in ranked.why_ranked:
            return "definition"
        if "support_role_hint:sanction" in ranked.why_ranked:
            return "sanction"
        if "support_role_hint:procedure" in ranked.why_ranked:
            return "procedure"
        if "support_role_hint:exception" in ranked.why_ranked:
            return "exception"
        if score.distractor_penalty >= 0.7:
            return "context"
        if score.core_issue_score >= 0.70 and score.distractor_penalty < 0.5:
            return "direct_basis"
        if score.target_object_score > 0 and score.core_issue_score < 0.25:
            return "context"

        if contract_modification_query and self._is_contract_modification_direct_basis(haystack):
            return "direct_basis"
        if self._contains_any(haystack, ("exception", "exceptie", "except", "cu exceptia", "exception_to")):
            return "exception"
        if self._contains_any(haystack, ("definition", "defineste", "in sensul", "se intelege", "defines")):
            return "definition"
        if self._contains_any(haystack, ("sanction", "sanctiune", "amenda", "contraventie", "sanctions")):
            return "sanction"
        if (plan and "procedure" in plan.query_types) or self._contains_any(
            haystack,
            ("procedura", "termen", "pasi", "contestatie", "cerere", "procedure_step"),
        ):
            return "procedure"
        if self._contains_any(haystack, ("daca", "in cazul", "cu conditia", "numai daca")):
            return "condition"
        if contract_modification_query and self._is_salary_context(haystack):
            return "context"
        if self._graph_node_type(unit) in {"root", "domain", "legal_act", "title", "chapter", "section"}:
            return "context"
        if contract_modification_query:
            return "context"
        return "direct_basis"

    def _is_contract_modification_query(self, plan: QueryPlan | None) -> bool:
        if plan is None:
            return False
        if self._normalize_text(str(plan.legal_domain or "")) != "munca":
            return False

        query = self._normalize_text(
            " ".join(
                [
                    plan.question,
                    plan.normalized_question,
                    " ".join(plan.query_types),
                    str(plan.retrieval_filters),
                ]
            )
        )
        if any(self._normalize_text(term) in query for term in CONTRACT_MODIFICATION_TRIGGERS):
            return True
        tokens = set(query.split())
        if {"act", "aditional"}.issubset(tokens):
            return True
        has_reduction = any(term in tokens for term in CONTRACT_MODIFICATION_REDUCTION_TERMS)
        has_target = any(term in tokens for term in CONTRACT_MODIFICATION_TARGET_TERMS)
        return has_reduction and has_target

    def _is_contract_modification_direct_basis(self, haystack: str) -> bool:
        agreement = self._is_contract_modification_agreement_rule(haystack)
        scope = any(
            self._normalize_text(term) in haystack
            for term in MODIFICATION_SCOPE_TERMS
        )
        if agreement:
            return True
        return scope and self._contains_any(
            haystack,
            ("modificarea", "modificare", "poate privi", "salariul", "elementele"),
        )

    def _is_contract_modification_agreement_rule(self, haystack: str) -> bool:
        return any(
            self._normalize_text(term) in haystack
            for term in AGREEMENT_RULE_TERMS
        )

    def _is_contract_modification_scope_parent(self, haystack: str) -> bool:
        scope = any(
            self._normalize_text(term) in haystack
            for term in MODIFICATION_SCOPE_TERMS
        )
        return scope and self._contains_any(
            haystack,
            (
                "modificarea",
                "modificare",
                "se refera",
                "poate privi",
                "elemente",
                "elementele",
                "urmatoarele elemente",
            ),
        )

    def _is_contract_modification_salary_scope(self, haystack: str) -> bool:
        return self._is_contract_modification_scope_parent(
            haystack
        ) and self._contains_any(haystack, CONTRACT_MODIFICATION_TARGET_TERMS)

    def _is_salary_target_child(
        self,
        candidate: _EvidenceCandidate,
        *,
        parent_ids: set[str],
    ) -> bool:
        parent_id = str(candidate.unit.get("parent_id") or "")
        if not parent_id or parent_id not in parent_ids:
            return False
        if not (candidate.unit.get("letter_number") or candidate.unit.get("point_number")):
            return False
        haystack = self._haystack(candidate.ranked, candidate.unit)
        if not self._contains_any(haystack, CONTRACT_MODIFICATION_TARGET_TERMS):
            return False
        return not self._is_salary_context(haystack)

    def _is_contract_modification_condition(self, haystack: str) -> bool:
        return (
            self._contains_any(
                haystack,
                (
                    "impune incheierea unui act aditional",
                    "incheierea unui act aditional",
                ),
            )
            and self._contains_any(
                haystack,
                (
                    "elementele prevazute",
                    "modificare",
                    "modificarea",
                ),
            )
            and not self._contains_any(
                haystack,
                (
                    "formare profesionala",
                    "durata formarii profesionale",
                ),
            )
        )

    def _is_salary_context(self, haystack: str) -> bool:
        return any(
            self._normalize_text(term) in haystack
            for term in SALARY_CONTEXT_TERMS
        )

    def _direct_basis_kind(
        self,
        candidate: _EvidenceCandidate,
        *,
        plan: QueryPlan | None,
    ) -> tuple[str | None, int]:
        if not self._is_contract_modification_query(plan):
            return (None, 0)

        if candidate.role_decision is not None and candidate.support_role == "direct_basis":
            matched_ids = set(candidate.role_decision.matched_requirement_ids)
            if "contract_modification_agreement_rule" in matched_ids:
                return (
                    "agreement_rule",
                    110 + self._unit_specificity(candidate.unit),
                )
            if "contract_modification_salary_scope" in matched_ids:
                return (
                    "modification_scope",
                    95 + self._unit_specificity(candidate.unit),
                )

        haystack = self._haystack(candidate.ranked, candidate.unit)
        has_agreement = any(
            self._normalize_text(term) in haystack
            for term in AGREEMENT_RULE_TERMS
        )
        has_contract_modification = any(
            self._normalize_text(term) in haystack
            for term in MODIFICATION_SCOPE_TERMS
        )
        has_target = self._contains_any(haystack, CONTRACT_MODIFICATION_TARGET_TERMS)
        has_scope_marker = self._contains_any(
            haystack,
            ("modificarea", "modificare", "poate privi", "elementele"),
        )

        if has_agreement:
            priority = 100
            if "numai prin acordul partilor" in haystack:
                priority += 10
            return ("agreement_rule", priority)

        if has_contract_modification and has_scope_marker:
            priority = 80
            if has_target:
                priority += 15
            if "poate privi" in haystack:
                priority += 5
            return ("modification_scope", priority)

        return (None, 0)

    def _unit_specificity(self, unit: dict[str, Any]) -> int:
        if unit.get("point_number"):
            return 5
        if unit.get("letter_number"):
            return 4
        if unit.get("paragraph_number"):
            return 3
        if unit.get("article_number"):
            return 2
        return 1

    def _base_selection_reasons(self, ranked: RankedCandidate) -> list[str]:
        reasons = list(ranked.why_ranked)
        if ranked.rerank_score >= 0.70:
            reasons.append("high_rerank_score")
        return reasons

    def _role_decision(
        self,
        *,
        ranked: RankedCandidate,
        unit: dict[str, Any],
        query_frame: QueryFrame | None,
    ) -> CandidateRoleDecision | None:
        if not self._should_use_role_classifier(query_frame):
            return None
        return self.role_classifier.classify(
            query_frame=query_frame,
            unit_id=ranked.unit_id,
            unit=unit,
            ranked_score_breakdown=ranked.score_breakdown.model_dump(mode="json"),
            existing_why_ranked=ranked.why_ranked,
        )

    def _should_use_role_classifier(self, query_frame: QueryFrame | None) -> bool:
        return (
            query_frame is not None
            and query_frame.confidence >= 0.35
            and "labor_contract_modification" in query_frame.intents
        )

    def _debug_payload(
        self,
        *,
        fallback_used: bool,
        input_ranked_candidate_count: int,
        candidate_pool_size: int,
        selected: list[_EvidenceCandidate],
        plan: QueryPlan | None,
        query_frame: QueryFrame | None,
        warnings: list[str],
    ) -> dict[str, Any]:
        return {
            "fallback_used": fallback_used,
            "input_ranked_candidate_count": input_ranked_candidate_count,
            "candidate_pool_size": candidate_pool_size,
            "selected_evidence_count": len(selected),
            "lambda": self.mmr_lambda,
            "target_evidence_units": self.target_evidence_units,
            "max_evidence_units": self.max_evidence_units,
            "selected_units": [
                {
                    "unit_id": candidate.ranked.unit_id,
                    "rerank_score": candidate.ranked.rerank_score,
                    "mmr_score": candidate.mmr_score,
                    "evidence_final_score": self._evidence_final_score(
                        candidate,
                        selected=selected,
                    )[0],
                    "evidence_final_score_breakdown": self._evidence_final_score(
                        candidate,
                        selected=selected,
                    )[1],
                    "support_role": candidate.support_role,
                    "why_selected": self._dedupe(candidate.why_selected),
                    "matched_requirement_ids": (
                        candidate.role_decision.matched_requirement_ids
                        if candidate.role_decision is not None
                        else []
                    ),
                    "role_decision": candidate.role_decision.model_dump(mode="json")
                    if candidate.role_decision is not None
                    else None,
                }
                for candidate in selected
            ],
            "requirement_coverage": self._requirement_coverage(
                selected,
                plan=plan,
                query_frame=query_frame,
            ),
            "warnings": warnings,
        }

    def _requirement_coverage(
        self,
        selected: list[_EvidenceCandidate],
        *,
        plan: QueryPlan | None,
        query_frame: QueryFrame | None,
    ) -> dict[str, Any]:
        required_ids: list[str] = []
        if self._should_use_role_classifier(query_frame):
            required_ids = [
                "contract_modification_agreement_rule",
                "contract_modification_salary_scope",
            ]
        elif self._is_contract_modification_query(plan):
            required_ids = [
                "contract_modification_agreement_rule",
                "contract_modification_salary_scope",
            ]
        if not required_ids:
            return {
                "enabled": False,
                "intent_id": None,
                "required_requirement_ids": [],
                "required_requirements_total": 0,
                "required_requirements_covered": 0,
                "covered_requirement_ids": [],
                "missing_required_requirements": [],
                "missing_required_requirement_ids": [],
                "coverage_passed": True,
            }

        covered: list[str] = []
        if self._contract_modification_agreement_covered(selected):
            covered.append("contract_modification_agreement_rule")
        if self._contract_modification_salary_scope_covered(selected):
            covered.append("contract_modification_salary_scope")
        missing = [
            requirement_id
            for requirement_id in required_ids
            if requirement_id not in covered
        ]
        return {
            "enabled": True,
            "intent_id": "labor_contract_modification",
            "required_requirement_ids": required_ids,
            "required_requirements_total": len(required_ids),
            "required_requirements_covered": len(covered),
            "covered_requirement_ids": covered,
            "missing_required_requirements": missing,
            "missing_required_requirement_ids": missing,
            "coverage_passed": not missing,
        }

    def _contract_modification_agreement_covered(
        self,
        selected: list[_EvidenceCandidate],
    ) -> bool:
        for candidate in selected:
            if candidate.role_decision is not None and (
                "contract_modification_agreement_rule"
                in candidate.role_decision.matched_requirement_ids
            ):
                return True
            if self._is_contract_modification_agreement_rule(
                self._haystack(candidate.ranked, candidate.unit)
            ):
                return True
        return False

    def _contract_modification_salary_scope_covered(
        self,
        selected: list[_EvidenceCandidate],
    ) -> bool:
        for candidate in selected:
            if candidate.role_decision is not None and (
                "contract_modification_salary_scope"
                in candidate.role_decision.matched_requirement_ids
            ):
                return True
            if self._is_contract_modification_salary_scope(
                self._haystack(candidate.ranked, candidate.unit)
            ):
                return True

        scope_parent_ids = {
            candidate.ranked.unit_id
            for candidate in selected
            if self._is_contract_modification_scope_parent(
                self._haystack(candidate.ranked, candidate.unit)
            )
        }
        if not scope_parent_ids:
            return False
        return any(
            self._is_salary_target_child(candidate, parent_ids=scope_parent_ids)
            for candidate in selected
        )

    def _candidate_text(self, unit: dict[str, Any]) -> str:
        value = unit.get("raw_text")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return ""

    def _has_required_metadata(self, unit: dict[str, Any]) -> bool:
        return all(unit.get(key) for key in ("law_id", "law_title", "legal_domain"))

    def _graph_node_type(self, unit: dict[str, Any]) -> str:
        unit_type = self._normalize_text(str(unit.get("type") or unit.get("unit_type") or ""))
        mapping = {
            "root": "root",
            "domain": "domain",
            "subdomain": "subdomain",
            "legal_act": "legal_act",
            "act": "legal_act",
            "title": "title",
            "titlu": "title",
            "chapter": "chapter",
            "capitol": "chapter",
            "section": "section",
            "sectiune": "section",
            "article": "article",
            "articol": "article",
            "paragraph": "paragraph",
            "paragraf": "paragraph",
            "alineat": "paragraph",
            "letter": "letter",
            "litera": "letter",
            "point": "point",
            "punct": "point",
            "annex": "annex",
            "anexa": "annex",
            "concept": "concept",
        }
        if unit_type in mapping:
            return mapping[unit_type]
        if unit.get("paragraph_number"):
            return "paragraph"
        if unit.get("letter_number"):
            return "letter"
        if unit.get("point_number"):
            return "point"
        if unit.get("article_number"):
            return "article"
        return "article"

    def _haystack(self, ranked: RankedCandidate, unit: dict[str, Any]) -> str:
        values: list[str] = [ranked.unit_id]
        values.extend(ranked.why_ranked)
        values.append(str(ranked.source or ""))
        values.append(str(unit))
        return self._normalize_text(" ".join(values))

    def _contains_any(self, haystack: str, terms: tuple[str, ...]) -> bool:
        return any(self._normalize_text(term) in haystack for term in terms)

    def _tokenize(self, text: str) -> set[str]:
        normalized = self._normalize_text(text)
        return {
            token
            for token in re.split(r"[^a-z0-9_]+", normalized)
            if len(token) > 1 and token not in STOPWORDS
        }

    def _jaccard_similarity(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    def _scalar_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in metadata.items()
            if isinstance(value, str | int | float | bool) or value is None
        }

    def _append_reason(self, candidate: _EvidenceCandidate, reason: str) -> None:
        if reason not in candidate.why_selected:
            candidate.why_selected.append(reason)

    def _dedupe(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped

    def _str_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value]

    def _optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

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
        stripped = re.sub(r"\badi[?\ufffd]ional(a?)\b", r"aditional\1", stripped)
        return " ".join(stripped.replace(".", " ").replace("-", "_").split())

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))
