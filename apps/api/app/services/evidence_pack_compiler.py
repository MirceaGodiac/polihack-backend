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


@dataclass
class _EvidenceCandidate:
    ranked: RankedCandidate
    unit: dict[str, Any]
    text: str
    tokens: set[str]
    support_role: str
    why_selected: list[str] = field(default_factory=list)
    mmr_score: float = 0.0


class EvidencePackCompiler:
    def __init__(
        self,
        mmr_lambda: float = 0.75,
        candidate_pool_size: int = 40,
        target_evidence_units: int = 12,
        max_evidence_units: int = 14,
    ) -> None:
        self.mmr_lambda = mmr_lambda
        self.candidate_pool_size = candidate_pool_size
        self.target_evidence_units = target_evidence_units
        self.max_evidence_units = max_evidence_units

    def compile(
        self,
        *,
        ranked_candidates: list[RankedCandidate],
        graph_expansion: GraphExpansionResult | None = None,
        plan: QueryPlan | None = None,
        debug: bool = False,
    ) -> EvidencePackResult:
        if not ranked_candidates:
            return self._fallback_result(debug=debug)

        warnings: list[str] = []
        unique_ranked = self._dedupe_ranked_candidates(ranked_candidates)
        candidate_pool = self._candidate_pool(unique_ranked, plan=plan, warnings=warnings)
        if not candidate_pool:
            warnings.append(EVIDENCE_PACK_NO_RANKED_CANDIDATES)
            return self._empty_result(
                debug=debug,
                input_ranked_candidate_count=len(ranked_candidates),
                warnings=self._dedupe(warnings),
            )

        selected = self._select_with_mmr(candidate_pool, plan=plan)
        self._ensure_role_candidate(selected, candidate_pool, "exception")
        self._ensure_role_candidate(selected, candidate_pool, "definition")
        self._ensure_role_candidate(selected, candidate_pool, "sanction")
        self._include_parent_context(selected, candidate_pool)

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
            pool.append(
                _EvidenceCandidate(
                    ranked=ranked,
                    unit=unit,
                    text=text,
                    tokens=self._tokenize(text),
                    support_role=self._support_role(ranked, unit, plan=plan),
                    why_selected=self._base_selection_reasons(ranked),
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
        target = min(self.target_evidence_units, len(candidate_pool))
        selected = self._priority_direct_basis_candidates(
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
    ) -> tuple[float, float, int, str]:
        return (
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
        haystack = self._haystack(ranked, unit)
        contract_modification_query = self._is_contract_modification_query(plan)
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
        agreement = any(
            self._normalize_text(term) in haystack
            for term in AGREEMENT_RULE_TERMS
        )
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

    def _debug_payload(
        self,
        *,
        fallback_used: bool,
        input_ranked_candidate_count: int,
        candidate_pool_size: int,
        selected: list[_EvidenceCandidate],
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
                    "support_role": candidate.support_role,
                    "why_selected": self._dedupe(candidate.why_selected),
                }
                for candidate in selected
            ],
            "warnings": warnings,
        }

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
        return " ".join(stripped.replace(".", " ").replace("-", "_").split())

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))
