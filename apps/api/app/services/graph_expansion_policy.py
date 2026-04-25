import math
import unicodedata
from typing import Any

from ..schemas import (
    ExpandedCandidate,
    GraphExpansionResult,
    GraphNode,
    QueryPlan,
    RawRetrievalResponse,
    RetrievalCandidate,
)

GRAPH_EXPANSION_NO_SEED_CANDIDATES = "graph_expansion_no_seed_candidates"
GRAPH_EXPANSION_NOT_CONFIGURED = "graph_expansion_not_configured"
GRAPH_EXPANSION_NEIGHBORS_UNAVAILABLE = "graph_expansion_neighbors_unavailable"

DEFAULT_ALLOWED_EDGE_TYPES = [
    "contains_parent",
    "contains_child",
    "references",
    "defines",
    "exception_to",
    "sanctions",
    "creates_obligation",
    "creates_right",
    "creates_prohibition",
    "procedure_step",
]

EDGE_TYPE_WEIGHTS = {
    "contains_parent": 0.90,
    "contains_child": 0.75,
    "references": 0.85,
    "defines": 0.80,
    "exception_to": 0.95,
    "sanctions": 0.85,
    "creates_obligation": 0.90,
    "creates_right": 0.90,
    "creates_prohibition": 0.90,
    "procedure_step": 0.80,
    "semantically_related": 0.45,
}


class GraphExpansionPolicy:
    def __init__(
        self,
        max_depth: int = 2,
        max_expanded_nodes: int = 80,
        lambda_decay: float = 0.7,
        neighbors_client: Any | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.max_expanded_nodes = max_expanded_nodes
        self.lambda_decay = lambda_decay
        self.neighbors_client = neighbors_client

    async def expand(
        self,
        *,
        plan: QueryPlan,
        retrieval_response: RawRetrievalResponse,
        debug: bool = False,
    ) -> GraphExpansionResult:
        seeds = [
            self._seed_candidate(candidate)
            for candidate in retrieval_response.candidates[: self.max_expanded_nodes]
        ]

        if not seeds:
            return self._result(
                plan=plan,
                seed_candidates=[],
                expanded_candidates=[],
                graph_nodes=[],
                warning=GRAPH_EXPANSION_NO_SEED_CANDIDATES,
                fallback_used=True,
                reason="graph expansion has no seed candidates",
                debug=debug,
            )

        if not self._neighbors_client_configured():
            return self._result(
                plan=plan,
                seed_candidates=seeds,
                expanded_candidates=seeds,
                graph_nodes=self._seed_graph_nodes(retrieval_response.candidates),
                warning=GRAPH_EXPANSION_NOT_CONFIGURED,
                fallback_used=True,
                reason="graph neighbors endpoint is not configured",
                debug=debug,
            )

        return self._result(
            plan=plan,
            seed_candidates=seeds,
            expanded_candidates=seeds,
            graph_nodes=self._seed_graph_nodes(retrieval_response.candidates),
            warning=GRAPH_EXPANSION_NEIGHBORS_UNAVAILABLE,
            fallback_used=True,
            reason="graph neighbors endpoint request failed",
            debug=debug,
        )

    def policy_for_plan(self, plan: QueryPlan) -> dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "max_expanded_nodes": self.max_expanded_nodes,
            "lambda_decay": self.lambda_decay,
            "allowed_edge_types": self.allowed_edge_types(plan),
            "edge_type_weights": EDGE_TYPE_WEIGHTS,
            "priority_edge_types": self.priority_edge_types(plan),
        }

    def allowed_edge_types(self, plan: QueryPlan) -> list[str]:
        allowed = list(DEFAULT_ALLOWED_EDGE_TYPES)
        if plan.exact_citations:
            return [
                edge_type
                for edge_type in allowed
                if edge_type != "semantically_related"
            ]
        return allowed

    def priority_edge_types(self, plan: QueryPlan) -> list[str]:
        match_text = self._match_text(plan.normalized_question)
        consequence_query = any(
            term in match_text
            for term in (
                "poate",
                "are voie",
                "ce se intampla daca",
                "fara",
                "amenda",
            )
        )

        priorities: list[str] = []
        if consequence_query or any(
            query_type in plan.query_types
            for query_type in ("exception", "sanction", "prohibition", "obligation")
        ):
            priorities.extend(
                [
                    "exception_to",
                    "sanctions",
                    "creates_obligation",
                    "creates_prohibition",
                ]
            )
        if "right" in plan.query_types:
            priorities.append("creates_right")
        if "definition" in plan.query_types or plan.ambiguity_flags:
            priorities.append("defines")
        if "procedure" in plan.query_types:
            priorities.append("procedure_step")

        return self._dedupe_allowed(priorities, self.allowed_edge_types(plan))

    def graph_proximity(self, edge_type: str, distance: int) -> float:
        edge_weight = EDGE_TYPE_WEIGHTS.get(edge_type, 0.0)
        return edge_weight * math.exp(-self.lambda_decay * distance)

    def _seed_candidate(self, candidate: RetrievalCandidate) -> ExpandedCandidate:
        return ExpandedCandidate(
            unit_id=candidate.unit_id,
            source="seed",
            graph_distance=0,
            graph_proximity=1.0,
            retrieval_candidate=candidate,
            score_breakdown=candidate.score_breakdown,
        )

    def _seed_graph_nodes(
        self,
        candidates: list[RetrievalCandidate],
    ) -> list[GraphNode]:
        nodes: list[GraphNode] = []
        seen: set[str] = set()
        for candidate in candidates:
            if not candidate.unit or candidate.unit_id in seen:
                continue
            seen.add(candidate.unit_id)
            label = self._node_label(candidate)
            nodes.append(
                GraphNode(
                    id=f"legal_unit:{candidate.unit_id}",
                    type=self._node_type(candidate.unit),
                    label=label,
                    legal_unit_id=candidate.unit_id,
                    domain=self._optional_str(
                        candidate.unit.get("legal_domain")
                        or candidate.unit.get("domain")
                    ),
                    status=self._optional_str(candidate.unit.get("status")),
                    metadata={
                        **self._scalar_metadata(candidate.unit),
                        "legal_unit_id": candidate.unit_id,
                        "seed": True,
                    },
                )
            )
        return nodes

    def _node_label(self, candidate: RetrievalCandidate) -> str:
        if not candidate.unit:
            return candidate.unit_id
        for key in ("label", "title", "legal_unit_id", "id"):
            value = candidate.unit.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return candidate.unit_id

    def _scalar_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in metadata.items()
            if isinstance(value, str | int | float | bool) or value is None
        }

    def _node_type(self, unit: dict[str, Any]) -> str:
        unit_type = self._match_text(
            str(unit.get("type") or unit.get("unit_type") or "")
        )
        mapping = {
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
        return "article"

    def _optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    def _neighbors_client_configured(self) -> bool:
        if self.neighbors_client is None:
            return False
        return bool(
            getattr(self.neighbors_client, "is_configured", False)
            or getattr(self.neighbors_client, "configured", False)
        )

    def _result(
        self,
        *,
        plan: QueryPlan,
        seed_candidates: list[ExpandedCandidate],
        expanded_candidates: list[ExpandedCandidate],
        graph_nodes: list[GraphNode],
        warning: str,
        fallback_used: bool,
        reason: str,
        debug: bool,
    ) -> GraphExpansionResult:
        result = GraphExpansionResult(
            seed_candidates=seed_candidates,
            expanded_candidates=expanded_candidates,
            graph_nodes=graph_nodes,
            graph_edges=[],
            warnings=[warning],
            debug=None,
        )
        if debug:
            result.debug = {
                "fallback_used": fallback_used,
                "reason": reason,
                "policy": self.policy_for_plan(plan),
                "seed_candidate_count": len(seed_candidates),
                "expanded_candidate_count": len(expanded_candidates),
                "expanded_candidates": [
                    candidate.model_dump(mode="json")
                    for candidate in expanded_candidates
                ],
                "graph_node_count": len(graph_nodes),
                "graph_edge_count": 0,
                "warnings": result.warnings,
            }
        return result

    def _dedupe_allowed(
        self,
        edge_types: list[str],
        allowed_edge_types: list[str],
    ) -> list[str]:
        priorities: list[str] = []
        for edge_type in edge_types:
            if edge_type in allowed_edge_types and edge_type not in priorities:
                priorities.append(edge_type)
        return priorities

    def _match_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFD", text)
        stripped = "".join(
            char for char in normalized if unicodedata.category(char) != "Mn"
        )
        return stripped.casefold()
