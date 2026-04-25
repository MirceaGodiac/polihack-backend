import math
import inspect
import unicodedata
from typing import Any

from ..schemas import (
    ExpandedCandidate,
    GraphEdge,
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

        try:
            return await self._expand_with_neighbors(
                plan=plan,
                retrieval_response=retrieval_response,
                seeds=seeds,
                debug=debug,
            )
        except Exception:
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

    async def _expand_with_neighbors(
        self,
        *,
        plan: QueryPlan,
        retrieval_response: RawRetrievalResponse,
        seeds: list[ExpandedCandidate],
        debug: bool,
    ) -> GraphExpansionResult:
        allowed_edge_types = set(self.allowed_edge_types(plan))
        expanded_by_id = {seed.unit_id: seed for seed in seeds}
        graph_nodes_by_id = {
            node.id: node for node in self._seed_graph_nodes(retrieval_response.candidates)
        }
        graph_edges_by_id: dict[str, GraphEdge] = {}

        for seed in seeds:
            records = await self._neighbor_records(
                seed.unit_id,
                allowed_edge_types=sorted(allowed_edge_types),
            )
            for record in records:
                if not isinstance(record, dict):
                    continue
                edge_type = self._neighbor_edge_type(record)
                if edge_type not in allowed_edge_types:
                    continue
                neighbor_id = self._neighbor_unit_id(seed.unit_id, record)
                if not neighbor_id or neighbor_id == seed.unit_id:
                    continue
                distance = self._neighbor_distance(record)
                proximity = self.graph_proximity(edge_type, distance)

                expanded_by_id.setdefault(
                    neighbor_id,
                    ExpandedCandidate(
                        unit_id=neighbor_id,
                        source="graph_expansion",
                        graph_distance=distance,
                        graph_proximity=proximity,
                        expansion_edge_type=edge_type,
                        expansion_reason=f"fixture_graph:{edge_type}",
                        score_breakdown={"graph_proximity": proximity},
                    ),
                )

                unit = record.get("unit")
                if isinstance(unit, dict):
                    node = self._graph_node_from_unit(
                        unit_id=neighbor_id,
                        unit=unit,
                        seed=False,
                    )
                    graph_nodes_by_id.setdefault(node.id, node)

                graph_edge = self._graph_edge_from_neighbor(seed.unit_id, record, edge_type)
                if graph_edge is not None:
                    graph_edges_by_id.setdefault(graph_edge.id, graph_edge)

        expanded_candidates = sorted(
            expanded_by_id.values(),
            key=lambda candidate: (
                candidate.graph_distance,
                candidate.source != "seed",
                candidate.unit_id,
            ),
        )
        graph_nodes = sorted(graph_nodes_by_id.values(), key=lambda node: node.id)
        graph_edges = sorted(graph_edges_by_id.values(), key=lambda edge: edge.id)

        return self._result(
            plan=plan,
            seed_candidates=seeds,
            expanded_candidates=expanded_candidates,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            warning=None,
            fallback_used=False,
            reason="graph neighbors expanded from configured client",
            debug=debug,
        )

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

    async def _neighbor_records(
        self,
        unit_id: str,
        *,
        allowed_edge_types: list[str],
    ) -> list[dict[str, Any]]:
        method = (
            getattr(self.neighbors_client, "neighbors_for", None)
            or getattr(self.neighbors_client, "get_neighbors", None)
        )
        if method is None:
            return []
        try:
            records = method(
                unit_id,
                allowed_edge_types=allowed_edge_types,
                max_depth=self.max_depth,
            )
        except TypeError:
            records = method(unit_id)
        if inspect.isawaitable(records):
            records = await records
        if records is None:
            return []
        return list(records)

    def _neighbor_edge_type(self, record: dict[str, Any]) -> str:
        edge_type = record.get("edge_type") or record.get("expansion_edge_type")
        if isinstance(edge_type, str) and edge_type:
            return edge_type
        edge = record.get("edge")
        if isinstance(edge, dict) and edge.get("type") == "contains":
            return "contains_child"
        return str(record.get("type") or "")

    def _neighbor_unit_id(self, seed_unit_id: str, record: dict[str, Any]) -> str | None:
        value = record.get("unit_id") or record.get("neighbor_unit_id")
        if isinstance(value, str) and value:
            return value
        edge = record.get("edge")
        edge_data = edge if isinstance(edge, dict) else record
        source_id = edge_data.get("source_id")
        target_id = edge_data.get("target_id")
        if source_id == seed_unit_id and isinstance(target_id, str):
            return target_id
        if target_id == seed_unit_id and isinstance(source_id, str):
            return source_id
        return None

    def _neighbor_distance(self, record: dict[str, Any]) -> int:
        try:
            distance = int(record.get("distance") or 1)
        except (TypeError, ValueError):
            distance = 1
        return max(1, distance)

    def _graph_node_from_unit(
        self,
        *,
        unit_id: str,
        unit: dict[str, Any],
        seed: bool,
    ) -> GraphNode:
        label = (
            unit.get("label")
            or unit.get("title")
            or self._hierarchy_label(unit)
            or unit.get("law_title")
            or unit.get("id")
            or unit_id
        )
        return GraphNode(
            id=f"legal_unit:{unit_id}",
            type=self._node_type(unit),
            label=str(label),
            legal_unit_id=unit_id,
            domain=self._optional_str(unit.get("legal_domain") or unit.get("domain")),
            status=self._optional_str(unit.get("status")),
            metadata={
                **self._scalar_metadata(unit),
                "legal_unit_id": unit_id,
                "seed": seed,
            },
        )

    def _graph_edge_from_neighbor(
        self,
        seed_unit_id: str,
        record: dict[str, Any],
        expansion_edge_type: str,
    ) -> GraphEdge | None:
        edge = record.get("edge")
        edge_data = edge if isinstance(edge, dict) else record
        neighbor_id = self._neighbor_unit_id(seed_unit_id, record)
        if not neighbor_id:
            return None
        source_id = edge_data.get("source_id")
        target_id = edge_data.get("target_id")
        direction = str(record.get("direction") or "")
        if not isinstance(source_id, str) or not isinstance(target_id, str):
            if direction == "parent":
                source_id, target_id = neighbor_id, seed_unit_id
            else:
                source_id, target_id = seed_unit_id, neighbor_id
        edge_type = "contains" if expansion_edge_type.startswith("contains_") else str(
            edge_data.get("type") or expansion_edge_type
        )
        if edge_type not in {"contains", "references", "defines", "exception_to", "sanctions"}:
            edge_type = "contains"
        return GraphEdge(
            id=str(edge_data.get("id") or f"edge:{edge_type}:{source_id}:{target_id}"),
            source=f"legal_unit:{source_id}",
            target=f"legal_unit:{target_id}",
            type=edge_type,
            weight=float(edge_data.get("weight") or 1.0),
            confidence=float(edge_data.get("confidence") or 1.0),
            explanation="Graph edge supplied by configured neighbors client.",
            metadata={
                **(
                    edge_data.get("metadata")
                    if isinstance(edge_data.get("metadata"), dict)
                    else {}
                ),
                "expansion_edge_type": expansion_edge_type,
                "seed_unit_id": seed_unit_id,
            },
        )

    def _hierarchy_label(self, unit: dict[str, Any]) -> str | None:
        hierarchy_path = unit.get("hierarchy_path")
        if isinstance(hierarchy_path, list) and hierarchy_path:
            return str(hierarchy_path[-1])
        return None

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
        warning: str | None,
        fallback_used: bool,
        reason: str,
        debug: bool,
        graph_edges: list[GraphEdge] | None = None,
    ) -> GraphExpansionResult:
        graph_edges = graph_edges or []
        result = GraphExpansionResult(
            seed_candidates=seed_candidates,
            expanded_candidates=expanded_candidates,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            warnings=[warning] if warning else [],
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
                "graph_edge_count": len(graph_edges),
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
