from typing import Any, Literal

from pydantic import BaseModel, Field

from .query import GraphEdge, GraphNode
from .retrieval import RetrievalCandidate


class ExpandedCandidate(BaseModel):
    unit_id: str
    source: Literal["seed", "graph_expansion"] = "seed"
    graph_distance: int = Field(default=0, ge=0)
    graph_proximity: float = Field(default=1.0, ge=0.0)
    expansion_edge_type: str | None = None
    expansion_reason: str | None = None
    retrieval_candidate: RetrievalCandidate | None = None
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class GraphExpansionResult(BaseModel):
    seed_candidates: list[ExpandedCandidate] = Field(default_factory=list)
    expanded_candidates: list[ExpandedCandidate] = Field(default_factory=list)
    graph_nodes: list[GraphNode] = Field(default_factory=list)
    graph_edges: list[GraphEdge] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    debug: dict[str, Any] | None = None
