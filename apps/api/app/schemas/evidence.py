from typing import Any

from pydantic import BaseModel, Field

from .query import EvidenceUnit, GraphEdge, GraphNode


class EvidencePackResult(BaseModel):
    evidence_units: list[EvidenceUnit] = Field(default_factory=list)
    graph_nodes: list[GraphNode] = Field(default_factory=list)
    graph_edges: list[GraphEdge] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    debug: dict[str, Any] | None = None
