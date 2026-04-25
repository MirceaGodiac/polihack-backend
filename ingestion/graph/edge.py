"""
edge.py
-------
Dataclass representing a directed edge between two legal units.

Mirrors the on-disk schema of ``legal_edges.json``:
  - ``contains``   : structural parent → child relationship
  - ``references`` : citation from one unit to another (carries confidence)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any


class EdgeType(str, Enum):
    CONTAINS = "contains"
    REFERENCES = "references"


@dataclass(slots=True)
class LegalEdge:
    source_id: str
    target_id: str
    type: EdgeType
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type.value,
        }
        if self.confidence is not None:
            d["confidence"] = self.confidence
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LegalEdge":
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            type=EdgeType(data["type"]),
            confidence=data.get("confidence"),
        )
