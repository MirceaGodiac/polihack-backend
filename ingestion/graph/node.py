"""
node.py
-------
Dataclass representing a single legal unit (node) in the corpus graph.

Mirrors the on-disk schema of ``legal_units.json`` produced by the parser.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class UnitType(str, Enum):
    TITLU = "titlu"
    CAPITOL = "capitol"
    ARTICOL = "articol"
    ALINEAT = "alineat"
    LITERA = "litera"


@dataclass(slots=True)
class LegalUnit:
    id: str
    type: UnitType
    raw_text: str
    hierarchy_path: list[str] = field(default_factory=list)
    corpus_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LegalUnit":
        return cls(
            id=data["id"],
            type=UnitType(data["type"]),
            raw_text=data["raw_text"],
            hierarchy_path=list(data.get("hierarchy_path", [])),
            corpus_id=data.get("corpus_id", ""),
        )
