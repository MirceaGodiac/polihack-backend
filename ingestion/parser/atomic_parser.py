"""
atomic_parser.py
----------------
Converts raw Romanian legal text into a flat list of atomic JSON units.

Each unit represents the smallest addressable piece of a legal document
(article, paragraph, letter, etc.) with:
  - A deterministic ID based on its hierarchy path
  - Its full human-readable hierarchy path
  - The raw text of that node
  - Any cross-references found within it

Output schema per unit:
{
    "id":         "ro.codul_muncii.titlu_i.capitol_i.art_2.lit_a",
    "path":       "Titlul I > Capitolul I > Art. 2 > a)",
    "type":       "litera",
    "text":       "a) cetatenilor romani cu domiciliul in Romania;",
    "references": [
        {
            "raw_reference":    "art. 4 alin. (1)",
            "target_article":   "4",
            "target_paragraph": "1",
            "target_letter":    None,
            "target_law_hint":  "same_act"
        }
    ]
}
"""

from __future__ import annotations

import json
from typing import List, Optional

from ingestion.parser_rules import RULES
from ingestion.legal_ids import make_unit_id
from ingestion.reference_extractor import extract_references


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_LEVEL_ORDER = ["titlu", "capitol", "sectiune", "articol", "alineat", "litera"]

# Human-readable labels for path rendering
_PATH_LABELS = {
    "titlu":    "Titlul",
    "capitol":  "Capitolul",
    "sectiune": "Secțiunea",
    "articol":  "Art.",
    "alineat":  lambda v: f"({v})",
    "litera":   lambda v: f"{v})",
}


def _render_path(hierarchy: list[tuple[str, str]]) -> str:
    """Convert internal hierarchy list to a human-readable breadcrumb string."""
    parts = []
    for level_type, level_val in hierarchy:
        label = _PATH_LABELS.get(level_type, level_type)
        if callable(label):
            parts.append(label(level_val))
        else:
            parts.append(f"{label} {level_val}")
    return " > ".join(parts)


# ---------------------------------------------------------------------------
# AtomicParser
# ---------------------------------------------------------------------------

class AtomicParser:
    """
    Parses a multiline string of Romanian legal text into a flat list of
    atomic unit dicts, ready for JSON serialisation and vector indexing.

    Usage:
        parser = AtomicParser(corpus_id="ro.codul_muncii")
        units  = parser.parse(raw_text)
        parser.save("output/units.json")
    """

    def __init__(self, corpus_id: str):
        self.corpus_id = corpus_id
        self._units: List[dict] = []
        # Current position in the hierarchy: {type -> value}
        self._state: dict[str, Optional[str]] = {k: None for k in _LEVEL_ORDER}

    # ── public API ────────────────────────────────────────────────────────────

    def parse(self, text: str) -> tuple[List[dict], List[dict]]:
        """
        Parse raw legal text (multiline string) into atomic units and edges.

        Args:
            text: Plain-text content of a legal document.

        Returns:
            Tuple of (list of atomic units, list of contains-edges).
        """
        self._units = []
        self._edges = []
        self._state = {k: None for k in _LEVEL_ORDER}

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            self._process_line(line)

        # Second pass: Extract references from finalized text
        for unit in self._units:
            unit["references"] = [
                {k: v for k, v in ref.items() if k != "source_unit_id"}
                for ref in extract_references({"id": unit["id"], "raw_text": unit["text"]})
            ]

        return self._units, self._edges

    def save(self, path: str = "atomic_units.json") -> None:
        """Serialise parsed units to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._units, f, indent=2, ensure_ascii=False)

    # ── internal processing ───────────────────────────────────────────────────

    def _process_line(self, line: str) -> None:
        for type_name, regex in RULES:
            match = regex.match(line)
            if match:
                val = match.group(1)
                self._update_state(type_name, val)
                unit = self._build_unit(type_name, line)
                self._units.append(unit)
                self._create_edge(unit["id"])
                return

        # Unmatched line: append as continuation text to the last unit
        if self._units:
            self._units[-1]["text"] += " " + line

    def _create_edge(self, unit_id: str) -> None:
        """Create a 'contains' edge from the parent unit to the newly created unit."""
        hierarchy = self._current_hierarchy()
        if len(hierarchy) > 1:
            parent_path = hierarchy[:-1]
            parent_id = make_unit_id(self.corpus_id, parent_path)
            self._edges.append({
                "source_id": parent_id,
                "target_id": unit_id,
                "type": "contains"
            })

    def _update_state(self, type_name: str, val: str) -> None:
        """Advance hierarchy state and reset all lower levels."""
        idx = _LEVEL_ORDER.index(type_name)
        self._state[type_name] = val
        for lower in _LEVEL_ORDER[idx + 1:]:
            self._state[lower] = None

    def _current_hierarchy(self) -> list[tuple[str, str]]:
        """Return the active hierarchy as an ordered list of (type, value) pairs."""
        return [
            (level, self._state[level])
            for level in _LEVEL_ORDER
            if self._state[level] is not None
        ]

    def _build_unit(self, type_name: str, line: str) -> dict:
        hierarchy = self._current_hierarchy()
        unit_id   = make_unit_id(self.corpus_id, hierarchy)
        path      = _render_path(hierarchy)

        return {
            "id":         unit_id,
            "path":       path,
            "type":       type_name,
            "text":       line,
            "references": [],
        }
