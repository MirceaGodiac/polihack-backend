from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apps.api.app.schemas import RawRetrievalResponse, RetrievalCandidate


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "corpus"
LEGAL_UNITS_PATH = FIXTURE_DIR / "codul_muncii_legal_units.json"
LEGAL_EDGES_PATH = FIXTURE_DIR / "codul_muncii_legal_edges.json"

DEMO_QUERY = "Poate angajatorul sa-mi scada salariul fara act aditional?"
DEMO_QUERY_WITH_DIACRITICS = "Poate angajatorul să-mi scadă salariul fără act adițional?"

DEMO_UNIT_IDS = [
    "ro.codul_muncii.art_41.alin_4",
    "ro.codul_muncii.art_41",
    "ro.codul_muncii.art_41.alin_3",
    "ro.codul_muncii.art_17.alin_3.lit_k",
]


def load_codul_muncii_units() -> dict[str, dict[str, Any]]:
    units = json.loads(LEGAL_UNITS_PATH.read_text(encoding="utf-8"))
    return {unit["id"]: unit for unit in units}


def load_codul_muncii_edges() -> list[dict[str, Any]]:
    return json.loads(LEGAL_EDGES_PATH.read_text(encoding="utf-8"))


class FixtureRawRetriever:
    def __init__(
        self,
        *,
        unit_ids: list[str] | None = None,
        units: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.units = units or load_codul_muncii_units()
        self.unit_ids = unit_ids or self._default_unit_ids()

    async def retrieve(
        self,
        plan,
        *,
        top_k: int = 50,
        debug: bool = False,
    ) -> RawRetrievalResponse:
        candidates: list[RetrievalCandidate] = []
        score_rows = self._score_rows(plan)
        for index, unit_id in enumerate(self.unit_ids[:top_k], start=1):
            unit = self.units[unit_id]
            scores = score_rows.get(unit_id, self._default_scores(index))
            retrieval_score = round(
                0.55 * scores["bm25_score"] + 0.45 * scores["dense_score"],
                6,
            )
            candidates.append(
                RetrievalCandidate(
                    unit_id=unit_id,
                    rank=index,
                    retrieval_score=retrieval_score,
                    score_breakdown=scores,
                    matched_terms=self._matched_terms(unit),
                    why_retrieved="codul_muncii_fixture_retriever",
                    unit=unit,
                )
            )

        debug_payload = None
        if debug:
            debug_payload = {
                "fallback_used": False,
                "fixture_units_path": str(LEGAL_UNITS_PATH),
                "candidate_count": len(candidates),
                "unit_ids": [candidate.unit_id for candidate in candidates],
                "score_summary": {
                    candidate.unit_id: candidate.score_breakdown
                    for candidate in candidates
                },
                "request_payload": {
                    "question": plan.question,
                    "retrieval_filters": plan.retrieval_filters,
                    "exact_citations": [
                        citation.model_dump(mode="json")
                        for citation in plan.exact_citations
                    ],
                    "top_k": top_k,
                    "debug": debug,
                },
            }

        return RawRetrievalResponse(
            candidates=candidates,
            retrieval_methods=["fixture_codul_muncii"],
            warnings=[],
            debug=debug_payload,
        )

    def _score_rows(self, plan) -> dict[str, dict[str, float]]:
        exact_boost = 1.0 if plan.exact_citations else 0.0
        return {
            "ro.codul_muncii.art_41.alin_4": {
                "bm25_score": 0.98,
                "dense_score": 0.88,
                "domain_match": 1.0,
                "exact_citation_boost": exact_boost,
            },
            "ro.codul_muncii.art_41": {
                "bm25_score": 0.94,
                "dense_score": 0.86,
                "domain_match": 1.0,
                "exact_citation_boost": exact_boost,
            },
            "ro.codul_muncii.art_41.alin_3": {
                "bm25_score": 0.91,
                "dense_score": 0.82,
                "domain_match": 1.0,
                "exact_citation_boost": 0.0,
            },
            "ro.codul_muncii.art_17.alin_3.lit_k": {
                "bm25_score": 0.84,
                "dense_score": 0.78,
                "domain_match": 1.0,
                "exact_citation_boost": 0.0,
            },
        }

    def _default_scores(self, index: int) -> dict[str, float]:
        base = max(0.1, 0.58 - index * 0.02)
        return {
            "bm25_score": round(base, 6),
            "dense_score": round(base - 0.05, 6),
            "domain_match": 1.0,
            "exact_citation_boost": 0.0,
        }

    def _default_unit_ids(self) -> list[str]:
        return [
            *DEMO_UNIT_IDS,
            *sorted(unit_id for unit_id in self.units if unit_id not in DEMO_UNIT_IDS),
        ]

    def _matched_terms(self, unit: dict[str, Any]) -> list[str]:
        text = str(unit.get("raw_text") or "").casefold()
        return [
            term
            for term in ("angajator", "salariu", "contract", "act aditional")
            if term in text
        ]


class FixtureGraphClient:
    is_configured = True

    def __init__(
        self,
        *,
        units: dict[str, dict[str, Any]] | None = None,
        edges: list[dict[str, Any]] | None = None,
    ) -> None:
        self.units = units or load_codul_muncii_units()
        self.edges = edges or load_codul_muncii_edges()

    def neighbors_for(
        self,
        unit_id: str,
        *,
        allowed_edge_types: list[str] | None = None,
        max_depth: int = 1,
    ) -> list[dict[str, Any]]:
        allowed = set(allowed_edge_types or [])
        records: list[dict[str, Any]] = []
        for edge in self.edges:
            if edge.get("type") != "contains":
                continue
            source_id = edge["source_id"]
            target_id = edge["target_id"]
            if source_id == unit_id and "contains_child" in allowed:
                records.append(
                    self._record(
                        seed_unit_id=unit_id,
                        neighbor_id=target_id,
                        direction="child",
                        edge_type="contains_child",
                        edge=edge,
                    )
                )
            if target_id == unit_id and "contains_parent" in allowed:
                records.append(
                    self._record(
                        seed_unit_id=unit_id,
                        neighbor_id=source_id,
                        direction="parent",
                        edge_type="contains_parent",
                        edge=edge,
                    )
                )
        return sorted(
            records,
            key=lambda record: (
                record["edge_type"],
                record["unit_id"],
                record["edge"]["id"],
            ),
        )[:max_depth * 20]

    def _record(
        self,
        *,
        seed_unit_id: str,
        neighbor_id: str,
        direction: str,
        edge_type: str,
        edge: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "seed_unit_id": seed_unit_id,
            "unit_id": neighbor_id,
            "unit": self.units[neighbor_id],
            "direction": direction,
            "edge_type": edge_type,
            "distance": 1,
            "edge": edge,
        }

