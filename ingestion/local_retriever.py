from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ingestion.bundle_loader import (
    CanonicalBundle,
    build_unit_index,
    load_canonical_bundle,
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
    "mi",
    "nr",
    "pe",
    "poate",
    "sa",
    "sau",
    "se",
    "si",
}


@dataclass(frozen=True)
class LocalRetrievalCandidate:
    unit_id: str
    chunk_id: str
    rank: int
    retrieval_score: float
    score_breakdown: dict[str, float]
    matched_terms: list[str]
    why_retrieved: str
    unit: dict[str, Any]
    evidence_text: str
    evidence_text_source: str
    scored_text_source: str

    def to_retrieval_candidate_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "rank": self.rank,
            "retrieval_score": self.retrieval_score,
            "score_breakdown": self.score_breakdown,
            "matched_terms": self.matched_terms,
            "why_retrieved": self.why_retrieved,
            "unit": self.unit,
        }


class LocalBundleRetriever:
    """Deterministic file-based retriever for local parser bundle checks."""

    def __init__(self, bundle: CanonicalBundle) -> None:
        self.bundle = bundle
        self.units_by_id = build_unit_index(bundle.legal_units)

    @classmethod
    def from_path(cls, path: str | Path) -> "LocalBundleRetriever":
        return cls(load_canonical_bundle(path))

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        legal_domain: str | None = None,
    ) -> list[LocalRetrievalCandidate]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        query_domain = legal_domain or _infer_legal_domain(query_tokens)
        rows: list[tuple[LocalRetrievalCandidate, float, float, str]] = []

        for chunk in self.bundle.legal_chunks:
            unit_id = str(chunk.get("legal_unit_id") or "")
            unit = self.units_by_id.get(unit_id)
            if not unit:
                continue
            retrieval_text = _chunk_retrieval_text(chunk)
            retrieval_tokens = _tokenize(retrieval_text)
            matched_terms = sorted(query_tokens & retrieval_tokens)
            exact_citation_score = _exact_citation_score(query, unit)
            lexical_score = round(
                len(matched_terms) / len(query_tokens),
                6,
            )
            if lexical_score == 0.0 and exact_citation_score == 0.0:
                continue
            domain_score = _domain_score(query_domain, unit)
            retrieval_score = round(
                0.70 * lexical_score
                + 0.20 * domain_score
                + 0.10 * exact_citation_score,
                6,
            )
            candidate = LocalRetrievalCandidate(
                unit_id=unit_id,
                chunk_id=str(chunk.get("chunk_id") or ""),
                rank=0,
                retrieval_score=retrieval_score,
                score_breakdown={
                    "lexical_overlap": lexical_score,
                    "domain_match": domain_score,
                    "exact_citation_boost": exact_citation_score,
                    "scored_text_retrieval_text": 1.0,
                },
                matched_terms=matched_terms,
                why_retrieved="local_bundle_retriever:token_overlap_retrieval_text",
                unit=unit,
                evidence_text=str(unit.get("raw_text") or ""),
                evidence_text_source="LegalUnit.raw_text",
                scored_text_source="LegalChunk.retrieval_text",
            )
            rows.append(
                (
                    candidate,
                    lexical_score,
                    exact_citation_score,
                    unit_id,
                )
            )

        rows.sort(
            key=lambda row: (
                -row[0].retrieval_score,
                -row[1],
                -row[2],
                row[3],
            )
        )
        ranked: list[LocalRetrievalCandidate] = []
        for rank, (candidate, _, _, _) in enumerate(rows[:top_k], start=1):
            ranked.append(
                LocalRetrievalCandidate(
                    unit_id=candidate.unit_id,
                    chunk_id=candidate.chunk_id,
                    rank=rank,
                    retrieval_score=candidate.retrieval_score,
                    score_breakdown=candidate.score_breakdown,
                    matched_terms=candidate.matched_terms,
                    why_retrieved=candidate.why_retrieved,
                    unit=candidate.unit,
                    evidence_text=candidate.evidence_text,
                    evidence_text_source=candidate.evidence_text_source,
                    scored_text_source=candidate.scored_text_source,
                )
            )
        return ranked


def _chunk_retrieval_text(chunk: dict[str, Any]) -> str:
    for key in ("retrieval_text", "embedding_text", "text", "raw_text"):
        value = chunk.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _domain_score(query_domain: str | None, unit: dict[str, Any]) -> float:
    if not query_domain:
        return 0.0
    unit_domain = str(unit.get("legal_domain") or "")
    return 1.0 if _normalize_text(unit_domain) == _normalize_text(query_domain) else 0.0


def _infer_legal_domain(query_tokens: set[str]) -> str | None:
    labor_terms = {"angajator", "salariat", "salariu", "contract", "munca"}
    if query_tokens & labor_terms:
        return "munca"
    return None


def _exact_citation_score(query: str, unit: dict[str, Any]) -> float:
    normalized_query = _normalize_text(query)
    article_number = unit.get("article_number")
    if article_number and re.search(rf"\bart\s*{re.escape(str(article_number))}\b", normalized_query):
        return 1.0
    return 0.0


def _tokenize(text: str) -> set[str]:
    normalized = _normalize_text(text)
    tokens = set()
    for token in re.split(r"[^a-z0-9_]+", normalized):
        token = _stem_token(token)
        if len(token) > 1 and token not in STOPWORDS:
            tokens.add(token)
    return tokens


def _stem_token(token: str) -> str:
    if len(token) > 6 and token.endswith("ului"):
        return token[:-4]
    if len(token) > 5 and token.endswith("ul"):
        return token[:-2]
    return token


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.casefold())
    stripped = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return " ".join(stripped.replace(".", " ").replace("-", "_").split())
