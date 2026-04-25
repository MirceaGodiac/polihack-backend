from typing import Any

from pydantic import BaseModel, Field


class RankerFeatureBreakdown(BaseModel):
    bm25_score: float = Field(default=0.0, ge=0.0, le=1.0)
    dense_score: float = Field(default=0.0, ge=0.0, le=1.0)
    exact_citation_match: float = Field(default=0.0, ge=0.0, le=1.0)
    domain_match: float = Field(default=0.0, ge=0.0, le=1.0)
    graph_proximity: float = Field(default=0.0, ge=0.0, le=1.0)
    concept_overlap: float = Field(default=0.0, ge=0.0, le=1.0)
    legal_term_overlap: float = Field(default=0.0, ge=0.0, le=1.0)
    temporal_validity: float = Field(default=0.0, ge=0.0, le=1.0)
    source_reliability: float = Field(default=0.0, ge=0.0, le=1.0)
    parent_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    is_exception: float = Field(default=0.0, ge=0.0, le=1.0)
    is_definition: float = Field(default=0.0, ge=0.0, le=1.0)
    is_sanction: float = Field(default=0.0, ge=0.0, le=1.0)


class RankedCandidate(BaseModel):
    unit_id: str
    rank: int = Field(..., ge=1)
    rerank_score: float = Field(..., ge=0.0, le=1.0)
    retrieval_score: float | None = None
    unit: dict[str, Any] | None = None
    score_breakdown: RankerFeatureBreakdown
    why_ranked: list[str] = Field(default_factory=list)
    source: str | None = None


class LegalRankerResult(BaseModel):
    ranked_candidates: list[RankedCandidate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    debug: dict[str, Any] | None = None
