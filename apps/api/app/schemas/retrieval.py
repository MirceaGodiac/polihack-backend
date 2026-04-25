from typing import Any

from pydantic import BaseModel, Field

from .query import ExactCitation


class RawRetrievalRequest(BaseModel):
    question: str
    retrieval_filters: dict[str, Any] = Field(default_factory=dict)
    exact_citations: list[ExactCitation] = Field(default_factory=list)
    top_k: int = Field(default=50, ge=1)
    debug: bool = False


class RetrievalCandidate(BaseModel):
    unit_id: str
    rank: int
    retrieval_score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    matched_terms: list[str] = Field(default_factory=list)
    why_retrieved: str | None = None
    unit: dict[str, Any] | None = None


class RawRetrievalResponse(BaseModel):
    candidates: list[RetrievalCandidate] = Field(default_factory=list)
    retrieval_methods: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    debug: dict[str, Any] | None = None
