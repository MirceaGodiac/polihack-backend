from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RawExactCitation(BaseModel):
    model_config = ConfigDict(extra="allow")

    law_id: str | None = None
    law_id_hint: str | None = None
    article_number: str | None = None
    article: str | None = None
    paragraph_number: str | None = None
    paragraph: str | None = None
    letter_number: str | None = None
    letter: str | None = None
    point_number: str | None = None
    point: str | None = None
    raw_text: str | None = None
    citation_type: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class RawRetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str
    filters: dict[str, Any] = Field(default_factory=dict)
    retrieval_filters: dict[str, Any] = Field(default_factory=dict)
    query_frame: dict[str, Any] | None = None
    exact_citations: list[RawExactCitation] = Field(default_factory=list)
    query_embedding: list[float] | None = None
    top_k: int = Field(default=50, ge=1, le=100)
    debug: bool = False

    @model_validator(mode="after")
    def merge_filter_aliases(self) -> "RawRetrievalRequest":
        if not self.filters and self.retrieval_filters:
            self.filters = dict(self.retrieval_filters)
        elif self.filters and not self.retrieval_filters:
            self.retrieval_filters = dict(self.filters)
        elif self.filters and self.retrieval_filters:
            merged = dict(self.retrieval_filters)
            merged.update(self.filters)
            self.filters = merged
            self.retrieval_filters = merged
        return self


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
