from pydantic import BaseModel, Field


class GenerationConstraints(BaseModel):
    language: str = "ro"
    mode: str = "strict_citations"
    max_evidence_units: int = Field(default=12, ge=1)
    require_citations: bool = True
    allow_uncited_legal_claims: bool = False


class DraftCitation(BaseModel):
    unit_id: str
    label: str
    snippet: str
    source_url: str | None = None
    support_score: float | None = None


class DraftAnswer(BaseModel):
    short_answer: str
    detailed_answer: str | None = None
    citations: list[DraftCitation] = Field(default_factory=list)
    used_evidence_unit_ids: list[str] = Field(default_factory=list)
    generation_mode: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
