from typing import Any, Literal

from pydantic import BaseModel, Field

JsonMetadata = dict[str, str | int | float | bool | None]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=10, max_length=4000)
    jurisdiction: Literal["RO"] = "RO"
    date: str = Field(default="current", min_length=1, max_length=64)
    mode: Literal["strict_citations"] = "strict_citations"
    debug: bool = False


class AnswerPayload(BaseModel):
    short_answer: str
    detailed_answer: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    not_legal_advice: bool = True
    refusal_reason: str | None = None


class LegalUnit(BaseModel):
    legal_unit_id: str
    legal_act: str
    article: str | None = None
    title: str | None = None
    jurisdiction: Literal["RO"] = "RO"
    source_url: str | None = None


class EvidenceUnit(BaseModel):
    evidence_id: str
    legal_unit: LegalUnit
    excerpt: str
    rank: int = Field(..., ge=1)
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    retrieval_method: str
    warnings: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    citation_id: str
    evidence_id: str
    legal_unit_id: str
    label: str
    quote: str
    source_url: str | None = None
    verified: bool = False


class GraphNode(BaseModel):
    node_id: str
    node_type: str
    label: str
    metadata: JsonMetadata = Field(default_factory=dict)


class GraphEdge(BaseModel):
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str
    metadata: JsonMetadata = Field(default_factory=dict)


class GraphPayload(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class ClaimResult(BaseModel):
    claim_id: str
    claim_text: str
    status: Literal["supported", "weakly_supported", "unsupported", "not_checked"]
    citation_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    warning: str | None = None


class VerifierStatus(BaseModel):
    groundedness_score: float = Field(..., ge=0.0, le=1.0)
    claims_total: int = Field(..., ge=0)
    claims_supported: int = Field(..., ge=0)
    claims_weakly_supported: int = Field(..., ge=0)
    claims_unsupported: int = Field(..., ge=0)
    citations_checked: int = Field(..., ge=0)
    verifier_passed: bool
    claim_results: list[ClaimResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    repair_applied: bool = False
    refusal_reason: str | None = None


class ExactCitation(BaseModel):
    raw_text: str
    citation_type: Literal[
        "article",
        "paragraph",
        "letter",
        "point",
        "thesis",
        "law",
        "ordinance",
        "government_decision",
        "named_code",
        "compound",
    ]
    article: str | None = None
    paragraph: str | None = None
    letter: str | None = None
    point: str | None = None
    thesis: str | None = None
    act_type: str | None = None
    act_number: str | None = None
    act_year: str | None = None
    act_hint: str | None = None
    law_id_hint: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_relative: bool = False
    needs_resolution: bool = True
    lookup_filters: dict[str, Any] = Field(default_factory=dict)
    span_start: int | None = None
    span_end: int | None = None


class QueryPlan(BaseModel):
    question: str
    normalized_question: str
    legal_domain: str | None = None
    domain_confidence: float = Field(..., ge=0.0, le=1.0)
    domain_scores: dict[str, float] = Field(default_factory=dict)
    query_types: list[
        Literal[
            "right",
            "obligation",
            "prohibition",
            "sanction",
            "procedure",
            "definition",
            "exception",
        ]
    ] = Field(default_factory=list)
    exact_citations: list[ExactCitation] = Field(default_factory=list)
    temporal_context: str | None = None
    ambiguity_flags: list[str] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    should_refuse_early: bool = False
    retrieval_filters: dict[str, Any] = Field(default_factory=dict)
    expansion_policy: dict[str, Any] = Field(default_factory=dict)


class QueryDebugData(BaseModel):
    orchestrator: str
    evidence_service: str
    retrieval_mode: str
    query_understanding: QueryPlan | None = None
    retrieval: dict[str, Any] | None = None
    graph_expansion: dict[str, Any] | None = None
    legal_ranker: dict[str, Any] | None = None
    evidence_units_count: int = Field(..., ge=0)
    citations_count: int = Field(..., ge=0)
    graph_nodes_count: int = Field(..., ge=0)
    graph_edges_count: int = Field(..., ge=0)
    notes: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    query_id: str
    question: str
    answer: AnswerPayload
    citations: list[Citation] = Field(default_factory=list)
    evidence_units: list[EvidenceUnit] = Field(default_factory=list)
    verifier: VerifierStatus
    graph: GraphPayload
    debug: QueryDebugData | None = None
    warnings: list[str] = Field(default_factory=list)
