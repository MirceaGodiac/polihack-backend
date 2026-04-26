from datetime import date
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
    id: str
    canonical_id: str | None = None
    source_id: str | None = None
    law_id: str
    law_title: str
    act_type: str | None = None
    act_number: str | None = None
    publication_date: date | None = None
    effective_date: date | None = None
    version_start: date | None = None
    version_end: date | None = None
    status: Literal["active", "historical", "repealed", "unknown"]
    hierarchy_path: list[str]
    article_number: str | None = None
    paragraph_number: str | None = None
    letter_number: str | None = None
    point_number: str | None = None
    raw_text: str
    normalized_text: str | None = None
    legal_domain: str
    legal_concepts: list[str] = Field(default_factory=list)
    source_url: str | None = None
    parent_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    outgoing_reference_ids: list[str] = Field(default_factory=list)
    incoming_reference_ids: list[str] = Field(default_factory=list)


class EvidenceUnit(LegalUnit):
    evidence_id: str
    excerpt: str
    rank: int = Field(..., ge=1)
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    retrieval_method: str
    retrieval_score: float = 0.0
    rerank_score: float = Field(default=0.0, ge=0.0, le=1.0)
    mmr_score: float | None = None
    support_role: Literal[
        "direct_basis",
        "definition",
        "condition",
        "exception",
        "sanction",
        "procedure",
        "context",
    ] = "direct_basis"
    why_selected: list[str] = Field(default_factory=list)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
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
    id: str
    label: str
    type: Literal[
        "root",
        "domain",
        "subdomain",
        "legal_act",
        "title",
        "chapter",
        "section",
        "article",
        "paragraph",
        "letter",
        "point",
        "annex",
        "concept",
        "query",
        "answer",
        "cited_claim",
    ]
    legal_unit_id: str | None = None
    domain: str | None = None
    status: str | None = None
    importance: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    type: Literal[
        "contains",
        "references",
        "amends",
        "repeals",
        "derogates_from",
        "exception_to",
        "defines",
        "sanctions",
        "creates_obligation",
        "creates_right",
        "creates_prohibition",
        "procedure_step",
        "same_topic_as",
        "semantically_related",
        "retrieved_for_query",
        "cited_in_answer",
        "supports_claim",
        "contradicts_claim",
        "historical_version_of",
    ]
    weight: float
    confidence: float
    explanation: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphPayload(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class QueryGraphResponse(BaseModel):
    query_id: str
    question: str
    graph: GraphPayload
    highlighted_node_ids: list[str] = Field(default_factory=list)
    highlighted_edge_ids: list[str] = Field(default_factory=list)
    cited_unit_ids: list[str] = Field(default_factory=list)
    reasoning_path: list[str] = Field(default_factory=list)
    verifier_summary: dict[str, int | float | bool] = Field(default_factory=dict)


class ClaimResult(BaseModel):
    claim_id: str
    claim_text: str
    status: Literal[
        "strongly_supported",
        "supported",
        "weakly_supported",
        "unsupported",
        "not_checked",
    ]
    citation_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    warning: str | None = None
    support_score: float | None = Field(default=None, ge=0.0, le=1.0)
    supporting_unit_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    score_breakdown: dict[str, float] = Field(default_factory=dict)


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
    query_frame: dict[str, Any] | None = None
    retrieval: dict[str, Any] | None = None
    graph_expansion: dict[str, Any] | None = None
    legal_ranker: dict[str, Any] | None = None
    evidence_pack: dict[str, Any] | None = None
    generation: dict[str, Any] | None = None
    verifier: dict[str, Any] | None = None
    answer_repair: dict[str, Any] | None = None
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
