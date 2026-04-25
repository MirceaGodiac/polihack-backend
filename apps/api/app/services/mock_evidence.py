from dataclasses import dataclass

from ..schemas import (
    AnswerPayload,
    Citation,
    EvidenceUnit,
    GraphEdge,
    GraphNode,
    GraphPayload,
    LegalUnit,
    QueryRequest,
    VerifierStatus,
)

MOCK_REFUSAL_REASON = "mock_evidence_pack_not_verified"
MOCK_WARNING = (
    "mock_unverified_answer: Phase 7 keeps answer generation and citation "
    "verification mocked; compiled evidence, if present, is not converted into "
    "a verified legal conclusion."
)


@dataclass(frozen=True)
class EvidencePack:
    answer: AnswerPayload
    citations: list[Citation]
    evidence_units: list[EvidenceUnit]
    verifier: VerifierStatus
    graph: GraphPayload
    warnings: list[str]
    debug_notes: list[str]


class MockEvidenceService:
    async def build_pack(self, request: QueryRequest, query_id: str) -> EvidencePack:
        evidence_units = self._build_evidence_units()
        citations = self._build_citations(evidence_units)
        graph = self._build_graph(request=request, query_id=query_id)
        warnings = [
            MOCK_WARNING,
            "mock_no_legal_conclusion: This response is not a verified legal analysis.",
        ]
        answer = AnswerPayload(
            short_answer=(
                "Phase 1 mock response only. No verified legal conclusion is provided; "
                "Phase 7 may compile evidence units separately, but generation is not configured."
            ),
            detailed_answer=None,
            confidence=0.0,
            not_legal_advice=True,
            refusal_reason=MOCK_REFUSAL_REASON,
        )
        verifier = VerifierStatus(
            groundedness_score=0.0,
            claims_total=0,
            claims_supported=0,
            claims_weakly_supported=0,
            claims_unsupported=0,
            citations_checked=0,
            verifier_passed=False,
            claim_results=[],
            warnings=warnings.copy(),
            repair_applied=False,
            refusal_reason=MOCK_REFUSAL_REASON,
        )

        return EvidencePack(
            answer=answer,
            citations=citations,
            evidence_units=evidence_units,
            verifier=verifier,
            graph=graph,
            warnings=warnings,
            debug_notes=[
                "MockEvidenceService.build_pack returned static fixture data.",
                "Generation and citation verification remain mocked and unverified.",
            ],
        )

    def _build_evidence_units(self) -> list[EvidenceUnit]:
        excerpts = [
            "Mock excerpt for art. 17. This placeholder was not retrieved from an official source.",
            "Mock excerpt for art. 41. This placeholder was not retrieved from an official source.",
            "Mock excerpt for art. 159. This placeholder was not retrieved from an official source.",
        ]
        legal_units = [
            LegalUnit(
                id="mock:ro:codul-muncii:art-17",
                law_id="ro.codul_muncii",
                law_title="Codul muncii",
                status="active",
                hierarchy_path=["Codul muncii", "art. 17"],
                article_number="17",
                raw_text=excerpts[0],
                legal_domain="muncă",
            ),
            LegalUnit(
                id="mock:ro:codul-muncii:art-41",
                law_id="ro.codul_muncii",
                law_title="Codul muncii",
                status="active",
                hierarchy_path=["Codul muncii", "art. 41"],
                article_number="41",
                raw_text=excerpts[1],
                legal_domain="muncă",
            ),
            LegalUnit(
                id="mock:ro:codul-muncii:art-159",
                law_id="ro.codul_muncii",
                law_title="Codul muncii",
                status="active",
                hierarchy_path=["Codul muncii", "art. 159"],
                article_number="159",
                raw_text=excerpts[2],
                legal_domain="muncă",
            ),
        ]

        return [
            EvidenceUnit(
                **legal_unit.model_dump(),
                evidence_id=f"mock-evidence-{index}",
                excerpt=excerpt,
                rank=index,
                relevance_score=0.0,
                retrieval_method="mock_static_fixture",
                retrieval_score=0.0,
                rerank_score=0.0,
                support_role="direct_basis",
                warnings=[MOCK_WARNING],
            )
            for index, (legal_unit, excerpt) in enumerate(
                zip(legal_units, excerpts, strict=True),
                start=1,
            )
        ]

    def _build_citations(self, evidence_units: list[EvidenceUnit]) -> list[Citation]:
        return [
            Citation(
                citation_id="mock-citation-1",
                evidence_id=evidence_units[0].evidence_id,
                legal_unit_id=evidence_units[0].id,
                label="Codul muncii art. 17 (mock, unverified)",
                quote=evidence_units[0].excerpt,
                verified=False,
            ),
            Citation(
                citation_id="mock-citation-2",
                evidence_id=evidence_units[1].evidence_id,
                legal_unit_id=evidence_units[1].id,
                label="Codul muncii art. 41 (mock, unverified)",
                quote=evidence_units[1].excerpt,
                verified=False,
            ),
        ]

    def _build_graph(self, request: QueryRequest, query_id: str) -> GraphPayload:
        nodes = [
            GraphNode(
                id=f"query:{query_id}",
                type="query",
                label=request.question,
                metadata={"jurisdiction": request.jurisdiction, "mode": request.mode},
            ),
            GraphNode(
                id="domain:employment_law",
                type="domain",
                label="Dreptul muncii (mock domain)",
                domain="muncă",
                metadata={"mock": True},
            ),
            GraphNode(
                id="legal_act:mock:codul-muncii",
                type="legal_act",
                label="Codul muncii (mock legal act)",
                metadata={"mock": True},
            ),
            GraphNode(
                id="article:mock:codul-muncii:art-41",
                type="article",
                label="art. 41 (mock article)",
                legal_unit_id="mock:ro:codul-muncii:art-41",
                metadata={"mock": True},
            ),
            GraphNode(
                id=f"answer:{query_id}",
                type="answer",
                label="Mock unverified answer",
                metadata={"verifier_passed": False},
            ),
        ]
        edges = [
            GraphEdge(
                id="edge:query-domain",
                source=nodes[0].id,
                target=nodes[1].id,
                type="retrieved_for_query",
                weight=0.0,
                confidence=0.0,
            ),
            GraphEdge(
                id="edge:domain-act",
                source=nodes[1].id,
                target=nodes[2].id,
                type="contains",
                weight=0.0,
                confidence=0.0,
            ),
            GraphEdge(
                id="edge:act-article",
                source=nodes[2].id,
                target=nodes[3].id,
                type="contains",
                weight=0.0,
                confidence=0.0,
            ),
            GraphEdge(
                id="edge:article-answer",
                source=nodes[3].id,
                target=nodes[4].id,
                type="cited_in_answer",
                weight=0.0,
                confidence=0.0,
                metadata={"verified": False},
            ),
        ]
        return GraphPayload(nodes=nodes, edges=edges)
