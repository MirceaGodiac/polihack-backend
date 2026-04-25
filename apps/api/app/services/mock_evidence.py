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
    "mock_unverified_evidence_pack: Phase 1 returns deterministic mock evidence "
    "only; retrieval, LegalRanker, generation, and citation verification have not run."
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
                "the evidence below is deterministic placeholder data for API contract testing."
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
                "No database, vector search, graph expansion, ranker, generation, or verifier was called.",
            ],
        )

    def _build_evidence_units(self) -> list[EvidenceUnit]:
        legal_units = [
            LegalUnit(
                legal_unit_id="mock:ro:codul-muncii:art-17",
                legal_act="Codul muncii",
                article="art. 17",
                title="Informarea salariatului - mock placeholder",
            ),
            LegalUnit(
                legal_unit_id="mock:ro:codul-muncii:art-41",
                legal_act="Codul muncii",
                article="art. 41",
                title="Modificarea contractului individual de munca - mock placeholder",
            ),
            LegalUnit(
                legal_unit_id="mock:ro:codul-muncii:art-159",
                legal_act="Codul muncii",
                article="art. 159",
                title="Salariul - mock placeholder",
            ),
        ]
        excerpts = [
            "Mock excerpt for art. 17. This placeholder was not retrieved from an official source.",
            "Mock excerpt for art. 41. This placeholder was not retrieved from an official source.",
            "Mock excerpt for art. 159. This placeholder was not retrieved from an official source.",
        ]

        return [
            EvidenceUnit(
                evidence_id=f"mock-evidence-{index}",
                legal_unit=legal_unit,
                excerpt=excerpt,
                rank=index,
                relevance_score=0.0,
                retrieval_method="mock_static_fixture",
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
                legal_unit_id=evidence_units[0].legal_unit.legal_unit_id,
                label="Codul muncii art. 17 (mock, unverified)",
                quote=evidence_units[0].excerpt,
                verified=False,
            ),
            Citation(
                citation_id="mock-citation-2",
                evidence_id=evidence_units[1].evidence_id,
                legal_unit_id=evidence_units[1].legal_unit.legal_unit_id,
                label="Codul muncii art. 41 (mock, unverified)",
                quote=evidence_units[1].excerpt,
                verified=False,
            ),
        ]

    def _build_graph(self, request: QueryRequest, query_id: str) -> GraphPayload:
        nodes = [
            GraphNode(
                node_id=f"query:{query_id}",
                node_type="query",
                label=request.question,
                metadata={"jurisdiction": request.jurisdiction, "mode": request.mode},
            ),
            GraphNode(
                node_id="domain:employment_law",
                node_type="domain",
                label="Dreptul muncii (mock domain)",
                metadata={"mock": True},
            ),
            GraphNode(
                node_id="legal_act:mock:codul-muncii",
                node_type="legal_act",
                label="Codul muncii (mock legal act)",
                metadata={"mock": True},
            ),
            GraphNode(
                node_id="article:mock:codul-muncii:art-41",
                node_type="article",
                label="art. 41 (mock article)",
                metadata={"mock": True},
            ),
            GraphNode(
                node_id=f"answer:{query_id}",
                node_type="answer",
                label="Mock unverified answer",
                metadata={"verifier_passed": False},
            ),
        ]
        edges = [
            GraphEdge(
                edge_id="edge:query-domain",
                source_node_id=nodes[0].node_id,
                target_node_id=nodes[1].node_id,
                edge_type="mock_classified_as",
            ),
            GraphEdge(
                edge_id="edge:domain-act",
                source_node_id=nodes[1].node_id,
                target_node_id=nodes[2].node_id,
                edge_type="mock_related_legal_act",
            ),
            GraphEdge(
                edge_id="edge:act-article",
                source_node_id=nodes[2].node_id,
                target_node_id=nodes[3].node_id,
                edge_type="mock_contains_article",
            ),
            GraphEdge(
                edge_id="edge:article-answer",
                source_node_id=nodes[3].node_id,
                target_node_id=nodes[4].node_id,
                edge_type="mock_support_candidate",
                metadata={"verified": False},
            ),
        ]
        return GraphPayload(nodes=nodes, edges=edges)
