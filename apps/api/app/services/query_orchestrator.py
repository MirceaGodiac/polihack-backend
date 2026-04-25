from uuid import NAMESPACE_URL, uuid5

from ..schemas import (
    AnswerPayload,
    Citation,
    DraftAnswer,
    EvidenceUnit,
    GenerationConstraints,
    GraphPayload,
    QueryDebugData,
    QueryRequest,
    QueryResponse,
    VerifierStatus,
)
from .evidence_pack_compiler import EvidencePackCompiler
from .generation_adapter import (
    GENERATION_FAILED,
    GENERATION_INSUFFICIENT_EVIDENCE,
    GENERATION_MODE_INSUFFICIENT_EVIDENCE,
    GENERATION_UNVERIFIED_WARNING,
    INSUFFICIENT_EVIDENCE_ANSWER,
    GenerationAdapter,
)
from .graph_expansion_policy import GraphExpansionPolicy
from .legal_ranker import LegalRanker
from .mock_evidence import MockEvidenceService
from .query_understanding import QueryUnderstanding
from .raw_retriever_client import RawRetrieverClient


class QueryOrchestrator:
    def __init__(
        self,
        evidence_service: MockEvidenceService | None = None,
        query_understanding: QueryUnderstanding | None = None,
        raw_retriever_client: RawRetrieverClient | None = None,
        graph_expansion_policy: GraphExpansionPolicy | None = None,
        legal_ranker: LegalRanker | None = None,
        evidence_pack_compiler: EvidencePackCompiler | None = None,
        generation_adapter: GenerationAdapter | None = None,
    ) -> None:
        self.evidence_service = evidence_service or MockEvidenceService()
        self.query_understanding = query_understanding or QueryUnderstanding()
        self.raw_retriever_client = raw_retriever_client or RawRetrieverClient()
        self.graph_expansion_policy = (
            graph_expansion_policy or GraphExpansionPolicy()
        )
        self.legal_ranker = legal_ranker or LegalRanker()
        self.evidence_pack_compiler = (
            evidence_pack_compiler or EvidencePackCompiler()
        )
        self.generation_adapter = generation_adapter or GenerationAdapter()

    async def run(self, request: QueryRequest) -> QueryResponse:
        query_id = self._query_id(request)
        query_plan = self.query_understanding.build_plan(request)
        raw_retrieval = await self.raw_retriever_client.retrieve(
            query_plan,
            top_k=50,
            debug=request.debug,
        )
        graph_expansion = await self.graph_expansion_policy.expand(
            plan=query_plan,
            retrieval_response=raw_retrieval,
            debug=request.debug,
        )
        legal_ranker = self.legal_ranker.rank(
            question=request.question,
            plan=query_plan,
            retrieval_response=raw_retrieval,
            graph_expansion=graph_expansion,
            debug=request.debug,
        )
        compiled_evidence = self.evidence_pack_compiler.compile(
            ranked_candidates=legal_ranker.ranked_candidates,
            graph_expansion=graph_expansion,
            plan=query_plan,
            debug=request.debug,
        )
        draft_answer = self._generate_answer(
            question=request.question,
            evidence_units=compiled_evidence.evidence_units,
            mode=request.mode,
        )
        evidence_pack = await self.evidence_service.build_pack(request, query_id)
        answer = self._answer_payload(draft_answer)
        citations = self._citations_from_draft(
            draft_answer,
            compiled_evidence.evidence_units,
        )
        verifier = self._unverified_status(
            evidence_pack.verifier,
            draft_answer=draft_answer,
            answer=answer,
        )
        graph = GraphPayload(
            nodes=compiled_evidence.graph_nodes,
            edges=compiled_evidence.graph_edges,
        )
        debug = None
        if request.debug:
            debug = QueryDebugData(
                orchestrator=self.__class__.__name__,
                evidence_service=self.evidence_service.__class__.__name__,
                retrieval_mode="mock_static_fixture",
                query_understanding=query_plan,
                retrieval=raw_retrieval.debug,
                graph_expansion=graph_expansion.debug,
                legal_ranker=legal_ranker.debug,
                evidence_pack=compiled_evidence.debug,
                generation=self._generation_debug(draft_answer),
                evidence_units_count=len(compiled_evidence.evidence_units),
                citations_count=len(citations),
                graph_nodes_count=len(graph.nodes),
                graph_edges_count=len(graph.edges),
                notes=evidence_pack.debug_notes,
            )

        return QueryResponse(
            query_id=query_id,
            question=request.question,
            answer=answer,
            citations=citations,
            evidence_units=compiled_evidence.evidence_units,
            verifier=verifier,
            graph=graph,
            debug=debug,
            warnings=(
                evidence_pack.warnings
                + raw_retrieval.warnings
                + graph_expansion.warnings
                + legal_ranker.warnings
                + compiled_evidence.warnings
                + draft_answer.warnings
            ),
        )

    def _generate_answer(
        self,
        *,
        question: str,
        evidence_units: list[EvidenceUnit],
        mode: str,
    ) -> DraftAnswer:
        try:
            return self.generation_adapter.generate(
                question=question,
                evidence_units=evidence_units,
                constraints=GenerationConstraints(mode=mode),
            )
        except Exception:
            return DraftAnswer(
                short_answer=INSUFFICIENT_EVIDENCE_ANSWER,
                detailed_answer=None,
                citations=[],
                used_evidence_unit_ids=[],
                generation_mode=GENERATION_MODE_INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                warnings=[
                    GENERATION_FAILED,
                    GENERATION_INSUFFICIENT_EVIDENCE,
                    GENERATION_UNVERIFIED_WARNING,
                ],
            )

    def _answer_payload(self, draft_answer: DraftAnswer) -> AnswerPayload:
        refusal_reason = (
            GENERATION_INSUFFICIENT_EVIDENCE
            if GENERATION_INSUFFICIENT_EVIDENCE in draft_answer.warnings
            else None
        )
        return AnswerPayload(
            short_answer=draft_answer.short_answer,
            detailed_answer=draft_answer.detailed_answer,
            confidence=0.0,
            not_legal_advice=True,
            refusal_reason=refusal_reason,
        )

    def _citations_from_draft(
        self,
        draft_answer: DraftAnswer,
        evidence_units: list[EvidenceUnit],
    ) -> list[Citation]:
        evidence_by_id = {unit.id: unit for unit in evidence_units}
        citations: list[Citation] = []
        for index, draft_citation in enumerate(draft_answer.citations, start=1):
            evidence = evidence_by_id.get(draft_citation.unit_id)
            if evidence is None:
                continue
            citations.append(
                Citation(
                    citation_id=f"citation:{index}",
                    evidence_id=evidence.evidence_id,
                    legal_unit_id=evidence.id,
                    label=draft_citation.label,
                    quote=draft_citation.snippet,
                    source_url=draft_citation.source_url,
                    verified=False,
                )
            )
        return citations

    def _unverified_status(
        self,
        verifier: VerifierStatus,
        *,
        draft_answer: DraftAnswer,
        answer: AnswerPayload,
    ) -> VerifierStatus:
        return verifier.model_copy(
            update={
                "groundedness_score": 0.0,
                "claims_total": 0,
                "claims_supported": 0,
                "claims_weakly_supported": 0,
                "claims_unsupported": 0,
                "citations_checked": 0,
                "verifier_passed": False,
                "claim_results": [],
                "warnings": self._dedupe(
                    verifier.warnings
                    + draft_answer.warnings
                    + [GENERATION_UNVERIFIED_WARNING]
                ),
                "repair_applied": False,
                "refusal_reason": answer.refusal_reason,
            }
        )

    def _generation_debug(self, draft_answer: DraftAnswer) -> dict[str, object]:
        return {
            "generation_mode": draft_answer.generation_mode,
            "evidence_unit_count_used": len(draft_answer.used_evidence_unit_ids),
            "warnings": draft_answer.warnings,
            "citation_unit_ids": [
                citation.unit_id for citation in draft_answer.citations
            ],
        }

    def _dedupe(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped

    def _query_id(self, request: QueryRequest) -> str:
        stable_input = "|".join(
            [
                request.question.strip(),
                request.jurisdiction,
                request.date,
                request.mode,
            ]
        )
        return str(uuid5(NAMESPACE_URL, stable_input))
