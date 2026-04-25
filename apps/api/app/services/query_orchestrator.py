from uuid import NAMESPACE_URL, uuid5

from ..schemas import QueryDebugData, QueryRequest, QueryResponse
from .mock_evidence import MockEvidenceService
from .query_understanding import QueryUnderstanding
from .raw_retriever_client import RawRetrieverClient


class QueryOrchestrator:
    def __init__(
        self,
        evidence_service: MockEvidenceService | None = None,
        query_understanding: QueryUnderstanding | None = None,
        raw_retriever_client: RawRetrieverClient | None = None,
    ) -> None:
        self.evidence_service = evidence_service or MockEvidenceService()
        self.query_understanding = query_understanding or QueryUnderstanding()
        self.raw_retriever_client = raw_retriever_client or RawRetrieverClient()

    async def run(self, request: QueryRequest) -> QueryResponse:
        query_id = self._query_id(request)
        query_plan = self.query_understanding.build_plan(request)
        raw_retrieval = await self.raw_retriever_client.retrieve(
            query_plan,
            top_k=50,
            debug=request.debug,
        )
        evidence_pack = await self.evidence_service.build_pack(request, query_id)
        debug = None
        if request.debug:
            debug = QueryDebugData(
                orchestrator=self.__class__.__name__,
                evidence_service=self.evidence_service.__class__.__name__,
                retrieval_mode="mock_static_fixture",
                query_understanding=query_plan,
                retrieval=raw_retrieval.debug,
                evidence_units_count=len(evidence_pack.evidence_units),
                citations_count=len(evidence_pack.citations),
                graph_nodes_count=len(evidence_pack.graph.nodes),
                graph_edges_count=len(evidence_pack.graph.edges),
                notes=evidence_pack.debug_notes,
            )

        return QueryResponse(
            query_id=query_id,
            question=request.question,
            answer=evidence_pack.answer,
            citations=evidence_pack.citations,
            evidence_units=evidence_pack.evidence_units,
            verifier=evidence_pack.verifier,
            graph=evidence_pack.graph,
            debug=debug,
            warnings=evidence_pack.warnings + raw_retrieval.warnings,
        )

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
