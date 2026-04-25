from uuid import NAMESPACE_URL, uuid5

from ..schemas import GraphPayload, QueryDebugData, QueryRequest, QueryResponse
from .evidence_pack_compiler import EvidencePackCompiler
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
        evidence_pack = await self.evidence_service.build_pack(request, query_id)
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
                evidence_units_count=len(compiled_evidence.evidence_units),
                citations_count=0,
                graph_nodes_count=len(graph.nodes),
                graph_edges_count=len(graph.edges),
                notes=evidence_pack.debug_notes,
            )

        return QueryResponse(
            query_id=query_id,
            question=request.question,
            answer=evidence_pack.answer,
            citations=[],
            evidence_units=compiled_evidence.evidence_units,
            verifier=evidence_pack.verifier,
            graph=graph,
            debug=debug,
            warnings=(
                evidence_pack.warnings
                + raw_retrieval.warnings
                + graph_expansion.warnings
                + legal_ranker.warnings
                + compiled_evidence.warnings
            ),
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
