import inspect
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
)
from .answer_repair import AnswerRepair
from .citation_verifier import CitationVerifier
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
from .query_frame import QueryFrameBuilder
from .query_understanding import QueryUnderstanding
from .raw_retriever_client import RAW_RETRIEVAL_NOT_CONFIGURED, RawRetrieverClient


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
        citation_verifier: CitationVerifier | None = None,
        answer_repair: AnswerRepair | None = None,
        query_frame_builder: QueryFrameBuilder | None = None,
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
        self.citation_verifier = citation_verifier or CitationVerifier()
        self.answer_repair = answer_repair or AnswerRepair()
        self.query_frame_builder = query_frame_builder or QueryFrameBuilder()

    async def run(self, request: QueryRequest) -> QueryResponse:
        query_id = self._query_id(request)
        query_plan = self.query_understanding.build_plan(request)
        query_frame = self.query_frame_builder.build(
            question=request.question,
            plan=query_plan,
        )
        raw_retrieval = await self._retrieve_raw(
            query_plan=query_plan,
            query_frame=query_frame,
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
            query_frame=query_frame,
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
        answer = self._answer_payload(draft_answer)
        citations = self._citations_from_draft(
            draft_answer,
            compiled_evidence.evidence_units,
        )
        verification = self.citation_verifier.verify(
            answer=answer,
            citations=citations,
            evidence_units=compiled_evidence.evidence_units,
            debug=request.debug,
        )
        answer = self._answer_after_verification(
            answer,
            verifier_passed=verification.verifier.verifier_passed,
        )
        citations = self._mark_verified_citations(
            citations,
            verification.verified_citation_ids,
        )
        pre_repair_warnings = self._dedupe(
            raw_retrieval.warnings
            + graph_expansion.warnings
            + legal_ranker.warnings
            + compiled_evidence.warnings
            + self._generation_warnings(draft_answer, verifier_ran=True)
            + verification.warnings
        )
        repair = self.answer_repair.repair(
            answer=answer,
            citations=citations,
            evidence_units=compiled_evidence.evidence_units,
            verifier=verification.verifier,
            warnings=pre_repair_warnings,
            debug=request.debug,
        )
        answer = repair.answer
        citations = repair.citations
        verifier = repair.verifier
        graph = GraphPayload(
            nodes=compiled_evidence.graph_nodes,
            edges=compiled_evidence.graph_edges,
        )
        debug = None
        if request.debug:
            debug = QueryDebugData(
                orchestrator=self.__class__.__name__,
                evidence_service=self.evidence_service.__class__.__name__,
                retrieval_mode=self._retrieval_mode(raw_retrieval),
                query_understanding=query_plan,
                query_frame=query_frame.model_dump(mode="json"),
                retrieval=raw_retrieval.debug,
                graph_expansion=graph_expansion.debug,
                legal_ranker=legal_ranker.debug,
                evidence_pack=compiled_evidence.debug,
                generation=self._generation_debug(draft_answer, verifier_ran=True),
                verifier=verification.debug,
                answer_repair=repair.debug,
                evidence_units_count=len(compiled_evidence.evidence_units),
                citations_count=len(citations),
                graph_nodes_count=len(graph.nodes),
                graph_edges_count=len(graph.edges),
                notes=[
                    "GenerationAdapter, CitationVerifier V1, and AnswerRepair V1 ran over compiled EvidencePack.",
                ],
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
            warnings=repair.warnings,
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

    def _answer_after_verification(
        self,
        answer: AnswerPayload,
        *,
        verifier_passed: bool,
    ) -> AnswerPayload:
        status = (
            "a trecut verificarea determinista CitationVerifier V1; "
            "LegalConfidence final nu este calculat inca"
            if verifier_passed
            else "a fost verificat determinist de CitationVerifier V1, "
            "dar verifier_passed=false; vezi warnings"
        )
        replacements = {
            "nu a fost verificat final de CitationVerifier": status,
            "CitationVerifier real nu a rulat inca": status,
        }
        return answer.model_copy(
            update={
                "short_answer": self._replace_status_text(
                    answer.short_answer,
                    replacements,
                ),
                "detailed_answer": self._replace_status_text(
                    answer.detailed_answer,
                    replacements,
                )
                if answer.detailed_answer
                else None,
            }
        )

    def _replace_status_text(
        self,
        text: str,
        replacements: dict[str, str],
    ) -> str:
        updated = text
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        return updated

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

    def _mark_verified_citations(
        self,
        citations: list[Citation],
        verified_citation_ids: set[str],
    ) -> list[Citation]:
        return [
            citation.model_copy(
                update={"verified": citation.citation_id in verified_citation_ids}
            )
            for citation in citations
        ]

    def _generation_debug(
        self,
        draft_answer: DraftAnswer,
        *,
        verifier_ran: bool,
    ) -> dict[str, object]:
        return {
            "generation_mode": draft_answer.generation_mode,
            "evidence_unit_count_used": len(draft_answer.used_evidence_unit_ids),
            "warnings": self._generation_warnings(
                draft_answer,
                verifier_ran=verifier_ran,
            ),
            "citation_unit_ids": [
                citation.unit_id for citation in draft_answer.citations
            ],
        }

    def _generation_warnings(
        self,
        draft_answer: DraftAnswer,
        *,
        verifier_ran: bool,
    ) -> list[str]:
        if not verifier_ran:
            return draft_answer.warnings
        return [
            warning
            for warning in draft_answer.warnings
            if warning != GENERATION_UNVERIFIED_WARNING
        ]

    def _dedupe(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped

    def _retrieval_mode(self, raw_retrieval) -> str:
        if RAW_RETRIEVAL_NOT_CONFIGURED in raw_retrieval.warnings:
            return "fallback_unconfigured"
        return f"raw_retriever_client:{self.raw_retriever_client.__class__.__name__}"

    async def _retrieve_raw(self, *, query_plan, query_frame, debug: bool):
        retrieve = self.raw_retriever_client.retrieve
        parameters = inspect.signature(retrieve).parameters
        if "query_frame" in parameters:
            return await retrieve(
                query_plan,
                query_frame=query_frame,
                top_k=50,
                debug=debug,
            )
        return await retrieve(query_plan, top_k=50, debug=debug)

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
