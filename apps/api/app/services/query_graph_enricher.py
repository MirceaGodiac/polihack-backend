from ..schemas import (
    Citation,
    ClaimResult,
    EvidenceUnit,
    GraphEdge,
    GraphNode,
    GraphPayload,
    QueryPlan,
    VerifierStatus,
)
from .query_frame import QueryFrame


class QueryGraphEnricher:
    def enrich(
        self,
        *,
        graph: GraphPayload,
        query_id: str,
        question: str,
        query_plan: QueryPlan,
        query_frame: QueryFrame,
        evidence_units: list[EvidenceUnit],
        citations: list[Citation],
        verifier: VerifierStatus,
    ) -> GraphPayload:
        nodes_by_id = {node.id: node for node in graph.nodes}
        edges_by_id = {edge.id: edge for edge in graph.edges}
        evidence_by_unit_id = {evidence.id: evidence for evidence in evidence_units}
        citations_by_id = {citation.citation_id: citation for citation in citations}
        cited_unit_ids = {citation.legal_unit_id for citation in citations}

        query_node = self._query_node(
            query_id=query_id,
            question=question,
            query_plan=query_plan,
            query_frame=query_frame,
        )
        nodes_by_id[query_node.id] = query_node
        self._mark_existing_legal_unit_nodes(
            nodes_by_id=nodes_by_id,
            cited_unit_ids=cited_unit_ids,
            evidence_by_unit_id=evidence_by_unit_id,
        )

        for evidence in evidence_units:
            self._upsert_legal_unit_node(
                nodes_by_id=nodes_by_id,
                evidence=evidence,
                is_cited=evidence.id in cited_unit_ids,
            )
            edges_by_id.setdefault(
                f"edge:retrieved_for_query:{query_id}:{evidence.id}",
                GraphEdge(
                    id=f"edge:retrieved_for_query:{query_id}:{evidence.id}",
                    source=query_node.id,
                    target=self._legal_unit_node_id(evidence.id),
                    type="retrieved_for_query",
                    weight=self._retrieval_weight(evidence),
                    confidence=evidence.rerank_score,
                    explanation="Evidence unit selected for this query.",
                    metadata={
                        "evidence_id": evidence.evidence_id,
                        "support_role": evidence.support_role,
                        "retrieval_score": evidence.retrieval_score,
                        "rerank_score": evidence.rerank_score,
                        "rank": evidence.rank,
                        "is_cited": evidence.id in cited_unit_ids,
                    },
                ),
            )

        cited_source_id = self._answer_node_id(nodes_by_id) or query_node.id
        for citation in citations:
            evidence = evidence_by_unit_id.get(citation.legal_unit_id)
            if evidence is None:
                continue
            self._upsert_legal_unit_node(
                nodes_by_id=nodes_by_id,
                evidence=evidence,
                is_cited=True,
            )
            edges_by_id.setdefault(
                f"edge:cited_in_answer:{query_id}:{citation.citation_id}:{citation.legal_unit_id}",
                GraphEdge(
                    id=f"edge:cited_in_answer:{query_id}:{citation.citation_id}:{citation.legal_unit_id}",
                    source=cited_source_id,
                    target=self._legal_unit_node_id(citation.legal_unit_id),
                    type="cited_in_answer",
                    weight=1.0,
                    confidence=1.0 if citation.verified else 0.8,
                    explanation="Citation used in the generated answer.",
                    metadata={
                        "citation_id": citation.citation_id,
                        "evidence_id": citation.evidence_id,
                        "legal_unit_id": citation.legal_unit_id,
                        "label": citation.label,
                        "verified": citation.verified,
                    },
                ),
            )

        for claim in verifier.claim_results:
            support_unit_ids = self._support_unit_ids(
                claim=claim,
                citations_by_id=citations_by_id,
                evidence_by_unit_id=evidence_by_unit_id,
            )
            if not support_unit_ids:
                continue
            claim_node = self._claim_node(claim)
            nodes_by_id[claim_node.id] = claim_node
            for unit_id in support_unit_ids:
                evidence = evidence_by_unit_id.get(unit_id)
                if evidence is None:
                    continue
                self._upsert_legal_unit_node(
                    nodes_by_id=nodes_by_id,
                    evidence=evidence,
                    is_cited=unit_id in cited_unit_ids,
                )
                edges_by_id.setdefault(
                    f"edge:supports_claim:{claim.claim_id}:{unit_id}",
                    GraphEdge(
                        id=f"edge:supports_claim:{claim.claim_id}:{unit_id}",
                        source=claim_node.id,
                        target=self._legal_unit_node_id(unit_id),
                        type="supports_claim",
                        weight=claim.support_score
                        if claim.support_score is not None
                        else claim.confidence,
                        confidence=claim.confidence,
                        explanation="Verifier mapped this claim to supporting evidence.",
                        metadata={
                            "claim_id": claim.claim_id,
                            "support_status": claim.status,
                        },
                    ),
                )

        return GraphPayload(
            nodes=list(nodes_by_id.values()),
            edges=list(edges_by_id.values()),
        )

    def _mark_existing_legal_unit_nodes(
        self,
        *,
        nodes_by_id: dict[str, GraphNode],
        cited_unit_ids: set[str],
        evidence_by_unit_id: dict[str, EvidenceUnit],
    ) -> None:
        for node_id, node in list(nodes_by_id.items()):
            if not node.legal_unit_id or node.legal_unit_id in evidence_by_unit_id:
                continue
            nodes_by_id[node_id] = node.model_copy(
                update={
                    "metadata": {
                        **node.metadata,
                        "is_cited": node.legal_unit_id in cited_unit_ids,
                    }
                }
            )

    def _query_node(
        self,
        *,
        query_id: str,
        question: str,
        query_plan: QueryPlan,
        query_frame: QueryFrame,
    ) -> GraphNode:
        return GraphNode(
            id=f"query:{query_id}",
            label=question,
            type="query",
            metadata={
                "query_id": query_id,
                "legal_domain": query_frame.domain or query_plan.legal_domain,
                "intents": query_frame.intents,
                "meta_intents": query_frame.meta_intents,
                "confidence": query_frame.confidence,
            },
        )

    def _upsert_legal_unit_node(
        self,
        *,
        nodes_by_id: dict[str, GraphNode],
        evidence: EvidenceUnit,
        is_cited: bool,
    ) -> None:
        node_id = self._legal_unit_node_id(evidence.id)
        existing = nodes_by_id.get(node_id)
        metadata = {
            **(existing.metadata if existing else {}),
            "is_cited": is_cited,
            "support_role": evidence.support_role,
            "retrieval_score": evidence.retrieval_score,
            "rerank_score": evidence.rerank_score,
            "rank": evidence.rank,
            "source_url": evidence.source_url,
            "article_number": evidence.article_number,
            "paragraph_number": evidence.paragraph_number,
            "letter_number": evidence.letter_number,
        }

        if existing is not None:
            nodes_by_id[node_id] = existing.model_copy(
                update={
                    "legal_unit_id": existing.legal_unit_id or evidence.id,
                    "domain": existing.domain or evidence.legal_domain,
                    "status": existing.status or evidence.status,
                    "importance": existing.importance
                    if existing.importance is not None
                    else evidence.rerank_score,
                    "metadata": metadata,
                }
            )
            return

        nodes_by_id[node_id] = GraphNode(
            id=node_id,
            label=self._legal_unit_label(evidence),
            type=self._legal_unit_node_type(evidence),
            legal_unit_id=evidence.id,
            domain=evidence.legal_domain,
            status=evidence.status,
            importance=evidence.rerank_score,
            metadata=metadata,
        )

    def _claim_node(self, claim: ClaimResult) -> GraphNode:
        return GraphNode(
            id=f"claim:{claim.claim_id}",
            label=self._short_label(claim.claim_text),
            type="cited_claim",
            metadata={
                "claim_id": claim.claim_id,
                "status": claim.status,
                "support_score": claim.support_score,
                "confidence": claim.confidence,
            },
        )

    def _support_unit_ids(
        self,
        *,
        claim: ClaimResult,
        citations_by_id: dict[str, Citation],
        evidence_by_unit_id: dict[str, EvidenceUnit],
    ) -> list[str]:
        support_unit_ids = [
            unit_id
            for unit_id in claim.supporting_unit_ids
            if unit_id in evidence_by_unit_id
        ]
        if support_unit_ids:
            return self._dedupe(support_unit_ids)

        citation_unit_ids = [
            citations_by_id[citation_id].legal_unit_id
            for citation_id in claim.citation_ids
            if citation_id in citations_by_id
            and citations_by_id[citation_id].legal_unit_id in evidence_by_unit_id
        ]
        return self._dedupe(citation_unit_ids)

    def _retrieval_weight(self, evidence: EvidenceUnit) -> float:
        if evidence.retrieval_score is not None:
            return evidence.retrieval_score
        return evidence.rerank_score

    def _answer_node_id(self, nodes_by_id: dict[str, GraphNode]) -> str | None:
        for node in nodes_by_id.values():
            if node.type == "answer":
                return node.id
        return None

    def _legal_unit_node_id(self, unit_id: str) -> str:
        return f"legal_unit:{unit_id}"

    def _legal_unit_label(self, evidence: EvidenceUnit) -> str:
        if evidence.hierarchy_path:
            return evidence.hierarchy_path[-1]
        return evidence.id

    def _legal_unit_node_type(self, evidence: EvidenceUnit) -> str:
        if evidence.point_number:
            return "point"
        if evidence.letter_number:
            return "letter"
        if evidence.paragraph_number:
            return "paragraph"
        if evidence.article_number:
            return "article"
        return "legal_act"

    def _short_label(self, text: str, *, max_length: int = 160) -> str:
        compact = " ".join(text.split())
        if len(compact) <= max_length:
            return compact
        return f"{compact[: max_length - 3].rstrip()}..."

    def _dedupe(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped
