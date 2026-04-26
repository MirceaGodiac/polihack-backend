from collections import OrderedDict

from ..schemas import QueryGraphResponse, QueryResponse


class QueryResponseStore:
    """Demo/dev in-memory query response store, not production persistence."""

    def __init__(self, *, max_size: int = 100) -> None:
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        self.max_size = max_size
        self._responses: OrderedDict[str, QueryResponse] = OrderedDict()

    def save(self, response: QueryResponse) -> None:
        if response.query_id in self._responses:
            self._responses.pop(response.query_id)
        self._responses[response.query_id] = response
        while len(self._responses) > self.max_size:
            self._responses.popitem(last=False)

    def get(self, query_id: str) -> QueryResponse | None:
        return self._responses.get(query_id)

    def get_graph(self, query_id: str) -> QueryGraphResponse | None:
        response = self.get(query_id)
        if response is None:
            return None

        cited_unit_ids = self._cited_unit_ids(response)
        highlighted_node_ids = self._highlighted_node_ids(response, cited_unit_ids)
        highlighted_edge_ids = self._highlighted_edge_ids(
            response,
            highlighted_node_ids,
        )
        reasoning_path = self._reasoning_path(
            response,
            cited_unit_ids,
        )

        return QueryGraphResponse(
            query_id=response.query_id,
            question=response.question,
            graph=response.graph,
            highlighted_node_ids=highlighted_node_ids,
            highlighted_edge_ids=highlighted_edge_ids,
            cited_unit_ids=cited_unit_ids,
            reasoning_path=reasoning_path,
            verifier_summary={
                "groundedness_score": response.verifier.groundedness_score,
                "claims_total": response.verifier.claims_total,
                "claims_supported": response.verifier.claims_supported,
                "claims_unsupported": response.verifier.claims_unsupported,
                "citations_checked": response.verifier.citations_checked,
                "verifier_passed": response.verifier.verifier_passed,
            },
        )

    def clear(self) -> None:
        self._responses.clear()

    def _cited_unit_ids(self, response: QueryResponse) -> list[str]:
        cited_unit_ids: list[str] = []
        for citation in response.citations:
            if citation.legal_unit_id not in cited_unit_ids:
                cited_unit_ids.append(citation.legal_unit_id)
        return cited_unit_ids

    def _highlighted_node_ids(
        self,
        response: QueryResponse,
        cited_unit_ids: list[str],
    ) -> list[str]:
        cited_unit_id_set = set(cited_unit_ids)
        highlighted_node_ids: list[str] = []
        for node in response.graph.nodes:
            if (
                node.type == "query"
                or node.legal_unit_id in cited_unit_id_set
                or node.metadata.get("is_cited") is True
            ):
                highlighted_node_ids.append(node.id)
        return highlighted_node_ids

    def _highlighted_edge_ids(
        self,
        response: QueryResponse,
        highlighted_node_ids: list[str],
    ) -> list[str]:
        highlighted_node_id_set = set(highlighted_node_ids)
        highlighted_edge_ids: list[str] = []
        for edge in response.graph.edges:
            if edge.type in {"cited_in_answer", "supports_claim"}:
                highlighted_edge_ids.append(edge.id)
                continue
            if (
                edge.type == "retrieved_for_query"
                and edge.source in highlighted_node_id_set
                and edge.target in highlighted_node_id_set
            ):
                highlighted_edge_ids.append(edge.id)
        return highlighted_edge_ids

    def _reasoning_path(
        self,
        response: QueryResponse,
        cited_unit_ids: list[str],
    ) -> list[str]:
        query_node_id = next(
            (node.id for node in response.graph.nodes if node.type == "query"),
            None,
        )
        if query_node_id is None:
            return []

        cited_unit_id_set = set(cited_unit_ids)
        cited_node_ids = [
            node.id
            for node in response.graph.nodes
            if node.legal_unit_id in cited_unit_id_set
        ]
        if not cited_node_ids:
            return []
        return [query_node_id, *cited_node_ids]
