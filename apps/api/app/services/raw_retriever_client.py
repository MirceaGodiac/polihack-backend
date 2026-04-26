from collections.abc import Callable
import inspect
from typing import Any

import httpx

from ..schemas import QueryPlan, RawRetrievalRequest, RawRetrievalResponse
from ..config import settings
from .query_frame import QueryFrame

RAW_RETRIEVAL_NOT_CONFIGURED = "raw_retrieval_not_configured"
RAW_RETRIEVAL_UNAVAILABLE = "raw_retrieval_unavailable"


class RawRetrieverClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: float = 5.0,
        *,
        use_internal: bool | None = None,
        internal_retriever_factory: Callable[[], Any] | None = None,
    ) -> None:
        configured_url = base_url
        if configured_url is None:
            configured_url = settings.raw_retrieval_base_url
        self.base_url = configured_url.strip().rstrip("/") if configured_url else None
        self.timeout_seconds = timeout_seconds
        self.use_internal = (self.base_url is None) if use_internal is None else use_internal
        self.internal_retriever_factory = internal_retriever_factory

    async def retrieve(
        self,
        plan: QueryPlan,
        *,
        query_frame: QueryFrame | None = None,
        top_k: int = 50,
        debug: bool = False,
    ) -> RawRetrievalResponse:
        request = self.build_request(
            plan=plan,
            query_frame=query_frame,
            top_k=top_k,
            debug=debug,
        )
        request_payload = request.model_dump(mode="json")

        if self.use_internal:
            return await self._retrieve_internal(
                request=request,
                request_payload=request_payload,
                debug=debug,
            )

        if not self.base_url:
            return self._fallback_response(
                warning=RAW_RETRIEVAL_NOT_CONFIGURED,
                reason="raw retrieval endpoint is not configured",
                request_payload=request_payload,
                debug=debug,
            )

        try:
            response_payload = await self._send_payload(request_payload)
            response = RawRetrievalResponse.model_validate(response_payload)
        except Exception:
            return self._fallback_response(
                warning=RAW_RETRIEVAL_UNAVAILABLE,
                reason="raw retrieval endpoint request failed",
                request_payload=request_payload,
                debug=debug,
            )

        if debug:
            raw_debug = response.debug
            response.debug = self._debug_payload(
                request_payload=request_payload,
                response=response,
                fallback_used=False,
                reason=None,
                backend="http",
                raw_retriever_debug=raw_debug,
            )
        return response

    def build_request(
        self,
        plan: QueryPlan,
        *,
        query_frame: QueryFrame | None = None,
        top_k: int = 50,
        debug: bool = False,
    ) -> RawRetrievalRequest:
        return RawRetrievalRequest(
            question=plan.question,
            filters=plan.retrieval_filters,
            retrieval_filters=plan.retrieval_filters,
            query_frame=(
                query_frame.model_dump(mode="json") if query_frame is not None else None
            ),
            exact_citations=[
                citation.model_dump(mode="json") for citation in plan.exact_citations
            ],
            top_k=top_k,
            debug=debug,
        )

    async def _retrieve_internal(
        self,
        *,
        request: RawRetrievalRequest,
        request_payload: dict[str, Any],
        debug: bool,
    ) -> RawRetrievalResponse:
        try:
            response = await self._call_internal_retriever(request)
        except Exception:
            return self._fallback_response(
                warning=RAW_RETRIEVAL_UNAVAILABLE,
                reason="internal raw retrieval failed",
                request_payload=request_payload,
                debug=debug,
                backend="internal",
            )

        if debug:
            raw_debug = response.debug
            response.debug = self._debug_payload(
                request_payload=request_payload,
                response=response,
                fallback_used=False,
                reason=None,
                backend="internal",
                raw_retriever_debug=raw_debug,
            )
        return response

    async def _call_internal_retriever(
        self,
        request: RawRetrievalRequest,
    ) -> RawRetrievalResponse:
        if self.internal_retriever_factory is not None:
            retriever = self.internal_retriever_factory()
            if inspect.isawaitable(retriever):
                retriever = await retriever
            return await retriever.retrieve(request)

        from ..db.session import session_context
        from .raw_retriever import PostgresRawRetrievalStore, RawRetriever

        async with session_context() as session:
            retriever = RawRetriever(PostgresRawRetrievalStore(session))
            return await retriever.retrieve(request)

    async def _send_payload(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(self._endpoint_url(), json=request_payload)
            response.raise_for_status()
            return response.json()

    def _endpoint_url(self) -> str:
        if not self.base_url:
            raise RuntimeError("raw retrieval endpoint is not configured")
        if self.base_url.endswith("/api/retrieve/raw"):
            return self.base_url
        return f"{self.base_url}/api/retrieve/raw"

    def _fallback_response(
        self,
        *,
        warning: str,
        reason: str,
        request_payload: dict[str, Any],
        debug: bool,
        backend: str = "http",
    ) -> RawRetrievalResponse:
        response = RawRetrievalResponse(
            candidates=[],
            retrieval_methods=[],
            warnings=[warning],
            debug=None,
        )
        if debug:
            response.debug = self._debug_payload(
                request_payload=request_payload,
                response=response,
                fallback_used=True,
                reason=reason,
                backend=backend,
                raw_retriever_debug=None,
            )
        return response

    def _debug_payload(
        self,
        *,
        request_payload: dict[str, Any],
        response: RawRetrievalResponse,
        fallback_used: bool,
        reason: str | None,
        backend: str,
        raw_retriever_debug: dict[str, Any] | None,
    ) -> dict[str, Any]:
        debug_payload: dict[str, Any] = {
            "backend": backend,
            "request_payload": request_payload,
            "response_summary": {
                "candidate_count": len(response.candidates),
                "retrieval_methods": response.retrieval_methods,
                "warnings": response.warnings,
            },
            "fallback_used": fallback_used,
        }
        if reason:
            debug_payload["reason"] = reason
        if raw_retriever_debug is not None:
            debug_payload["raw_retriever_debug"] = raw_retriever_debug
        return debug_payload
