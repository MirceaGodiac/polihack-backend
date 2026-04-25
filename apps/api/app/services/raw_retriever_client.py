from typing import Any

import httpx

from ..schemas import QueryPlan, RawRetrievalRequest, RawRetrievalResponse
from ..config import settings

RAW_RETRIEVAL_NOT_CONFIGURED = "raw_retrieval_not_configured"
RAW_RETRIEVAL_UNAVAILABLE = "raw_retrieval_unavailable"


class RawRetrieverClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        configured_url = base_url
        if configured_url is None:
            configured_url = settings.raw_retrieval_base_url
        self.base_url = configured_url.strip().rstrip("/") if configured_url else None
        self.timeout_seconds = timeout_seconds

    async def retrieve(
        self,
        plan: QueryPlan,
        *,
        top_k: int = 50,
        debug: bool = False,
    ) -> RawRetrievalResponse:
        request = self.build_request(plan=plan, top_k=top_k, debug=debug)
        request_payload = request.model_dump(mode="json")

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
            response.debug = self._debug_payload(
                request_payload=request_payload,
                response=response,
                fallback_used=False,
                reason=None,
            )
        return response

    def build_request(
        self,
        plan: QueryPlan,
        *,
        top_k: int = 50,
        debug: bool = False,
    ) -> RawRetrievalRequest:
        return RawRetrievalRequest(
            question=plan.question,
            retrieval_filters=plan.retrieval_filters,
            exact_citations=plan.exact_citations,
            top_k=top_k,
            debug=debug,
        )

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
            )
        return response

    def _debug_payload(
        self,
        *,
        request_payload: dict[str, Any],
        response: RawRetrievalResponse,
        fallback_used: bool,
        reason: str | None,
    ) -> dict[str, Any]:
        debug_payload: dict[str, Any] = {
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
        return debug_payload
