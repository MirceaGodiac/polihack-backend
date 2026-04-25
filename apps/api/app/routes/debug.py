from __future__ import annotations

import os
import secrets
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse


router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/embeddings-health", include_in_schema=False)
async def embeddings_health(secret: str | None = Query(default=None)) -> JSONResponse:
    """Temporary debug endpoint for Backend Railway -> Ollama embeddings checks."""
    _authorize_debug_request(secret)

    base_url = _env_text("EMBEDDING_BASE_URL")
    model = _env_text("EMBEDDING_MODEL")
    expected_dim_raw = _env_text("EMBEDDING_DIM")
    response_payload: dict[str, Any] = {
        "ok": False,
        "base_url": base_url,
        "model": model,
        "expected_dim": None,
        "models_status": None,
        "embeddings_status": None,
    }

    missing = [
        name
        for name, value in (
            ("EMBEDDING_BASE_URL", base_url),
            ("EMBEDDING_MODEL", model),
            ("EMBEDDING_DIM", expected_dim_raw),
        )
        if not value
    ]
    if missing:
        response_payload["error"] = "missing env vars: " + ", ".join(missing)
        return JSONResponse(response_payload)

    try:
        expected_dim = int(expected_dim_raw)
    except (TypeError, ValueError):
        response_payload["error"] = "EMBEDDING_DIM must be an integer"
        return JSONResponse(response_payload)

    if expected_dim <= 0:
        response_payload["expected_dim"] = expected_dim
        response_payload["error"] = "EMBEDDING_DIM must be positive"
        return JSONResponse(response_payload)

    response_payload["expected_dim"] = expected_dim
    models_url = _endpoint_url(base_url, "models")
    embeddings_url = _endpoint_url(base_url, "embeddings")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
            models_response = await client.get(models_url)
            response_payload["models_status"] = models_response.status_code

            embeddings_response = await client.post(
                embeddings_url,
                json={"model": model, "input": ["test"]},
            )
            response_payload["embeddings_status"] = embeddings_response.status_code
    except httpx.HTTPError as exc:
        response_payload["error"] = f"embedding provider request failed: {exc.__class__.__name__}"
        return JSONResponse(response_payload)

    if not 200 <= models_response.status_code < 300:
        response_payload["error"] = f"/models returned HTTP {models_response.status_code}"
        return JSONResponse(response_payload)

    if not 200 <= embeddings_response.status_code < 300:
        response_payload["error"] = (
            f"/embeddings returned HTTP {embeddings_response.status_code}"
        )
        return JSONResponse(response_payload)

    try:
        body = embeddings_response.json()
    except ValueError:
        response_payload["error"] = "embeddings response is not valid JSON"
        return JSONResponse(response_payload)

    embedding = _extract_first_embedding(body)
    if embedding is None:
        response_payload["error"] = "embeddings response missing data[0].embedding"
        return JSONResponse(response_payload)

    embedding_dim = len(embedding)
    response_payload["embedding_dim"] = embedding_dim
    if embedding_dim != expected_dim:
        response_payload["error"] = (
            f"embedding dimension mismatch: expected {expected_dim}, got {embedding_dim}"
        )
        return JSONResponse(response_payload)

    response_payload["ok"] = True
    return JSONResponse(response_payload)


def _authorize_debug_request(secret: str | None) -> None:
    configured_secret = _env_text("ADMIN_INGEST_SECRET")
    if configured_secret:
        if not secret or not secrets.compare_digest(secret, configured_secret):
            raise HTTPException(status_code=403, detail="Invalid debug secret")
        return

    if _env_text("APP_ENV").casefold() == "production":
        raise HTTPException(
            status_code=503,
            detail="ADMIN_INGEST_SECRET is required for debug endpoint in production",
        )


def _env_text(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _endpoint_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _extract_first_embedding(payload: object) -> list[object] | None:
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if not isinstance(first, dict):
        return None
    embedding = first.get("embedding")
    return embedding if isinstance(embedding, list) else None
