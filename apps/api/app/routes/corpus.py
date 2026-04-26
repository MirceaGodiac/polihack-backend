from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from sqlalchemy import text

from ingestion.normalizer import contains_romanian_mojibake

from ..config import settings
from ..db import session_context


router = APIRouter(prefix="/corpus", tags=["corpus"])


@router.get("/stats")
async def corpus_stats() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "degraded",
        "database_url_configured": bool((settings.database_url or "").strip()),
        "db_reachable": False,
        "tables": {
            "legal_units": {"exists": False},
            "legal_edges": {"exists": False},
        },
        "counts": {
            "total_legal_units": None,
            "active_units": None,
            "munca_active_units": None,
            "codul_muncii_units": None,
            "legal_edges": None,
            "empty_raw_text": None,
            "mojibake_raw_text": None,
        },
        "mojibake_sample_unit_ids": [],
        "warnings": [],
    }
    if not payload["database_url_configured"]:
        payload["warnings"].append("database_url_not_configured")
        return payload

    try:
        async with session_context() as session:
            await session.execute(text("SELECT 1"))
            payload["db_reachable"] = True
            legal_units_exists = await _table_exists(session, "legal_units")
            legal_edges_exists = await _table_exists(session, "legal_edges")
            payload["tables"] = {
                "legal_units": {"exists": legal_units_exists},
                "legal_edges": {"exists": legal_edges_exists},
            }
            if not legal_units_exists:
                payload["warnings"].append("table_missing:legal_units")
            if not legal_edges_exists:
                payload["warnings"].append("table_missing:legal_edges")
            if legal_units_exists:
                payload["counts"].update(await _legal_unit_counts(session))
                mojibake_count, mojibake_samples = await _mojibake_raw_text_stats(
                    session,
                )
                payload["counts"]["mojibake_raw_text"] = mojibake_count
                payload["mojibake_sample_unit_ids"] = mojibake_samples
                if mojibake_count:
                    payload["warnings"].append("raw_text_contains_romanian_mojibake")
            if legal_edges_exists:
                payload["counts"]["legal_edges"] = await _count(
                    session,
                    "SELECT COUNT(*) FROM legal_edges",
                )
    except Exception as exc:
        payload["warnings"].append(f"corpus_stats_unavailable:{type(exc).__name__}")

    payload["status"] = (
        "ok"
        if payload["db_reachable"]
        and payload["tables"]["legal_units"]["exists"]
        and payload["tables"]["legal_edges"]["exists"]
        else "degraded"
    )
    return payload


async def _table_exists(session: Any, table_name: str) -> bool:
    result = await session.execute(
        text("SELECT to_regclass(:table_name)"),
        {"table_name": f"public.{table_name}"},
    )
    return result.scalar_one() is not None


async def _legal_unit_counts(session: Any) -> dict[str, int]:
    return {
        "total_legal_units": await _count(
            session,
            "SELECT COUNT(*) FROM legal_units",
        ),
        "active_units": await _count(
            session,
            "SELECT COUNT(*) FROM legal_units WHERE status = 'active'",
        ),
        "munca_active_units": await _count(
            session,
            (
                "SELECT COUNT(*) FROM legal_units "
                "WHERE status = 'active' AND legal_domain = 'munca'"
            ),
        ),
        "codul_muncii_units": await _count(
            session,
            (
                "SELECT COUNT(*) FROM legal_units "
                "WHERE law_id = 'ro.codul_muncii' "
                "OR lower(law_title) LIKE '%codul muncii%'"
            ),
        ),
        "empty_raw_text": await _count(
            session,
            (
                "SELECT COUNT(*) FROM legal_units "
                "WHERE raw_text IS NULL OR btrim(raw_text) = ''"
            ),
        ),
    }


async def _mojibake_raw_text_stats(session: Any) -> tuple[int, list[str]]:
    result = await session.execute(text("SELECT id, raw_text FROM legal_units"))
    mojibake_count = 0
    sample_unit_ids: list[str] = []
    for row in result.all():
        unit_id = str(row[0])
        raw_text = str(row[1] or "")
        if not contains_romanian_mojibake(raw_text):
            continue
        mojibake_count += 1
        if len(sample_unit_ids) < 10:
            sample_unit_ids.append(unit_id)
    return mojibake_count, sample_unit_ids


async def _count(session: Any, sql: str) -> int:
    result = await session.execute(text(sql))
    return int(result.scalar_one() or 0)
