from fastapi import APIRouter, HTTPException, status
from sqlalchemy import text

from ..db import session_context

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "lexai-api"}


@router.get("/health/db")
async def db_health() -> dict:
    try:
        async with session_context() as session:
            select_one = (await session.execute(text("SELECT 1"))).scalar_one()
            server_version = (
                await session.execute(text("SHOW server_version"))
            ).scalar_one()
            extensions = (
                await session.execute(
                    text("SELECT extname FROM pg_extension ORDER BY extname")
                )
            ).scalars().all()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "error",
                "service": "postgres",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "error",
                "service": "postgres",
                "message": "database unavailable",
            },
        ) from exc

    return {
        "status": "ok",
        "service": "postgres",
        "select_1": select_one,
        "server_version": server_version,
        "extensions": list(extensions),
    }
