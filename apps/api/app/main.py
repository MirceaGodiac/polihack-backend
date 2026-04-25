from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.app.db import engine, get_session


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    yield
    await engine.dispose()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health(session: AsyncSession = Depends(get_session)) -> dict:
    version = (await session.execute(text("SHOW server_version"))).scalar_one()
    extensions = (
        await session.execute(
            text("SELECT extname FROM pg_extension ORDER BY extname")
        )
    ).scalars().all()
    return {"status": "ok", "postgres_version": version, "extensions": extensions}
