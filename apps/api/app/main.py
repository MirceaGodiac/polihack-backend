import asyncio
from contextlib import asynccontextmanager
from typing import Callable
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def _configure_windows_event_loop_policy(
    *,
    platform: str | None = None,
    set_policy: Callable[[asyncio.AbstractEventLoopPolicy], None] | None = None,
) -> bool:
    platform_name = platform if platform is not None else sys.platform
    if not platform_name.startswith("win"):
        return False

    policy_cls = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy_cls is None:
        return False

    setter = set_policy or asyncio.set_event_loop_policy
    setter(policy_cls())
    return True


_configure_windows_event_loop_policy()

from .db import dispose_engine
from .routes.health import health, router as health_router
from .routes.query import router as query_router
from .routes.ingest import router as ingest_router
from .routes.corpus import router as corpus_router
from .routes.legal_units import router as legal_units_router
from .routes.admin import router as admin_router
from .routes.debug import router as debug_router
from .routes.retrieve_raw import router as retrieve_raw_router
from .config import settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await dispose_engine()


def create_app() -> FastAPI:
    app = FastAPI(
        title="LexAI API",
        description="API foundation for LexAI legal assistant services.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router, prefix="/api")
    app.include_router(query_router, prefix="/api")
    app.include_router(ingest_router, prefix="/api")
    app.include_router(corpus_router, prefix="/api")
    app.include_router(legal_units_router, prefix="/api")
    app.include_router(admin_router, prefix="/api")
    app.include_router(debug_router, prefix="/api")
    app.include_router(retrieve_raw_router, prefix="/api")
    app.add_api_route(
        "/health",
        health,
        methods=["GET"],
        tags=["health"],
        include_in_schema=False,
    )
    return app


app = create_app()
