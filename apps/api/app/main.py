from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import dispose_engine
from .routes.health import health, router as health_router
from .routes.query import router as query_router
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
    app.add_api_route(
        "/health",
        health,
        methods=["GET"],
        tags=["health"],
        include_in_schema=False,
    )
    return app


app = create_app()
