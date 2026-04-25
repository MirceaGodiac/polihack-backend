import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.app.main import app
from apps.api.app.routes import health as health_module

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeScalarResult:
    def __init__(self, scalar=None, scalar_list=None):
        self._scalar = scalar
        self._scalar_list = scalar_list or []

    def scalar_one(self):
        return self._scalar

    def scalars(self):
        return self

    def all(self):
        return self._scalar_list


class FakeSession:
    async def execute(self, statement):
        query = str(statement)
        if query == "SELECT 1":
            return FakeScalarResult(scalar=1)
        if query == "SHOW server_version":
            return FakeScalarResult(scalar="16.0")
        if query == "SELECT extname FROM pg_extension ORDER BY extname":
            return FakeScalarResult(scalar_list=["plpgsql", "vector"])
        raise AssertionError(f"Unexpected query: {query}")


def test_app_imports_without_database_url(tmp_path):
    env = os.environ.copy()
    env.pop("DATABASE_URL", None)
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from apps.api.app.main import app; print(app.title)",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "LexAI API" in result.stdout


def test_api_health_does_not_require_database():
    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "lexai-api"}


def test_legacy_health_does_not_require_database():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "lexai-api"}


def test_api_health_db_returns_503_without_database_url(monkeypatch):
    import apps.api.app.db.session as db_module

    monkeypatch.setattr(db_module.settings, "database_url", None)
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_sessionmaker", None)

    with TestClient(app) as client:
        response = client.get("/api/health/db")

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "status": "error",
            "service": "postgres",
            "message": "DATABASE_URL is not configured",
        }
    }


def test_api_health_db_uses_separate_database_check(monkeypatch):
    @asynccontextmanager
    async def fake_session_context():
        yield FakeSession()

    monkeypatch.setattr(health_module, "session_context", fake_session_context)

    with TestClient(app) as client:
        response = client.get("/api/health/db")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "postgres",
        "select_1": 1,
        "server_version": "16.0",
        "extensions": ["plpgsql", "vector"],
    }
