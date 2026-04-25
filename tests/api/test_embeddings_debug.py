import httpx
from fastapi.testclient import TestClient

from apps.api.app.main import app
from apps.api.app.routes import debug as debug_module


def test_embeddings_health_returns_403_with_wrong_secret(monkeypatch):
    monkeypatch.setenv("ADMIN_INGEST_SECRET", "correct-secret")
    monkeypatch.setenv("APP_ENV", "production")

    with TestClient(app) as client:
        response = client.get("/api/debug/embeddings-health?secret=wrong-secret")

    assert response.status_code == 403


def test_embeddings_health_returns_ok_with_mocked_httpx(monkeypatch):
    calls: list[tuple] = []

    class FakeAsyncClient:
        def __init__(self, *, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            calls.append(("timeout", self.timeout.read))
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url):
            calls.append(("GET", url))
            return httpx.Response(200, json={"data": [{"id": "qwen3-embedding:4b"}]})

        async def post(self, url, *, json):
            calls.append(("POST", url, json))
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]},
            )

    _set_embedding_env(monkeypatch, expected_dim="3")
    monkeypatch.setattr(debug_module.httpx, "AsyncClient", FakeAsyncClient)

    with TestClient(app) as client:
        response = client.get("/api/debug/embeddings-health")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "ok": True,
        "base_url": "http://ollama.railway.internal:11434/v1/",
        "model": "qwen3-embedding:4b",
        "expected_dim": 3,
        "models_status": 200,
        "embeddings_status": 200,
        "embedding_dim": 3,
    }
    assert calls == [
        ("timeout", 120.0),
        ("GET", "http://ollama.railway.internal:11434/v1/models"),
        (
            "POST",
            "http://ollama.railway.internal:11434/v1/embeddings",
            {"model": "qwen3-embedding:4b", "input": ["test"]},
        ),
    ]


def test_embeddings_health_does_not_return_full_vector(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, *, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url):
            return httpx.Response(200, json={"data": []})

        async def post(self, url, *, json):
            return httpx.Response(
                200,
                json={"data": [{"embedding": [0.11, 0.22, 0.33]}]},
            )

    _set_embedding_env(monkeypatch, expected_dim="3")
    monkeypatch.setattr(debug_module.httpx, "AsyncClient", FakeAsyncClient)

    with TestClient(app) as client:
        response = client.get("/api/debug/embeddings-health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert "embedding" not in payload
    assert "[0.11" not in response.text
    assert "0.22" not in response.text


def test_embeddings_health_returns_error_on_dimension_mismatch(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, *, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url):
            return httpx.Response(200, json={"data": []})

        async def post(self, url, *, json):
            return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})

    _set_embedding_env(monkeypatch, expected_dim="3")
    monkeypatch.setattr(debug_module.httpx, "AsyncClient", FakeAsyncClient)

    with TestClient(app) as client:
        response = client.get("/api/debug/embeddings-health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert payload["expected_dim"] == 3
    assert payload["embedding_dim"] == 2
    assert "dimension mismatch" in payload["error"]


def test_embeddings_health_returns_error_for_missing_env_vars(monkeypatch):
    monkeypatch.delenv("ADMIN_INGEST_SECRET", raising=False)
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)

    with TestClient(app) as client:
        response = client.get("/api/debug/embeddings-health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert "missing env vars" in payload["error"]
    assert "EMBEDDING_BASE_URL" in payload["error"]
    assert "EMBEDDING_MODEL" in payload["error"]
    assert "EMBEDDING_DIM" in payload["error"]


def _set_embedding_env(monkeypatch, *, expected_dim: str) -> None:
    monkeypatch.delenv("ADMIN_INGEST_SECRET", raising=False)
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://ollama.railway.internal:11434/v1/")
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen3-embedding:4b")
    monkeypatch.setenv("EMBEDDING_DIM", expected_dim)
