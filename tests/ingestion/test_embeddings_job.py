import json
import sys

import httpx
import pytest

import scripts.generate_embeddings as generate_embeddings_cli
from ingestion.chunks import stable_text_hash
from ingestion.embeddings import (
    DeterministicFakeEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    generate_embeddings,
)


class StaticEmbeddingProvider:
    def __init__(self, vectors):
        self.vectors = vectors
        self.received_texts = []

    def embed_texts(self, texts: list[str], model_name: str) -> list[list[float]]:
        self.received_texts.extend(texts)
        return self.vectors


def test_valid_jsonl_writes_chunk_based_embedding_output(tmp_path):
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    records = [
        _input_record("1", embedding_text="Retrieval text 1"),
        _input_record("2", embedding_text="Retrieval text 2"),
    ]
    _write_jsonl(input_path, records)

    provider = DeterministicFakeEmbeddingProvider(dimension=4)
    summary = generate_embeddings(
        input_path=input_path,
        output_path=output_path,
        provider=provider,
        model_name="fake-model",
        batch_size=2,
        expected_dim=4,
    )

    output_records = _read_jsonl(output_path)
    assert summary.read_count == 2
    assert summary.written_count == 2
    assert summary.embedding_dim == 4
    assert len(output_records) == 2
    assert output_records[0]["record_id"] == records[0]["record_id"]
    assert output_records[0]["chunk_id"] == records[0]["chunk_id"]
    assert output_records[0]["legal_unit_id"] == records[0]["legal_unit_id"]
    assert output_records[0]["law_id"] == records[0]["law_id"]
    assert output_records[0]["model_name"] == "fake-model"
    assert output_records[0]["embedding_dim"] == 4
    assert output_records[0]["text_hash"] == records[0]["text_hash"]
    assert "unit_id" not in output_records[0]
    assert len(output_records[0]["embedding"]) == 4
    assert all(isinstance(value, float) for value in output_records[0]["embedding"])


def test_provider_receives_embedding_text_not_citable_text(tmp_path):
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    record = _input_record(
        "1",
        text="LegalUnit.raw_text citable source",
        embedding_text="LegalChunk.embedding_text retrieval only",
    )
    _write_jsonl(input_path, [record])

    provider = DeterministicFakeEmbeddingProvider(dimension=4)
    generate_embeddings(
        input_path=input_path,
        output_path=output_path,
        provider=provider,
        model_name="fake-model",
        expected_dim=4,
    )

    assert provider.received_texts == ["LegalChunk.embedding_text retrieval only"]


def test_empty_embedding_text_is_skipped_with_warning(tmp_path):
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    record = _input_record("1", embedding_text="", text_hash=stable_text_hash(""))
    _write_jsonl(input_path, [record])

    provider = DeterministicFakeEmbeddingProvider(dimension=4)
    summary = generate_embeddings(
        input_path=input_path,
        output_path=output_path,
        provider=provider,
        model_name="fake-model",
        expected_dim=4,
    )

    assert summary.read_count == 1
    assert summary.written_count == 0
    assert summary.skipped_empty_text_count == 1
    assert provider.received_texts == []
    assert "skipped empty embedding_text" in summary.warnings[0]
    assert "record_id=embedding.chunk.ro.test.1.0" in summary.warnings[0]
    assert "embedding_text_length=0" in summary.warnings[0]
    assert output_path.read_text(encoding="utf-8") == ""


def test_expected_dim_mismatch_fails_explicitly(tmp_path):
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(input_path, [_input_record("1", embedding_text="Retrieval text")])

    provider = DeterministicFakeEmbeddingProvider(dimension=3)
    with pytest.raises(ValueError, match="dimension mismatch"):
        generate_embeddings(
            input_path=input_path,
            output_path=output_path,
            provider=provider,
            model_name="fake-model",
            expected_dim=4,
        )

    assert not output_path.exists()


def test_expected_dim_absent_is_deduced_from_first_valid_vector(tmp_path):
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(
        input_path,
        [
            _input_record("1", embedding_text="Retrieval text 1"),
            _input_record("2", embedding_text="Retrieval text 2"),
        ],
    )

    provider = DeterministicFakeEmbeddingProvider(dimension=5)
    summary = generate_embeddings(
        input_path=input_path,
        output_path=output_path,
        provider=provider,
        model_name="fake-model",
        expected_dim=None,
    )

    output_records = _read_jsonl(output_path)
    assert summary.embedding_dim == 5
    assert [record["embedding_dim"] for record in output_records] == [5, 5]
    assert [len(record["embedding"]) for record in output_records] == [5, 5]


def test_resume_does_not_duplicate_existing_output(tmp_path):
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(input_path, [_input_record("1", embedding_text="Retrieval text")])

    first_provider = DeterministicFakeEmbeddingProvider(dimension=4)
    generate_embeddings(
        input_path=input_path,
        output_path=output_path,
        provider=first_provider,
        model_name="fake-model",
        expected_dim=4,
    )

    second_provider = DeterministicFakeEmbeddingProvider(dimension=4)
    summary = generate_embeddings(
        input_path=input_path,
        output_path=output_path,
        provider=second_provider,
        model_name="fake-model",
        expected_dim=4,
        resume=True,
    )

    assert summary.read_count == 1
    assert summary.written_count == 0
    assert summary.skipped_resume_count == 1
    assert second_provider.received_texts == []
    assert len(_read_jsonl(output_path)) == 1


def test_dry_run_does_not_call_provider_or_write_output(tmp_path):
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(input_path, [_input_record("1", embedding_text="Retrieval text")])

    provider = DeterministicFakeEmbeddingProvider(dimension=4)
    summary = generate_embeddings(
        input_path=input_path,
        output_path=output_path,
        provider=provider,
        model_name="fake-model",
        expected_dim=4,
        dry_run=True,
    )

    assert summary.read_count == 1
    assert summary.written_count == 0
    assert provider.received_texts == []
    assert not output_path.exists()


def test_job_does_not_require_database_url(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(input_path, [_input_record("1", embedding_text="Retrieval text")])

    provider = DeterministicFakeEmbeddingProvider(dimension=4)
    summary = generate_embeddings(
        input_path=input_path,
        output_path=output_path,
        provider=provider,
        model_name="fake-model",
        expected_dim=4,
    )

    assert summary.written_count == 1


def test_output_preserves_identity_hash_and_metadata(tmp_path):
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    record = _input_record(
        "1",
        embedding_text="Retrieval text",
        metadata={"retrieval_text_is_citable": False, "source": "unit-test"},
    )
    _write_jsonl(input_path, [record])

    provider = DeterministicFakeEmbeddingProvider(dimension=4)
    generate_embeddings(
        input_path=input_path,
        output_path=output_path,
        provider=provider,
        model_name="fake-model",
        expected_dim=4,
    )

    output_record = _read_jsonl(output_path)[0]
    for key in ("record_id", "chunk_id", "legal_unit_id", "law_id", "text_hash"):
        assert output_record[key] == record[key]
    assert output_record["metadata"] == record["metadata"]


@pytest.mark.parametrize(
    "vector",
    [
        [float("nan")],
        [float("inf")],
        ["not-a-number"],
        [True],
    ],
)
def test_invalid_vector_values_fail_validation(tmp_path, vector):
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(input_path, [_input_record("1", embedding_text="Retrieval text")])

    provider = StaticEmbeddingProvider([vector])
    with pytest.raises(ValueError):
        generate_embeddings(
            input_path=input_path,
            output_path=output_path,
            provider=provider,
            model_name="fake-model",
        )

    assert not output_path.exists()


def test_openai_compatible_provider_posts_to_embeddings_with_body():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _embedding_response([[0.1, 0.2]])

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:11434/v1/",
        transport=httpx.MockTransport(handler),
        sleep_func=lambda delay: None,
    )

    embeddings = provider.embed_texts(["retrieval text"], "Qwen/Qwen3-Embedding-4B")

    assert embeddings == [[0.1, 0.2]]
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "http://localhost:11434/v1/embeddings"
    assert request.headers["content-type"] == "application/json"
    body = json.loads(request.content.decode("utf-8"))
    assert body == {
        "model": "Qwen/Qwen3-Embedding-4B",
        "input": ["retrieval text"],
    }


def test_openai_compatible_provider_sends_authorization_when_api_key_exists():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _embedding_response([[0.1]])

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://provider.example.com/v1",
        api_key="secret-key",
        transport=httpx.MockTransport(handler),
        sleep_func=lambda delay: None,
    )

    provider.embed_texts(["retrieval text"], "model")

    assert requests[0].headers["authorization"] == "Bearer secret-key"


def test_openai_compatible_provider_omits_authorization_without_api_key():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _embedding_response([[0.1]])

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:8000/v1",
        api_key=None,
        transport=httpx.MockTransport(handler),
        sleep_func=lambda delay: None,
    )

    provider.embed_texts(["retrieval text"], "model")

    assert "authorization" not in requests[0].headers


def test_openai_compatible_provider_parses_standard_response_without_index():
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:11434/v1",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "data": [
                        {"embedding": [0.1, 0.2]},
                        {"embedding": [0.3, 0.4]},
                    ],
                    "model": "model",
                },
            )
        ),
        sleep_func=lambda delay: None,
    )

    assert provider.embed_texts(["text 1", "text 2"], "model") == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]


def test_openai_compatible_provider_sorts_embeddings_by_index():
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:11434/v1",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "data": [
                        {"embedding": [1.0], "index": 1},
                        {"embedding": [0.0], "index": 0},
                    ]
                },
            )
        ),
        sleep_func=lambda delay: None,
    )

    assert provider.embed_texts(["text 0", "text 1"], "model") == [[0.0], [1.0]]


@pytest.mark.parametrize(
    "data",
    [
        [
            {"embedding": [0.0], "index": 0},
            {"embedding": [1.0], "index": 0},
        ],
        [
            {"embedding": [0.0], "index": 0},
            {"embedding": [1.0], "index": 2},
        ],
        [
            {"embedding": [0.0], "index": -1},
            {"embedding": [1.0], "index": 1},
        ],
    ],
)
def test_openai_compatible_provider_fails_on_non_contiguous_indexes(data):
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:11434/v1",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"data": data})
        ),
        sleep_func=lambda delay: None,
    )

    with pytest.raises(
        ValueError,
        match="indexes must be unique and contiguous from 0",
    ):
        provider.embed_texts(["text 0", "text 1"], "model")


def test_openai_compatible_provider_fails_on_embedding_count_mismatch():
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:11434/v1",
        transport=httpx.MockTransport(lambda request: _embedding_response([[0.1]])),
        sleep_func=lambda delay: None,
    )

    with pytest.raises(ValueError, match="embedding count mismatch"):
        provider.embed_texts(["text 1", "text 2"], "model")


def test_openai_compatible_provider_fails_on_http_400_without_retry():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(400, json={"error": "bad request"})

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:11434/v1",
        max_retries=3,
        transport=httpx.MockTransport(handler),
        sleep_func=lambda delay: None,
    )

    with pytest.raises(RuntimeError, match="HTTP status 400"):
        provider.embed_texts(["retrieval text"], "model")

    assert len(requests) == 1


def test_openai_compatible_provider_retries_http_429_then_succeeds():
    requests = []
    sleeps = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return _embedding_response([[0.1]])

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:11434/v1",
        max_retries=2,
        transport=httpx.MockTransport(handler),
        sleep_func=sleeps.append,
    )

    assert provider.embed_texts(["retrieval text"], "model") == [[0.1]]
    assert len(requests) == 2
    assert sleeps == [0.5]


def test_openai_compatible_provider_retries_timeout_then_succeeds():
    requests = []
    sleeps = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            raise httpx.ReadTimeout("timeout", request=request)
        return _embedding_response([[0.1]])

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:11434/v1",
        max_retries=2,
        transport=httpx.MockTransport(handler),
        sleep_func=sleeps.append,
    )

    assert provider.embed_texts(["retrieval text"], "model") == [[0.1]]
    assert len(requests) == 2
    assert sleeps == [0.5]


def test_cli_openai_compatible_requires_base_url(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_embeddings.py",
            "--input",
            "missing.jsonl",
            "--output",
            "out.jsonl",
            "--provider",
            "openai-compatible",
            "--model",
            "model",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        generate_embeddings_cli.main()

    assert exc.value.code == 2
    assert "base URL missing" in capsys.readouterr().err


def test_cli_api_key_env_missing_fails_explicitly(monkeypatch, capsys):
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_embeddings.py",
            "--input",
            "missing.jsonl",
            "--output",
            "out.jsonl",
            "--provider",
            "openai-compatible",
            "--base-url",
            "https://provider.example.com/v1",
            "--api-key-env",
            "EMBEDDING_API_KEY",
            "--model",
            "model",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        generate_embeddings_cli.main()

    assert exc.value.code == 2
    assert "env var missing or empty: EMBEDDING_API_KEY" in capsys.readouterr().err


def test_cli_dry_run_openai_compatible_does_not_create_http_provider(
    tmp_path,
    monkeypatch,
):
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(input_path, [_input_record("1", embedding_text="Retrieval text")])

    def fail_if_created(*args, **kwargs):
        raise AssertionError("HTTP provider must not be created during dry-run")

    monkeypatch.setattr(
        generate_embeddings_cli,
        "OpenAICompatibleEmbeddingProvider",
        fail_if_created,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_embeddings.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--provider",
            "openai-compatible",
            "--base-url",
            "http://localhost:11434/v1",
            "--model",
            "model",
            "--dry-run",
        ],
    )

    generate_embeddings_cli.main()

    assert not output_path.exists()


def test_http_provider_end_to_end_writes_chunk_output_and_uses_embedding_text(tmp_path):
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    input_record = _input_record(
        "1",
        text="LegalUnit.raw_text must not be embedded",
        embedding_text="LegalChunk.embedding_text retrieval only",
    )
    _write_jsonl(input_path, [input_record])
    request_bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.content.decode("utf-8")))
        return _embedding_response([[0.7, 0.8]])

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:11434/v1",
        transport=httpx.MockTransport(handler),
        sleep_func=lambda delay: None,
    )

    summary = generate_embeddings(
        input_path=input_path,
        output_path=output_path,
        provider=provider,
        model_name="Qwen/Qwen3-Embedding-4B",
        expected_dim=2,
    )

    output_record = _read_jsonl(output_path)[0]
    assert summary.written_count == 1
    assert request_bodies[0]["input"] == ["LegalChunk.embedding_text retrieval only"]
    assert output_record["record_id"] == input_record["record_id"]
    assert output_record["chunk_id"] == input_record["chunk_id"]
    assert output_record["legal_unit_id"] == input_record["legal_unit_id"]
    assert output_record["law_id"] == input_record["law_id"]
    assert output_record["text_hash"] == input_record["text_hash"]
    assert output_record["model_name"] == "Qwen/Qwen3-Embedding-4B"
    assert output_record["embedding_dim"] == 2
    assert output_record["embedding"] == [0.7, 0.8]
    assert "unit_id" not in output_record


def _input_record(
    suffix: str,
    *,
    text: str = "LegalUnit.raw_text",
    embedding_text: str = "LegalChunk.embedding_text",
    text_hash: str | None = None,
    metadata: dict | None = None,
) -> dict:
    if text_hash is None:
        text_hash = stable_text_hash(embedding_text)
    return {
        "record_id": f"embedding.chunk.ro.test.{suffix}.0",
        "chunk_id": f"chunk.ro.test.{suffix}.0",
        "legal_unit_id": f"ro.test.{suffix}",
        "law_id": "ro.test",
        "text": text,
        "embedding_text": embedding_text,
        "text_hash": text_hash,
        "model_hint": None,
        "metadata": metadata or {},
    }


def _write_jsonl(path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _read_jsonl(path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _embedding_response(vectors: list[list[float]]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": [
                {"embedding": vector, "index": index}
                for index, vector in enumerate(vectors)
            ],
            "model": "model",
        },
    )
