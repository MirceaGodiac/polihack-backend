import json

import pytest

from ingestion.chunks import stable_text_hash
from ingestion.embeddings import (
    DeterministicFakeEmbeddingProvider,
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
