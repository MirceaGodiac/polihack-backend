import json
import sys

import pytest

import scripts.validate_embeddings_output as validation_cli
from ingestion.embeddings import validate_embeddings_output


def test_valid_output_returns_summary(tmp_path):
    output_path = tmp_path / "embeddings_output.jsonl"
    records = [
        _output_record("1", law_id="ro.test.a"),
        _output_record("2", law_id="ro.test.b"),
    ]
    _write_jsonl(output_path, records)

    summary = validate_embeddings_output(output_path, expected_model="model-a", expected_dim=3)

    assert summary.read_count == 2
    assert summary.valid_count == 2
    assert summary.invalid_count == 0
    assert summary.model_names == ["model-a"]
    assert summary.embedding_dims == [3]
    assert summary.law_ids == ["ro.test.a", "ro.test.b"]
    assert summary.errors == []


def test_missing_required_field_is_error(tmp_path):
    output_path = tmp_path / "embeddings_output.jsonl"
    record = _output_record("1")
    del record["chunk_id"]
    _write_jsonl(output_path, [record])

    summary = validate_embeddings_output(output_path, strict=False)

    assert summary.invalid_count == 1
    assert any("missing required field chunk_id" in error for error in summary.errors)


def test_empty_required_string_field_is_error(tmp_path):
    output_path = tmp_path / "embeddings_output.jsonl"
    record = _output_record("1")
    record["record_id"] = ""
    _write_jsonl(output_path, [record])

    summary = validate_embeddings_output(output_path, strict=False)

    assert summary.invalid_count == 1
    assert any("field record_id must be a non-empty string" in error for error in summary.errors)


def test_embedding_dim_must_match_embedding_length(tmp_path):
    output_path = tmp_path / "embeddings_output.jsonl"
    record = _output_record("1", embedding=[0.1, 0.2], embedding_dim=3)
    _write_jsonl(output_path, [record])

    summary = validate_embeddings_output(output_path, strict=False)

    assert summary.invalid_count == 1
    assert any("dimension mismatch" in error for error in summary.errors)


def test_expected_dim_mismatch_is_error(tmp_path):
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(output_path, [_output_record("1", embedding_dim=3)])

    summary = validate_embeddings_output(output_path, expected_dim=1024, strict=False)

    assert summary.invalid_count == 1
    assert any("expected_dim mismatch" in error for error in summary.errors)


def test_expected_model_mismatch_is_error(tmp_path):
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(output_path, [_output_record("1", model_name="model-a")])

    summary = validate_embeddings_output(output_path, expected_model="model-b", strict=False)

    assert summary.invalid_count == 1
    assert any("expected_model mismatch" in error for error in summary.errors)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), True, "not-a-number"],
)
def test_invalid_embedding_values_are_errors(tmp_path, value):
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(output_path, [_output_record("1", embedding=[value])])

    summary = validate_embeddings_output(output_path, strict=False)

    assert summary.invalid_count == 1
    assert any("invalid embedding vector" in error for error in summary.errors)


def test_duplicate_resume_key_is_error(tmp_path):
    output_path = tmp_path / "embeddings_output.jsonl"
    record = _output_record("1")
    _write_jsonl(output_path, [record, dict(record)])

    summary = validate_embeddings_output(output_path, strict=False)

    assert summary.duplicate_resume_key_count == 1
    assert summary.invalid_count == 1
    assert any("duplicate resume key" in error for error in summary.errors)


def test_duplicate_record_id_with_different_hash_or_model_is_warning_by_default(tmp_path):
    output_path = tmp_path / "embeddings_output.jsonl"
    records = [
        _output_record("1", text_hash="hash-a", model_name="model-a"),
        _output_record("1", text_hash="hash-b", model_name="model-a"),
    ]
    _write_jsonl(output_path, records)

    summary = validate_embeddings_output(output_path)

    assert summary.valid_count == 2
    assert summary.duplicate_record_id_count == 1
    assert any("duplicate record_id" in warning for warning in summary.warnings)
    assert summary.errors == []


def test_duplicate_record_id_is_error_when_required_unique(tmp_path):
    output_path = tmp_path / "embeddings_output.jsonl"
    records = [
        _output_record("1", text_hash="hash-a"),
        _output_record("1", text_hash="hash-b"),
    ]
    _write_jsonl(output_path, records)

    summary = validate_embeddings_output(
        output_path,
        require_unique_record_ids=True,
        strict=False,
    )

    assert summary.duplicate_record_id_count == 1
    assert summary.invalid_count == 1
    assert any("duplicate record_id" in error for error in summary.errors)


def test_missing_metadata_is_error(tmp_path):
    output_path = tmp_path / "embeddings_output.jsonl"
    record = _output_record("1")
    del record["metadata"]
    _write_jsonl(output_path, [record])

    summary = validate_embeddings_output(output_path, strict=False)

    assert summary.invalid_count == 1
    assert any("missing required field metadata" in error for error in summary.errors)


def test_empty_metadata_is_valid_and_counted(tmp_path):
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(output_path, [_output_record("1", metadata={})])

    summary = validate_embeddings_output(output_path)

    assert summary.valid_count == 1
    assert summary.empty_metadata_count == 1
    assert summary.errors == []


def test_cli_human_readable_passes_on_valid_file(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(output_path, [_output_record("1")])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_embeddings_output.py",
            "--output",
            str(output_path),
            "--expected-model",
            "model-a",
            "--expected-dim",
            "3",
            "--strict",
        ],
    )

    validation_cli.main()

    stdout = capsys.readouterr().out
    assert "Embeddings output validation" in stdout
    assert "valid_count: 1" in stdout
    assert "invalid_count: 0" in stdout


def test_cli_json_outputs_valid_json(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(output_path, [_output_record("1")])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_embeddings_output.py",
            "--output",
            str(output_path),
            "--expected-dim",
            "3",
            "--json",
        ],
    )

    validation_cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["read_count"] == 1
    assert payload["valid_count"] == 1
    assert payload["embedding_dims"] == [3]


def test_cli_strict_returns_nonzero_on_invalid_file(tmp_path, monkeypatch):
    output_path = tmp_path / "embeddings_output.jsonl"
    record = _output_record("1")
    del record["metadata"]
    _write_jsonl(output_path, [record])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_embeddings_output.py",
            "--output",
            str(output_path),
            "--strict",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        validation_cli.main()

    assert exc.value.code == 1


def test_validator_does_not_require_database_url(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(output_path, [_output_record("1")])

    summary = validate_embeddings_output(output_path)

    assert summary.valid_count == 1


def test_cli_output_does_not_include_complete_embedding_vector(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(
        output_path,
        [_output_record("1", embedding=[0.1, 0.2, 0.3], embedding_dim=2)],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_embeddings_output.py",
            "--output",
            str(output_path),
        ],
    )

    validation_cli.main()

    stdout = capsys.readouterr().out
    assert "errors:" in stdout
    assert "[0.1, 0.2, 0.3]" not in stdout


def test_strict_validator_exception_does_not_include_complete_embedding_vector(tmp_path):
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(
        output_path,
        [_output_record("1", embedding=[0.1, 0.2, 0.3], embedding_dim=2)],
    )

    with pytest.raises(ValueError) as exc:
        validate_embeddings_output(output_path)

    assert "[0.1, 0.2, 0.3]" not in str(exc.value)


def _output_record(
    suffix: str,
    *,
    record_id: str | None = None,
    chunk_id: str | None = None,
    legal_unit_id: str | None = None,
    law_id: str = "ro.test",
    model_name: str = "model-a",
    embedding_dim: int = 3,
    text_hash: str = "hash-a",
    embedding: list | None = None,
    metadata: dict | None = None,
) -> dict:
    if embedding is None:
        embedding = [0.1, 0.2, 0.3]
    if metadata is None:
        metadata = {"retrieval_text_is_citable": False}
    return {
        "record_id": record_id or f"embedding.chunk.ro.test.{suffix}.0",
        "chunk_id": chunk_id or f"chunk.ro.test.{suffix}.0",
        "legal_unit_id": legal_unit_id or f"ro.test.{suffix}",
        "law_id": law_id,
        "model_name": model_name,
        "embedding_dim": embedding_dim,
        "text_hash": text_hash,
        "embedding": embedding,
        "metadata": metadata,
    }


def _write_jsonl(path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
