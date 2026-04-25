import json
import sys

import pytest

import scripts.write_embeddings_manifest as manifest_cli
from ingestion.embeddings import (
    EMBEDDINGS_IMPORT_READINESS_VALIDATOR_VERSION,
    build_embeddings_import_manifest,
)


MODEL = "Qwen/Qwen3-Embedding-4B"


def test_manifest_is_built_from_valid_input_and_output(tmp_path):
    input_path, output_path = _write_pair(tmp_path)

    manifest = build_embeddings_import_manifest(
        input_path=input_path,
        output_path=output_path,
        expected_model=MODEL,
        expected_dim=2,
    )

    assert manifest.input_path == str(input_path)
    assert manifest.output_path == str(output_path)
    assert manifest.model_name == MODEL
    assert manifest.embedding_dim == 2
    assert manifest.input_read_count == 1
    assert manifest.output_read_count == 1
    assert manifest.embeddable_input_count == 1
    assert manifest.matched_output_count == 1
    assert manifest.missing_output_count == 0
    assert manifest.orphan_output_count == 0
    assert manifest.law_ids == ["ro.test"]
    assert manifest.model_names == [MODEL]
    assert manifest.embedding_dims == [2]
    assert manifest.validator_version == EMBEDDINGS_IMPORT_READINESS_VALIDATOR_VERSION
    assert manifest.validated_at.endswith("Z")


def test_manifest_written_to_disk_is_valid_json(tmp_path):
    input_path, output_path = _write_pair(tmp_path)
    manifest_path = tmp_path / "manifests" / "validated_embeddings_manifest.json"

    manifest = build_embeddings_import_manifest(
        input_path=input_path,
        output_path=output_path,
        expected_model=MODEL,
        expected_dim=2,
        manifest_path=manifest_path,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload == manifest.model_dump()
    assert payload["ready_for_pgvector_import"] is True


def test_ready_for_pgvector_import_is_true_when_validation_passes(tmp_path):
    input_path, output_path = _write_pair(tmp_path)

    manifest = build_embeddings_import_manifest(
        input_path=input_path,
        output_path=output_path,
        expected_model=MODEL,
        expected_dim=2,
    )

    assert manifest.ready_for_pgvector_import is True


def test_invalid_pair_does_not_write_manifest(tmp_path):
    input_record = _input_record("1")
    output_record = _output_record_from_input(input_record)
    output_record["chunk_id"] = "changed"
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    manifest_path = tmp_path / "validated_embeddings_manifest.json"
    _write_jsonl(input_path, [input_record])
    _write_jsonl(output_path, [output_record])

    with pytest.raises(ValueError):
        build_embeddings_import_manifest(
            input_path=input_path,
            output_path=output_path,
            expected_model=MODEL,
            expected_dim=2,
            manifest_path=manifest_path,
        )

    assert not manifest_path.exists()


def test_manifest_does_not_include_text_embedding_text_raw_text_or_embedding(tmp_path):
    input_path, output_path = _write_pair(
        tmp_path,
        text="SECRET_RAW_TEXT",
        embedding_text="SECRET_EMBEDDING_TEXT",
    )

    manifest = build_embeddings_import_manifest(
        input_path=input_path,
        output_path=output_path,
        expected_model=MODEL,
        expected_dim=2,
    )
    payload = manifest.model_dump()
    serialized = json.dumps(payload, ensure_ascii=False)

    assert "text" not in payload
    assert "embedding_text" not in payload
    assert "raw_text" not in payload
    assert "embedding" not in payload
    assert "SECRET_RAW_TEXT" not in serialized
    assert "SECRET_EMBEDDING_TEXT" not in serialized
    assert "[0.1, 0.2]" not in serialized


def test_cli_writes_valid_manifest(tmp_path, monkeypatch, capsys):
    input_path, output_path = _write_pair(tmp_path)
    manifest_path = tmp_path / "validated_embeddings_manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "write_embeddings_manifest.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--manifest",
            str(manifest_path),
            "--expected-model",
            MODEL,
            "--expected-dim",
            "2",
        ],
    )

    manifest_cli.main()

    stdout = capsys.readouterr().out
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["ready_for_pgvector_import"] is True
    assert payload["model_name"] == MODEL
    assert "ready_for_pgvector_import: True" in stdout
    assert "matched_output_count: 1" in stdout


def test_cli_json_prints_valid_json(tmp_path, monkeypatch, capsys):
    input_path, output_path = _write_pair(tmp_path)
    manifest_path = tmp_path / "validated_embeddings_manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "write_embeddings_manifest.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--manifest",
            str(manifest_path),
            "--expected-model",
            MODEL,
            "--expected-dim",
            "2",
            "--json",
        ],
    )

    manifest_cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["model_name"] == MODEL
    assert payload["embedding_dim"] == 2
    assert payload["ready_for_pgvector_import"] is True
    assert manifest_path.exists()


def test_cli_invalid_pair_returns_nonzero(tmp_path, monkeypatch):
    input_record = _input_record("1")
    output_record = _output_record_from_input(input_record)
    output_record["law_id"] = "changed"
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    manifest_path = tmp_path / "validated_embeddings_manifest.json"
    _write_jsonl(input_path, [input_record])
    _write_jsonl(output_path, [output_record])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "write_embeddings_manifest.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--manifest",
            str(manifest_path),
            "--expected-model",
            MODEL,
            "--expected-dim",
            "2",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        manifest_cli.main()

    assert exc.value.code == 1
    assert not manifest_path.exists()


def test_manifest_builder_does_not_require_database_url(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    input_path, output_path = _write_pair(tmp_path)

    manifest = build_embeddings_import_manifest(
        input_path=input_path,
        output_path=output_path,
        expected_model=MODEL,
        expected_dim=2,
    )

    assert manifest.ready_for_pgvector_import is True


def test_manifest_builder_does_not_make_network_calls(tmp_path, monkeypatch):
    input_path, output_path = _write_pair(tmp_path)

    def fail_http_client(*args, **kwargs):
        raise AssertionError("network calls are not allowed")

    monkeypatch.setattr("httpx.Client", fail_http_client)

    manifest = build_embeddings_import_manifest(
        input_path=input_path,
        output_path=output_path,
        expected_model=MODEL,
        expected_dim=2,
    )

    assert manifest.matched_output_count == 1


def _write_pair(
    tmp_path,
    *,
    text: str = "LegalUnit.raw_text",
    embedding_text: str = "LegalChunk.embedding_text",
):
    input_record = _input_record("1", text=text, embedding_text=embedding_text)
    output_record = _output_record_from_input(input_record)
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(input_path, [input_record])
    _write_jsonl(output_path, [output_record])
    return input_path, output_path


def _input_record(
    suffix: str,
    *,
    text: str = "LegalUnit.raw_text",
    embedding_text: str = "LegalChunk.embedding_text",
) -> dict:
    return {
        "record_id": f"embedding.chunk.ro.test.{suffix}.0",
        "chunk_id": f"chunk.ro.test.{suffix}.0",
        "legal_unit_id": f"ro.test.{suffix}",
        "law_id": "ro.test",
        "text": text,
        "embedding_text": embedding_text,
        "text_hash": f"hash-{suffix}",
        "model_hint": None,
        "metadata": {"retrieval_text_is_citable": False},
    }


def _output_record_from_input(input_record: dict) -> dict:
    return {
        "record_id": input_record["record_id"],
        "chunk_id": input_record["chunk_id"],
        "legal_unit_id": input_record["legal_unit_id"],
        "law_id": input_record["law_id"],
        "model_name": MODEL,
        "embedding_dim": 2,
        "text_hash": input_record["text_hash"],
        "embedding": [0.1, 0.2],
        "metadata": {"retrieval_text_is_citable": False},
    }


def _write_jsonl(path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
