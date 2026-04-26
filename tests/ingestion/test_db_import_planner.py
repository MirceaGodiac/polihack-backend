import json
from pathlib import Path

import pytest

import scripts.plan_db_import as plan_cli
from ingestion.imports import (
    NullImportRepository,
    build_import_plan,
    validate_import_bundle,
)


def test_valid_bundle_minimal_produces_safe_plan_without_embeddings(tmp_path):
    source_dir = _write_bundle(tmp_path)

    plan = build_import_plan(source_dir)

    assert plan.safe_for_db_import is True
    assert plan.mode == "validate_only"
    assert plan.counts.legal_units == 2
    assert plan.counts.legal_edges == 1
    assert plan.counts.embedding_records == 0
    assert plan.validation.passed is True
    assert plan.validation.validation_report_passed is True
    assert plan.idempotency == {
        "legal_units_key": "id",
        "legal_edges_key": "id",
        "embeddings_key": "record_id + model_name + text_hash",
    }


def test_missing_legal_units_produces_error_and_invalid_plan(tmp_path):
    source_dir = _write_bundle(tmp_path, missing={"legal_units.json"})

    plan = build_import_plan(source_dir)

    assert plan.safe_for_db_import is False
    assert any(error.code == "missing_required_artifact" for error in plan.errors)


def test_missing_optional_artifacts_produce_warning_not_error(tmp_path):
    source_dir = _write_bundle(
        tmp_path,
        missing={"reference_candidates.json", "legal_chunks.json"},
    )

    plan = build_import_plan(source_dir)

    assert plan.safe_for_db_import is True
    assert plan.errors == []
    warning_codes = {warning.code for warning in plan.warnings}
    assert warning_codes == {"missing_optional_artifact"}


def test_duplicate_legal_unit_id_produces_error(tmp_path):
    source_dir = _write_bundle(
        tmp_path,
        legal_units=[
            _legal_unit("ro.test"),
            _legal_unit("ro.test"),
        ],
        legal_edges=[],
    )

    plan = build_import_plan(source_dir)

    assert plan.safe_for_db_import is False
    assert any(error.code == "legal_units_duplicate_id" for error in plan.errors)


def test_duplicate_legal_edge_id_produces_error(tmp_path):
    source_dir = _write_bundle(
        tmp_path,
        legal_edges=[
            _legal_edge("edge.1", "ro.test", "ro.test.art_1"),
            _legal_edge("edge.1", "ro.test", "ro.test.art_1"),
        ],
    )

    plan = build_import_plan(source_dir)

    assert plan.safe_for_db_import is False
    assert any(error.code == "legal_edges_duplicate_id" for error in plan.errors)


def test_orphan_legal_edge_produces_warning_by_default(tmp_path):
    source_dir = _write_bundle(
        tmp_path,
        legal_edges=[_legal_edge("edge.orphan", "ro.test", "ro.test.missing")],
    )

    plan = build_import_plan(source_dir)

    assert plan.safe_for_db_import is True
    assert any(warning.code == "orphan_legal_edges" for warning in plan.warnings)


def test_orphan_legal_edge_produces_error_when_configured(tmp_path):
    source_dir = _write_bundle(
        tmp_path,
        legal_edges=[_legal_edge("edge.orphan", "ro.test", "ro.test.missing")],
    )

    plan = build_import_plan(source_dir, fail_on_orphan_edges=True)

    assert plan.safe_for_db_import is False
    assert any(error.code == "orphan_legal_edges" for error in plan.errors)


def test_with_embeddings_requires_embedding_artifacts(tmp_path):
    source_dir = _write_bundle(tmp_path)

    plan = build_import_plan(source_dir, with_embeddings=True, embedding_dim=2)

    assert plan.safe_for_db_import is False
    error_codes = {error.code for error in plan.errors}
    assert "missing_required_embedding_artifact" in error_codes


def test_with_embeddings_validates_manifest_and_counts_records(tmp_path):
    source_dir = _write_bundle(tmp_path, with_embeddings=True)

    plan = build_import_plan(source_dir, with_embeddings=True, embedding_dim=2)

    assert plan.safe_for_db_import is True
    assert plan.counts.embedding_records == 1
    assert plan.validation.embeddings_manifest_present is True
    assert plan.validation.pair_validation_assumed_from_manifest is True


def test_with_embeddings_accepts_official_embeddings_manifest_filename(tmp_path):
    source_dir = _write_bundle(
        tmp_path,
        with_embeddings=True,
        embedding=[0.0] * 2560,
        manifest_embedding_dim=2560,
    )

    plan = build_import_plan(source_dir, with_embeddings=True, embedding_dim=2560)

    assert plan.safe_for_db_import is True
    assert plan.errors == []
    assert plan.counts.embedding_records == 1
    assert Path(plan.artifact_paths.embeddings_manifest).name == "embeddings_manifest.json"


def test_with_embeddings_accepts_legacy_embeddings_import_manifest_filename(tmp_path):
    source_dir = _write_bundle(
        tmp_path,
        with_embeddings=True,
        manifest_filename="embeddings_import_manifest.json",
    )

    plan = build_import_plan(source_dir, with_embeddings=True, embedding_dim=2)

    assert plan.safe_for_db_import is True
    assert plan.errors == []
    assert Path(plan.artifact_paths.embeddings_manifest).name == "embeddings_import_manifest.json"


def test_embedding_dim_mismatch_produces_error_without_logging_vector(tmp_path):
    source_dir = _write_bundle(
        tmp_path,
        with_embeddings=True,
        embedding=[0.1, 0.2, 0.3],
        manifest_embedding_dim=2,
    )

    plan = build_import_plan(source_dir, with_embeddings=True, embedding_dim=2)
    serialized = json.dumps(plan.model_dump(), ensure_ascii=False)

    assert plan.safe_for_db_import is False
    assert any(error.code == "embedding_dim_mismatch" for error in plan.errors)
    assert "[0.1, 0.2, 0.3]" not in serialized


def test_duplicate_embedding_identity_produces_error(tmp_path):
    output_records = [_embedding_output_record(), _embedding_output_record()]
    source_dir = _write_bundle(
        tmp_path,
        with_embeddings=True,
        embedding_output_records=output_records,
    )

    plan = build_import_plan(source_dir, with_embeddings=True, embedding_dim=2)

    assert plan.safe_for_db_import is False
    assert any(error.code == "duplicate_embedding_identity" for error in plan.errors)


def test_import_plan_does_not_include_raw_text_embedding_text_or_vectors(tmp_path):
    source_dir = _write_bundle(tmp_path, with_embeddings=True)

    plan = build_import_plan(source_dir, with_embeddings=True, embedding_dim=2)
    serialized = json.dumps(plan.model_dump(), ensure_ascii=False)

    assert "raw_text" not in serialized
    assert "embedding_text" not in serialized
    assert "SECRET_RAW_TEXT" not in serialized
    assert "SECRET_EMBEDDING_TEXT" not in serialized
    assert "[0.1, 0.2]" not in serialized


def test_cli_validate_only_returns_zero_on_valid_bundle(tmp_path, capsys):
    source_dir = _write_bundle(tmp_path)

    exit_code = plan_cli.main(["--source-dir", str(source_dir), "--pretty"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["safe_for_db_import"] is True
    assert payload["counts"]["legal_units"] == 2


def test_cli_apply_returns_two_and_does_not_write_db(tmp_path, capsys):
    source_dir = _write_bundle(tmp_path)

    exit_code = plan_cli.main(["--source-dir", str(source_dir), "--mode", "apply"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "apply mode is reserved for H08 Phase D2/D3" in captured.err


def test_validate_import_bundle_handles_missing_source_dir(tmp_path):
    result = validate_import_bundle(tmp_path / "missing")

    assert result.passed is False
    assert result.errors[0].code == "source_dir_missing"


def test_null_import_repository_returns_simulated_counts():
    repository = NullImportRepository()

    result = repository.upsert_legal_units([{"id": "ro.test"}, {"id": "ro.test.art_1"}])

    assert result.attempted == 2
    assert result.inserted == 0
    assert result.updated == 0


def _write_bundle(
    tmp_path: Path,
    *,
    legal_units: list[dict] | None = None,
    legal_edges: list[dict] | None = None,
    missing: set[str] | None = None,
    with_embeddings: bool = False,
    embedding: list[float] | None = None,
    embedding_output_records: list[dict] | None = None,
    manifest_embedding_dim: int = 2,
    manifest_filename: str = "embeddings_manifest.json",
) -> Path:
    source_dir = tmp_path / "bundle"
    source_dir.mkdir(parents=True, exist_ok=True)
    missing = missing or set()
    legal_units = legal_units or [
        _legal_unit("ro.test"),
        _legal_unit("ro.test.art_1"),
    ]
    legal_edges = legal_edges or [_legal_edge("edge.1", "ro.test", "ro.test.art_1")]
    files = {
        "legal_units.json": legal_units,
        "legal_edges.json": legal_edges,
        "reference_candidates.json": [],
        "legal_chunks.json": [],
        "validation_report.json": {"import_blocking_passed": True},
        "corpus_manifest.json": {
            "sources": [
                {
                    "source_id": "fixture_source",
                    "source_url": "https://example.test/legal",
                }
            ]
        },
    }
    for filename, payload in files.items():
        if filename not in missing:
            _write_json(source_dir / filename, payload)

    if with_embeddings:
        input_record = _embedding_input_record()
        output_record = _embedding_output_record(embedding=embedding)
        _write_jsonl(source_dir / "embeddings_input.jsonl", [input_record])
        _write_jsonl(
            source_dir / "embeddings_output.jsonl",
            embedding_output_records or [output_record],
        )
        _write_json(
            source_dir / manifest_filename,
            {
                "input_path": str(source_dir / "embeddings_input.jsonl"),
                "output_path": str(source_dir / "embeddings_output.jsonl"),
                "model_name": "qwen3-embedding:4b",
                "embedding_dim": manifest_embedding_dim,
                "input_read_count": 1,
                "output_read_count": 1,
                "embeddable_input_count": 1,
                "matched_output_count": 1,
                "missing_output_count": 0,
                "orphan_output_count": 0,
                "law_ids": ["ro.test"],
                "model_names": ["qwen3-embedding:4b"],
                "embedding_dims": [manifest_embedding_dim],
                "ready_for_pgvector_import": True,
                "validated_at": "2026-04-26T00:00:00Z",
                "validator_version": "test",
                "warnings": [],
            },
        )
    return source_dir


def _legal_unit(unit_id: str) -> dict:
    return {
        "id": unit_id,
        "law_id": "ro.test",
        "source_id": "fixture_source",
        "source_url": "https://example.test/legal",
        "raw_text": "SECRET_RAW_TEXT",
    }


def _legal_edge(edge_id: str, source_id: str, target_id: str) -> dict:
    return {
        "id": edge_id,
        "source_id": source_id,
        "target_id": target_id,
        "type": "contains",
    }


def _embedding_input_record() -> dict:
    return {
        "record_id": "embedding.chunk.ro.test.art_1.0",
        "chunk_id": "chunk.ro.test.art_1.0",
        "legal_unit_id": "ro.test.art_1",
        "law_id": "ro.test",
        "text": "SECRET_RAW_TEXT",
        "embedding_text": "SECRET_EMBEDDING_TEXT",
        "text_hash": "hash-1",
        "model_hint": None,
        "metadata": {"retrieval_text_is_citable": False},
    }


def _embedding_output_record(*, embedding: list[float] | None = None) -> dict:
    return {
        "record_id": "embedding.chunk.ro.test.art_1.0",
        "chunk_id": "chunk.ro.test.art_1.0",
        "legal_unit_id": "ro.test.art_1",
        "law_id": "ro.test",
        "model_name": "qwen3-embedding:4b",
        "embedding_dim": len(embedding or [0.1, 0.2]),
        "text_hash": "hash-1",
        "embedding": embedding or [0.1, 0.2],
        "metadata": {"retrieval_text_is_citable": False},
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
