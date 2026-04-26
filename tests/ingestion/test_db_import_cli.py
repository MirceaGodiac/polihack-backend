import json
from pathlib import Path

import scripts.import_db_bundle as import_cli


def test_cli_dry_run_does_not_require_database_url_or_write_db(tmp_path, monkeypatch, capsys):
    source_dir = _write_bundle(tmp_path, with_embedding_artifacts=True)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    def fail_repository(*args, **kwargs):
        raise AssertionError("dry_run must not construct a DB repository")

    monkeypatch.setattr(import_cli, "PostgresImportRepository", fail_repository)

    exit_code = import_cli.main(["--source-dir", str(source_dir), "--mode", "dry_run", "--pretty"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "dry_run"
    assert payload["counts"]["legal_units"]["attempted"] == 2
    assert "SECRET_RAW_TEXT" not in captured.out
    assert "SECRET_EMBEDDING_TEXT" not in captured.out
    assert "[0.1, 0.2]" not in captured.out


def test_cli_dry_run_with_embeddings_does_not_require_database_url(
    tmp_path,
    monkeypatch,
    capsys,
):
    source_dir = _write_bundle(tmp_path, with_embedding_artifacts=True)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    def fail_repository(*args, **kwargs):
        raise AssertionError("dry_run must not construct a DB repository")

    monkeypatch.setattr(import_cli, "PostgresImportRepository", fail_repository)

    exit_code = import_cli.main(
        [
            "--source-dir",
            str(source_dir),
            "--mode",
            "dry_run",
            "--with-embeddings",
            "--embedding-dim",
            "2",
            "--pretty",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "dry_run"
    assert payload["embedding_dim"] == 2
    assert payload["model_name"] == "fake-model"
    assert payload["counts"]["embeddings"]["attempted"] == 1
    assert "SECRET_RAW_TEXT" not in captured.out
    assert "SECRET_EMBEDDING_TEXT" not in captured.out
    assert "[0.1, 0.2]" not in captured.out


def test_cli_apply_without_database_url_returns_exit_code_2(tmp_path, monkeypatch, capsys):
    source_dir = _write_bundle(tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    exit_code = import_cli.main(["--source-dir", str(source_dir), "--mode", "apply"])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 2
    assert payload["error"] == "DATABASE_URL is required for apply mode"


def test_cli_apply_with_embeddings_without_database_url_returns_exit_code_2(
    tmp_path,
    monkeypatch,
    capsys,
):
    source_dir = _write_bundle(tmp_path, with_embedding_artifacts=True)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    exit_code = import_cli.main(
        [
            "--source-dir",
            str(source_dir),
            "--mode",
            "apply",
            "--with-embeddings",
            "--embedding-dim",
            "2",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 2
    assert payload["error"] == "DATABASE_URL is required for apply mode"


def test_cli_apply_invalid_plan_returns_exit_code_1_without_repository(
    tmp_path,
    monkeypatch,
    capsys,
):
    source_dir = _write_bundle(tmp_path, missing={"legal_units.json"})

    def fail_repository(*args, **kwargs):
        raise AssertionError("invalid plans must not construct a DB repository")

    monkeypatch.setattr(import_cli, "PostgresImportRepository", fail_repository)

    exit_code = import_cli.main(
        [
            "--source-dir",
            str(source_dir),
            "--mode",
            "apply",
            "--database-url",
            "postgresql://lexai:test@localhost/lexai",
            "--pretty",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["status"] == "invalid_bundle"
    assert any(error["code"] == "missing_required_artifact" for error in payload["errors"])


def test_cli_with_embeddings_valid_bundle_no_longer_returns_exit_code_2(tmp_path, capsys):
    source_dir = _write_bundle(tmp_path, with_embedding_artifacts=True)

    exit_code = import_cli.main(
        [
            "--source-dir",
            str(source_dir),
            "--with-embeddings",
            "--embedding-dim",
            "2",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["safe_for_db_import"] is True
    assert payload["counts"]["embeddings"]["attempted"] == 1


def test_cli_missing_embeddings_output_produces_invalid_plan(tmp_path, capsys):
    source_dir = _write_bundle(
        tmp_path,
        with_embedding_artifacts=True,
        missing={"embeddings_output.jsonl"},
    )

    exit_code = import_cli.main(
        [
            "--source-dir",
            str(source_dir),
            "--mode",
            "dry_run",
            "--with-embeddings",
            "--embedding-dim",
            "2",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["status"] == "invalid_bundle"
    assert any(
        error["code"] == "missing_required_embedding_artifact"
        for error in payload["errors"]
    )


def test_cli_apply_embedding_dim_mismatch_returns_exit_code_4_without_repository(
    tmp_path,
    monkeypatch,
    capsys,
):
    source_dir = _write_bundle(tmp_path, with_embedding_artifacts=True)

    def fail_repository(*args, **kwargs):
        raise AssertionError("embedding validation errors must stop before DB")

    monkeypatch.setattr(import_cli, "PostgresImportRepository", fail_repository)

    exit_code = import_cli.main(
        [
            "--source-dir",
            str(source_dir),
            "--mode",
            "apply",
            "--with-embeddings",
            "--embedding-dim",
            "3",
            "--database-url",
            "postgresql://lexai:test@localhost/lexai",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 4
    assert payload["status"] == "invalid_bundle"
    assert any("dim" in error["code"] for error in payload["errors"])


def test_cli_duplicate_embedding_identity_produces_invalid_plan(tmp_path, capsys):
    embedding_record = _embedding_output_record()
    source_dir = _write_bundle(
        tmp_path,
        with_embedding_artifacts=True,
        embedding_output_records=[embedding_record, embedding_record],
    )

    exit_code = import_cli.main(
        [
            "--source-dir",
            str(source_dir),
            "--mode",
            "dry_run",
            "--with-embeddings",
            "--embedding-dim",
            "2",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert any(error["code"] == "duplicate_embedding_identity" for error in payload["errors"])


def test_cli_duplicate_ids_are_caught_before_db(tmp_path, monkeypatch, capsys):
    source_dir = _write_bundle(
        tmp_path,
        legal_units=[_legal_unit("ro.test"), _legal_unit("ro.test")],
        legal_edges=[],
    )

    def fail_repository(*args, **kwargs):
        raise AssertionError("duplicate IDs must stop before DB repository construction")

    monkeypatch.setattr(import_cli, "PostgresImportRepository", fail_repository)

    exit_code = import_cli.main(
        [
            "--source-dir",
            str(source_dir),
            "--mode",
            "apply",
            "--database-url",
            "postgresql://lexai:test@localhost/lexai",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert any(error["code"] == "legal_units_duplicate_id" for error in payload["errors"])


def test_cli_orphan_edges_are_warning_by_default(tmp_path, capsys):
    source_dir = _write_bundle(
        tmp_path,
        legal_edges=[_legal_edge("edge.orphan", "ro.test", "ro.test.missing")],
    )

    exit_code = import_cli.main(["--source-dir", str(source_dir), "--mode", "dry_run"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert any(warning["code"] == "orphan_legal_edges" for warning in payload["warnings"])


def test_cli_orphan_edges_are_error_when_configured(tmp_path, capsys):
    source_dir = _write_bundle(
        tmp_path,
        legal_edges=[_legal_edge("edge.orphan", "ro.test", "ro.test.missing")],
    )

    exit_code = import_cli.main(
        ["--source-dir", str(source_dir), "--mode", "dry_run", "--fail-on-orphan-edges"]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert any(error["code"] == "orphan_legal_edges" for error in payload["errors"])


def test_cli_apply_success_uses_repository_and_sanitizes_summary(tmp_path, monkeypatch, capsys):
    source_dir = _write_bundle(tmp_path, with_embedding_artifacts=True)
    FakeRepository.instances = []
    monkeypatch.setattr(import_cli, "PostgresImportRepository", FakeRepository)

    exit_code = import_cli.main(
        [
            "--source-dir",
            str(source_dir),
            "--mode",
            "apply",
            "--database-url",
            "postgresql://lexai:test@localhost/lexai",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "succeeded"
    assert payload["counts"]["legal_units"]["inserted"] == 2
    assert payload["counts"]["legal_edges"]["inserted"] == 1
    assert "SECRET_RAW_TEXT" not in captured.out
    assert "SECRET_EMBEDDING_TEXT" not in captured.out
    assert "[0.1, 0.2]" not in captured.out


def test_cli_apply_with_embeddings_success_uses_repository_and_sanitizes_summary(
    tmp_path,
    monkeypatch,
    capsys,
):
    source_dir = _write_bundle(tmp_path, with_embedding_artifacts=True)
    FakeRepository.instances = []
    monkeypatch.setattr(import_cli, "PostgresImportRepository", FakeRepository)

    exit_code = import_cli.main(
        [
            "--source-dir",
            str(source_dir),
            "--mode",
            "apply",
            "--with-embeddings",
            "--embedding-dim",
            "2",
            "--database-url",
            "postgresql://lexai:test@localhost/lexai",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "succeeded"
    assert payload["counts"]["embeddings"]["inserted"] == 1
    assert payload["counts"]["embeddings"]["inserted_count"] == 1
    assert FakeRepository.instances[0].embedding_records_count == 1
    assert FakeRepository.instances[0].expected_embedding_dim == 2
    assert "SECRET_RAW_TEXT" not in captured.out
    assert "SECRET_EMBEDDING_TEXT" not in captured.out
    assert "[0.1, 0.2]" not in captured.out


def test_cli_apply_db_error_returns_exit_code_5_without_raw_text(
    tmp_path,
    monkeypatch,
    capsys,
):
    source_dir = _write_bundle(tmp_path)

    class FailingRepository(FakeRepository):
        def apply_import_plan(self, plan, *, legal_units, legal_edges):
            raise RuntimeError("SECRET_RAW_TEXT")

    monkeypatch.setattr(import_cli, "PostgresImportRepository", FailingRepository)

    exit_code = import_cli.main(
        [
            "--source-dir",
            str(source_dir),
            "--mode",
            "apply",
            "--database-url",
            "postgresql://lexai:test@localhost/lexai",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 5
    assert payload["status"] == "db_error"
    assert payload["error"] == "database import failed; transaction rolled back"
    assert "SECRET_RAW_TEXT" not in captured.err


class FakeApplyResult:
    def __init__(self, *, embeddings_attempted: int = 0):
        self.status = "succeeded"
        self.embeddings_attempted = embeddings_attempted

    @property
    def counts(self):
        return {
            "legal_units": {
                "attempted": 2,
                "inserted": 2,
                "updated": 0,
                "unchanged": 0,
                "skipped": 0,
            },
            "legal_edges": {
                "attempted": 1,
                "inserted": 1,
                "updated": 0,
                "unchanged": 0,
                "skipped": 0,
            },
            "embeddings": {
                "attempted": self.embeddings_attempted,
                "inserted": self.embeddings_attempted,
                "updated": 0,
                "unchanged": 0,
                "failed": 0,
                "skipped": 0,
            },
        }


class FakeRepository:
    instances = []

    def __init__(self, database_url):
        self.database_url = database_url
        self.schema_ensured = False
        self.embedding_records_count = 0
        self.expected_embedding_dim = None
        FakeRepository.instances.append(self)

    def ensure_schema(self):
        self.schema_ensured = True

    def apply_import_plan(
        self,
        plan,
        *,
        legal_units,
        legal_edges,
        embeddings=None,
        expected_embedding_dim=2560,
    ):
        assert self.schema_ensured is True
        assert len(list(legal_units)) == 2
        assert len(list(legal_edges)) == 1
        embedding_records = list(embeddings or [])
        self.embedding_records_count = len(embedding_records)
        self.expected_embedding_dim = expected_embedding_dim
        return FakeApplyResult(embeddings_attempted=len(embedding_records))


def _write_bundle(
    tmp_path: Path,
    *,
    legal_units: list[dict] | None = None,
    legal_edges: list[dict] | None = None,
    embedding_output_records: list[dict] | None = None,
    missing: set[str] | None = None,
    with_embedding_artifacts: bool = False,
) -> Path:
    source_dir = tmp_path / "bundle"
    source_dir.mkdir()
    missing = missing or set()
    legal_units = legal_units or [_legal_unit("ro.test"), _legal_unit("ro.test.art_1")]
    legal_edges = legal_edges or [_legal_edge("edge.1", "ro.test", "ro.test.art_1")]
    payloads = {
        "legal_units.json": legal_units,
        "legal_edges.json": legal_edges,
        "reference_candidates.json": [],
        "legal_chunks.json": [],
        "validation_report.json": {"import_blocking_passed": True},
        "corpus_manifest.json": {"sources": []},
    }
    for filename, payload in payloads.items():
        if filename not in missing:
            _write_json(source_dir / filename, payload)
    if with_embedding_artifacts:
        if "embeddings_input.jsonl" not in missing:
            _write_jsonl(source_dir / "embeddings_input.jsonl", [_embedding_input_record()])
        if "embeddings_output.jsonl" not in missing:
            _write_jsonl(
                source_dir / "embeddings_output.jsonl",
                embedding_output_records or [_embedding_output_record()],
            )
        if "embeddings_manifest.json" not in missing:
            _write_json(
                source_dir / "embeddings_manifest.json",
                {
                    "ready_for_pgvector_import": True,
                    "embedding_dim": 2,
                    "model_name": "fake-model",
                    "law_ids": ["ro.test"],
                },
            )
    return source_dir


def _legal_unit(unit_id: str) -> dict:
    return {
        "id": unit_id,
        "canonical_id": unit_id,
        "source_id": "fixture_source",
        "law_id": "ro.test",
        "law_title": "Lege test",
        "status": "unknown",
        "hierarchy_path": ["Lege test"],
        "raw_text": "SECRET_RAW_TEXT",
        "normalized_text": "SECRET_RAW_TEXT",
        "legal_domain": "test",
        "legal_concepts": [],
        "parser_warnings": [],
    }


def _legal_edge(edge_id: str, source_id: str, target_id: str) -> dict:
    return {
        "id": edge_id,
        "source_id": source_id,
        "target_id": target_id,
        "type": "contains",
        "weight": 1.0,
        "confidence": 1.0,
        "metadata": {"source": "unit-test"},
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
        "metadata": {"retrieval_text_is_citable": False},
    }


def _embedding_output_record(*, embedding: list[float] | None = None) -> dict:
    return {
        "record_id": "embedding.chunk.ro.test.art_1.0",
        "chunk_id": "chunk.ro.test.art_1.0",
        "legal_unit_id": "ro.test.art_1",
        "law_id": "ro.test",
        "model_name": "fake-model",
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
