import hashlib
import json
from pathlib import Path

import pytest

import scripts.run_daily_ingestion_pipeline as daily_pipeline


def test_load_legal_sources_reads_valid_json(tmp_path):
    sources_path = _write_sources(tmp_path, [_source("ro.codul_muncii")])

    sources = daily_pipeline.load_legal_sources(sources_path)

    assert sources == [
        {
            "source_id": "source_ro_codul_muncii",
            "law_id": "ro.codul_muncii",
            "law_title": "Codul muncii",
            "source_url": "https://legislatie.just.ro/Public/DetaliiDocument/128647",
            "status": "active",
            "enabled": True,
        }
    ]


def test_run_daily_ingestion_skips_disabled_sources(tmp_path, monkeypatch):
    calls = []

    def fake_run_pipeline(**kwargs):
        calls.append(kwargs)
        _write_bundle(kwargs["out_dir"], law_id=kwargs["law_id"])

    sources_path = _write_sources(
        tmp_path,
        [
            _source("ro.codul_muncii", enabled=True),
            _source("ro.codul_civil", enabled=False),
        ],
    )
    monkeypatch.setattr(daily_pipeline, "run_pipeline", fake_run_pipeline)

    summary = daily_pipeline.run_daily_ingestion(
        sources_path=sources_path,
        output_root=tmp_path / "output",
    )

    assert summary["sources_total"] == 2
    assert summary["sources_enabled"] == 1
    assert summary["sources_succeeded"] == 1
    assert summary["sources_failed"] == 0
    assert [call["law_id"] for call in calls] == ["ro.codul_muncii"]


def test_dry_run_does_not_write_output_or_call_pipeline(tmp_path, monkeypatch):
    def fail_run_pipeline(**kwargs):
        raise AssertionError("dry-run must not call run_pipeline")

    output_root = tmp_path / "output"
    sources_path = _write_sources(tmp_path, [_source("ro.codul_muncii")])
    monkeypatch.setattr(daily_pipeline, "run_pipeline", fail_run_pipeline)

    summary = daily_pipeline.run_daily_ingestion(
        sources_path=sources_path,
        output_root=output_root,
        dry_run=True,
        with_embeddings=True,
        embedding_dim=2,
    )

    assert summary["sources_total"] == 1
    assert summary["sources_enabled"] == 1
    assert summary["sources_succeeded"] == 0
    assert summary["sources_failed"] == 0
    assert summary["output_dirs"] == [str(output_root / "ro_codul_muncii")]
    assert summary["embedding_config"]["with_embeddings"] is True
    assert summary["embedding_config"]["provider"] == "fake"
    assert summary["embedding_config"]["model"] == "qwen3-embedding:4b"
    assert summary["embedding_config"]["dim"] == 2
    assert not output_root.exists()


def test_output_dir_is_deterministic(tmp_path, monkeypatch):
    calls = []

    def fake_run_pipeline(**kwargs):
        calls.append(kwargs)
        _write_bundle(kwargs["out_dir"], law_id=kwargs["law_id"])

    output_root = tmp_path / "output"
    sources_path = _write_sources(tmp_path, [_source("ro.codul_muncii")])
    monkeypatch.setattr(daily_pipeline, "run_pipeline", fake_run_pipeline)

    summary = daily_pipeline.run_daily_ingestion(
        sources_path=sources_path,
        output_root=output_root,
    )

    expected = output_root / "ro_codul_muncii"
    assert calls[0]["out_dir"] == expected
    assert summary["output_dirs"] == [str(expected)]


def test_invalid_source_produces_clear_error(tmp_path):
    sources_path = _write_sources(
        tmp_path,
        [
            {
                "source_id": "broken",
                "law_id": "ro.broken",
                "law_title": "Broken",
                "source_url": "not-a-url",
                "enabled": True,
            }
        ],
    )

    with pytest.raises(ValueError, match=r"source\[0\]\.source_url must be an http\(s\) URL"):
        daily_pipeline.load_legal_sources(sources_path)


def test_artifact_check_passes_when_required_files_exist(tmp_path, monkeypatch):
    def fake_run_pipeline(**kwargs):
        _write_bundle(kwargs["out_dir"], law_id=kwargs["law_id"])

    sources_path = _write_sources(tmp_path, [_source("ro.codul_muncii")])
    monkeypatch.setattr(daily_pipeline, "run_pipeline", fake_run_pipeline)

    summary = daily_pipeline.run_daily_ingestion(
        sources_path=sources_path,
        output_root=tmp_path / "output",
    )

    source_summary = summary["sources"][0]
    assert summary["sources_succeeded"] == 1
    assert source_summary["artifact_check_passed"] is True
    assert source_summary["validation_gate_passed"] is True


def test_missing_artifact_marks_source_failed(tmp_path, monkeypatch):
    def fake_run_pipeline(**kwargs):
        _write_bundle(kwargs["out_dir"], law_id=kwargs["law_id"], missing={"legal_edges.json"})

    sources_path = _write_sources(tmp_path, [_source("ro.codul_muncii")])
    monkeypatch.setattr(daily_pipeline, "run_pipeline", fake_run_pipeline)

    summary = daily_pipeline.run_daily_ingestion(
        sources_path=sources_path,
        output_root=tmp_path / "output",
    )

    assert summary["sources_succeeded"] == 0
    assert summary["sources_failed"] == 1
    assert "missing required bundle artifacts: legal_edges.json" in summary["errors"][0]["error"]


def test_validation_report_import_blocking_false_marks_failed(tmp_path, monkeypatch):
    def fake_run_pipeline(**kwargs):
        _write_bundle(kwargs["out_dir"], law_id=kwargs["law_id"], import_blocking_passed=False)

    sources_path = _write_sources(tmp_path, [_source("ro.codul_muncii")])
    monkeypatch.setattr(daily_pipeline, "run_pipeline", fake_run_pipeline)

    summary = daily_pipeline.run_daily_ingestion(
        sources_path=sources_path,
        output_root=tmp_path / "output",
    )

    assert summary["sources_failed"] == 1
    assert summary["sources"][0]["artifact_check_passed"] is True
    assert summary["sources"][0]["validation_gate_passed"] is False
    assert "import_blocking_passed=false" in summary["errors"][0]["error"]


def test_skip_validation_gate_turns_failure_into_warning(tmp_path, monkeypatch):
    def fake_run_pipeline(**kwargs):
        _write_bundle(kwargs["out_dir"], law_id=kwargs["law_id"], import_blocking_passed=False)

    sources_path = _write_sources(tmp_path, [_source("ro.codul_muncii")])
    monkeypatch.setattr(daily_pipeline, "run_pipeline", fake_run_pipeline)

    summary = daily_pipeline.run_daily_ingestion(
        sources_path=sources_path,
        output_root=tmp_path / "output",
        skip_validation_gate=True,
    )

    assert summary["sources_succeeded"] == 1
    assert summary["sources_failed"] == 0
    assert summary["warnings"] == [
        "source_ro_codul_muncii: validation_gate_skipped_import_blocking_passed_false"
    ]


def test_with_embeddings_fake_provider_writes_output_and_validates(tmp_path, monkeypatch):
    def fake_run_pipeline(**kwargs):
        _write_bundle(kwargs["out_dir"], law_id=kwargs["law_id"])

    sources_path = _write_sources(tmp_path, [_source("ro.codul_muncii")])
    monkeypatch.setattr(daily_pipeline, "run_pipeline", fake_run_pipeline)

    summary = daily_pipeline.run_daily_ingestion(
        sources_path=sources_path,
        output_root=tmp_path / "output",
        with_embeddings=True,
        embedding_dim=2,
        embedding_batch_size=1,
        write_manifest=False,
    )

    source_summary = summary["sources"][0]
    output_path = Path(source_summary["embeddings_output_path"])
    assert summary["sources_succeeded"] == 1
    assert summary["embeddings_sources_succeeded"] == 1
    assert output_path.name == "embeddings_output.jsonl"
    assert output_path.exists()
    assert source_summary["embeddings_generated"] is True
    assert source_summary["embeddings_written_count"] == 2
    assert source_summary["embeddings_validation_passed"] is True
    assert source_summary["pair_validation_passed"] is True


def test_embedding_limit_allows_partial_pair_validation(tmp_path, monkeypatch):
    def fake_run_pipeline(**kwargs):
        _write_bundle(kwargs["out_dir"], law_id=kwargs["law_id"])

    sources_path = _write_sources(tmp_path, [_source("ro.codul_muncii")])
    monkeypatch.setattr(daily_pipeline, "run_pipeline", fake_run_pipeline)

    summary = daily_pipeline.run_daily_ingestion(
        sources_path=sources_path,
        output_root=tmp_path / "output",
        with_embeddings=True,
        embedding_dim=2,
        embedding_limit=1,
        write_manifest=False,
    )

    source_summary = summary["sources"][0]
    assert summary["sources_succeeded"] == 1
    assert source_summary["embeddings_written_count"] == 1
    assert source_summary["pair_validation_passed"] is True
    assert any("missing embedding output" in warning for warning in source_summary["warnings"])


def test_embedding_limit_does_not_write_import_ready_manifest(tmp_path, monkeypatch):
    def fake_run_pipeline(**kwargs):
        _write_bundle(kwargs["out_dir"], law_id=kwargs["law_id"])

    sources_path = _write_sources(tmp_path, [_source("ro.codul_muncii")])
    monkeypatch.setattr(daily_pipeline, "run_pipeline", fake_run_pipeline)

    summary = daily_pipeline.run_daily_ingestion(
        sources_path=sources_path,
        output_root=tmp_path / "output",
        with_embeddings=True,
        embedding_dim=2,
        embedding_limit=1,
    )

    source_summary = summary["sources"][0]
    out_dir = Path(source_summary["out_dir"])
    assert source_summary["manifest_path"] is None
    assert not (out_dir / "validated_embeddings_manifest.json").exists()
    assert "manifest_skipped_for_partial_embeddings" in source_summary["warnings"]


def test_complete_embeddings_writes_manifest(tmp_path, monkeypatch):
    def fake_run_pipeline(**kwargs):
        _write_bundle(kwargs["out_dir"], law_id=kwargs["law_id"])

    sources_path = _write_sources(tmp_path, [_source("ro.codul_muncii")])
    monkeypatch.setattr(daily_pipeline, "run_pipeline", fake_run_pipeline)

    summary = daily_pipeline.run_daily_ingestion(
        sources_path=sources_path,
        output_root=tmp_path / "output",
        with_embeddings=True,
        embedding_dim=2,
    )

    manifest_path = Path(summary["sources"][0]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_path.name == "validated_embeddings_manifest.json"
    assert manifest["ready_for_pgvector_import"] is True
    assert manifest["model_name"] == "qwen3-embedding:4b"
    assert manifest["embedding_dim"] == 2
    assert "embedding" not in manifest


def test_openai_compatible_provider_can_be_monkeypatched_without_network(tmp_path, monkeypatch):
    provider_calls = []

    class FakeOpenAICompatibleProvider:
        def __init__(self, *, base_url: str, timeout_seconds: float):
            provider_calls.append((base_url, timeout_seconds))

        def embed_texts(self, texts: list[str], model_name: str) -> list[list[float]]:
            return [[0.1, 0.2] for _ in texts]

    def fake_run_pipeline(**kwargs):
        _write_bundle(kwargs["out_dir"], law_id=kwargs["law_id"])

    sources_path = _write_sources(tmp_path, [_source("ro.codul_muncii")])
    monkeypatch.setattr(daily_pipeline, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        daily_pipeline,
        "OpenAICompatibleEmbeddingProvider",
        FakeOpenAICompatibleProvider,
    )

    summary = daily_pipeline.run_daily_ingestion(
        sources_path=sources_path,
        output_root=tmp_path / "output",
        with_embeddings=True,
        embedding_provider="openai-compatible",
        embedding_base_url="http://ollama.railway.internal:11434/v1",
        embedding_dim=2,
        write_manifest=False,
    )

    assert summary["sources_succeeded"] == 1
    assert provider_calls == [("http://ollama.railway.internal:11434/v1", 120.0)]


def test_summary_counts_success_and_failure(tmp_path, monkeypatch):
    def fake_run_pipeline(**kwargs):
        if kwargs["law_id"] == "ro.fail":
            raise RuntimeError("fixture failure")
        _write_bundle(kwargs["out_dir"], law_id=kwargs["law_id"])

    sources_path = _write_sources(
        tmp_path,
        [
            _source("ro.ok"),
            _source("ro.fail"),
            _source("ro.disabled", enabled=False),
        ],
    )
    monkeypatch.setattr(daily_pipeline, "run_pipeline", fake_run_pipeline)

    summary = daily_pipeline.run_daily_ingestion(
        sources_path=sources_path,
        output_root=tmp_path / "output",
    )

    assert summary["sources_total"] == 3
    assert summary["sources_enabled"] == 2
    assert summary["sources_succeeded"] == 1
    assert summary["sources_failed"] == 1
    assert len(summary["output_dirs"]) == 2
    assert summary["errors"] == [
        {
            "source_id": "source_ro_fail",
            "law_id": "ro.fail",
            "error": "fixture failure",
        }
    ]


def test_no_database_url_required(tmp_path, monkeypatch):
    def fake_run_pipeline(**kwargs):
        _write_bundle(kwargs["out_dir"], law_id=kwargs["law_id"])

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(daily_pipeline, "run_pipeline", fake_run_pipeline)
    sources_path = _write_sources(tmp_path, [_source("ro.codul_muncii")])

    summary = daily_pipeline.run_daily_ingestion(
        sources_path=sources_path,
        output_root=tmp_path / "output",
    )

    assert summary["sources_succeeded"] == 1


def _write_sources(tmp_path: Path, sources: list[dict]) -> Path:
    path = tmp_path / "legal_sources.json"
    path.write_text(json.dumps(sources, ensure_ascii=False), encoding="utf-8")
    return path


def _source(law_id: str, *, enabled: bool = True) -> dict:
    return {
        "source_id": f"source_{law_id.replace('.', '_')}",
        "law_id": law_id,
        "law_title": "Codul muncii",
        "source_url": "https://legislatie.just.ro/Public/DetaliiDocument/128647",
        "status": "active",
        "enabled": enabled,
    }


def _write_bundle(
    out_dir: Path,
    *,
    law_id: str,
    import_blocking_passed: bool = True,
    missing: set[str] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = missing or set()
    payloads = {
        "legal_units.json": [],
        "legal_edges.json": [],
        "legal_chunks.json": [],
        "corpus_manifest.json": {},
        "validation_report.json": {"import_blocking_passed": import_blocking_passed},
        "reference_candidates.json": [],
    }
    for filename, payload in payloads.items():
        if filename not in missing:
            (out_dir / filename).write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
    if "embeddings_input.jsonl" not in missing:
        records = [
            _embedding_input_record(law_id=law_id, suffix="1", text="retrieval text one"),
            _embedding_input_record(law_id=law_id, suffix="2", text="retrieval text two"),
        ]
        (out_dir / "embeddings_input.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )


def _embedding_input_record(*, law_id: str, suffix: str, text: str) -> dict:
    return {
        "record_id": f"embedding.chunk.{law_id}.art_{suffix}.0",
        "chunk_id": f"chunk.{law_id}.art_{suffix}.0",
        "legal_unit_id": f"{law_id}.art_{suffix}",
        "law_id": law_id,
        "text": text,
        "embedding_text": text,
        "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "model_hint": None,
        "metadata": {"retrieval_text_is_citable": False},
    }
