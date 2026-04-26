import json

import ingestion.pipeline as pipeline


SAMPLE_LEGISLATIE_HTML = """
<html>
  <head><title>Codul muncii</title></head>
  <body>
    <nav>Meniu Cautare Acasa Tipareste</nav>
    <main id="textdocumentleg">
      <h1>Codul muncii</h1>
      <p>Art. 17</p>
      <p>(3) Persoana selectată va fi informată cu privire la:</p>
      <p>k) salariul de bază, alte elemente constitutive ale veniturilor salariale.</p>
      <p>Art. 41</p>
      <p>(1) Contractul individual de muncă poate fi modificat numai prin acordul părților.</p>
      <p>(3) Modificarea contractului individual de muncă se referă la durata contractului,
      locul muncii, felul muncii, condițiile de muncă, salariul și timpul de muncă.</p>
      <p>(4) Modificarea contractului individual de muncă se face prin act adițional,
      în condițiile prevăzute la alin. (3).</p>
    </main>
    <script>window.noise = true;</script>
    <footer>Footer legislatie</footer>
  </body>
</html>
"""

DIACRITICS_LEGISLATIE_HTML = """
<html>
  <head><title>Codul muncii</title></head>
  <body>
    <header>CÄƒutare Meniu AcasÄƒ TipÄƒreÈ™te</header>
    <nav>AcasÄƒ CÄƒutare Meniu</nav>
    <main id="textdocumentleg">
      <h1>Codul muncii</h1>
      <p>Art. 1</p>
      <p>(1) Legea este publicatÄƒ Ã®n Monitorul Oficial.</p>
      <p>(2) InformaÈ›iile privind muncÄƒ sunt stabilite prin lege.</p>
    </main>
  </body>
</html>
"""


def test_run_pipeline_from_url_exports_canonical_bundle(tmp_path, monkeypatch):
    url = "https://example.gov/Public/DetaliiDocument/123456"
    monkeypatch.setattr(
        pipeline,
        "scrape_html_source",
        lambda requested_url: SAMPLE_LEGISLATIE_HTML,
    )

    result = pipeline.run_pipeline(
        url=url,
        out_dir=tmp_path,
        law_id="ro.codul_muncii",
        law_title="Codul muncii",
        generated_at="2026-04-25T00:00:00+00:00",
        write_debug=True,
    )

    assert result.import_blocking_passed is True
    assert result.law_id == "ro.codul_muncii"
    assert result.intermediate_units_count >= 7
    assert set(result.artifact_paths) == {
        "legal_units",
        "legal_edges",
        "legal_chunks",
        "embeddings_input",
        "corpus_manifest",
        "validation_report",
        "reference_candidates",
    }
    for path in result.artifact_paths.values():
        assert path.exists()

    legal_units = json.loads(
        result.artifact_paths["legal_units"].read_text(encoding="utf-8")
    )
    units_by_id = {unit["id"]: unit for unit in legal_units}
    assert "ro.codul_muncii.art_41" in units_by_id
    assert "ro.codul_muncii.art_41.alin_4" in units_by_id
    assert "ro.codul_muncii.art_17.alin_3.lit_k" in units_by_id
    assert units_by_id["ro.codul_muncii.art_41.alin_4"]["source_url"] == url
    assert (
        units_by_id["ro.codul_muncii.art_41.alin_4"]["raw_text"]
        == "(4) Modificarea contractului individual de muncă se face prin act adițional,\n"
        "în condițiile prevăzute la alin. (3)."
    )

    legal_edges = json.loads(
        result.artifact_paths["legal_edges"].read_text(encoding="utf-8")
    )
    assert all(edge["type"] == "contains" for edge in legal_edges)
    assert {
        (edge["source_id"], edge["target_id"])
        for edge in legal_edges
    } >= {
        ("ro.codul_muncii", "ro.codul_muncii.art_41"),
        ("ro.codul_muncii.art_41", "ro.codul_muncii.art_41.alin_4"),
        ("ro.codul_muncii.art_17.alin_3", "ro.codul_muncii.art_17.alin_3.lit_k"),
    }

    legal_chunks = json.loads(
        result.artifact_paths["legal_chunks"].read_text(encoding="utf-8")
    )
    chunk_by_unit = {chunk["legal_unit_id"]: chunk for chunk in legal_chunks}
    assert chunk_by_unit["ro.codul_muncii.art_41.alin_4"]["text"] == units_by_id[
        "ro.codul_muncii.art_41.alin_4"
    ]["raw_text"]
    assert (
        "act adițional"
        in chunk_by_unit["ro.codul_muncii.art_41.alin_4"]["retrieval_text"]
    )

    embeddings_lines = result.artifact_paths["embeddings_input"].read_text(
        encoding="utf-8"
    ).splitlines()
    assert embeddings_lines
    first_embedding_record = json.loads(embeddings_lines[0])
    assert {"record_id", "chunk_id", "legal_unit_id", "text_hash"} <= set(
        first_embedding_record
    )

    manifest = json.loads(
        result.artifact_paths["corpus_manifest"].read_text(encoding="utf-8")
    )
    assert manifest["input_files"] == [url]
    assert manifest["sources"][0]["source_type"] == "url:example.gov"
    assert manifest["contextual_retrieval_enabled"] is True

    validation_report = json.loads(
        result.artifact_paths["validation_report"].read_text(encoding="utf-8")
    )
    assert validation_report["import_blocking_passed"] is True
    assert validation_report["chunks_count"] == len(legal_chunks)

    assert (tmp_path / "cleaned_lines.txt").exists()
    assert (tmp_path / "cleaner_report.json").exists()
    assert (tmp_path / "intermediate_units.json").exists()


def test_run_pipeline_repairs_romanian_mojibake_in_embeddings_input(
    tmp_path,
    monkeypatch,
):
    url = "https://example.gov/Public/DetaliiDocument/123456"
    monkeypatch.setattr(
        pipeline,
        "scrape_html_source",
        lambda requested_url: DIACRITICS_LEGISLATIE_HTML,
    )

    result = pipeline.run_pipeline(
        url=url,
        out_dir=tmp_path,
        law_id="ro.codul_muncii",
        law_title="Codul muncii",
        source_id="fixture_diacritics",
        status="active",
        generated_at="2026-04-25T00:00:00+00:00",
        write_debug=True,
    )

    cleaned_lines = (tmp_path / "cleaned_lines.txt").read_text(encoding="utf-8")
    legal_units = result.artifact_paths["legal_units"].read_text(encoding="utf-8")
    embeddings_input = result.artifact_paths["embeddings_input"].read_text(
        encoding="utf-8"
    )

    for payload in (cleaned_lines, legal_units, embeddings_input):
        assert "publicată în" in payload
        assert "Informațiile" in payload
        assert "muncă" in payload
        assert "publicatÄƒ" not in payload
        assert "InformaÈ›iile" not in payload
        assert "muncÄƒ" not in payload

    embedding_records = [json.loads(line) for line in embeddings_input.splitlines()]
    serialized_records = json.dumps(embedding_records, ensure_ascii=False)
    assert "muncă" in serialized_records
    assert "muncÄƒ" not in serialized_records


def test_run_pipeline_accepts_non_legislatie_http_url(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "scrape_html_source",
        lambda requested_url: SAMPLE_LEGISLATIE_HTML,
    )

    result = pipeline.run_pipeline(
        url="https://example.com/Public/DetaliiDocument/123456",
        out_dir=tmp_path,
        law_id="ro.codul_muncii",
        law_title="Codul muncii",
    )

    assert result.import_blocking_passed is True


def test_run_pipeline_rejects_non_http_url(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "scrape_html_source",
        lambda requested_url: SAMPLE_LEGISLATIE_HTML,
    )

    try:
        pipeline.run_pipeline(
            url="file:///tmp/source.html",
            out_dir=tmp_path,
            law_id="ro.codul_muncii",
            law_title="Codul muncii",
        )
    except ValueError as exc:
        assert "http or https" in str(exc)
    else:
        raise AssertionError("Expected non-http URL to be rejected")


def test_run_pipeline_can_infer_title_and_law_id_from_html(tmp_path, monkeypatch):
    url = "https://legislatie.just.ro/Public/DetaliiDocument/123456"
    monkeypatch.setattr(
        pipeline,
        "scrape_html_source",
        lambda requested_url: SAMPLE_LEGISLATIE_HTML,
    )

    result = pipeline.run_pipeline(
        url=url,
        out_dir=tmp_path,
        generated_at="2026-04-25T00:00:00+00:00",
    )

    assert result.law_title == "Codul muncii"
    assert result.law_id == "ro.codul_muncii"


def test_run_pipeline_normalizes_numbered_legislatie_title():
    assert (
        pipeline._make_law_id_from_title(
            "LEGE nr. 53 din 24 ianuarie 2003 - Portal Legislativ"
        )
        == "ro.lege_53_2003"
    )
