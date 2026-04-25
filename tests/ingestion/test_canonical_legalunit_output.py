from apps.api.app.schemas.query import LegalUnit
from ingestion.exporters import (
    build_parsed_legal_unit,
    legacy_unit_to_legal_unit_dict,
    parsed_legal_unit_to_legal_unit_dict,
)


def test_article_legal_unit_output_is_backend_compatible():
    raw_text = "Art. 41\nDreptul la munca nu poate fi ingradit."
    parsed = build_parsed_legal_unit(
        law_title="Codul muncii",
        raw_text=raw_text,
        hierarchy_path=[("articol", "41")],
    )

    unit = parsed_legal_unit_to_legal_unit_dict(parsed)
    validated = LegalUnit.model_validate(unit)

    assert validated.id == "ro.codul_muncii.art_41"
    assert unit["canonical_id"] == "codul_muncii:art_41"
    assert unit["parent_id"] == "ro.codul_muncii"
    assert unit["hierarchy_path"] == ["Art. 41"]
    assert unit["article_number"] == "41"
    assert unit["paragraph_number"] is None
    assert unit["raw_text"] == raw_text
    assert unit["normalized_text"] == "Art. 41 Dreptul la munca nu poate fi ingradit."
    assert unit["legal_domain"] == "munca"
    assert unit["legal_concepts"] == []
    assert unit["source_url"] is None
    assert unit["status"] == "unknown"
    assert "source_url_unknown" in unit["parser_warnings"]
    assert "status_unknown" in unit["parser_warnings"]


def test_paragraph_legal_unit_parent_and_hierarchy_are_coherent():
    parsed = build_parsed_legal_unit(
        law_title="Codul muncii",
        raw_text="(1) Dreptul la munca este garantat.",
        hierarchy_path=[("articol", "41"), ("alineat", "(1)")],
        source_url="https://legislatie.just.ro/Public/DetaliiDocument/53",
        source_id="legislatie.just.ro:53/2003",
    )

    unit = parsed_legal_unit_to_legal_unit_dict(parsed)

    assert LegalUnit.model_validate(unit).id == "ro.codul_muncii.art_41.alin_1"
    assert unit["canonical_id"] == "codul_muncii:art_41:alin_1"
    assert unit["parent_id"] == "ro.codul_muncii.art_41"
    assert unit["hierarchy_path"] == ["Art. 41", "Alin. (1)"]
    assert unit["legal_concepts"] == []
    assert "source_url_unknown" not in unit["parser_warnings"]
    assert "source_id_unknown" not in unit["parser_warnings"]


def test_letter_legal_unit_handles_uppercase_letter():
    parsed = build_parsed_legal_unit(
        law_title="Codul muncii",
        raw_text="k) un criteriu enumerat de lege.",
        hierarchy_path=[("articol", "17"), ("alineat", "(3)"), ("litera", "K)")],
    )

    unit = parsed_legal_unit_to_legal_unit_dict(parsed)

    assert LegalUnit.model_validate(unit).id == "ro.codul_muncii.art_17.alin_3.lit_k"
    assert unit["canonical_id"] == "codul_muncii:art_17:alin_3:lit_k"
    assert unit["parent_id"] == "ro.codul_muncii.art_17.alin_3"
    assert unit["hierarchy_path"] == ["Art. 17", "Alin. (3)", "Lit. k)"]
    assert unit["letter_number"] == "k"


def test_article_superscript_id_is_stable_and_raw_text_is_faithful():
    raw_text = "Art. 41¹\nText introdus pentru articol suplimentar."
    parsed = build_parsed_legal_unit(
        law_title="Codul muncii",
        raw_text=raw_text,
        hierarchy_path=[("articol", "41¹"), ("alineat", "(2)")],
    )

    unit = parsed_legal_unit_to_legal_unit_dict(parsed)

    assert LegalUnit.model_validate(unit).id == "ro.codul_muncii.art_41_1.alin_2"
    assert unit["canonical_id"] == "codul_muncii:art_41_1:alin_2"
    assert unit["parent_id"] == "ro.codul_muncii.art_41_1"
    assert unit["raw_text"] == raw_text
    assert unit["normalized_text"] != unit["raw_text"]


def test_unknown_policy_uses_unknown_null_and_empty_defaults():
    parsed = build_parsed_legal_unit(
        law_title="Legea nr. 999/2099",
        raw_text="Art. 1\nText fara metadata sigura.",
        hierarchy_path=[("articol", "1")],
    )

    unit = parsed_legal_unit_to_legal_unit_dict(parsed)

    assert LegalUnit.model_validate(unit).id == "ro.lege_999_2099.art_1"
    assert unit["legal_domain"] == "unknown"
    assert unit["legal_concepts"] == []
    assert unit["source_id"] is None
    assert unit["source_url"] is None
    assert unit["publication_date"] is None
    assert unit["effective_date"] is None
    assert unit["version_start"] is None
    assert unit["version_end"] is None
    assert unit["status"] == "unknown"
    assert "legal_domain_unknown" in unit["parser_warnings"]
    assert "source_url_unknown" in unit["parser_warnings"]
    assert "status_unknown" in unit["parser_warnings"]


def test_legacy_structural_unit_can_be_exported_without_changing_parser_output():
    legacy_unit = {
        "id": "ro.codul_muncii.art_41.alin_1",
        "type": "alineat",
        "raw_text": "(1) Dreptul la munca este garantat.",
        "hierarchy_path": ["41", "1"],
        "corpus_id": "ro.codul_muncii",
    }

    unit = legacy_unit_to_legal_unit_dict(
        legacy_unit,
        {
            "law_id": "ro.codul_muncii",
            "law_title": "Codul muncii",
            "status": "unknown",
            "source_url": "https://legislatie.just.ro/Public/DetaliiDocument/53",
        },
    )

    assert LegalUnit.model_validate(unit).id == "ro.codul_muncii.art_41.alin_1"
    assert unit["raw_text"] == legacy_unit["raw_text"]
    assert unit["parent_id"] == "ro.codul_muncii.art_41"
    assert unit["legal_domain"] == "munca"
    assert "legacy_unit_converted" in unit["parser_warnings"]
    assert "references" not in unit
