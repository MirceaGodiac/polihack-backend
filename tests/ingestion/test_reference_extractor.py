from ingestion.reference_extractor import extract_references


SOURCE_UNIT_ID = "ro.codul_muncii.art_100"


def _extract(text: str):
    return extract_references({"id": SOURCE_UNIT_ID, "raw_text": text})


def _by_raw(candidates: list[dict], raw_reference: str) -> dict:
    return next(
        candidate
        for candidate in candidates
        if candidate["raw_reference"] == raw_reference
    )


def test_detects_article_reference():
    candidate = _by_raw(_extract("Conform art. 41, regula se aplică."), "art. 41")

    assert candidate["reference_type"] == "article"
    assert candidate["target_article"] == "41"
    assert candidate["target_law_hint"] == "same_act"
    assert candidate["resolved_target_id"] is None
    assert candidate["resolution_status"] == "candidate_only"


def test_detects_superscript_and_caret_articles_with_same_normalization():
    candidates = _extract("A se vedea Art. 41^1 și Art. 41\u00b9.")

    assert _by_raw(candidates, "Art. 41^1")["target_article"] == "41_1"
    assert _by_raw(candidates, "Art. 41\u00b9")["target_article"] == "41_1"


def test_detects_paragraph_letter_point_and_thesis_references():
    candidates = _extract("Potrivit alin. (3), lit. k), pct. 2 și teza a II-a.")

    paragraph = _by_raw(candidates, "alin. (3)")
    letter = _by_raw(candidates, "lit. k)")
    point = _by_raw(candidates, "pct. 2")
    thesis = _by_raw(candidates, "teza a II-a")

    assert paragraph["reference_type"] == "paragraph"
    assert paragraph["target_paragraph"] == "3"
    assert paragraph["resolution_status"] == "unresolved_needs_context"
    assert letter["target_letter"] == "k"
    assert point["target_point"] == "2"
    assert thesis["target_thesis"] == "II"


def test_detects_compound_article_paragraph_letter_without_independent_letter():
    candidates = _extract("Se aplică art. 17 alin. (3) lit. k).")

    assert candidates == [
        {
            "source_unit_id": SOURCE_UNIT_ID,
            "raw_reference": "art. 17 alin. (3) lit. k)",
            "reference_type": "compound",
            "target_law_hint": "same_act",
            "target_article": "17",
            "target_paragraph": "3",
            "target_letter": "k",
            "target_point": None,
            "target_thesis": None,
            "resolved_target_id": None,
            "resolution_status": "candidate_only",
            "resolution_confidence": 0.0,
            "resolver_notes": ["compound_reference_candidate_only"],
        }
    ]


def test_detects_numbered_laws_and_ordinances():
    candidates = _extract(
        "Conform Legea nr. 53/2003, O.U.G. nr. 195/2002, "
        "OUG nr. 195/2002, O.G. nr. 2/2001, OG nr. 2/2001, "
        "H.G. nr. 1/2016 și HG nr. 1/2016."
    )

    assert _by_raw(candidates, "Legea nr. 53/2003")["target_law_hint"] == "ro.lege_53_2003"
    assert _by_raw(candidates, "O.U.G. nr. 195/2002")["reference_type"] == "oug"
    assert _by_raw(candidates, "O.U.G. nr. 195/2002")["target_law_hint"] == "ro.oug_195_2002"
    assert _by_raw(candidates, "OUG nr. 195/2002")["target_law_hint"] == "ro.oug_195_2002"
    assert _by_raw(candidates, "O.G. nr. 2/2001")["target_law_hint"] == "ro.og_2_2001"
    assert _by_raw(candidates, "OG nr. 2/2001")["target_law_hint"] == "ro.og_2_2001"
    assert _by_raw(candidates, "H.G. nr. 1/2016")["target_law_hint"] == "ro.hg_1_2016"
    assert _by_raw(candidates, "HG nr. 1/2016")["target_law_hint"] == "ro.hg_1_2016"


def test_detects_named_codes_with_diacritics_and_without():
    candidates = _extract(
        "Codul muncii, Codul civil, Codul fiscal, Codul penal, "
        "Codul de procedură civilă și Codul de procedura penala."
    )

    assert _by_raw(candidates, "Codul muncii")["target_law_hint"] == "ro.codul_muncii"
    assert _by_raw(candidates, "Codul civil")["target_law_hint"] == "ro.codul_civil"
    assert _by_raw(candidates, "Codul fiscal")["target_law_hint"] == "ro.codul_fiscal"
    assert _by_raw(candidates, "Codul penal")["target_law_hint"] == "ro.codul_penal"
    assert (
        _by_raw(candidates, "Codul de procedură civilă")["target_law_hint"]
        == "ro.codul_de_procedura_civila"
    )
    assert (
        _by_raw(candidates, "Codul de procedura penala")["target_law_hint"]
        == "ro.codul_de_procedura_penala"
    )


def test_detects_same_act_references():
    candidates = _extract(
        "prezentul cod, prezenta lege, prezentul act normativ, "
        "prezenta ordonanță, prezenta ordonanta, prezenta hotărâre și prezenta hotarare"
    )

    raw_refs = {candidate["raw_reference"] for candidate in candidates}
    assert "prezentul cod" in raw_refs
    assert "prezenta lege" in raw_refs
    assert "prezentul act normativ" in raw_refs
    assert "prezenta ordonanță" in raw_refs
    assert "prezenta ordonanta" in raw_refs
    assert "prezenta hotărâre" in raw_refs
    assert "prezenta hotarare" in raw_refs
    assert {candidate["target_law_hint"] for candidate in candidates} == {"same_act"}


def test_dedupes_exact_same_reference_in_same_unit():
    candidates = _extract("Vezi art. 41 și art. 41.")

    assert [candidate["raw_reference"] for candidate in candidates] == ["art. 41"]


def test_does_not_match_obvious_numbers_without_reference_labels():
    candidates = _extract("Termenul este de 20 de zile, suma este 100 lei, procentul este 5%.")

    assert candidates == []


def test_preserves_raw_reference_and_output_is_deterministic():
    text = "Conform art. 41 alin. (3), Legea nr. 53/2003 și prezentul cod."

    first = _extract(text)
    second = _extract(text)

    assert first == second
    assert [candidate["raw_reference"] for candidate in first] == [
        "art. 41 alin. (3)",
        "Legea nr. 53/2003",
        "prezentul cod",
    ]
