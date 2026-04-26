from ingestion.normalizer import (
    contains_romanian_mojibake,
    normalize_legal_text,
    repair_romanian_mojibake,
)


def test_repair_romanian_mojibake_repairs_demo_contract_text():
    broken = (
        "Contractul individual de muncÄ poate fi modificat numai prin acordul "
        "pÄrÈilor."
    )

    repaired = repair_romanian_mojibake(broken)

    assert (
        repaired
        == "Contractul individual de muncă poate fi modificat numai prin acordul părților."
    )


def test_repair_romanian_mojibake_is_idempotent_for_correct_text():
    text = "Contractul individual de muncă poate fi modificat numai prin acordul părților."

    assert repair_romanian_mojibake(text) == text


def test_contains_romanian_mojibake_flags_citable_raw_text_only_when_broken():
    assert contains_romanian_mojibake("muncÄ") is True
    assert contains_romanian_mojibake("pÄrÈilor") is True
    assert contains_romanian_mojibake("muncă") is False
    assert contains_romanian_mojibake("părților") is False


def test_normalize_legal_text_repairs_before_normalizing():
    assert (
        normalize_legal_text("Contractul individual de muncÄ poate fi modificat.")
        == "Contractul individual de muncă poate fi modificat."
    )
