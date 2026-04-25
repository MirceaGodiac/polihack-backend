import json
from pathlib import Path

from ingestion.html_cleaner import clean_html_to_lines
from ingestion.structural_parser import StructuralParser


FIXTURE_DIR = Path("tests/fixtures/html")
NAVIGATION_TERMS = ("Meniu", "Căutare", "Acasă", "Tipărește")


def _fixture_text(name: str) -> str:
    return (FIXTURE_DIR / f"{name}.html").read_text(encoding="utf-8")


def _expected_lines(name: str) -> list[str]:
    return json.loads(
        (FIXTURE_DIR / f"{name}.expected_lines.json").read_text(encoding="utf-8")
    )


def test_cleaner_removes_script_style_nav_footer_and_matches_golden_lines():
    result = clean_html_to_lines(_fixture_text("codul_muncii_fragment"))
    joined = "\n".join(result.lines)

    assert result.lines == _expected_lines("codul_muncii_fragment")
    assert result.selected_container == "#textdocumentleg"
    assert result.removed_blocks_count >= 4
    assert "removed_navigation_blocks" in result.warnings
    assert "window.fakeNavigation" not in joined
    assert "display: block" not in joined
    assert "Contact" not in joined
    for term in NAVIGATION_TERMS:
        assert term not in joined


def test_cleaner_preserves_legal_diacritics_and_numbering():
    result = clean_html_to_lines(_fixture_text("codul_muncii_fragment"))

    assert "(1) Contractul individual de muncă poate fi modificat numai prin acordul părților." in result.lines
    assert "(4) Orice modificare a unuia dintre elementele prevăzute la alin. (3), în timpul executării contractului individual de muncă, impune încheierea unui act adițional la contract." in result.lines
    assert "k) salariul de bază, alte elemente constitutive ale veniturilor salariale, precum și periodicitatea plății salariului la care salariatul are dreptul." in result.lines
    assert "Art. 41" in result.lines
    assert "Art. 41¹" in result.lines


def test_cleaner_body_fallback_is_explicit_and_still_clean():
    result = clean_html_to_lines(_fixture_text("noisy_legislatie_page"))
    joined = "\n".join(result.lines)

    assert result.lines == _expected_lines("noisy_legislatie_page")
    assert result.selected_container == "body"
    assert "legal_container_not_found" in result.warnings
    assert "used_body_fallback" in result.warnings
    assert "removed_navigation_blocks" in result.warnings
    assert "possible_navigation_residue" not in result.warnings
    for term in NAVIGATION_TERMS:
        assert term not in joined
    assert "Art. 41^1" in result.lines


def test_cleaner_output_is_deterministic():
    first = clean_html_to_lines(_fixture_text("codul_muncii_fragment"))
    second = clean_html_to_lines(_fixture_text("codul_muncii_fragment"))

    assert first == second
    assert first.text_hash == second.text_hash
    assert first.text_hash is not None


def test_cleaner_does_not_empty_simple_legal_html():
    html = """
    <html>
      <body>
        <p>Articolul 1</p>
        <p>(1) Dreptul la muncă este garantat.</p>
      </body>
    </html>
    """

    result = clean_html_to_lines(html)

    assert result.lines == ["Articolul 1", "(1) Dreptul la muncă este garantat."]
    assert "legal_container_not_found" in result.warnings
    assert "used_body_fallback" in result.warnings


def test_cleaned_lines_remain_structural_parser_compatible():
    result = clean_html_to_lines(_fixture_text("codul_muncii_fragment"))
    parser = StructuralParser("ro.codul_muncii")

    units, edges = parser.parse(result.lines)
    unit_ids = {unit["id"] for unit in units}
    edge_pairs = {(edge["source_id"], edge["target_id"]) for edge in edges}

    assert "ro.codul_muncii.art_41" in unit_ids
    assert "ro.codul_muncii.art_41.alin_1" in unit_ids
    assert "ro.codul_muncii.art_41.alin_4" in unit_ids
    assert "ro.codul_muncii.art_17.alin_3.lit_k" in unit_ids
    assert "ro.codul_muncii.art_41_1" in unit_ids
    assert ("ro.codul_muncii.art_41", "ro.codul_muncii.art_41.alin_1") in edge_pairs
    assert (
        "ro.codul_muncii.art_17.alin_3",
        "ro.codul_muncii.art_17.alin_3.lit_k",
    ) in edge_pairs
