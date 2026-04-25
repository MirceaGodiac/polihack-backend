import pytest
from ingestion.parser.atomic_parser import AtomicParser

# ── Fixture ───────────────────────────────────────────────────────────────────

SAMPLE_TEXT = """\
TITLUL I
Dispozitii generale

CAPITOLUL I
Domeniul de aplicare

Art. 1
Prezentul cod reglementeaza domeniul raporturilor de munca.

Art. 2
Dispozitiile prezentului cod se aplica:
a) cetatenilor romani cu domiciliul in Romania;
b) cetatenilor straini sau apatrizilor.

CAPITOLUL II
Principiile fundamentale

Art. 3
Libertatea muncii este garantata prin Constitutie.

Art. 4
(1) Munca fortata este interzisa.
(2) Prin munca fortata se intelege orice munca impusa unui individ.

Art. 5
Orice discriminare directa sau indirecta este interzisa, conform art. 4 alin. (1) din prezentul cod.
"""


@pytest.fixture
def parser():
    return AtomicParser(corpus_id="ro.codul_muncii")


@pytest.fixture
def result(parser):
    return parser.parse(SAMPLE_TEXT)


@pytest.fixture
def units(result):
    return result[0]


@pytest.fixture
def edges(result):
    return result[1]


# ── Basic structure ───────────────────────────────────────────────────────────

def test_units_is_list(units):
    assert isinstance(units, list)
    assert len(units) > 0


def test_edges_is_list(edges):
    assert isinstance(edges, list)
    assert len(edges) > 0


def test_unit_has_required_fields(units):
    required = {"id", "path", "type", "text", "references"}
    for unit in units:
        assert required.issubset(unit.keys()), f"Missing fields in unit: {unit}"


def test_types_are_valid(units):
    valid_types = {"titlu", "capitol", "sectiune", "articol", "alineat", "litera"}
    for unit in units:
        assert unit["type"] in valid_types


# ── IDs and paths ─────────────────────────────────────────────────────────────

def test_titlu_id(units):
    titlu = next(u for u in units if u["type"] == "titlu")
    assert titlu["id"] == "ro.codul_muncii.titlu_i"


def test_capitol_id(units):
    cap = next(u for u in units if u["type"] == "capitol")
    assert cap["id"] == "ro.codul_muncii.titlu_i.capitol_i"


def test_articol_id(units):
    art1 = next(u for u in units if u["type"] == "articol" and "Art. 1" in u["text"])
    assert art1["id"] == "ro.codul_muncii.titlu_i.capitol_i.art_1"


def test_litera_id(units):
    lit_a = next(u for u in units if u["type"] == "litera" and u["text"].startswith("a)"))
    assert lit_a["id"] == "ro.codul_muncii.titlu_i.capitol_i.art_2.lit_a"


def test_alineat_id(units):
    alin = next(u for u in units if u["type"] == "alineat" and "(1)" in u["text"])
    assert "alin_1" in alin["id"]


def test_path_breadcrumb(units):
    lit_a = next(u for u in units if u["type"] == "litera" and u["text"].startswith("a)"))
    assert lit_a["path"] == "Titlul I > Capitolul I > Art. 2 > a)"


# ── Hierarchy reset ───────────────────────────────────────────────────────────

def test_litera_resets_on_new_article(units):
    """Letters from Art.2 must not bleed into Art.3."""
    art3_and_after = [u for u in units if "art_3" in u["id"] or "art_4" in u["id"] or "art_5" in u["id"]]
    for u in art3_and_after:
        # No unit after Art.3 should reference Art.2 parent in its ID
        assert "art_2" not in u["id"]


def test_capitol_ii_resets_capitol(units):
    cap2 = next(u for u in units if u["type"] == "capitol" and "II" in u["text"])
    assert cap2["id"] == "ro.codul_muncii.titlu_i.capitol_ii"


# ── Cross-references ──────────────────────────────────────────────────────────

def test_references_extracted(units):
    art5 = next(u for u in units if u["type"] == "articol" and "Art. 5" in u["text"])
    assert len(art5["references"]) > 0


def test_reference_fields(units):
    art5 = next(u for u in units if u["type"] == "articol" and "Art. 5" in u["text"])
    ref = art5["references"][0]
    assert "raw_reference" in ref
    assert "target_article" in ref
    assert "target_law_hint" in ref
    # source_unit_id should NOT be in the output (it's redundant — we already have the unit id)
    assert "source_unit_id" not in ref


def test_reference_values(units):
    art5 = next(u for u in units if u["type"] == "articol" and "Art. 5" in u["text"])
    # Find the reference that targets article 4
    ref = next(r for r in art5["references"] if r["target_article"] == "4")
    assert ref["target_paragraph"] == "1"


# ── Empty / edge cases ────────────────────────────────────────────────────────

def test_empty_text(parser):
    assert parser.parse("") == ([], [])


def test_only_whitespace(parser):
    assert parser.parse("   \n\n\t  ") == ([], [])


def test_no_structure_appends_to_last_unit(parser):
    text = "Art. 1\nSome content.\nMore content."
    units, edges = parser.parse(text)
    assert len(units) == 1
    assert "More content." in units[0]["text"]


def test_contains_edges(units, edges):
    # Capitolul I should contain Art. 1
    cap1 = next(u for u in units if u["type"] == "capitol" and "I" in u["text"])
    art1 = next(u for u in units if u["type"] == "articol" and "Art. 1" in u["text"])
    
    edge = next(e for e in edges if e["target_id"] == art1["id"])
    assert edge["source_id"] == cap1["id"]
    assert edge["type"] == "contains"
