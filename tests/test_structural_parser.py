import pytest
from ingestion.structural_parser import StructuralParser
from ingestion.legal_ids import make_unit_id
from ingestion.validators import validate_corpus

def test_make_unit_id_normalization():
    # Test normalization of ^
    id1 = make_unit_id("ro.codul_muncii", [("articol", "41^1")])
    assert id1 == "ro.codul_muncii.art_41_1"
    
    # Test hierarchy
    path = [("articol", "41"), ("alineat", "1")]
    id2 = make_unit_id("ro.codul_muncii", path)
    assert id2 == "ro.codul_muncii.art_41.alin_1"

def test_parser_basic_hierarchy():
    parser = StructuralParser("ro.codul_muncii")
    mock_text = [
        "Art. 41 Libertatea muncii",
        "(1) Dreptul la muncă este garantat.",
        "a) Primul punct al literei."
    ]
    
    units, edges = parser.parse(mock_text)
    
    assert len(units) == 3
    assert units[0]['id'] == "ro.codul_muncii.art_41"
    assert units[1]['id'] == "ro.codul_muncii.art_41.alin_1"
    assert units[2]['id'] == "ro.codul_muncii.art_41.alin_1.lit_a"
    
    assert len(edges) == 2
    assert edges[0]['source_id'] == "ro.codul_muncii.art_41"
    assert edges[0]['target_id'] == "ro.codul_muncii.art_41.alin_1"
    assert edges[1]['source_id'] == "ro.codul_muncii.art_41.alin_1"
    assert edges[1]['target_id'] == "ro.codul_muncii.art_41.alin_1.lit_a"

def test_validator_duplicates():
    units = [
        {"id": "ro.test.1", "raw_text": "text1"},
        {"id": "ro.test.1", "raw_text": "text2"}
    ]
    with pytest.raises(ValueError, match="Duplicate ID found"):
        validate_corpus(units)

def test_complex_state_reset():
    parser = StructuralParser("ro.test")
    lines = [
        "Art. 1",
        "(1) Para 1",
        "Art. 2",
        "(1) Para 2"
    ]
    units, _ = parser.parse(lines)
    assert units[2]['id'] == "ro.test.art_2"
    assert units[3]['id'] == "ro.test.art_2.alin_1"
