from ingestion.legal_ids import (
    make_article_segment,
    make_canonical_id,
    make_law_id,
    make_letter_segment,
    make_parent_unit_id,
    make_paragraph_segment,
    make_unit_id,
)


def test_law_id_codul_muncii():
    assert make_law_id("Codul muncii") == "ro.codul_muncii"


def test_law_id_legea_nr_53_2003():
    assert make_law_id("Legea nr. 53/2003") == "ro.lege_53_2003"


def test_article_id_segments_are_stable():
    assert make_article_segment("41") == "art_41"
    assert make_article_segment("41^1") == "art_41_1"
    assert make_article_segment("41¹") == "art_41_1"


def test_paragraph_and_letter_segments_are_stable():
    assert make_paragraph_segment("(1)") == "alin_1"
    assert make_letter_segment("K)") == "lit_k"


def test_unit_id_examples_are_stable():
    assert make_unit_id("ro.codul_muncii", []) == "ro.codul_muncii"
    assert make_unit_id("ro.codul_muncii", [("articol", "41")]) == "ro.codul_muncii.art_41"
    assert (
        make_unit_id("ro.codul_muncii", [("articol", "41"), ("alineat", "(1)")])
        == "ro.codul_muncii.art_41.alin_1"
    )
    assert (
        make_unit_id(
            "ro.codul_muncii",
            [("articol", "17"), ("alineat", "(3)"), ("litera", "K)")],
        )
        == "ro.codul_muncii.art_17.alin_3.lit_k"
    )
    assert make_unit_id("ro.codul_muncii", [("articol", "41¹")]) == "ro.codul_muncii.art_41_1"
    assert (
        make_unit_id("ro.codul_muncii", [("articol", "41^1"), ("alineat", "(2)")])
        == "ro.codul_muncii.art_41_1.alin_2"
    )


def test_canonical_id_examples_are_stable():
    assert make_canonical_id("ro.codul_muncii", []) == "codul_muncii"
    assert make_canonical_id("ro.codul_muncii", [("articol", "41")]) == "codul_muncii:art_41"
    assert (
        make_canonical_id("ro.codul_muncii", [("articol", "41"), ("alineat", "(1)")])
        == "codul_muncii:art_41:alin_1"
    )
    assert (
        make_canonical_id(
            "ro.codul_muncii",
            [("articol", "17"), ("alineat", "(3)"), ("litera", "K)")],
        )
        == "codul_muncii:art_17:alin_3:lit_k"
    )


def test_parent_id_is_coherent():
    path = [("articol", "17"), ("alineat", "(3)"), ("litera", "K)")]
    assert make_parent_unit_id("ro.codul_muncii", path) == "ro.codul_muncii.art_17.alin_3"
    assert make_parent_unit_id("ro.codul_muncii", [("articol", "17")]) == "ro.codul_muncii"
    assert make_parent_unit_id("ro.codul_muncii", []) is None
