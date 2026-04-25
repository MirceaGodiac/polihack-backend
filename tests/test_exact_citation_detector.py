from apps.api.app.services.exact_citation_detector import ExactCitationDetector


def detect(question: str):
    return ExactCitationDetector().detect(question)


def test_detects_article_and_named_code():
    citations = detect("Ce spune art. 41 Codul muncii?")

    assert len(citations) == 1
    citation = citations[0]
    assert citation.article == "41"
    assert citation.act_hint == "Codul muncii"
    assert citation.law_id_hint == "ro.codul_muncii"
    assert citation.lookup_filters["law_id"] == "ro.codul_muncii"
    assert citation.lookup_filters["article_number"] == "41"


def test_groups_article_paragraph_letter_and_named_code():
    citations = detect("Ce prevede art. 17 alin. (3) lit. k) din Codul muncii?")

    assert len(citations) == 1
    citation = citations[0]
    assert citation.citation_type == "compound"
    assert citation.article == "17"
    assert citation.paragraph == "3"
    assert citation.letter == "k"
    assert citation.law_id_hint == "ro.codul_muncii"


def test_detects_law_number_and_year_alias():
    citations = detect("Ce prevede Legea nr. 53/2003?")

    assert len(citations) == 1
    citation = citations[0]
    assert citation.citation_type == "law"
    assert citation.act_type == "lege"
    assert citation.act_number == "53"
    assert citation.act_year == "2003"
    assert citation.law_id_hint == "ro.codul_muncii"


def test_detects_oug_number_and_year():
    citations = detect("Ce spune O.U.G. nr. 57/2019?")

    assert len(citations) == 1
    citation = citations[0]
    assert citation.citation_type == "ordinance"
    assert citation.act_type == "oug"
    assert citation.act_number == "57"
    assert citation.act_year == "2019"
    assert citation.law_id_hint == "ro.oug_57_2019"


def test_detects_hg_number_and_year_without_alias():
    citations = detect("Ce spune HG nr. 500/2011?")

    assert len(citations) == 1
    citation = citations[0]
    assert citation.citation_type == "government_decision"
    assert citation.act_type == "hg"
    assert citation.act_number == "500"
    assert citation.act_year == "2011"
    assert citation.law_id_hint is None
    assert citation.needs_resolution is True


def test_detects_relative_paragraph():
    citations = detect("Ce înseamnă alin. (2)?")

    assert len(citations) == 1
    citation = citations[0]
    assert citation.citation_type == "paragraph"
    assert citation.paragraph == "2"
    assert citation.is_relative is True
    assert citation.needs_resolution is True


def test_detects_relative_present_law_without_law_id():
    citations = detect("Cum se aplică prezenta lege?")

    assert len(citations) == 1
    citation = citations[0]
    assert citation.is_relative is True
    assert citation.needs_resolution is True
    assert citation.law_id_hint is None


def test_does_not_detect_plain_text_without_legal_citation():
    assert detect("Ce drepturi are un salariat într-o situație obișnuită?") == []
