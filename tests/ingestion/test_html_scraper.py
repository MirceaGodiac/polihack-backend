from unittest.mock import MagicMock, patch

import requests
from bs4 import BeautifulSoup

from ingestion.html_scraper import (
    decode_html_content,
    repair_romanian_mojibake,
    scrape_html_source,
    scrape_with_soup,
)


BROKEN_PUBLICATION = "publicatÄƒ Ã®n MONITORUL OFICIAL"
FIXED_PUBLICATION = "publicat\u0103 \u00een MONITORUL OFICIAL"
BROKEN_LEGAL_TERMS = "InformaÈ›iile privind muncÄƒ"
FIXED_LEGAL_TERMS = "Informa\u021biile privind munc\u0103"
BROKEN_DEMO_TEXT = (
    "Contractul individual de muncÄ poate fi modificat numai prin acordul pÄrÈilor."
)
FIXED_DEMO_TEXT = (
    "Contractul individual de munc\u0103 poate fi modificat numai prin acordul p\u0103r\u021bilor."
)


def test_scrape_html_source_success():
    mock_html = "<html><body><h1>Test</h1></body></html>"
    with patch("ingestion.html_scraper.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = scrape_html_source("https://example.com")

        assert result == mock_html


def test_repair_romanian_mojibake_publication_text():
    assert repair_romanian_mojibake(BROKEN_PUBLICATION) == FIXED_PUBLICATION


def test_repair_romanian_mojibake_repairs_truncated_demo_words():
    assert repair_romanian_mojibake(BROKEN_DEMO_TEXT) == FIXED_DEMO_TEXT


def test_repair_romanian_mojibake_legal_terms():
    assert repair_romanian_mojibake(BROKEN_LEGAL_TERMS) == FIXED_LEGAL_TERMS


def test_repair_romanian_mojibake_leaves_correct_text_unchanged():
    assert repair_romanian_mojibake(FIXED_DEMO_TEXT) == FIXED_DEMO_TEXT


def test_decode_html_content_repairs_utf8_bytes_decoded_as_latin1():
    html_bytes = (
        b"<html><body>publicat\xc4\x83 \xc3\xaen MONITORUL OFICIAL "
        b"Informa\xc8\x9biile privind munc\xc4\x83</body></html>"
    )

    decoded = decode_html_content(
        html_bytes,
        headers={"Content-Type": "text/html; charset=iso-8859-1"},
    )

    assert FIXED_PUBLICATION in decoded
    assert FIXED_LEGAL_TERMS in decoded
    assert BROKEN_PUBLICATION not in decoded
    assert BROKEN_LEGAL_TERMS not in decoded


def test_scrape_html_source_decodes_response_content_bytes():
    html_bytes = (
        b"<html><body>publicat\xc4\x83 \xc3\xaen MONITORUL OFICIAL "
        b"Informa\xc8\x9biile privind munc\xc4\x83</body></html>"
    )
    with patch("ingestion.html_scraper.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = html_bytes
        mock_response.headers = {"Content-Type": "text/html; charset=iso-8859-1"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = scrape_html_source("https://example.com")

        assert result is not None
        assert FIXED_PUBLICATION in result
        assert FIXED_LEGAL_TERMS in result


def test_scrape_html_source_failure():
    with patch("ingestion.html_scraper.requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        result = scrape_html_source("https://example.com")

        assert result is None


def test_scrape_with_soup_returns_beautifulsoup():
    mock_html = "<html><body><p>Hello</p></body></html>"
    with patch("ingestion.html_scraper.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = scrape_with_soup("https://example.com")

        assert isinstance(result, BeautifulSoup)
        assert result.find("p").text == "Hello"


def test_scrape_with_soup_returns_none_on_failure():
    with patch("ingestion.html_scraper.requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.ConnectionError()

        result = scrape_with_soup("https://example.com")

        assert result is None
