import pytest
from unittest.mock import patch, MagicMock
from ingestion.html_scraper import (
    decode_html_content,
    repair_romanian_mojibake,
    scrape_html_source,
    scrape_with_soup,
)
from bs4 import BeautifulSoup
import requests


# ── scrape_html_source ────────────────────────────────────────────────────────

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
    assert (
        repair_romanian_mojibake("publicatÄƒ Ã®n MONITORUL OFICIAL")
        == "publicată în MONITORUL OFICIAL"
    )


def test_repair_romanian_mojibake_legal_terms():
    assert (
        repair_romanian_mojibake("informaÈ›iile privind muncÄƒ")
        == "informațiile privind muncă"
    )


def test_repair_romanian_mojibake_leaves_correct_text_unchanged():
    assert repair_romanian_mojibake("publicată în muncă") == "publicată în muncă"


def test_decode_html_content_repairs_utf8_bytes_decoded_as_latin1():
    html = "<html><body>publicată în informațiile privind muncă</body></html>"

    decoded = decode_html_content(
        html.encode("utf-8"),
        headers={"Content-Type": "text/html; charset=iso-8859-1"},
    )

    assert "publicată în" in decoded
    assert "informațiile" in decoded
    assert "muncă" in decoded
    assert "publicatÄƒ" not in decoded


def test_scrape_html_source_decodes_response_content_bytes():
    html = "<html><body>publicată în informațiile privind muncă</body></html>"
    with patch("ingestion.html_scraper.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = html.encode("utf-8")
        mock_response.headers = {"Content-Type": "text/html; charset=iso-8859-1"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = scrape_html_source("https://example.com")

        assert result is not None
        assert "publicată în" in result
        assert "informațiile" in result
        assert "muncă" in result


def test_scrape_html_source_failure():
    with patch("ingestion.html_scraper.requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        result = scrape_html_source("https://example.com")

        assert result is None


# ── scrape_with_soup ──────────────────────────────────────────────────────────

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
