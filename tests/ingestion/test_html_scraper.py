import pytest
from unittest.mock import patch, MagicMock
from ingestion.html_scraper import scrape_html_source, scrape_with_soup
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
