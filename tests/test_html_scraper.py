import pytest
from unittest.mock import patch, MagicMock
from ingestion.parser.html_scraper import scrape_html_source
import requests

def test_scrape_html_source_success():
    mock_html = "<html><body><h1>Test</h1></body></html>"
    with patch("ingestion.parser.html_scraper.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        url = "https://example.com"
        result = scrape_html_source(url)
        
        assert result == mock_html
        mock_get.assert_called_once_with(url, timeout=10)

def test_scrape_html_source_failure():
    with patch("ingestion.parser.html_scraper.requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        url = "https://example.com"
        result = scrape_html_source(url)
        
        assert result is None
        mock_get.assert_called_once_with(url, timeout=10)
