from __future__ import annotations

from typing import Optional

import requests
from bs4 import BeautifulSoup


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def scrape_html_source(url: str) -> Optional[str]:
    """
    Fetch raw HTML source from the given URL.

    This helper only fetches HTML. Canonical parsing happens in
    scripts/run_parser_pipeline.py via html_cleaner + StructuralParser.
    """
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as exc:
        print(f"[scraper] Error fetching URL {url}: {exc}")
        return None


def scrape_with_soup(url: str) -> Optional[BeautifulSoup]:
    html = scrape_html_source(url)
    if html is None:
        return None
    return BeautifulSoup(html, "html.parser")
