import requests
from bs4 import BeautifulSoup
from typing import Optional

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def scrape_html_source(url: str) -> Optional[str]:
    """
    Fetches the raw HTML source code from the given URL.

    Args:
        url (str): The link to scrape.

    Returns:
        Optional[str]: The raw HTML string if successful, None otherwise.
    """
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"[scraper] Error fetching URL {url}: {e}")
        return None


def scrape_with_soup(url: str) -> Optional[BeautifulSoup]:
    """
    Fetches a webpage and parses it directly with BeautifulSoup.

    Args:
        url (str): The link to scrape.

    Returns:
        Optional[BeautifulSoup]: A parsed BeautifulSoup object, or None on failure.
    """
    html = scrape_html_source(url)
    if html is None:
        return None
    return BeautifulSoup(html, "html.parser")
