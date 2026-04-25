import re
import unicodedata
from bs4 import BeautifulSoup
from typing import Optional

# Tags whose content is never meaningful printable text
_NON_PRINTABLE_TAGS = {"script", "style", "noscript", "head", "meta", "link", "iframe"}

# Containers targeting legislatie.just.ro document body
_CONTENT_SELECTORS = [
    {"id": "textdocumentleg"},
    {"class_": "textdocumentleg"},
]


def parse_html_to_text(html_content: str) -> Optional[str]:
    """
    Parses HTML content and extracts the main legal text.
    Targets the Romanian Ministry of Justice portal structure, with body fallback.

    Args:
        html_content (str): The raw HTML string.

    Returns:
        Optional[str]: The extracted plain text, or None if extraction fails.
    """
    if not html_content:
        return None

    soup = BeautifulSoup(html_content, "html.parser")

    main_content = None
    for selector in _CONTENT_SELECTORS:
        main_content = soup.find(**selector)
        if main_content:
            break

    if not main_content:
        main_content = soup.find("body")

    if main_content:
        return main_content.get_text(separator="\n", strip=True)

    return None


def parse_printable_text(html_content: str) -> Optional[str]:
    """
    Extracts only clean, human-readable printable text from HTML.
    Strips all scripts, styles, navigation, and non-text noise.
    Collapses blank lines and normalises unicode characters.

    Suitable for pages that are entirely printable/readable documents
    (e.g. legislation portals, plain content pages).

    Args:
        html_content (str): The raw HTML string.

    Returns:
        Optional[str]: Cleaned printable text, or None if nothing is found.
    """
    if not html_content:
        return None

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove non-printable tag blocks entirely
    for tag in soup.find_all(_NON_PRINTABLE_TAGS):
        tag.decompose()

    # Try the known content container first, fall back to body
    main_content = None
    for selector in _CONTENT_SELECTORS:
        main_content = soup.find(**selector)
        if main_content:
            break

    if not main_content:
        main_content = soup.find("body")

    if not main_content:
        return None

    raw_text = main_content.get_text(separator="\n", strip=True)

    # Normalise unicode (e.g. Romanian diacritics stay intact, garbage collapsed)
    raw_text = unicodedata.normalize("NFC", raw_text)

    # Keep only lines that contain at least one printable word character
    lines = raw_text.splitlines()
    printable_lines = [
        line.strip()
        for line in lines
        if re.search(r"\w", line)
    ]

    # Collapse consecutive blank lines
    result_lines = []
    prev_blank = False
    for line in printable_lines:
        if line == "":
            if not prev_blank:
                result_lines.append(line)
            prev_blank = True
        else:
            result_lines.append(line)
            prev_blank = False

    return "\n".join(result_lines) or None


def extract_metadata(html_content: str) -> dict:
    """
    Extracts basic metadata (title, description) from the HTML head.

    Args:
        html_content (str): The raw HTML string.

    Returns:
        dict: A mapping of metadata fields found.
    """
    if not html_content:
        return {}

    soup = BeautifulSoup(html_content, "html.parser")
    metadata = {}

    title_tag = soup.find("title")
    if title_tag:
        metadata["title"] = title_tag.get_text(strip=True)

    desc_tag = soup.find("meta", attrs={"name": "description"})
    if desc_tag and desc_tag.get("content"):
        metadata["description"] = desc_tag["content"].strip()

    return metadata
