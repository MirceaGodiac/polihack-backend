import pytest
from ingestion.parser.html_parser import parse_html_to_text, parse_printable_text, extract_metadata


# ── parse_html_to_text ────────────────────────────────────────────────────────

def test_parse_html_to_text_with_id():
    html = """
    <html><body>
        <div id="textdocumentleg"><p>Articolul 1</p><p>Prevederi generale.</p></div>
        <div>Ignore this</div>
    </body></html>
    """
    text = parse_html_to_text(html)
    assert "Articolul 1" in text
    assert "Prevederi generale." in text
    assert "Ignore this" not in text


def test_parse_html_to_text_fallback_class():
    html = """
    <html><body>
        <div class="textdocumentleg"><p>Articolul 2</p></div>
    </body></html>
    """
    text = parse_html_to_text(html)
    assert text == "Articolul 2"


def test_parse_html_to_text_fallback_body():
    html = "<html><body><p>Articolul 3</p></body></html>"
    text = parse_html_to_text(html)
    assert text == "Articolul 3"


def test_parse_html_to_text_empty():
    assert parse_html_to_text("") is None
    assert parse_html_to_text(None) is None


# ── parse_printable_text ──────────────────────────────────────────────────────

def test_parse_printable_text_strips_scripts():
    html = """
    <html><body>
        <script>alert('XSS')</script>
        <style>body { color: red; }</style>
        <p>Articolul 4 - Definitii</p>
    </body></html>
    """
    text = parse_printable_text(html)
    assert "alert" not in text
    assert "color: red" not in text
    assert "Articolul 4" in text


def test_parse_printable_text_collapses_blank_lines():
    html = """
    <html><body>
        <p>Line one</p>
        <p></p>
        <p></p>
        <p>Line two</p>
    </body></html>
    """
    text = parse_printable_text(html)
    # Should not have more than one consecutive blank line
    assert "\n\n\n" not in text


def test_parse_printable_text_prefers_content_container():
    html = """
    <html><body>
        <nav>Nav noise here</nav>
        <div id="textdocumentleg"><p>Legal content</p></div>
    </body></html>
    """
    text = parse_printable_text(html)
    assert "Legal content" in text
    assert "Nav noise here" not in text


def test_parse_printable_text_empty():
    assert parse_printable_text("") is None
    assert parse_printable_text(None) is None


def test_parse_printable_text_only_whitespace():
    html = "<html><body>   \n   \t   </body></html>"
    result = parse_printable_text(html)
    assert result is None


# ── extract_metadata ──────────────────────────────────────────────────────────

def test_extract_metadata_title_and_description():
    html = """
    <html>
        <head>
            <title>ORDIN 745 23/06/2020</title>
            <meta name="description" content="Ordin privind aprobarea normelor.">
        </head>
        <body></body>
    </html>
    """
    metadata = extract_metadata(html)
    assert metadata.get("title") == "ORDIN 745 23/06/2020"
    assert metadata.get("description") == "Ordin privind aprobarea normelor."


def test_extract_metadata_no_title():
    html = "<html><head></head><body></body></html>"
    assert extract_metadata(html) == {}


def test_extract_metadata_empty():
    assert extract_metadata("") == {}
    assert extract_metadata(None) == {}
