import pytest
from ingestion.parser.html_parser import parse_html_to_text, extract_metadata

def test_parse_html_to_text_with_id():
    html_content = """
    <html>
        <body>
            <div id="textdocumentleg">
                <p>Articolul 1</p>
                <p>Prevederi generale.</p>
            </div>
            <div>Ignore this</div>
        </body>
    </html>
    """
    text = parse_html_to_text(html_content)
    assert text == "Articolul 1\nPrevederi generale."

def test_parse_html_to_text_fallback_class():
    html_content = """
    <html>
        <body>
            <div class="textdocumentleg">
                <p>Articolul 2</p>
            </div>
        </body>
    </html>
    """
    text = parse_html_to_text(html_content)
    assert text == "Articolul 2"

def test_parse_html_to_text_fallback_body():
    html_content = """
    <html>
        <body>
            <p>Articolul 3</p>
        </body>
    </html>
    """
    text = parse_html_to_text(html_content)
    assert text == "Articolul 3"

def test_parse_html_to_text_empty():
    assert parse_html_to_text("") is None
    assert parse_html_to_text(None) is None

def test_extract_metadata():
    html_content = """
    <html>
        <head>
            <title>ORDIN 745 23/06/2020</title>
        </head>
        <body></body>
    </html>
    """
    metadata = extract_metadata(html_content)
    assert metadata.get("title") == "ORDIN 745 23/06/2020"

def test_extract_metadata_no_title():
    html_content = "<html><head></head><body></body></html>"
    metadata = extract_metadata(html_content)
    assert metadata == {}
