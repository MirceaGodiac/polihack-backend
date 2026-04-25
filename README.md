# Legal Parser Pipeline

A robust pipeline for scraping and parsing Romanian legal documents into atomic, addressable JSON units.

## Features
- **Scraper:** Fetches legal documents from `legislatie.just.ro` or archived versions.
- **HTML Parser:** Cleans and normalizes Romanian legal text.
- **Atomic Parser:** Splits text into Articles, Paragraphs, and Letters with deterministic IDs.
- **Reference Extractor:** Automatically identifies and extracts cross-references within the text.

## Installation
```bash
pip install -r requirements.txt
```

## Running the Pipeline

You can ingest a law using either a live URL or a locally saved HTML file.

### Option 1: Using a URL
Run the following command to scrape and parse a law directly from a link:

```powershell
python scripts/ingest_single.py `
    --url "https://web.archive.org/web/20250405171446mp_/https://legislatie.just.ro/Public/FormaPrintabila/00000G1KT5P4C47BW0232MK2YPXFN5DI" `
    --law-id "ro.archive_lege_test" `
    --out-dir ingestion/output/archive_test_bundle

```

### Option 2: Using a Local HTML File
If you have an HTML file saved locally, use the `--file` flag:

```powershell
python scripts/ingest_single.py `
    --file "path/to/law.html" `
    --law-id "ro.my_custom_law" `
    --out-dir ingestion/output/my_law_bundle
```

## Output Structure
The pipeline generates a single folder containing:
- `legal_units.json`: A flat list of all parsed legal units (Articles, Paragraphs, etc.) with their full hierarchy paths and text.

## Configuration
- Parsing rules (regex) are defined in `ingestion/parser_rules.py`.
- Hierarchy logic is managed by `ingestion/parser/atomic_parser.py`.
