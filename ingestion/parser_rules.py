import re

# Regex for Romanian legal structures
# Handles variations like "Art. 41", "Art. 41^1", etc.
ARTICLE_RE = re.compile(r'^Art\.\s*(\d+(?:\^\d+)?)', re.IGNORECASE)

# Handles paragraph numbers like "(1)", "(2)"
PARA_RE = re.compile(r'^\((\d+)\)')

# Handles letters like "a)", "b)"
LETTER_RE = re.compile(r'^([a-z])\)')

# Handles Titles like "TITLUL I", "TITLUL II"
TITLE_RE = re.compile(r'^TITLUL\s+([IVXLCDM]+)', re.IGNORECASE)

# Handles Chapters like "CAPITOLUL I"
CHAPTER_RE = re.compile(r'^CAPITOLUL\s+([IVXLCDM]+)', re.IGNORECASE)

# Handles Sections like "SECŢIUNEA 1" or "SECŢIUNEA I"
SECTION_RE = re.compile(r'^SEC[ŢT]IUNEA\s+([0-9IVXLCDM]+)', re.IGNORECASE)

# Mapping for parser to identify which regex to check
RULES = [
    ('titlu', TITLE_RE),
    ('capitol', CHAPTER_RE),
    ('sectiune', SECTION_RE),
    ('articol', ARTICLE_RE),
    ('alineat', PARA_RE),
    ('litera', LETTER_RE),
]
