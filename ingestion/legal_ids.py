import re

def normalize_number(text: str) -> str:
    """
    Normalizes Romanian legal numbering.
    Example: '41^1' -> '41_1'
    """
    if not text:
        return ""
    # Replace ^ with _
    normalized = text.replace('^', '_')
    # Remove any dots or non-alphanumeric except underscore
    normalized = re.sub(r'[^a-zA-Z0-9_]', '', normalized)
    return normalized.lower()

def make_unit_id(corpus_id: str, hierarchy_path: list) -> str:
    """
    Generates a deterministic ID for a legal unit.
    Example: ro.codul_muncii.art_41.alin_1
    """
    parts = [corpus_id]
    for level_type, level_val in hierarchy_path:
        norm_val = normalize_number(level_val)
        # Map Romanian types to standard prefixes
        prefix_map = {
            'titlu': 'titlu',
            'capitol': 'capitol',
            'sectiune': 'sectiune',
            'articol': 'art',
            'alineat': 'alin',
            'litera': 'lit'
        }
        prefix = prefix_map.get(level_type.lower(), level_type.lower())
        parts.append(f"{prefix}_{norm_val}")
    
    return ".".join(parts)
