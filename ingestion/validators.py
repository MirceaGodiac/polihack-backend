def validate_corpus(units: list):
    """
    Checks for duplicate IDs in the generated legal units.
    Throws a ValueError if duplicates are found.
    """
    seen_ids = set()
    for unit in units:
        unit_id = unit.get('id')
        if not unit_id:
            raise ValueError("Unit missing ID field")
        
        if unit_id in seen_ids:
            raise ValueError(f"Blocking Error: Duplicate ID found: {unit_id}")
        
        seen_ids.add(unit_id)
    
    return True
