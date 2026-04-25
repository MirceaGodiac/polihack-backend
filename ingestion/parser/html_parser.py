from bs4 import BeautifulSoup
from typing import Optional

def parse_html_to_text(html_content: str) -> Optional[str]:
    """
    Parses HTML content and extracts the main legal text.
    Targeting the Romanian Ministry of Justice portal structure.
    
    Args:
        html_content (str): The raw HTML string.
        
    Returns:
        Optional[str]: The extracted plain text, or None if extraction fails.
    """
    if not html_content:
        return None
        
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Target the main document container typically found on legislatie.just.ro
    main_content = soup.find(id='textdocumentleg')
    
    if not main_content:
        # Fallback to a generic class if ID is missing
        main_content = soup.find(class_='textdocumentleg')
        
    if not main_content:
        # Fallback to the body if specific containers are missing
        main_content = soup.find('body')
        
    if main_content:
        # Extract text, joining with newlines to preserve some structure
        text = main_content.get_text(separator='\n', strip=True)
        return text
        
    return None

def extract_metadata(html_content: str) -> dict:
    """
    Extracts basic metadata like title from the HTML.
    """
    if not html_content:
        return {}
        
    soup = BeautifulSoup(html_content, 'html.parser')
    metadata = {}
    
    title_tag = soup.find('title')
    if title_tag:
        metadata['title'] = title_tag.get_text(strip=True)
        
    return metadata
