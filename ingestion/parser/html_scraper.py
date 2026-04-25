import requests
from typing import Optional

def scrape_html_source(url: str) -> Optional[str]:
    """
    Fetches the HTML source code from the given URL.
    
    Args:
        url (str): The link to scrape.
        
    Returns:
        Optional[str]: The HTML source code if successful, None otherwise.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None
