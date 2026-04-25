from ingestion.html_scraper import scrape_html_source

def test_scraper():
    test_url = "https://example.com"
    print(f"Testing scraper with URL: {test_url}")
    html = scrape_html_source(test_url)
    
    if html:
        print(f"Success! Retrieved {len(html)} characters of HTML.")
        print("First 200 characters:")
        print(html[:200])
    else:
        print("Failed to retrieve HTML.")

if __name__ == "__main__":
    test_scraper()
