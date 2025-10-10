"""
Step 1: Fetch a genre page HTML to inspect parent genre location
"""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import sys

def fetch_genre_page(genre_slug):
    """
    Fetch the HTML of a genre page from Chosic
    genre_slug: e.g., 'post-grunge', 'southern-metal', etc.
    """
    url = f"https://www.chosic.com/genre-chart/{genre_slug}/"
    
    chrome_options = Options()
    # chrome_options.add_argument('--headless')  # Disabled to avoid Cloudflare blocking
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        print(f"Fetching: {url}")
        driver.get(url)
        print("Waiting for Cloudflare challenge to complete...")
        time.sleep(10)  # Wait longer for Cloudflare and page to load
        
        html = driver.page_source
        
        # Save to file
        filename = f"genre_page_{genre_slug}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✓ Saved HTML to '{filename}' ({len(html)} characters)")
        print(f"\n→ Now examine '{filename}' to find where the parent genre is!")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_genre_page.py <genre-slug>")
        print("Example: python fetch_genre_page.py post-grunge")
        sys.exit(1)
    
    genre_slug = sys.argv[1]
    fetch_genre_page(genre_slug)
