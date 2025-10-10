"""
Test the updated scraper with precise selectors
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import re
import time

TRACK_ID = "0Prct5TDjAnEgIqbxcldY9"
URL = f"https://www.chosic.com/music-genre-finder/?track={TRACK_ID}"

chrome_options = Options()
# chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(options=chrome_options)
wait = WebDriverWait(driver, 20)

try:
    print(f"Loading: {URL}")
    driver.get(URL)
    time.sleep(2)
    
    # Click the search button
    print("Finding search button...")
    search_btn = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "btn-search")))
    print("Clicking search button...")
    search_btn.click()
    
    # Wait for results
    print("Waiting for results to load...")
    time.sleep(5)
    
    # Save HTML after click
    html = driver.page_source
    with open('page_after_click.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n✓ Saved HTML after click to 'page_after_click.html' ({len(html)} characters)")
    
    # Parse and extract data
    soup = BeautifulSoup(html, 'html.parser')
    
    print("\n=== Extracted Data ===")
    
    # Genres
    all_genres = []
    tag_containers = soup.select('#spotify-tags .pl-tags.tagcloud, .wiki-tags .pl-tags.tagcloud')
    for container in tag_containers:
        genre_links = container.find_all('a')
        for link in genre_links:
            genre_text = link.get_text().strip()
            normalized_genre = genre_text.lower().replace(' ', '-')
            all_genres.append(normalized_genre)
    print(f"Genres found: {all_genres}")
    
    # Explicit
    explicit_span = soup.select_one('span.span-explicit')
    if explicit_span:
        explicit_text = explicit_span.get_text().strip()
        explicit_value = 1 if 'Yes' in explicit_text else 0
        print(f"Explicit: {explicit_value} ({explicit_text})")
    
    # Year
    album_data = soup.select_one('p.album-data')
    if album_data:
        album_text = album_data.get_text()
        year_match = re.search(r'\(.*?(\d{4})\)', album_text)
        if year_match:
            print(f"Year: {year_match.group(1)} (from: {album_text.strip()})")
    
    # Popularity
    progressbars_div = soup.select_one('div.progressbars-div')
    if progressbars_div:
        progressbars_text = progressbars_div.get_text()
        popularity_match = re.search(r'Popularity:\s*(\d+)/100', progressbars_text)
        if popularity_match:
            print(f"Popularity: {popularity_match.group(1)}/100")
    
    print("\nKeeping browser open for 20 seconds...")
    time.sleep(20)
    
finally:
    driver.quit()
    print("Browser closed")
