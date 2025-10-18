"""
Genre Mapping System (HTTP Version)
Maps sub-genres to main genres using a CSV file and HTTP requests when needed
NO SELENIUM - Uses requests library for web scraping

==================== HOW TO GET COOKIES ====================
If you get 403 errors, you need to provide valid cookies from your browser:

1. Open Firefox/Chrome and go to: https://www.chosic.com/
2. Open Developer Tools (F12) > Network tab
3. Refresh the page
4. Click on any request to chosic.com
5. Copy the entire "Cookie" header value
6. Save it to 'cookies.txt' in the same directory

Your cookies should include: pll_language, cf_clearance, r_c1062550
============================================================
"""
import csv
import os
from collections import Counter
import requests
from bs4 import BeautifulSoup
import time

# Define main genres
MAIN_GENRES = [
    "Classical", "Folk", "Blues", "Jazz", "Country", 
    "R&B", "Rock", "Pop", "Hip-Hop", "Electronic"
]

GENRE_MAP_FILE = "../genre_mappings.csv"
COOKIES_FILE = "cookies.txt"

# Global session with cookies (loaded once)
_session = None


def get_session():
    """Get or create a requests session with cookies loaded"""
    global _session
    
    if _session is None:
        _session = requests.Session()
        
        # Load cookies if available
        if os.path.exists(COOKIES_FILE):
            try:
                with open(COOKIES_FILE, 'r') as f:
                    cookies_string = f.read().strip()
                
                # Parse and add cookies
                for cookie in cookies_string.split(';'):
                    cookie = cookie.strip()
                    if '=' in cookie:
                        key, value = cookie.split('=', 1)
                        _session.cookies.set(key.strip(), value.strip(), domain='.chosic.com')
                
                print(f"✓ Loaded cookies from {COOKIES_FILE}: {list(_session.cookies.keys())}")
            except Exception as e:
                print(f"⚠ Could not load cookies: {e}")
        
        # Set headers
        _session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:143.0) Gecko/20100101 Firefox/143.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Sec-GPC': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    return _session


def load_genre_mappings():
    """Load genre mappings from CSV file"""
    mappings = {}
    if os.path.exists(GENRE_MAP_FILE):
        with open(GENRE_MAP_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mappings[row['subgenre'].lower()] = row['main_genre']
    return mappings


def save_genre_mapping(subgenre, main_genre):
    """
    Save a new genre mapping to CSV file
    Saves mappings for valid MAIN_GENRES or "Unknown"
    """
    # Check if file exists before opening
    file_exists = os.path.exists(GENRE_MAP_FILE)
    
    # Validate that main_genre is actually a main genre (or Unknown)
    if main_genre != "Unknown":
        is_main_genre = False
        for main in MAIN_GENRES:
            if main.lower() == main_genre.lower():
                is_main_genre = True
                main_genre = main  # Use the proper casing from MAIN_GENRES
                break
        
        if not is_main_genre:
            print(f"✗ Cannot save mapping: '{main_genre}' is not a valid main genre")
            print(f"   Valid main genres: {', '.join(MAIN_GENRES)}")
            main_genre = "Unknown"
    
    # Save the mapping to CSV
    with open(GENRE_MAP_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['subgenre', 'main_genre'])
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'subgenre': subgenre,
            'main_genre': main_genre
        })
    
    print(f"✓ Saved mapping: '{subgenre}' → '{main_genre}'")
    return main_genre != "Unknown"


def scrape_parent_genre(genre_slug, max_retries=3):
    """
    Scrape the FIRST parent genre from Chosic genre page using HTTP requests
    Returns the first parent genre text from the first <a> tag
    
    Args:
        genre_slug: Genre slug (e.g., 'metalcore')
        max_retries: Maximum number of retry attempts
    """
    url = f"https://www.chosic.com/genre-chart/{genre_slug}/"
    session = get_session()  # Use session with cookies
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"      Attempt {attempt + 1}/{max_retries}")
            
            response = session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Find parent genre element
                parent_element = soup.find(class_='parent-genre')
                
                if parent_element:
                    # Find first <a> tag
                    first_link = parent_element.find('a')
                    if first_link:
                        parent_genre = first_link.get_text().strip()
                        return parent_genre
                print(soup.prettify()[:500])  # Print first 500 chars for debugging
                return None
            
            elif response.status_code == 403:
                # Cloudflare block
                print(f"      ❌ Blocked by Cloudflare (403)")
                print(f"      💡 Solution: Add valid cookies to {COOKIES_FILE}")
                print(f"         See instructions at top of this file")
                return None
            
            elif response.status_code == 429:
                # Rate limit hit
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)  # Exponential backoff: 2s, 4s, 8s
                    print(f"      ⚠️  Rate limit (429), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"      ⚠️  Rate limit (429), max retries reached")
                    return None
            else:
                # Other HTTP error
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"      ⚠️  HTTP {response.status_code}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return None
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"      ⚠️  Timeout, retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            return None
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"      ⚠️  Error: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            print(f"      ✗ Error scraping parent genre: {e}")
            return None
    
    # All retries failed
    return None


def map_subgenre_to_main(subgenre, genre_mappings):
    """
    Map a subgenre to a main genre
    Returns the main genre if found, None otherwise
    """
    subgenre_lower = subgenre.lower()
    
    # Check if already in mappings
    if subgenre_lower in genre_mappings:
        return genre_mappings[subgenre_lower]
    
    # If subgenre is already a main genre
    for main in MAIN_GENRES:
        if main.lower() == subgenre_lower:
            return main
    
    return None


def get_main_genre(subgenre, genre_mappings=None):
    """
    Get the main genre for a subgenre
    If not found, scrape from Chosic and save to CSV
    Returns tuple: (main_genre, scraped_flag)
    - main_genre: The mapped genre or "Unknown"
    - scraped_flag: True if we made an HTTP request, False if cached
    """
    if genre_mappings is None:
        genre_mappings = load_genre_mappings()
    
    # Try to find in existing mappings
    main_genre = map_subgenre_to_main(subgenre, genre_mappings)
    
    if main_genre:
        print(f"      ✓ Found in mappings: '{subgenre}' → '{main_genre}'")
        return main_genre, False  # Found in cache, no scraping
    
    # Not found, need to scrape
    print(f"      🔍 '{subgenre}' not in mappings. Scraping parent genre...")
    
    # Convert subgenre to slug (lowercase, spaces to hyphens)
    genre_slug = subgenre.lower().replace(' ', '-')
    
    parent_genre = scrape_parent_genre(genre_slug)
    
    if parent_genre:
        # Try to map parent genre to main genre (this will look up the chain in CSV)
        main_genre = map_subgenre_to_main(parent_genre, genre_mappings)
        
        if main_genre:
            # Save the mapping (will only save if main_genre is valid)
            print(f"      ✓ Mapped '{subgenre}' → '{main_genre}' (via {parent_genre})")
            save_genre_mapping(subgenre, main_genre)
            return main_genre, True  # Scraped
        else:
            # Parent genre is not mapped yet, return "Unknown"
            print(f"      ⚠ Could not map '{subgenre}', using 'Unknown'")
            save_genre_mapping(subgenre, "Unknown")
            return "Unknown", True  # Scraped
    else:
        print(f"      ⚠ Could not scrape parent genre for '{subgenre}'")
        save_genre_mapping(subgenre, "Unknown")
        return "Unknown", True  # Attempted to scrape

def get_main_genre_for_list(subgenres):
    """
    Get the most common main genre from a list of subgenres.
    It processes all subgenres, counts the resulting main genres,
    and returns the one that appears most frequently.
    
    Includes rate limiting delays ONLY when making HTTP requests (not for cached results).
    """
    main_genres_found = []
    genre_mappings = load_genre_mappings()  # Load cache once

    for subgenre in subgenres:
        # Pass mappings to get_main_genre to avoid re-reading the file
        main_genre, was_scraped = get_main_genre(subgenre, genre_mappings=genre_mappings)
        genre_mappings[subgenre] = main_genre
        if main_genre and main_genre != "Unknown":
            main_genres_found.append(main_genre)
        
        # Add delay ONLY if we actually made an HTTP request
        if was_scraped:
            time.sleep(0.3)  # 0.3 second delay after scraping to avoid rate limiting

    if not main_genres_found:
        return "Unknown"

    # Count the occurrences of each main genre and return the most common
    genre_counts = Counter(main_genres_found)
    most_common_genre = genre_counts.most_common(1)[0][0]
    
    return most_common_genre


def get_main_genre_for_list_static(subgenres):
    """
    STATIC VERSION: Get the most common main genre from a list of subgenres.
    Uses ONLY pre-existing mappings - NO dynamic scraping.
    This prevents browser crashes during scraping.
    """
    main_genres_found = []
    genre_mappings = load_genre_mappings()  # Load cache once

    for subgenre in subgenres:
        subgenre_lower = subgenre.lower()
        
        # Check if in mappings
        if subgenre_lower in genre_mappings:
            main_genre = genre_mappings[subgenre_lower]
            if main_genre and main_genre != "Unknown":
                main_genres_found.append(main_genre)
        else:
            # Check if subgenre is already a main genre
            for main in MAIN_GENRES:
                if main.lower() == subgenre_lower:
                    main_genres_found.append(main)
                    break

    if not main_genres_found:
        return "Unknown"

    # Count the occurrences of each main genre and return the most common
    genre_counts = Counter(main_genres_found)
    most_common_genre = genre_counts.most_common(1)[0][0]
    
    return most_common_genre

# Test the system
if __name__ == "__main__":
    # Example usage
    test_genres = ['vietnamese-hip-hop', 'vietnam-indie', 'viet chill rap', 'vietnamese-trap', 'v-pop', 'vietnamese-melodic-rap']
    
    print("Testing genre mapper...")
    print("=" * 50)
    
    print(get_main_genre_for_list(test_genres))
