"""
Unknown Genre Rechecking System
Rechecks all genres mapped to 'Unknown' in genre_mappings.csv
and saves corrected mappings to a new CSV file.

This script:
1. Reads genre_mappings.csv
2. Filters only mappings to 'Unknown'
3. Scrapes Chosic.com to find valid main genre for each
4. Saves only corrected mappings to corrected_genre_mappings.csv

INDEPENDENT SCRIPT - Does NOT use genre_mapper.py functions
Implements its own HTTP scraping logic
"""
import csv
import os
import time
import requests
from bs4 import BeautifulSoup

# Define main genres
MAIN_GENRES = [
    "Classical", "Folk", "Blues", "Jazz", "Country", 
    "R&B", "Rock", "Pop", "Hip-Hop", "Electronic"
]

GENRE_MAP_FILE = "../genre_mappings.csv"
CORRECTED_MAP_FILE = "../corrected_genre_mappings.csv"
COOKIES_FILE = "cookies.txt"

# Global session
_session = None


def get_session():
    """Create and configure a requests session with cookies"""
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
                
                print(f"✓ Loaded cookies: {list(_session.cookies.keys())}")
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


def load_all_genre_mappings():
    """
    Load ALL genre mappings from CSV file into a dictionary
    Returns dict: {subgenre_lower: main_genre}
    """
    mappings = {}
    
    if not os.path.exists(GENRE_MAP_FILE):
        print(f"✗ Genre mappings file not found: {GENRE_MAP_FILE}")
        return mappings
    
    with open(GENRE_MAP_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mappings[row['subgenre'].lower()] = row['main_genre']
    
    return mappings


def get_unknown_genres():
    """
    Load all genres that are currently mapped to 'Unknown'
    Returns list of subgenres (original casing)
    """
    unknown_genres = []
    
    if not os.path.exists(GENRE_MAP_FILE):
        print(f"✗ Genre mappings file not found: {GENRE_MAP_FILE}")
        return unknown_genres
    
    with open(GENRE_MAP_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['main_genre'] == 'Unknown':
                unknown_genres.append(row['subgenre'])
    
    return unknown_genres


def scrape_parent_genre(genre_slug, max_retries=3):
    """
    Scrape the FIRST parent genre from Chosic genre page
    Returns the first parent genre text or None
    """
    url = f"https://www.chosic.com/genre-chart/{genre_slug}/"
    session = get_session()
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"      Attempt {attempt + 1}/{max_retries}")
            
            response = session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                parent_element = soup.find(class_='parent-genre')
                
                if parent_element:
                    # Find all <a> tags and select the last one
                    all_links = parent_element.find_all('a')
                    if all_links:
                        if len(all_links) > 1:
                            last_link = all_links[1]
                        else:
                            last_link = all_links[0]
                        parent_genre = last_link.get_text().strip()
                        return parent_genre
                
                # No parent genre found on page
                return None
            
            elif response.status_code == 403:
                print(f"      ❌ Blocked by Cloudflare (403)")
                print(f"      💡 Add cookies to {COOKIES_FILE}")
                return None
            
            elif response.status_code == 429:
                # Rate limit hit
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    print(f"      ⚠️  Rate limit (429), waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return None
            elif response.status_code == 404:
                print(f"      ❌ Not Found (404)")
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
            print(f"      ✗ Error: {e}")
            return None
    
    return None


def is_main_genre(genre_text):
    """Check if a genre text matches one of the main genres (case-insensitive)"""
    genre_lower = genre_text.lower()
    for main in MAIN_GENRES:
        if main.lower() == genre_lower:
            return main  # Return with proper casing
    return None


def find_main_genre_for_subgenre(subgenre, all_mappings):
    """
    Find main genre for a subgenre by:
    1. Scraping parent genre from Chosic (one level only)
    2. Checking if parent is a main genre
    3. If not, looking up parent in all_mappings (which already point to main genres)
    
    Returns: (main_genre, chain) or (None, chain)
    - main_genre: The found main genre or None
    - chain: List of genres traversed [subgenre, parent, main_genre]
    """
    chain = [subgenre]
    
    # Convert to slug
    genre_slug = subgenre.lower().replace(' ', '-')
    
    # Scrape parent genre (one level only)
    print(f"      🔍 Scraping parent of '{subgenre}'...")
    parent_genre = scrape_parent_genre(genre_slug)
    
    if not parent_genre:
        print(f"      ✗ No parent genre found")
        return None, chain
    
    chain.append(parent_genre)
    print(f"      → Found parent: '{parent_genre}'")
    
    # Check if parent is already a main genre
    main_genre = is_main_genre(parent_genre)
    if main_genre:
        print(f"      ✓ Parent IS a main genre: '{main_genre}'")
        return main_genre, chain
    
    # Check if parent is mapped in our existing mappings
    # (all mappings already point directly to main genres or Unknown)
    parent_lower = parent_genre.lower()
    if parent_lower in all_mappings:
        mapped_genre = all_mappings[parent_lower]
        if mapped_genre != "Unknown":
            print(f"      ✓ Parent mapped in CSV: '{parent_genre}' → '{mapped_genre}'")
            chain.append(mapped_genre)
            return mapped_genre, chain
        else:
            print(f"      ⚠️  Parent also mapped to 'Unknown'")
            return None, chain
    
    # Parent not found in mappings
    print(f"      ⚠️  Parent '{parent_genre}' not found in existing mappings")
    return None, chain


def save_corrected_mapping(subgenre, main_genre):
    """
    Save a corrected genre mapping to the corrected CSV file
    Only saves if main_genre is NOT 'Unknown'
    """
    if main_genre == "Unknown" or not main_genre:
        return False
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(CORRECTED_MAP_FILE)
    
    with open(CORRECTED_MAP_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['subgenre', 'main_genre'])
        
        if not file_exists:
            writer.writeheader()
            print(f"✓ Created new file: {CORRECTED_MAP_FILE}")
        
        writer.writerow({
            'subgenre': subgenre,
            'main_genre': main_genre
        })
    
    return True


def recheck_unknown_genres():
    """
    Main function to recheck all 'Unknown' genre mappings
    """
    print("=" * 60)
    print("UNKNOWN GENRE RECHECKING SYSTEM")
    print("=" * 60)
    
    # Load ALL genre mappings (for lookup during traversal)
    all_mappings = load_all_genre_mappings()
    print(f"✓ Loaded {len(all_mappings)} existing genre mappings")
    
    # Get all genres mapped to 'Unknown'
    unknown_genres = get_unknown_genres()
    
    if not unknown_genres:
        print("\n✓ No genres mapped to 'Unknown' found!")
        return
    
    print(f"\n📊 Found {len(unknown_genres)} genres mapped to 'Unknown'")
    print(f"🔍 Starting recheck process...\n")
    
    # Statistics
    corrected_count = 0
    still_unknown_count = 0
    error_count = 0
    
    # Process each unknown genre
    for i, subgenre in enumerate(unknown_genres, 1):
        print(f"\n[{i}/{len(unknown_genres)}] Processing: '{subgenre}'")
        
        try:
            # Try to find main genre by scraping
            main_genre, chain = find_main_genre_for_subgenre(subgenre, all_mappings)
            
            if main_genre and main_genre != "Unknown":
                # We found a valid mapping!
                if save_corrected_mapping(subgenre, main_genre):
                    corrected_count += 1
                    chain_str = " → ".join(chain)
                    print(f"      ✅ CORRECTED: {chain_str}")
                else:
                    still_unknown_count += 1
                    print(f"      ⚠️  Still Unknown: '{subgenre}'")
            else:
                still_unknown_count += 1
                print(f"      ⚠️  Still Unknown: '{subgenre}'")
            
            # Rate limiting between genres
            time.sleep(0.5)
        
        except Exception as e:
            error_count += 1
            print(f"      ❌ Error processing '{subgenre}': {e}")
            continue
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total genres processed:     {len(unknown_genres)}")
    print(f"✅ Corrected mappings:      {corrected_count}")
    print(f"⚠️  Still Unknown:           {still_unknown_count}")
    print(f"❌ Errors:                   {error_count}")
    
    if corrected_count > 0:
        print(f"\n✓ Corrected mappings saved to: {CORRECTED_MAP_FILE}")
    else:
        print(f"\n⚠️  No new mappings found")
    
    print("=" * 60)


if __name__ == "__main__":
    recheck_unknown_genres()
