"""
Genre Mapping System
Maps sub-genres to main genres using a CSV file and web scraping when needed
"""
import csv
import os
from collections import Counter
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time

# Define main genres
MAIN_GENRES = [
    "Classical", "Folk", "Blues", "Jazz", "Country", 
    "R&B", "Rock", "Pop", "Hip-Hop", "Electronic"
]

GENRE_MAP_FILE = "genre_mappings.csv"


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
    Only saves if main_genre is one of the valid MAIN_GENRES or "Unknown"
    """
    # Allow "Unknown" as a special case
    if main_genre == "Unknown":
        print(f"⚠ Mapping '{subgenre}' to 'Unknown' (not saved to CSV)")
        with open(GENRE_MAP_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['subgenre', 'main_genre'])
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'subgenre': subgenre,
                'main_genre': main_genre
            })
        
        print(f"✓ Saved mapping: '{subgenre}' → '{main_genre}'")
        return False
    
    # Validate that main_genre is actually a main genre
    is_main_genre = False
    for main in MAIN_GENRES:
        if main.lower() == main_genre.lower():
            is_main_genre = True
            main_genre = main  # Use the proper casing from MAIN_GENRES
            break
    
    if not is_main_genre:
        print(f"✗ Cannot save mapping: '{main_genre}' is not a valid main genre")
        print(f"   Valid main genres: {', '.join(MAIN_GENRES)}")
        with open(GENRE_MAP_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['subgenre', 'main_genre'])
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'subgenre': subgenre,
                'main_genre': "Unknown"
            })

        print(f"✓ Saved mapping: '{subgenre}' → 'Unknown'")
        return False
    
    file_exists = os.path.exists(GENRE_MAP_FILE)
    
    with open(GENRE_MAP_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['subgenre', 'main_genre'])
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'subgenre': subgenre,
            'main_genre': main_genre
        })
    
    print(f"✓ Saved mapping: '{subgenre}' → '{main_genre}'")
    return True


def scrape_parent_genre(genre_slug):
    """
    Scrape the FIRST parent genre from Chosic genre page
    Returns the first parent genre text from the first <a> tag
    """
    url = f"https://www.chosic.com/genre-chart/{genre_slug}/"
    
    chrome_options = Options()
    # chrome_options.add_argument('--headless')  # Disabled to avoid Cloudflare
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        print(f"Fetching: {url}")
        driver.get(url)
        print("Waiting for parent genre element to load...")
        try:
            # User-requested additional wait for rate limiting
            print("Waiting 8 seconds...")
            time.sleep(8)

            # Wait up to 15 seconds for the element with class 'parent-genre' to be present
            print("Looking for element with class 'parent-genre'...")
            parent_element = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CLASS_NAME, "parent-genre"))
            )
            

            # Now find the <a> tag within the located element
            first_link = parent_element.find_elements(By.TAG_NAME, "a")
            if first_link:
                parent_genre = first_link[0].text.strip()
                print(f"✓ Found first parent genre: '{parent_genre}'")
                return parent_genre
            else:
                print("✗ No <a> tag found in parent genre element")
                return None

        except TimeoutException:
            print("✗ Timed out waiting for parent genre element to appear.")
            return None
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return None
    finally:
        driver.quit()


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
    Returns "Unknown" if parent genre is not in main genres list
    """
    if genre_mappings is None:
        genre_mappings = load_genre_mappings()
    
    # Try to find in existing mappings
    main_genre = map_subgenre_to_main(subgenre, genre_mappings)
    
    if main_genre:
        print(f"✓ Found in mappings: '{subgenre}' → '{main_genre}'")
        return main_genre
    
    # Not found, need to scrape
    print(f"⚠ '{subgenre}' not in mappings. Scraping parent genre...")
    
    # Convert subgenre to slug (lowercase, spaces to hyphens)
    genre_slug = subgenre.lower().replace(' ', '-')
    
    parent_genre = scrape_parent_genre(genre_slug)
    
    if parent_genre:
        # Try to map parent genre to main genre (this will look up the chain in CSV)
        main_genre = map_subgenre_to_main(parent_genre, genre_mappings)
        
        if main_genre:
            # Save the mapping (will only save if main_genre is valid)
            save_genre_mapping(subgenre, main_genre)
            return main_genre
        else:
            # Parent genre is not mapped yet, return "Unknown"
            print(f"⚠ Parent genre '{parent_genre}' is not in main genres list.")
            print(f"Returning 'Unknown' for '{subgenre}'.")
            save_genre_mapping(subgenre, "Unknown")
            return "Unknown"
    else:
        print(f"⚠ Could not scrape parent genre for '{subgenre}'.")
        print("Returning 'Unknown'.")
        save_genre_mapping(subgenre, "Unknown")
        return "Unknown"

def get_main_genre_for_list(subgenres):
    """
    Get the most common main genre from a list of subgenres.
    It processes all subgenres, counts the resulting main genres,
    and returns the one that appears most frequently.
    """
    main_genres_found = []
    genre_mappings = load_genre_mappings()  # Load cache once

    for subgenre in subgenres:
        # Pass mappings to get_main_genre to avoid re-reading the file
        main_genre = get_main_genre(subgenre, genre_mappings=genre_mappings)
        genre_mappings[subgenre] = main_genre
        if main_genre and main_genre != "Unknown":
            main_genres_found.append(main_genre)

    if not main_genres_found:
        return "Unknown"

    # Count the occurrences of each main genre and return the most common
    genre_counts = Counter(main_genres_found)
    most_common_genre = genre_counts.most_common(1)[0][0]
    
    return most_common_genre

# Test the system
if __name__ == "__main__":
    # Example usage
    test_genres = ['southern-metal', 'groove-metal', 'alternative-metal', 'groove-metal', 'alternative-metal']
    
    print("Testing genre mapper...")
    print("=" * 50)
    
    print(get_main_genre_for_list(test_genres))
