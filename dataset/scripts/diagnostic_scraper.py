#!/usr/bin/env python3
"""
Diagnostic scraper to identify why scraping fails after 7 songs.
This runs with visible browser and detailed logging.
"""
import pandas as pd
from bs4 import BeautifulSoup
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from genre_mapper import get_main_genre_for_list

class DiagnosticScraper:
    def __init__(self):
        """Initialize Chrome WebDriver with VISIBLE browser for debugging"""
        chrome_options = Options()
        # REMOVED headless mode to see what's happening
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        
        # More realistic user agent
        chrome_options.add_argument('user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # Disable automation flags
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = webdriver.Chrome(options=chrome_options)
        
        # Execute script to remove webdriver property
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        self.wait = WebDriverWait(self.driver, 15)  # Increased timeout
        self.delay = 3  # Increased delay between requests
        
        print("✓ Browser initialized (VISIBLE mode for debugging)")
    
    def __del__(self):
        """Clean up WebDriver on deletion"""
        if hasattr(self, 'driver'):
            print("\nClosing browser...")
            input("Press Enter to close the browser...")  # Wait for user
            self.driver.quit()
    
    def check_for_cloudflare(self, page_source):
        """Check if Cloudflare challenge is present"""
        cloudflare_indicators = [
            'Checking your browser',
            'Please wait',
            'Cloudflare',
            'cf-challenge',
            'Just a moment'
        ]
        
        for indicator in cloudflare_indicators:
            if indicator in page_source:
                print(f"⚠️  CLOUDFLARE DETECTED: '{indicator}' found in page")
                return True
        return False
    
    def check_for_bot_detection(self, page_source):
        """Check for other bot detection mechanisms"""
        bot_indicators = [
            'Access Denied',
            'Bot detected',
            'Automated access',
            'Rate limit exceeded',
            'Too many requests'
        ]
        
        for indicator in bot_indicators:
            if indicator in page_source:
                print(f"⚠️  BOT DETECTION: '{indicator}' found in page")
                return True
        return False
    
    def scrape_track_metadata(self, track_id, track_number):
        """
        Scrape track metadata with extensive debugging
        """
        url = f"https://www.chosic.com/music-genre-finder/?track={track_id}"
        
        print("\n" + "=" * 70)
        print(f"TRACK #{track_number}: {track_id}")
        print("=" * 70)
        
        try:
            # Load the page
            print(f"1️⃣  Loading URL: {url}")
            self.driver.get(url)
            time.sleep(2)
            
            # Check for Cloudflare or bot detection
            page_source = self.driver.page_source
            
            if self.check_for_cloudflare(page_source):
                print("❌ PROBLEM: Cloudflare challenge detected!")
                print("   Waiting 10 seconds to see if it resolves...")
                time.sleep(10)
                page_source = self.driver.page_source
                
                if self.check_for_cloudflare(page_source):
                    print("❌ Cloudflare still present. Skipping this track.")
                    return None
            
            if self.check_for_bot_detection(page_source):
                print("❌ PROBLEM: Bot detection triggered!")
                return None
            
            # Try to find and click the search button
            print("2️⃣  Looking for search button...")
            try:
                search_button = self.wait.until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "btn-search"))
                )
                print("   ✓ Search button found!")
                
                # Scroll to button
                self.driver.execute_script("arguments[0].scrollIntoView(true);", search_button)
                time.sleep(1)
                
                # Click button
                search_button.click()
                print("   ✓ Search button clicked!")
                
            except TimeoutException:
                print("   ❌ Search button not found! Checking page content...")
                print(f"   Page title: {self.driver.title}")
                print(f"   Current URL: {self.driver.current_url}")
                
                # Save page source for analysis
                with open(f'../failed_page_{track_number}.html', 'w', encoding='utf-8') as f:
                    f.write(page_source)
                print(f"   Saved page source to: failed_page_{track_number}.html")
                
                return None
            
            # Wait for results to load
            print("3️⃣  Waiting for results to load...")
            time.sleep(5)
            
            # Get the page source after JavaScript has executed
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            print("4️⃣  Parsing metadata...")
            
            metadata = {
                'year': None,
                'genre': None,
                'explicit': None,
                'popularity': None
            }
            
            # Extract genres
            all_genres = []
            tag_containers = soup.select('#spotify-tags .pl-tags.tagcloud, .wiki-tags .pl-tags.tagcloud')
            print(f"   Found {len(tag_containers)} tag containers")
            
            for container in tag_containers:
                genre_links = container.find_all('a')
                for link in genre_links:
                    genre_text = link.get_text().strip()
                    normalized_genre = genre_text.lower().replace(' ', '-')
                    all_genres.append(normalized_genre)
            
            if all_genres:
                main_genre = get_main_genre_for_list(all_genres)
                metadata['genre'] = main_genre
                print(f"   ✓ Genres: {all_genres[:5]}... → Main: {main_genre}")
            else:
                print(f"   ❌ No genres found!")
            
            # Extract explicit flag
            explicit_span = soup.select_one('span.span-explicit')
            if explicit_span:
                explicit_text = explicit_span.get_text().strip()
                if 'Yes' in explicit_text:
                    metadata['explicit'] = 1
                elif 'No' in explicit_text:
                    metadata['explicit'] = 0
                print(f"   ✓ Explicit: {metadata['explicit']}")
            else:
                print(f"   ❌ Explicit flag not found!")
            
            # Extract year
            album_data = soup.select_one('p.album-data')
            if album_data:
                album_text = album_data.get_text()
                year_match = re.search(r'\(.*?(\d{4})\)', album_text)
                if year_match:
                    metadata['year'] = int(year_match.group(1))
                    print(f"   ✓ Year: {metadata['year']}")
                else:
                    print(f"   ❌ Year not found in: {album_text}")
            else:
                print(f"   ❌ Album data not found!")
            
            # Extract popularity
            progressbars_div = soup.select_one('div.progressbars-div')
            if progressbars_div:
                progressbars_text = progressbars_div.get_text()
                popularity_match = re.search(r'Popularity:\s*(\d+)/100', progressbars_text)
                if popularity_match:
                    metadata['popularity'] = int(popularity_match.group(1))
                    print(f"   ✓ Popularity: {metadata['popularity']}")
                else:
                    print(f"   ❌ Popularity not found in: {progressbars_text}")
            else:
                print(f"   ❌ Progressbars div not found!")
            
            # Check if we got any data
            has_data = any(v is not None for v in metadata.values())
            
            if not has_data:
                print("\n⚠️  WARNING: No metadata extracted! Saving page for analysis...")
                with open(f'../failed_parse_{track_number}.html', 'w', encoding='utf-8') as f:
                    f.write(page_source)
                print(f"   Saved to: failed_parse_{track_number}.html")
                
                # Print some of the page to see what we got
                print("\n   Page excerpt (first 500 chars):")
                print("   " + page_source[:500].replace('\n', '\n   '))
            
            print("\n5️⃣  Results:")
            print(f"   Genre: {metadata['genre']}")
            print(f"   Year: {metadata['year']}")
            print(f"   Explicit: {metadata['explicit']}")
            print(f"   Popularity: {metadata['popularity']}")
            
            return metadata
            
        except Exception as e:
            print(f"\n❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        finally:
            # Rate limiting
            print(f"\n⏱️  Waiting {self.delay} seconds before next request...")
            time.sleep(self.delay)


def test_diagnostic():
    """Test the diagnostic scraper on problematic tracks"""
    
    # Load the tracks that failed (around position 8+)
    print("Loading dataset...")
    df = pd.read_csv('../songs_with_attributes_and_lyrics.csv')
    
    print(f"Total tracks: {len(df)}")
    
    # Test tracks 1-15 (including the problematic ones)
    test_tracks = df.iloc[7:15]  # Tracks 8-15 (0-indexed: 7-14)
    
    print(f"\nTesting {len(test_tracks)} tracks (positions 8-15)...")
    print("\nPress Enter to start testing...")
    input()
    
    scraper = DiagnosticScraper()
    
    results = []
    
    for idx, row in enumerate(test_tracks.iterrows(), start=8):
        index, data = row
        track_id = data['id']
        
        metadata = scraper.scrape_track_metadata(track_id, idx)
        
        result = {
            'track_number': idx,
            'track_id': track_id,
            'success': metadata is not None and any(v is not None for v in metadata.values()) if metadata else False,
            'metadata': metadata
        }
        results.append(result)
        
        print(f"\n{'✓' if result['success'] else '❌'} Track #{idx}: {'SUCCESS' if result['success'] else 'FAILED'}")
        
        # Ask user if they want to continue
        print("\nOptions:")
        print("1. Continue to next track")
        print("2. Retry this track")
        print("3. Stop testing")
        
        choice = input("Choice (default: 1): ").strip() or '1'
        
        if choice == '2':
            print("\nRetrying...")
            time.sleep(5)
            metadata = scraper.scrape_track_metadata(track_id, idx)
            result['metadata'] = metadata
            result['success'] = metadata is not None and any(v is not None for v in metadata.values()) if metadata else False
        elif choice == '3':
            print("\nStopping test.")
            break
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"✓ Successful: {successful}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")
    
    if failed > 0:
        print("\nFailed tracks:")
        for r in results:
            if not r['success']:
                print(f"  - Track #{r['track_number']}: {r['track_id']}")
    
    print("\n" + "=" * 70)
    print("Check the saved HTML files for failed tracks:")
    print("  - failed_page_X.html (page before clicking search)")
    print("  - failed_parse_X.html (page after clicking search)")
    print("=" * 70)


if __name__ == "__main__":
    test_diagnostic()
