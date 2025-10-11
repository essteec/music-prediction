#!/usr/bin/env python3
"""
ROBUST Chosic Scraper with Browser Recovery
Fixes:
- Browser crash/freeze after N requests
- NaN validation before saving
- Auto browser restart on failure
- Retry logic with exponential backoff
"""
import pandas as pd
from bs4 import BeautifulSoup
import time
import re
import os
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

# Import genre mapper utilities
from genre_mapper import load_genre_mappings, save_genre_mapping, MAIN_GENRES
from collections import Counter

class RobustChosicScraper:
    def __init__(self, headless=True, restart_every_n=50):
        """
        Initialize scraper with browser recovery
        
        Args:
            headless: Run in headless mode
            restart_every_n: Restart browser every N successful scrapes (prevents memory leaks)
        """
        self.headless = headless
        self.restart_every_n = restart_every_n
        self.scrapes_since_restart = 0
        self.driver = None
        self.wait = None
        
        self._init_browser()
    
    def _init_browser(self):
        """Initialize or reinitialize the Chrome browser"""
        # Close existing browser if any
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
        
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument('--headless=new')
        
        # Anti-detection options
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # Disable automation flags
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Set page load timeout to prevent freezing
        chrome_options.add_argument('--page-load-strategy=eager')  # Don't wait for all resources
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.set_page_load_timeout(30)  # 30 second max page load
        self.driver.set_script_timeout(30)  # 30 second max script execution
        
        # Remove webdriver property
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        self.wait = WebDriverWait(self.driver, 15)
        self.scrapes_since_restart = 0
        
        print(f"✓ Browser {'re' if self.scrapes_since_restart > 0 else ''}initialized")
    
    def __del__(self):
        """Clean up browser on deletion"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
    
    def _check_and_restart_browser(self):
        """Check if browser needs restart and restart if necessary"""
        if self.scrapes_since_restart >= self.restart_every_n:
            print(f"\n♻️  Restarting browser after {self.scrapes_since_restart} scrapes (memory management)...")
            self._init_browser()
    
    def _validate_metadata(self, metadata):
        """
        Validate that all metadata fields are present and valid (not None)
        Returns: (is_valid, missing_fields)
        """
        if not metadata:
            return False, "metadata is None"
        
        # Check if ANY field is None
        missing_fields = [k for k, v in metadata.items() if v is None]
        if missing_fields:
            return False, f"missing or invalid fields: {', '.join(missing_fields)}"
        
        return True, None
    
    def _scrape_parent_genre(self, genre_slug, timeout=10):
        """
        Scrape parent genre from Chosic USING THE SAME BROWSER
        This prevents creating new browser instances that crash
        
        Args:
            genre_slug: Genre slug (e.g., 'metalcore')
            timeout: Timeout in seconds
            
        Returns:
            Parent genre string or None
        """
        url = f"https://www.chosic.com/genre-chart/{genre_slug}/"
        
        try:
            # Use the existing browser
            self.driver.get(url)
            time.sleep(1)
            
            # Wait for parent genre element
            parent_element = self.wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "parent-genre"))
            )
            
            time.sleep(1)  # Small delay
            
            # Find <a> tag within parent element
            first_link = parent_element.find_elements(By.TAG_NAME, "a")
            if first_link:
                parent_genre = first_link[0].text.strip()
                return parent_genre
            
            return None
            
        except TimeoutException:
            print(f"      ⚠️  Timeout getting parent genre for '{genre_slug}'")
            return None
        except Exception as e:
            print(f"      ⚠️  Error getting parent genre: {e}")
            return None
    
    def _map_genre_to_main(self, subgenre, genre_mappings):
        """
        Map a subgenre to main genre with dynamic scraping
        SAFELY reuses the same browser instance
        
        Args:
            subgenre: The subgenre to map
            genre_mappings: Current genre mappings dict
            
        Returns:
            Main genre string
        """
        subgenre_lower = subgenre.lower()
        
        # Check if already in mappings
        if subgenre_lower in genre_mappings:
            return genre_mappings[subgenre_lower]
        
        # Check if subgenre is already a main genre
        for main in MAIN_GENRES:
            if main.lower() == subgenre_lower:
                return main
        
        # Need to scrape parent genre
        print(f"      🔍 '{subgenre}' not mapped, scraping parent genre...")
        
        parent_genre = self._scrape_parent_genre(subgenre_lower)
        
        if parent_genre:
            # Map parent genre to main genre
            parent_lower = parent_genre.lower()
            
            # Check if parent is in our main genres
            for main in MAIN_GENRES:
                if main.lower() == parent_lower or parent_lower in main.lower():
                    print(f"      ✓ Mapped '{subgenre}' → '{main}' (via {parent_genre})")
                    save_genre_mapping(subgenre_lower, main)
                    genre_mappings[subgenre_lower] = main
                    return main
            
            # Try common mappings
            parent_to_main = {
                'classical': 'Classical',
                'folk': 'Folk',
                'blues': 'Blues',
                'jazz': 'Jazz',
                'country': 'Country',
                'r&b': 'R&B',
                'rock': 'Rock',
                'pop': 'Pop',
                'hip hop': 'Hip-Hop',
                'electronic': 'Electronic',
                'folk/acoustic': 'Folk',
                'metal': 'Rock',
                'traditional-music': 'Folk',
                'reggae': 'Folk',
                'easy-listening': 'Pop'
            }
            
            for key, value in parent_to_main.items():
                if key in parent_lower:
                    print(f"      ✓ Mapped '{subgenre}' → '{value}' (via {parent_genre})")
                    save_genre_mapping(subgenre_lower, value)
                    genre_mappings[subgenre_lower] = value
                    return value
        
        # Failed to map, use Unknown
        print(f"      ⚠️  Could not map '{subgenre}', using 'Unknown'")
        save_genre_mapping(subgenre_lower, 'Unknown')
        genre_mappings[subgenre_lower] = 'Unknown'
        return 'Unknown'
    
    def _get_main_genre_for_list(self, subgenres):
        """
        Get the most common main genre from a list of subgenres
        Uses dynamic scraping but SAFELY with the same browser
        
        Args:
            subgenres: List of subgenre strings
            
        Returns:
            Most common main genre string
        """
        main_genres_found = []
        genre_mappings = load_genre_mappings()
        
        for subgenre in subgenres:
            main_genre = self._map_genre_to_main(subgenre, genre_mappings)
            if main_genre and main_genre != "Unknown":
                main_genres_found.append(main_genre)
        
        if not main_genres_found:
            return "Unknown"
        
        # Return most common genre
        genre_counts = Counter(main_genres_found)
        most_common_genre = genre_counts.most_common(1)[0][0]
        
        return most_common_genre
    
    def scrape_track_metadata(self, track_id, max_retries=3):
        """
        Scrape track metadata with retry logic and browser recovery
        
        Args:
            track_id: Spotify track ID
            max_retries: Maximum number of retry attempts
            
        Returns:
            dict with year, genre, explicit, popularity (or None if failed after retries)
        """
        url = f"https://www.chosic.com/music-genre-finder/?track={track_id}"
        
        for attempt in range(max_retries):
            try:
                # Check if browser needs restart
                # self._check_and_restart_browser()
                
                print(f"  Attempt {attempt + 1}/{max_retries} for {track_id}")
                
                # Load the page with timeout protection
                try:
                    self.driver.get(url)
                except TimeoutException:
                    print(f"  ⚠️  Page load timeout, retrying...")
                    continue
                except WebDriverException as e:
                    if "invalid session id" in str(e) or "connection" in str(e).lower():
                        print(f"  ⚠️  Browser crashed! Restarting...")
                        self._init_browser()
                        continue
                    raise
                
                time.sleep(random.uniform(1.5, 2.5))  # Random delay
                
                # Click the search button
                try:
                    search_button = self.wait.until(
                        EC.element_to_be_clickable((By.CLASS_NAME, "btn-search"))
                    )
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", search_button)
                    time.sleep(0.5)
                    search_button.click()
                except TimeoutException:
                    print(f"  ⚠️  Search button not found, retrying...")
                    continue
                
                # Wait for results with shorter timeout
                time.sleep(4)
                
                # Get page source
                page_source = self.driver.page_source
                soup = BeautifulSoup(page_source, 'html.parser')
                
                metadata = {
                    'year': None,
                    'genre': None,
                    'explicit': None,
                    'popularity': None
                }
                
                # Extract genres (STATIC MAPPING ONLY - no dynamic scraping)
                all_genres = []
                tag_containers = soup.select('#spotify-tags .pl-tags.tagcloud, .wiki-tags .pl-tags.tagcloud')
                
                for container in tag_containers:
                    genre_links = container.find_all('a')
                    for link in genre_links:
                        genre_text = link.get_text().strip()
                        normalized_genre = genre_text.lower().replace(' ', '-')
                        all_genres.append(normalized_genre)
                
                if all_genres:
                    # Use DYNAMIC mapping with SAFE browser reuse
                    main_genre = self._get_main_genre_for_list(all_genres)
                    metadata['genre'] = main_genre
                
                # Extract explicit flag
                explicit_span = soup.select_one('span.span-explicit')
                if explicit_span:
                    explicit_text = explicit_span.get_text().strip()
                    if 'Yes' in explicit_text:
                        metadata['explicit'] = 1
                    elif 'No' in explicit_text:
                        metadata['explicit'] = 0
                
                # Extract year
                album_data = soup.select_one('p.album-data')
                if album_data:
                    album_text = album_data.get_text()
                    year_match = re.search(r'\(.*?(\d{4})\)', album_text)
                    if year_match:
                        metadata['year'] = int(year_match.group(1))
                
                # Extract popularity
                progressbars_div = soup.select_one('div.progressbars-div')
                if progressbars_div:
                    progressbars_text = progressbars_div.get_text()
                    popularity_match = re.search(r'Popularity:\s*(\d+)/100', progressbars_text)
                    if popularity_match:
                        metadata['popularity'] = int(popularity_match.group(1))
                
                # Validate metadata
                is_valid, error = self._validate_metadata(metadata)
                
                if not is_valid:
                    print(f"  ⚠️  Invalid metadata ({error}), retrying...")
                    continue
                
                # Success!
                self.scrapes_since_restart += 1
                
                # Random delay between requests (2-3 seconds)
                delay = random.uniform(2.0, 3.5)
                time.sleep(delay)
                
                return metadata
                
            except WebDriverException as e:
                if "invalid session id" in str(e) or "connection" in str(e).lower():
                    print(f"  ⚠️  Browser connection lost! Restarting...")
                    self._init_browser()
                    continue
                else:
                    print(f"  ❌ WebDriver error: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
            
            except Exception as e:
                print(f"  ❌ Unexpected error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
        
        # All retries failed
        print(f"  ❌ Failed after {max_retries} attempts")
        return None
    
    def get_processed_count(self, output_file, failed_tracks_file):
        """Count how many rows have already been processed (successful + failed)"""
        successful_count = 0
        failed_count = 0
        
        # Count successful rows
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file)
                successful_count = len(existing_df)
            except Exception as e:
                print(f"Error reading existing output file: {e}")
        
        # Count failed rows
        if os.path.exists(failed_tracks_file):
            try:
                failed_df = pd.read_csv(failed_tracks_file)
                failed_count = len(failed_df)
            except Exception as e:
                print(f"Error reading failed tracks file: {e}")
        
        total_processed = successful_count + failed_count
        print(f"Previously processed: {successful_count} successful + {failed_count} failed = {total_processed} total")
        
        return total_processed
    
    def process_csv(self, input_file, output_file, failed_tracks_file='failed_tracks.csv', sample_size=None):
        """
        Process CSV with checkpoint resumption and failed track logging
        """
        print(f"Reading {input_file}...")
        
        # Check for existing progress
        processed_count = self.get_processed_count(output_file, failed_tracks_file)
        
        # Read the full dataset
        total_df = pd.read_csv(input_file)
        total_tracks = len(total_df)
        
        print(f"Total tracks in dataset: {total_tracks:,}")
        print(f"Already processed: {processed_count:,}")
        print(f"Remaining: {total_tracks - processed_count:,}")
        
        # Determine range
        if sample_size:
            end_row = min(processed_count + sample_size, total_tracks)
            print(f"Processing sample: rows {processed_count} to {end_row}")
        else:
            end_row = total_tracks
        
        # All done?
        if processed_count >= total_tracks:
            print("✓ All tracks processed!")
            return pd.read_csv(output_file)
        
        # Read unprocessed portion
        if processed_count > 0:
            df_to_process = total_df.iloc[processed_count:end_row].copy()
        else:
            df_to_process = total_df.iloc[:end_row].copy()
        
        print(f"Loading {len(df_to_process):,} tracks to process...")
        
        # Initialize columns
        df_to_process['year'] = None
        df_to_process['genre'] = None
        df_to_process['explicit'] = None
        df_to_process['popularity'] = None
        
        # Track stats
        success_count = 0
        failed_count = 0
        
        # Check if failed tracks file already exists
        failed_file_exists = os.path.exists(failed_tracks_file)
        
        # Process tracks
        for idx, (index, row) in enumerate(df_to_process.iterrows()):
            track_id = row['id']
            
            print(f"\n[{idx + 1}/{len(df_to_process)}] {track_id}")
            
            # Scrape with retries
            metadata = self.scrape_track_metadata(track_id)
            
            if metadata:
                # Validate before saving
                is_valid, error = self._validate_metadata(metadata)
                
                if is_valid:
                    df_to_process.at[index, 'year'] = metadata['year']
                    df_to_process.at[index, 'genre'] = metadata['genre']
                    df_to_process.at[index, 'explicit'] = metadata['explicit']
                    df_to_process.at[index, 'popularity'] = metadata['popularity']
                    success_count += 1
                    
                    # Append to output file
                    single_row_df = df_to_process.loc[[index]]
                    
                    if processed_count == 0 and idx == 0:
                        single_row_df.to_csv(output_file, mode='w', index=False, header=True)
                    else:
                        single_row_df.to_csv(output_file, mode='a', index=False, header=False)
                    
                    processed_count += 1
                    print(f"  ✓ Saved to success file ({success_count} successful)")
                else:
                    print(f"  ❌ Invalid metadata ({error}), saving to failed file")
                    # Save failed track to CSV
                    failed_row = df_to_process.loc[[index]]
                    if not failed_file_exists and failed_count == 0:
                        failed_row.to_csv(failed_tracks_file, mode='w', index=False, header=True)
                        failed_file_exists = True
                    else:
                        failed_row.to_csv(failed_tracks_file, mode='a', index=False, header=False)
                    failed_count += 1
                    processed_count += 1
            else:
                print(f"  ❌ Scraping failed, saving to failed file")
                # Save failed track to CSV
                failed_row = df_to_process.loc[[index]]
                if not failed_file_exists and failed_count == 0:
                    failed_row.to_csv(failed_tracks_file, mode='w', index=False, header=True)
                    failed_file_exists = True
                else:
                    failed_row.to_csv(failed_tracks_file, mode='a', index=False, header=False)
                failed_count += 1
                processed_count += 1
            
            # Progress report every 10 tracks
            if (idx + 1) % 10 == 0:
                print(f"\n{'='*60}")
                print(f"Progress: {idx + 1}/{len(df_to_process)} ({100*(idx+1)/len(df_to_process):.1f}%)")
                print(f"Success rate: {success_count}/{idx+1} ({100*success_count/(idx+1):.1f}%)")
                print(f"Failed: {failed_count}/{idx+1}")
                print(f"{'='*60}")
        
        print(f"\n{'='*60}")
        print(f"Session completed!")
        print(f"Successfully scraped: {success_count}/{len(df_to_process)}")
        print(f"Failed: {failed_count}/{len(df_to_process)}")
        print(f"Total progress: {processed_count}/{total_tracks} ({100*processed_count/total_tracks:.1f}%)")
        print(f"{'='*60}")
        
        return pd.read_csv(output_file) if os.path.exists(output_file) else None


def main():
    """Main function"""
    INPUT_FILE = '../songs_with_attributes_and_lyrics.csv'
    OUTPUT_FILE = '../songs_enhanced_full.csv'
    FAILED_TRACKS_FILE = '../failed_tracks.csv'
    
    print("=" * 60)
    print("ROBUST Chosic Scraper with Browser Recovery")
    print("=" * 60)
    
    # Initialize scraper
    print("\nInitializing scraper...")
    print("  - Browser restart: Every 50 tracks")
    print("  - Retry attempts: 3 per track")
    print("  - NaN validation: Enabled")
    print("  - Failed track logging: Enabled")
    
    scraper = RobustChosicScraper(headless=True, restart_every_n=50)
    
    # Check existing progress
    if os.path.exists(OUTPUT_FILE):
        print(f"\n✓ Found existing progress file")
    else:
        print(f"\n→ Starting fresh")
    
    print("\nOptions:")
    print("1. Process full dataset (or resume if interrupted)")
    print("2. Process a sample batch")
    print("3. Exit")
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    if choice == '1':
        print("\n" + "=" * 60)
        print("Processing full dataset...")
        print("=" * 60)
        print("Note: You can stop anytime (Ctrl+C). Progress is saved after each track.")
        print()
        
        try:
            scraper.process_csv(INPUT_FILE, OUTPUT_FILE, FAILED_TRACKS_FILE)
            print("\n✓ Complete!")
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            print("Progress has been saved. Run again to resume.")
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            print("Progress has been saved. Run again to resume.")
    
    elif choice == '2':
        batch_size = input("\nBatch size (default: 100): ").strip()
        batch_size = int(batch_size) if batch_size.isdigit() else 100
        
        print("\n" + "=" * 60)
        print(f"Processing batch of {batch_size} tracks...")
        print("=" * 60)
        
        try:
            scraper.process_csv(INPUT_FILE, OUTPUT_FILE, FAILED_TRACKS_FILE, sample_size=batch_size)
            print("\n✓ Batch complete!")
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
    
    elif choice == '3':
        print("\nExiting...")
    
    else:
        print("\nInvalid choice.")


if __name__ == "__main__":
    main()
