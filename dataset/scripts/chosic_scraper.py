import pandas as pd
from bs4 import BeautifulSoup
import time
import json
import re
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from genre_mapper import get_main_genre_for_list

class ChosicScraper:
    def __init__(self, headless=True):
        """Initialize Chrome WebDriver with options"""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 10)
        self.delay = 2  # seconds between requests
    
    def __del__(self):
        """Clean up WebDriver on deletion"""
        if hasattr(self, 'driver'):
            self.driver.quit()
    
    def get_processed_count(self, output_file):
        """
        Count how many rows have already been processed in the output file.
        Returns: Number of data rows (excluding header)
        """
        if not os.path.exists(output_file):
            return 0
        
        try:
            # Read only to count rows (efficient)
            existing_df = pd.read_csv(output_file)
            processed_count = len(existing_df)
            print(f"Found existing output file with {processed_count} processed tracks")
            return processed_count
        except Exception as e:
            print(f"Error reading existing output file: {e}")
            return 0
        
    def scrape_track_metadata(self, track_id):
        """
        Scrape track metadata from chosic.com using Selenium
        Returns: dict with year, genre, explicit, popularity
        """
        url = f"https://www.chosic.com/music-genre-finder/?track={track_id}"
        
        try:
            print(f"Scraping: {track_id}")
            
            # Load the page
            self.driver.get(url)
            time.sleep(2)
            
            # Click the search button
            search_button = self.wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "btn-search")))
            search_button.click()
            print("Clicked search button")
            
            # Wait for results to load
            time.sleep(5)
            
            # Get the page source after JavaScript has executed
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            metadata = {
                'year': None,
                'genre': None,
                'explicit': None,
                'popularity': None
            }
            
            # Extract genres from Spotify and Wikipedia tags
            all_genres = []
            
            # Get genres from both Spotify and Wikipedia sections
            tag_containers = soup.select('#spotify-tags .pl-tags.tagcloud, .wiki-tags .pl-tags.tagcloud')
            for container in tag_containers:
                genre_links = container.find_all('a')
                for link in genre_links:
                    genre_text = link.get_text().strip()
                    # Normalize: lowercase and replace spaces with hyphens
                    normalized_genre = genre_text.lower().replace(' ', '-')
                    all_genres.append(normalized_genre)
            
            # Get main genre using genre_mapper
            if all_genres:
                main_genre = get_main_genre_for_list(all_genres)
                metadata['genre'] = main_genre
                print(f"Found genres: {all_genres} -> Main genre: {main_genre}")
            
            # Extract explicit flag
            explicit_span = soup.select_one('span.span-explicit')
            if explicit_span:
                explicit_text = explicit_span.get_text().strip()
                # Extract "Yes" or "No" from "Explicit: Yes" or "Explicit: No"
                if 'Yes' in explicit_text:
                    metadata['explicit'] = 1
                elif 'No' in explicit_text:
                    metadata['explicit'] = 0
                print(f"Explicit: {metadata['explicit']}")
            
            # Extract year from album data
            album_data = soup.select_one('p.album-data')
            if album_data:
                album_text = album_data.get_text()
                # Extract year from format: "from album: NAME (Month DD, YYYY)"
                year_match = re.search(r'\(.*?(\d{4})\)', album_text)
                if year_match:
                    metadata['year'] = int(year_match.group(1))
                    print(f"Year: {metadata['year']}")
            
            # Extract popularity
            progressbars_div = soup.select_one('div.progressbars-div')
            if progressbars_div:
                progressbars_text = progressbars_div.get_text()
                # Extract "Popularity: X/100" from the text
                popularity_match = re.search(r'Popularity:\s*(\d+)/100', progressbars_text)
                if popularity_match:
                    metadata['popularity'] = int(popularity_match.group(1))
                    print(f"Popularity: {metadata['popularity']}")
            
            return metadata
            
        except TimeoutException as e:
            print(f"Timeout error for {track_id}: {e}")
            return None
        except Exception as e:
            print(f"Error scraping {track_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_csv(self, input_file, output_file, sample_size=None):
        """
        Process the CSV file and add new columns with checkpoint resumption support.
        
        How it works:
        1. Check if output_file exists and count processed rows
        2. Skip already processed rows in input_file
        3. Process remaining rows and append to output_file one by one
        4. Can be stopped and resumed at any time
        """
        print(f"Reading {input_file}...")
        
        # Step 1: Check for existing progress
        processed_count = self.get_processed_count(output_file)
        
        # Step 2: Determine how many rows to skip
        skip_rows = processed_count
        
        # Read the full dataset to get total count
        total_df = pd.read_csv(input_file)
        total_tracks = len(total_df)
        
        print(f"Total tracks in dataset: {total_tracks}")
        print(f"Already processed: {processed_count}")
        print(f"Remaining to process: {total_tracks - processed_count}")
        
        # If sample_size is specified, adjust it
        if sample_size:
            # Process only sample_size tracks starting from the checkpoint
            end_row = min(skip_rows + sample_size, total_tracks)
            print(f"Processing sample: rows {skip_rows} to {end_row}")
        else:
            end_row = total_tracks
        
        # If all tracks are already processed
        if skip_rows >= total_tracks:
            print("All tracks have been processed!")
            return pd.read_csv(output_file)
        
        # Step 3: Read only the unprocessed portion
        if skip_rows > 0:
            # Read from checkpoint onwards
            df_to_process = total_df.iloc[skip_rows:end_row].copy()
        else:
            # First run - read from beginning
            df_to_process = total_df.iloc[:end_row].copy()
        
        print(f"Loading {len(df_to_process)} tracks to process...")
        
        # Initialize new columns
        df_to_process['year'] = None
        df_to_process['genre'] = None
        df_to_process['explicit'] = None
        df_to_process['popularity'] = None
        
        # Track progress
        tracks_to_process = len(df_to_process)
        success_count = 0
        
        # Step 4: Process and append one by one
        for idx, (index, row) in enumerate(df_to_process.iterrows()):
            track_id = row['id']
            
            # Scrape metadata
            metadata = self.scrape_track_metadata(track_id)
            
            if metadata:
                df_to_process.at[index, 'year'] = metadata['year']
                df_to_process.at[index, 'genre'] = metadata['genre']
                df_to_process.at[index, 'explicit'] = metadata['explicit']
                df_to_process.at[index, 'popularity'] = metadata['popularity']
                success_count += 1
            
            # Append this single row to the output file
            single_row_df = df_to_process.loc[[index]]
            
            if processed_count == 0 and idx == 0:
                # First row ever - create new file with header
                single_row_df.to_csv(output_file, mode='w', index=False, header=True)
            else:
                # Append without header
                single_row_df.to_csv(output_file, mode='a', index=False, header=False)
            
            processed_count += 1
            
            # Progress update
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{tracks_to_process} tracks in this session ({success_count} successful)")
                print(f"Total progress: {processed_count}/{total_tracks} ({100*processed_count/total_tracks:.1f}%)")
            
            # Rate limiting
            time.sleep(self.delay)
        
        print(f"\nSession completed! Successfully scraped {success_count}/{tracks_to_process} tracks")
        print(f"Total progress: {processed_count}/{total_tracks} tracks ({100*processed_count/total_tracks:.1f}%)")
        
        if processed_count >= total_tracks:
            print("✓ All tracks have been processed!")
        else:
            print(f"→ Run the script again to continue from row {processed_count}")
        
        # Return the complete output file
        return pd.read_csv(output_file)

def main():
    """
    Main function with checkpoint resumption support.
    You can run this script multiple times - it will always resume from where it left off.
    """
    # Paths relative to dataset/ directory (parent of scripts/)
    INPUT_FILE = '../songs_with_attributes_and_lyrics.csv'
    OUTPUT_FILE = '../songs_enhanced_full.csv'
    
    print("=" * 60)
    print("Chosic Scraper with Checkpoint Resumption")
    print("=" * 60)
    
    scraper = ChosicScraper()
    
    # Check if there's existing progress
    if os.path.exists(OUTPUT_FILE):
        print(f"\n✓ Found existing progress file: {OUTPUT_FILE}")
        print("The script will resume from where it left off.")
    else:
        print(f"\n→ No existing progress found. Starting fresh.")
    
    print("\nOptions:")
    print("1. Process full dataset (or resume if interrupted)")
    print("2. Process a sample batch (e.g., 100 tracks)")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == '1':
        print("\n" + "=" * 60)
        print("Processing full dataset (with auto-resume)...")
        print("=" * 60)
        print("Note: You can stop this at any time (Ctrl+C).")
        print("Just run the script again to resume.\n")
        
        try:
            full_df = scraper.process_csv(INPUT_FILE, OUTPUT_FILE)
            print("\n" + "=" * 60)
            print("✓ Full dataset processing completed!")
            print("=" * 60)
        except KeyboardInterrupt:
            print("\n\n" + "=" * 60)
            print("⚠ Process interrupted by user")
            print("=" * 60)
            print("Your progress has been saved.")
            print("Run the script again to resume from where you left off.")
        except Exception as e:
            print(f"\n\n⚠ Error occurred: {e}")
            print("Your progress has been saved.")
            print("Run the script again to resume.")
    
    elif choice == '2':
        batch_size = input("\nEnter batch size (default: 100): ").strip()
        batch_size = int(batch_size) if batch_size.isdigit() else 100
        
        print("\n" + "=" * 60)
        print(f"Processing batch of {batch_size} tracks...")
        print("=" * 60)
        
        try:
            sample_df = scraper.process_csv(INPUT_FILE, OUTPUT_FILE, sample_size=batch_size)
            print("\n" + "=" * 60)
            print(f"✓ Batch processing completed!")
            print("=" * 60)
            print("Run the script again to process the next batch.")
        except KeyboardInterrupt:
            print("\n\n" + "=" * 60)
            print("⚠ Process interrupted by user")
            print("=" * 60)
            print("Your progress has been saved.")
        except Exception as e:
            print(f"\n\n⚠ Error occurred: {e}")
            print("Your progress has been saved.")
    
    elif choice == '3':
        print("\nExiting...")
    
    else:
        print("\nInvalid choice. Exiting...")

if __name__ == "__main__":
    main()