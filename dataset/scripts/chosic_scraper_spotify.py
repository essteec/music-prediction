"""
Spotify API Direct Scraper
Uses Spotify Web API for reliable and efficient data collection
Handles batch requests (up to 50 tracks/artists per request)
Preserves existing checkpoint system and data collection logic
"""

import pandas as pd
import time
import json
import re
import os
import requests
from genre_mapper_http import get_main_genre_for_list

class SpotifyAPIScraper:
    def __init__(self, client_id=None, client_secret=None, access_token=None):
        """
        Initialize Spotify API client
        
        Args:
            client_id: Spotify Client ID (preferred - auto-refreshes token)
            client_secret: Spotify Client Secret (preferred - auto-refreshes token)
            access_token: Manual access token (will prompt if credentials not provided)
        """
        self.session = requests.Session()
        self.base_url = 'https://api.spotify.com/v1'
        
        # Credentials for auto-refresh
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token
        self.token_expires_at = 0  # Timestamp when token expires
        
        # Rate limiting settings
        self.delay = 5  # 5 seconds between requests (Spotify allows ~180 req/min)
        self.batch_size = 50  # Maximum IDs per request
        self.last_request_time = 0
        
        # Track API usage
        self.requests_made = 0
        self.rate_limit_remaining = None
        self.rate_limit_reset = None
        
        # Get credentials from environment if not provided
        if not self.client_id:
            self.client_id = os.environ.get('SPOTIFY_CLIENT_ID')
        if not self.client_secret:
            self.client_secret = os.environ.get('SPOTIFY_CLIENT_SECRET')
        
        # If we have credentials, get token automatically
        if self.client_id and self.client_secret:
            print("✓ Using Client ID and Secret for automatic token management")
            self._refresh_access_token()
        elif not self.access_token:
            # Fall back to manual token
            self.access_token = self._get_manual_access_token()
            # Set authorization header
            self.session.headers.update({
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
        else:
            # Use provided token
            self.session.headers.update({
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
    
    def _refresh_access_token(self):
        """
        Get a fresh access token using Client Credentials flow
        Automatically called when token expires
        """
        print("🔄 Requesting access token from Spotify...")
        
        url = "https://accounts.spotify.com/api/token"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        
        try:
            response = requests.post(url, headers=headers, data=data, timeout=10)
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                expires_in = token_data['expires_in']  # Seconds (usually 3600 = 1 hour)
                
                # Set expiration time (with 5 minute buffer)
                self.token_expires_at = time.time() + expires_in - 300
                
                # Update session header
                self.session.headers.update({
                    'Authorization': f'Bearer {self.access_token}',
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                })
                
                print(f"✓ Access token obtained! Valid for {expires_in/60:.0f} minutes")
                return True
            else:
                print(f"❌ Failed to get access token: {response.status_code}")
                print(f"   Response: {response.text}")
                raise ValueError("Failed to obtain access token. Check your Client ID and Secret.")
                
        except Exception as e:
            print(f"❌ Error getting access token: {e}")
            raise
    
    def _check_token_expiration(self):
        """
        Check if token is about to expire and refresh if needed
        """
        if self.client_id and self.client_secret:
            # We can auto-refresh
            if time.time() >= self.token_expires_at:
                print("\n⚠️  Access token expired, refreshing...")
                self._refresh_access_token()
    
    def _get_manual_access_token(self):
        """
        Get Spotify access token from environment or user input (fallback method)
        """
        # Try environment variable first
        token = os.environ.get('SPOTIFY_ACCESS_TOKEN')
        if token:
            print("✓ Using access token from environment variable")
            return token
        
        # Prompt user
        print("\n" + "="*60)
        print("Spotify API Credentials Required")
        print("="*60)
        print("RECOMMENDED: Use Client ID + Secret for automatic token refresh")
        print("")
        print("Option 1: Set environment variables (best):")
        print("  export SPOTIFY_CLIENT_ID='your_client_id'")
        print("  export SPOTIFY_CLIENT_SECRET='your_client_secret'")
        print("")
        print("Option 2: Enter manual token (expires in 1 hour):")
        print("  Get from: https://developer.spotify.com/console/get-track/")
        print("="*60)
        
        choice = input("\nDo you have Client ID and Secret? (y/n): ").strip().lower()
        
        if choice == 'y':
            self.client_id = input("Enter Client ID: ").strip()
            self.client_secret = input("Enter Client Secret: ").strip()
            
            if self.client_id and self.client_secret:
                self._refresh_access_token()
                return self.access_token
        
        # Fall back to manual token
        token = input("\nEnter your Spotify access token: ").strip()
        
        if not token:
            raise ValueError("Access token is required to use Spotify API")
        
        return token
    
    def _handle_rate_limit(self, response):
        """
        Handle rate limiting by checking response headers
        """
        # Update rate limit info from headers
        if 'X-RateLimit-Remaining' in response.headers:
            self.rate_limit_remaining = int(response.headers['X-RateLimit-Remaining'])
        
        if 'X-RateLimit-Reset' in response.headers:
            self.rate_limit_reset = int(response.headers['X-RateLimit-Reset'])
        
        # If rate limited, wait until reset
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"⚠️  Rate limited! Waiting {retry_after} seconds...")
            time.sleep(retry_after + 1)
            return True
        
        return False
    
    def _make_request(self, url, max_retries=6):
        """
        Make API request with rate limiting and retry logic
        
        Args:
            url: API endpoint URL
            max_retries: Maximum number of retry attempts
            
        Returns: Response object or None if failed
        """
        # Check if token needs refresh
        self._check_token_expiration()
        
        # Respect rate limiting delay
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=10)
                self.last_request_time = time.time()
                self.requests_made += 1
                
                # Handle rate limiting
                if self._handle_rate_limit(response):
                    continue  # Retry after waiting
                
                # Check for token expiration
                if response.status_code == 401:
                    # Try to refresh token if we have credentials
                    if self.client_id and self.client_secret:
                        print("⚠️  Token expired, refreshing...")
                        self._refresh_access_token()
                        continue  # Retry with new token
                    else:
                        print("❌ Access token expired or invalid")
                        print("   Please restart with valid credentials")
                        raise ValueError("Invalid access token")
                
                # Success
                if response.status_code == 200:
                    return response
                
                # Client error (4xx) - don't retry
                if 400 <= response.status_code < 500:
                    print(f"⚠️  Client error {response.status_code}: {response.text[:200]}")
                    return None
                
                # Server error (5xx) - retry with backoff
                if response.status_code >= 500:
                    wait_time = 2 ** attempt
                    print(f"⚠️  Server error {response.status_code}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                return response
                
            except requests.exceptions.Timeout:
                print(f"⚠️  Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            except requests.exceptions.RequestException as e:
                print(f"⚠️  Request error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
        
        return None
    
    def get_tracks_batch(self, track_ids):
        """
        Get metadata for multiple tracks (up to 50) in one request
        
        Args:
            track_ids: List of Spotify track IDs (max 50)
            
        Returns: dict mapping track_id -> track data, or None if failed
        """
        if len(track_ids) > 50:
            raise ValueError(f"Maximum 50 track IDs per request (got {len(track_ids)})")
        
        # Build URL with comma-separated IDs
        ids_param = ','.join(track_ids)
        url = f'{self.base_url}/tracks?ids={ids_param}'
        
        response = self._make_request(url)
        
        if not response:
            return None
        
        try:
            data = response.json()
            
            # Create mapping of track_id -> track data
            tracks_map = {}
            for track in data.get('tracks', []):
                if track:  # API returns null for invalid IDs
                    tracks_map[track['id']] = track
            
            return tracks_map
            
        except Exception as e:
            print(f"⚠️  Error parsing tracks response: {e}")
            return None
    
    def get_artists_batch(self, artist_ids):
        """
        Get metadata for multiple artists (up to 50) in one request
        
        Args:
            artist_ids: List of Spotify artist IDs (max 50)
            
        Returns: dict mapping artist_id -> artist data, or None if failed
        """
        if len(artist_ids) > 50:
            raise ValueError(f"Maximum 50 artist IDs per request (got {len(artist_ids)})")
        
        # Build URL with comma-separated IDs
        ids_param = ','.join(artist_ids)
        url = f'{self.base_url}/artists?ids={ids_param}'
        
        response = self._make_request(url)
        
        if not response:
            return None
        
        try:
            data = response.json()
            
            # Create mapping of artist_id -> artist data
            artists_map = {}
            for artist in data.get('artists', []):
                if artist:  # API returns null for invalid IDs
                    artists_map[artist['id']] = artist
            
            return artists_map
            
        except Exception as e:
            print(f"⚠️  Error parsing artists response: {e}")
            return None
    
    def extract_track_metadata(self, track_data):
        """
        Extract metadata from a single track object
        
        Args:
            track_data: Track object from Spotify API
            
        Returns: dict with year, explicit, popularity, artist_ids
        """
        metadata = {
            'year': None,
            'explicit': 0,
            'popularity': None,
            'artist_ids': []
        }
        
        if not track_data:
            return metadata
        
        # Extract popularity
        metadata['popularity'] = track_data.get('popularity')
        
        # Extract explicit flag
        metadata['explicit'] = 1 if track_data.get('explicit', False) else 0
        
        # Extract year from album release date
        album = track_data.get('album', {})
        release_date = album.get('release_date', '')
        if release_date:
            year_match = re.search(r'(\d{4})', release_date)
            if year_match:
                metadata['year'] = int(year_match.group(1))
        
        # Extract artist IDs
        artists = track_data.get('artists', [])
        metadata['artist_ids'] = [artist['id'] for artist in artists if 'id' in artist]
        
        return metadata
    
    def extract_artist_genres(self, artists_data):
        """
        Extract and aggregate genres from multiple artist objects
        
        Args:
            artists_data: dict mapping artist_id -> artist data
            
        Returns: list of all genres from all artists
        """
        all_genres = []
        
        for artist_id, artist_data in artists_data.items():
            if artist_data and 'genres' in artist_data:
                all_genres.extend(artist_data['genres'])
        
        return all_genres
    
    def process_tracks_batch(self, track_ids):
        """
        Process a batch of tracks efficiently
        
        Args:
            track_ids: List of track IDs to process (max 50)
            
        Returns: dict mapping track_id -> metadata dict
        """
        results = {}
        
        # Step 1: Get all track data in one request
        tracks_data = self.get_tracks_batch(track_ids)
        
        if not tracks_data:
            print(f"⚠️  Failed to fetch tracks batch")
            return results
        
        # Step 2: Collect all unique artist IDs from all tracks
        all_artist_ids = set()
        track_to_artists = {}  # Map track_id -> artist_ids
        
        for track_id in track_ids:
            track_data = tracks_data.get(track_id)
            
            if not track_data:
                # Track not found or invalid
                results[track_id] = None
                continue
            
            # Extract basic metadata
            metadata = self.extract_track_metadata(track_data)
            
            # Store artist IDs for this track
            track_to_artists[track_id] = metadata['artist_ids']
            all_artist_ids.update(metadata['artist_ids'])
            
            # Store partial metadata (genre to be added later)
            results[track_id] = metadata
        
        # Step 3: Get all artist data in batches (if any artists)
        artists_data = {}
        if all_artist_ids:
            artist_ids_list = list(all_artist_ids)
            
            # Process artists in batches of 50
            for i in range(0, len(artist_ids_list), 50):
                batch = artist_ids_list[i:i+50]
                batch_data = self.get_artists_batch(batch)
                
                if batch_data:
                    artists_data.update(batch_data)
        
        # Step 4: Add genre information to each track
        for track_id, metadata in results.items():
            if metadata is None:
                continue
            
            # Get artist IDs for this track
            artist_ids = track_to_artists.get(track_id, [])
            
            # Get genres from these artists
            track_artists_data = {aid: artists_data.get(aid) for aid in artist_ids}
            all_genres = self.extract_artist_genres(track_artists_data)
            
            # Normalize genres (lowercase, spaces to hyphens)
            normalized_genres = [g.lower().replace(' ', '-') for g in all_genres]
            
            # Get main genre using genre_mapper
            if normalized_genres:
                main_genre = get_main_genre_for_list(normalized_genres)
                metadata['genre'] = main_genre
            else:
                metadata['genre'] = None
        
        return results
    
    def _validate_metadata(self, metadata):
        """
        Validate that all metadata fields are present and valid (not None)
        Returns: (is_valid, missing_fields)
        """
        if not metadata:
            return False, "metadata is None"
        
        # Check if ANY required field is None
        missing_fields = [k for k, v in metadata.items() if v is None and k != 'artist_ids']
        if missing_fields:
            return False, f"missing or invalid fields: {', '.join(missing_fields)}"
        
        return True, None
    
    def get_processed_count(self, output_file, failed_file='../failed_tracks.csv'):
        """
        Count how many rows have already been processed (successful + failed).
        Returns: Number of data rows (excluding header)
        """
        successful_count = 0
        failed_count = 0
        
        # Count successful tracks
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file)
                successful_count = len(existing_df)
            except Exception as e:
                print(f"Error reading existing output file: {e}")
        
        # Count failed tracks
        if os.path.exists(failed_file):
            try:
                failed_df = pd.read_csv(failed_file)
                failed_count = len(failed_df)
            except Exception as e:
                print(f"Error reading failed tracks file: {e}")
        
        total_processed = successful_count + failed_count
        
        if failed_count > 0:
            print(f"Found existing progress: {successful_count} successful + {failed_count} failed = {total_processed} total processed tracks")
        else:
            print(f"Found existing output file with {successful_count} processed tracks")
        
        return total_processed
    
    def process_csv(self, input_file, output_file, failed_file='../failed_tracks.csv', sample_size=None):
        """
        Process the CSV file and add new columns with checkpoint resumption support.
        Uses efficient batch processing (50 tracks per API call).
        
        How it works:
        1. Check if output_file exists and count processed rows (including failed)
        2. Skip already processed rows in input_file
        3. Process remaining rows in batches of 50 for efficiency
        4. Append results to output_file one by one
        5. Failed tracks are logged to failed_file
        6. Can be stopped and resumed at any time
        """
        print(f"Reading {input_file}...")
        
        # Step 1: Check for existing progress (successful + failed)
        processed_count = self.get_processed_count(output_file, failed_file)
        
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
        failed_count = 0
        start_time = time.time()
        
        # Check if files exist for proper headers
        success_file_exists = os.path.exists(output_file)
        failed_file_exists = os.path.exists(failed_file)
        
        # Step 4: Process in batches of 50 for efficiency
        total_to_process = len(df_to_process)
        
        for batch_start in range(0, total_to_process, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_to_process)
            batch_df = df_to_process.iloc[batch_start:batch_end]
            
            # Get track IDs for this batch
            track_ids = batch_df['id'].tolist()
            
            print(f"\n{'='*60}")
            print(f"Processing batch: tracks {batch_start + 1}-{batch_end} ({len(track_ids)} tracks)")
            print(f"{'='*60}")
            
            # Process entire batch with 2-3 API calls total
            batch_results = self.process_tracks_batch(track_ids)
            
            # Save results for each track in batch
            for idx, (index, row) in enumerate(batch_df.iterrows()):
                track_id = row['id']
                metadata = batch_results.get(track_id)
                
                print(f"[{batch_start + idx + 1}/{total_to_process}] {track_id}: ", end="")
                
                if metadata and self._validate_metadata(metadata)[0]:
                    # Valid metadata - save to output file
                    df_to_process.at[index, 'year'] = metadata['year']
                    df_to_process.at[index, 'genre'] = metadata['genre']
                    df_to_process.at[index, 'explicit'] = metadata['explicit']
                    df_to_process.at[index, 'popularity'] = metadata['popularity']
                    success_count += 1
                    
                    # Append to success file
                    single_row_df = df_to_process.loc[[index]]
                    
                    if not success_file_exists:
                        single_row_df.to_csv(output_file, mode='w', index=False, header=True)
                        success_file_exists = True
                    else:
                        single_row_df.to_csv(output_file, mode='a', index=False, header=False)
                    
                    print(f"✓ Year: {metadata['year']}, Pop: {metadata['popularity']}, Genre: {metadata['genre']}")
                else:
                    # Failed to scrape - log to failed file
                    failed_count += 1
                    failed_row_df = df_to_process.loc[[index]]
                    
                    if not failed_file_exists:
                        failed_row_df.to_csv(failed_file, mode='w', index=False, header=True)
                        failed_file_exists = True
                    else:
                        failed_row_df.to_csv(failed_file, mode='a', index=False, header=False)
                    
                    print(f"❌ Failed to scrape")
                
                processed_count += 1
            
            # Progress report after each batch
            elapsed = time.time() - start_time
            speed = (batch_end) / elapsed if elapsed > 0 else 0
            
            print(f"\n{'='*60}")
            print(f"Progress: {batch_end}/{total_to_process} ({100*batch_end/total_to_process:.1f}%)")
            print(f"Success rate: {success_count}/{batch_end} ({100*success_count/batch_end:.1f}%)")
            print(f"Failed: {failed_count}/{batch_end}")
            print(f"Speed: {speed:.2f} tracks/sec")
            print(f"API requests made: {self.requests_made}")
            if self.rate_limit_remaining is not None:
                print(f"Rate limit remaining: {self.rate_limit_remaining}")
            print(f"{'='*60}")
        
        elapsed = time.time() - start_time
        avg_speed = total_to_process / elapsed if elapsed > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"Session completed!")
        print(f"Successfully scraped: {success_count}/{total_to_process}")
        print(f"Failed: {failed_count}/{total_to_process}")
        print(f"Average speed: {avg_speed:.2f} tracks/sec ({elapsed/60:.1f} minutes total)")
        print(f"Total API requests: {self.requests_made}")
        print(f"Efficiency: {total_to_process/self.requests_made:.1f} tracks per request")
        print(f"Total progress: {processed_count}/{total_tracks} ({100*processed_count/total_tracks:.1f}%)")
        print(f"{'='*60}")
        
        if processed_count >= total_tracks:
            print("\n✓ All tracks have been processed!")
        else:
            print(f"\n→ Run the script again to continue from row {processed_count}")
        
        # Return the complete output file
        return pd.read_csv(output_file)

def main():
    """
    Main function with checkpoint resumption support.
    Uses Spotify API directly for reliable and efficient data collection.
    Supports automatic token refresh with Client Credentials.
    You can run this script multiple times - it will always resume from where it left off.
    """
    # Paths relative to dataset/ directory (parent of scripts/)
    INPUT_FILE = '../songs_with_attributes_and_lyrics.csv'
    OUTPUT_FILE = '../songs_enhanced_full.csv'
    FAILED_FILE = '../failed_tracks.csv'
    
    print("=" * 60)
    print("Spotify API Direct Scraper")
    print("=" * 60)
    print()
    
    # Check for credentials in environment
    has_client_id = os.environ.get('SPOTIFY_CLIENT_ID')
    has_client_secret = os.environ.get('SPOTIFY_CLIENT_SECRET')
    
    if has_client_id and has_client_secret:
        print("✓ Found credentials in environment variables")
        print("  → Token will auto-refresh every hour")
    else:
        print("💡 TIP: For automatic token refresh, set these environment variables:")
        print("   export SPOTIFY_CLIENT_ID='your_client_id'")
        print("   export SPOTIFY_CLIENT_SECRET='your_client_secret'")
        print()
    
    # Initialize scraper
    try:
        scraper = SpotifyAPIScraper()
    except ValueError as e:
        print(f"\n❌ {e}")
        return
    
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
            full_df = scraper.process_csv(INPUT_FILE, OUTPUT_FILE, FAILED_FILE)
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
            sample_df = scraper.process_csv(INPUT_FILE, OUTPUT_FILE, FAILED_FILE, sample_size=batch_size)
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
