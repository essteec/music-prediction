#!/usr/bin/env python3
"""
Audio Download Pilot - YouTube Music Pipeline (Optimized)

Downloads and validates songs from YouTube using yt-dlp with concurrent processing.
Uses producer-consumer pattern for 5x speedup over sequential approach.

Key optimizations:
- yt-dlp Python API (32% faster than subprocess)
- ThreadPoolExecutor for concurrent search (8 workers) and download (4 workers)
- Producer-consumer pattern: search queue feeds download queue
- Reduced rate limiting (0.5s between batches vs 2s per song)

Usage:
    python 01_pilot_download.py [--start-row N] [--limit M] [--no-resume]
    python 01_pilot_download.py --workers 8  # Adjust concurrency
    
Examples:
    # Start pilot from beginning
    python 01_pilot_download.py
    
    # Test on 100 songs
    python 01_pilot_download.py --limit 100
"""
import os
import sys
import csv
import json
import time
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import safe_eval_artists, format_query, sanitize_filename, seconds_to_readable, estimate_remaining_time
from validation import calculate_confidence_score

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm not installed. Run: pip install tqdm")
    sys.exit(1)

try:
    import yt_dlp
except ImportError:
    print("ERROR: yt-dlp not installed. Run: pip install yt-dlp")
    sys.exit(1)


# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio" / "pilot"
LOG_DIR = DATA_DIR / "logs"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
LOG_FILE = LOG_DIR / "download_log_pilot.csv"
CHECKPOINT_FILE = LOG_DIR / "checkpoint_pilot.json"

# Download settings
DEFAULT_LIMIT = 55000
CONFIDENCE_THRESHOLD = 60  # Minimum score for auto-download

# Concurrency settings (tuned from benchmarks)
SEARCH_WORKERS = 4   # Search is CPU/network light, can parallelize more
DOWNLOAD_WORKERS = 4 # Download is bandwidth limited
BATCH_SIZE = 50      # Process in batches for checkpoint stability
BATCH_DELAY = 30    # Small delay between batches to avoid rate limiting

# Thread-safe logging
log_lock = threading.Lock()


def ensure_directories():
    """Create required directories if they don't exist."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_checkpoint() -> Dict[str, Any]:
    """Load checkpoint data if exists."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Corrupt checkpoint file, starting fresh: {e}")
    return {'last_row': -1, 'completed': 0, 'successful': 0, 'failed': 0}


def save_checkpoint(row_idx: int, stats: Dict[str, int]):
    """Save checkpoint for resume capability (atomic write)."""
    checkpoint = {
        'last_row': row_idx,
        'completed': stats['completed'],
        'successful': stats['successful'],
        'failed': stats['failed'],
        'timestamp': datetime.now().isoformat()
    }
    # Atomic write: write to temp file then rename
    temp_file = CHECKPOINT_FILE.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    temp_file.replace(CHECKPOINT_FILE)  # Atomic on POSIX


def init_log_file():
    """Initialize CSV log file with headers if it doesn't exist."""
    if not LOG_FILE.exists():
        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'row_idx', 'song_id', 'song_name', 'artists', 'duration_ms',
                'query', 'youtube_id', 'youtube_title', 'youtube_duration',
                'confidence_score', 'confidence_level', 'title_similarity',
                'duration_match', 'duration_diff', 'artist_matches',
                'download_success', 'file_size_mb', 'time_taken_sec', 'error_msg'
            ])


def log_result(row_idx: int, csv_row: Dict, result: Dict):
    """Append result to log file (thread-safe, with CSV formula injection protection)."""
    def sanitize_csv_cell(value):
        """Prevent formula injection by escaping dangerous prefixes."""
        if value and isinstance(value, str) and value[0] in ('=', '+', '-', '@'):
            return "'" + value
        return value
    
    with log_lock:
        with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                row_idx,
                sanitize_csv_cell(csv_row.get('id', '')),
                sanitize_csv_cell(csv_row.get('name', '')),
                sanitize_csv_cell(csv_row.get('artists', '')),
                csv_row.get('duration_ms', ''),
                sanitize_csv_cell(result.get('query', '')),
                result.get('youtube_id', ''),
                sanitize_csv_cell(result.get('youtube_title', '')),
                result.get('youtube_duration', ''),
                result.get('confidence_score', ''),
                result.get('confidence_level', ''),
                result.get('title_similarity', ''),
                result.get('duration_match', ''),
                result.get('duration_diff', ''),
                result.get('artist_matches', ''),
                result.get('download_success', False),
                result.get('file_size_mb', ''),
                result.get('time_taken', ''),
                sanitize_csv_cell(result.get('error_msg', ''))
            ])


def search_youtube(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Search YouTube using yt-dlp Python API (faster than subprocess).
    
    Args:
        query: Search query string
        max_results: Number of results to return
        
    Returns:
        List of dicts with keys: id, title, duration, uploader, url
    """
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'noprogress': True,
            'extract_flat': 'in_playlist',  # Fast search, avoid JS challenge errors
            'ignoreerrors': True,           # Skip failing videos instead of aborting
            'skip_download': True,
            'cookiesfrombrowser': ('firefox',),  # Use Firefox cookies to bypass bot detection
            'remote_components': ['ejs:github'], # Download EJS component for JS challenges
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f'ytsearch{max_results}:{query}', download=False)
        
        if not info or 'entries' not in info:
            return []
        
        videos = []
        for entry in info['entries']:
            if not entry:
                continue
            videos.append({
                'id': entry.get('id', ''),
                'title': entry.get('title', ''),
                'duration': entry.get('duration', 0),
                'uploader': entry.get('uploader', ''),
                'url': entry.get('webpage_url', f"https://www.youtube.com/watch?v={entry.get('id', '')}")
            })
        
        return videos
    
    except Exception as e:
        return []


def download_audio(video_id: str, output_path: Path) -> Dict[str, Any]:
    """
    Download audio using yt-dlp Python API.
    
    Args:
        video_id: YouTube video ID
        output_path: Path to save audio file
        
    Returns:
        Dict with 'success' (bool) and optional 'error' (str), 'file_size_mb' (float)
    """
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'noprogress': True,
            'format': '251/bestaudio',  # Opus/WebM ~3-4MB per song
            'outtmpl': str(output_path),
            'cookiesfrombrowser': ('firefox',),  # Use Firefox cookies to bypass bot detection
            'remote_components': ['ejs:github'], # Download EJS component for JS challenges
        }
        
        url = f'https://www.youtube.com/watch?v={video_id}'
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            return {'success': True, 'file_size_mb': round(file_size_mb, 2)}
        else:
            return {'success': False, 'error': 'File not created'}
    
    except Exception as e:
        return {'success': False, 'error': str(e)[:100]}


def process_song_search(row_idx: int, csv_row: Dict) -> Dict[str, Any]:
    """
    Phase 1: Search and validate (CPU/network light, highly parallelizable).
    
    Returns result dict with validation data and video_id if match found.
    """
    result = {
        'row_idx': row_idx,
        'csv_row': csv_row,
        'download_success': False,
        'ready_for_download': False,
        'time_taken': 0,
        'error_msg': ''
    }
    
    start_time = time.time()
    
    # Check if already downloaded FIRST - skip entire search
    output_filename = f"{row_idx:06d}_opus.webm"
    output_path = AUDIO_DIR / output_filename
    
    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        result['download_success'] = True
        result['ready_for_download'] = False  # Don't need to download
        result['file_size_mb'] = round(file_size_mb, 2)
        result['time_taken'] = round(time.time() - start_time, 2)
        return result
    
    try:
        # Parse CSV data
        song_id = csv_row.get('id', '')
        track_name = csv_row.get('name', '')
        artists_str = csv_row.get('artists', '[]')
        duration_ms = int(csv_row.get('duration_ms', 0))
        
        artists = safe_eval_artists(artists_str)
        if not artists or not track_name:
            result['error_msg'] = 'Missing track name or artists'
            return result
        
        if duration_ms <= 0:
            result['error_msg'] = 'Invalid duration_ms in CSV'
            return result
        
        # Format search query
        query = format_query(track_name, artists, max_artists=3)
        result['query'] = query
        
        # Search YouTube
        search_results = search_youtube(query, max_results=5)
        if not search_results:
            result['error_msg'] = 'No YouTube results found'
            return result
        
        # Validate and select best match
        best_match = None
        best_score = -1
        best_validation = None
        
        csv_data = {
            'name': track_name,
            'artists': artists,
            'duration_ms': duration_ms
        }
        
        for video in search_results:
            if video.get('duration') is None or not isinstance(video['duration'], (int, float)):
                continue
            
            validation = calculate_confidence_score(csv_data, video)
            
            if validation['total_score'] > best_score:
                best_score = validation['total_score']
                best_match = video
                best_validation = validation
        
        if not best_match:
            result['error_msg'] = 'No valid matches (all too long or invalid)'
            return result
        
        # Store validation results
        result.update({
            'youtube_id': best_match['id'],
            'youtube_title': best_match['title'],
            'youtube_duration': best_match['duration'],
            'confidence_score': best_validation['total_score'],
            'confidence_level': best_validation['confidence'],
            'title_similarity': best_validation['title_similarity'],
            'duration_match': best_validation['duration_match'],
            'duration_diff': best_validation['duration_diff'],
            'artist_matches': best_validation['artist_matches']
        })
        
        # Check if confidence is high enough for download
        if best_validation['total_score'] >= CONFIDENCE_THRESHOLD:
            result['ready_for_download'] = True
        else:
            result['error_msg'] = f'Low confidence ({best_validation["total_score"]:.1f} < {CONFIDENCE_THRESHOLD})'
    
    except Exception as e:
        result['error_msg'] = f'Search error: {str(e)[:100]}'
    
    finally:
        result['time_taken'] = round(time.time() - start_time, 2)
    
    return result


def process_song_download(search_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 2: Download audio (bandwidth limited, fewer parallel workers).
    
    Takes search result and performs download if ready_for_download is True.
    """
    if not search_result.get('ready_for_download'):
        return search_result
    
    start_time = time.time()
    
    try:
        row_idx = search_result['row_idx']
        video_id = search_result['youtube_id']
        
        # Check if already downloaded (skip if exists)
        output_filename = f"{row_idx:06d}_opus.webm"
        output_path = AUDIO_DIR / output_filename
        
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            search_result['download_success'] = True
            search_result['file_size_mb'] = round(file_size_mb, 2)
            search_result['time_taken'] = round(time.time() - start_time, 2)
            return search_result
        
        download_result = download_audio(video_id, output_path)
        
        if download_result['success']:
            search_result['download_success'] = True
            search_result['file_size_mb'] = download_result['file_size_mb']
        else:
            search_result['error_msg'] = download_result.get('error', 'Download failed')
    
    except Exception as e:
        search_result['error_msg'] = f'Download error: {str(e)[:100]}'
    
    finally:
        # Add download time to total
        search_result['time_taken'] = round(search_result.get('time_taken', 0) + (time.time() - start_time), 2)
    
    return search_result


def process_batch(batch: List[tuple], stats: Dict, pbar: tqdm, 
                  search_workers: int = SEARCH_WORKERS, 
                  download_workers: int = DOWNLOAD_WORKERS) -> int:
    """
    Process a batch of songs using producer-consumer pattern.
    
    Phase 1: Parallel search (8 workers) - populates download queue
    Phase 2: Parallel download (4 workers) - processes download queue
    
    Returns the last processed row index.
    """
    last_row = batch[0][0] if batch else 0
    
    # Phase 1: Parallel search
    search_results = []
    with ThreadPoolExecutor(max_workers=search_workers) as executor:
        futures = {executor.submit(process_song_search, row_idx, csv_row): (row_idx, csv_row) 
                   for row_idx, csv_row in batch}
        
        for future in as_completed(futures):
            result = future.result()
            search_results.append(result)
    
    # Separate into: downloadable, already exists, and failed
    to_download = [r for r in search_results if r.get('ready_for_download')]
    already_exists = [r for r in search_results if not r.get('ready_for_download') and r.get('download_success')]
    failed = [r for r in search_results if not r.get('ready_for_download') and not r.get('download_success')]
    
    # Count already exists as success (no YouTube search needed, do not log again)
    for result in already_exists:
        stats['completed'] += 1
        stats['successful'] += 1
        pbar.update(1)
    
    # Log failed searches
    for result in failed:
        log_result(result['row_idx'], result['csv_row'], result)
        stats['completed'] += 1
        stats['failed'] += 1
        
        conf_level = result.get('confidence_level', '')
        if conf_level == 'high':
            stats['high_conf'] += 1
        elif conf_level == 'medium':
            stats['medium_conf'] += 1
        elif conf_level == 'low':
            stats['low_conf'] += 1
        
        pbar.update(1)
    
    # Phase 2: Parallel download (bandwidth limited)
    if to_download:
        with ThreadPoolExecutor(max_workers=download_workers) as executor:
            futures = {executor.submit(process_song_download, result): result for result in to_download}
            
            for future in as_completed(futures):
                result = future.result()
                log_result(result['row_idx'], result['csv_row'], result)
                
                stats['completed'] += 1
                stats['total_time'] += result.get('time_taken', 0)
                
                if result['download_success']:
                    stats['successful'] += 1
                else:
                    stats['failed'] += 1
                
                conf_level = result.get('confidence_level', '')
                if conf_level == 'high':
                    stats['high_conf'] += 1
                elif conf_level == 'medium':
                    stats['medium_conf'] += 1
                elif conf_level == 'low':
                    stats['low_conf'] += 1
                
                pbar.update(1)
    
    # Update progress bar stats
    if stats['completed'] > 0:
        success_rate = (stats['successful'] / stats['completed']) * 100
        avg_time = stats['total_time'] / stats['completed'] if stats['total_time'] > 0 else 0
        pbar.set_postfix({
            'success': f"{success_rate:.1f}%",
            'avg_time': f"{avg_time:.1f}s"
        })
    
    # Return last processed row
    return max(r['row_idx'] for r in search_results) if search_results else last_row


def main():
    parser = argparse.ArgumentParser(description='Download pilot: concurrent YouTube download pipeline')
    parser.add_argument('--start-row', type=int, default=None, help='Starting row index (overrides checkpoint if set)')
    parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT, help='Number of songs to process')
    parser.add_argument('--no-resume', action='store_true', help='Ignore checkpoint and start fresh')
    parser.add_argument('--retry-file', type=str, default=None, help='Path to file with row indices to retry')
    parser.add_argument('--workers', type=int, default=SEARCH_WORKERS, help='Search workers (default: 8)')
    parser.add_argument('--dl-workers', type=int, default=DOWNLOAD_WORKERS, help='Download workers (default: 4)')
    args = parser.parse_args()
    
    # Update worker counts from args
    search_workers = args.workers
    download_workers = args.dl_workers
    
    ensure_directories()
    init_log_file()
    
    # Load retry rows if requested
    retry_rows = set()
    if args.retry_file:
        retry_path = Path(args.retry_file)
        if retry_path.exists():
            with open(retry_path, 'r') as f:
                retry_rows = {int(line.strip()) for line in f if line.strip().isdigit()}
            print(f"Retry mode: Loaded {len(retry_rows)} rows from {args.retry_file}")
        else:
            print(f"ERROR: Retry file not found: {args.retry_file}")
            sys.exit(1)
    
    is_retry_mode = len(retry_rows) > 0
    
    # Determine starting row
    if is_retry_mode:
        # Retry mode: checkpoint and start-row are ignored for collection, but stats are shown
        start_row = 0
        checkpoint = load_checkpoint()
    elif args.no_resume:
        # Explicit --no-resume: start from 0 or user's --start-row
        start_row = args.start_row if args.start_row is not None else 0
        checkpoint = {'last_row': -1, 'completed': 0, 'successful': 0, 'failed': 0}
    elif args.start_row is not None:
        # User explicitly set --start-row: use it (ignore checkpoint)
        start_row = args.start_row
        checkpoint = load_checkpoint()  # Load for stats display
    else:
        # Default behavior: resume from checkpoint
        checkpoint = load_checkpoint()
        start_row = checkpoint['last_row'] + 1
    
    if is_retry_mode:
        print(f"Audio Download Pilot (Optimized) - RETRY MODE")
        print(f"Retrying {len(retry_rows)} specific songs from {args.retry_file}")
    else:
        print(f"Audio Download Pilot (Optimized) - Starting from row {start_row}")
        print(f"Target: {args.limit} songs")
        
    print(f"Concurrency: {search_workers} search workers, {download_workers} download workers")
    print(f"Output: {AUDIO_DIR}")
    print(f"Log: {LOG_FILE}")
    print(f"Confidence threshold: ≥{CONFIDENCE_THRESHOLD}")
    print("=" * 60)
    
    # Statistics
    stats = {
        'completed': checkpoint.get('completed', 0) if not is_retry_mode else 0,
        'successful': checkpoint.get('successful', 0) if not is_retry_mode else 0,
        'failed': checkpoint.get('failed', 0) if not is_retry_mode else 0,
        'high_conf': 0,
        'medium_conf': 0,
        'low_conf': 0,
        'total_time': 0
    }
    
    pipeline_start = time.time()
    last_processed_row = start_row - 1
    
    # Read CSV and process in batches
    try:
        with open(SONGS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Collect rows for batch processing
            all_rows = []
            
            if is_retry_mode:
                max_retry = max(retry_rows)
                for row_idx, csv_row in enumerate(reader):
                    if row_idx in retry_rows:
                        all_rows.append((row_idx, csv_row))
                    if row_idx >= max_retry:
                        break
            else:
                # Skip to start row
                for _ in range(start_row):
                    next(reader, None)
                
                end_row = start_row + args.limit
                for row_idx in range(start_row, end_row):
                    try:
                        csv_row = next(reader)
                        all_rows.append((row_idx, csv_row))
                    except StopIteration:
                        break
            
            if not all_rows:
                print("No rows to process")
                return
            
            # Process in batches with progress bar
            total_songs = len(all_rows)
            with tqdm(total=total_songs, desc="Downloading", unit="song") as pbar:
                for batch_start in range(0, total_songs, BATCH_SIZE):
                    batch = all_rows[batch_start:batch_start + BATCH_SIZE]
                    
                    last_processed_row = process_batch(batch, stats, pbar, search_workers, download_workers)
                    
                    # Checkpoint after each batch (SKIP in retry mode to avoid rewinding progress)
                    if not is_retry_mode:
                        save_checkpoint(last_processed_row, stats)
                    
                    # Small delay between batches to avoid rate limiting
                    if batch_start + BATCH_SIZE < total_songs:
                        time.sleep(BATCH_DELAY)
    
    except FileNotFoundError:
        print(f"\nERROR: CSV file not found: {SONGS_CSV}")
        print("Please ensure data/processed/songs.csv exists.")
        sys.exit(1)
    except PermissionError:
        print(f"\nERROR: Permission denied reading: {SONGS_CSV}")
        sys.exit(1)
    except UnicodeDecodeError as e:
        print(f"\nERROR: Encoding error in CSV file: {e}")
        sys.exit(1)
    
    # Final checkpoint
    if not is_retry_mode:
        save_checkpoint(last_processed_row, stats)
    
    # Print summary
    elapsed = time.time() - pipeline_start
    
    print("\n" + "=" * 60)
    print("PILOT COMPLETE")
    print("=" * 60)
    
    if stats['completed'] > 0:
        success_rate = (stats['successful'] / stats['completed']) * 100
        fail_rate = (stats['failed'] / stats['completed']) * 100
        print(f"Total processed: {stats['completed']}")
        print(f"Successful: {stats['successful']} ({success_rate:.1f}%)")
        print(f"Failed: {stats['failed']} ({fail_rate:.1f}%)")
        
        print(f"\nConfidence distribution:")
        print(f"  High (≥80): {stats['high_conf']}")
        print(f"  Medium (60-79): {stats['medium_conf']}")
        print(f"  Low (<60): {stats['low_conf']}")
        
        print(f"\nTiming:")
        print(f"  Total time: {seconds_to_readable(elapsed)}")
        avg_per_song = elapsed / stats['completed']
        print(f"  Avg per song: {avg_per_song:.2f}s")
        
        # Projection for full dataset
        remaining = 550000 - stats['completed']
        projected_time = remaining * avg_per_song
        print(f"\nProjection for 550K songs:")
        print(f"  Remaining: {remaining:,} songs")
        print(f"  Estimated time: {seconds_to_readable(projected_time)}")
    else:
        print("No songs were processed.")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
