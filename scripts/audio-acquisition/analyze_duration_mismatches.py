#!/usr/bin/env python3
"""
Duration Mismatch Analyzer

Analyzes download_log_pilot.csv to find songs that were successfully downloaded
but have duration mismatches according to the new stricter validation rules.

The new rule rejects if: diff > max(60 seconds, 30% of expected duration)

Usage:
    python analyze_duration_mismatches.py
    python analyze_duration_mismatches.py --delete-mismatched  # Delete bad files
"""
import csv
import sys
from pathlib import Path
from typing import Dict, List

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_FILE = PROJECT_ROOT / "data" / "logs" / "download_log_pilot.csv"
AUDIO_DIR = PROJECT_ROOT / "data" / "audio" / "pilot"


def check_duration_mismatch(expected_ms: int, actual_sec: float) -> Dict:
    """
    Check if duration would be rejected by new stricter rules.
    
    Args:
        expected_ms: Expected duration in milliseconds
        actual_sec: Actual YouTube duration in seconds
        
    Returns:
        Dict with 'reject' (bool), 'diff' (float), 'max_allowed' (float)
    """
    expected_sec = expected_ms / 1000.0
    diff = abs(expected_sec - actual_sec)
    max_allowed = max(60, expected_sec * 0.30)
    reject = diff > max_allowed
    
    return {
        'reject': reject,
        'diff': diff,
        'max_allowed': max_allowed,
        'percent_diff': (diff / expected_sec * 100) if expected_sec > 0 else 0
    }


def analyze_log() -> Dict:
    """Analyze download log for duration mismatches."""
    
    if not LOG_FILE.exists():
        print(f"ERROR: Log file not found: {LOG_FILE}")
        sys.exit(1)
    
    stats = {
        'total_downloaded': 0,
        'mismatched': 0,
        'correct': 0,
        'missing_data': 0
    }
    
    mismatched_songs = []
    
    print(f"Analyzing: {LOG_FILE}")
    print("=" * 80)
    
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Only check successfully downloaded songs
            download_success = row.get('download_success', '').strip().lower()
            if download_success != 'true':
                continue
            
            stats['total_downloaded'] += 1
            
            # Get durations
            try:
                duration_ms = int(row.get('duration_ms', 0))
                youtube_duration = float(row.get('youtube_duration', 0))
            except (ValueError, TypeError):
                stats['missing_data'] += 1
                continue
            
            if duration_ms <= 0 or youtube_duration <= 0:
                stats['missing_data'] += 1
                continue
            
            # Check if it would be rejected
            check = check_duration_mismatch(duration_ms, youtube_duration)
            
            if check['reject']:
                stats['mismatched'] += 1
                
                mismatched_songs.append({
                    'row_idx': row.get('row_idx', '?'),
                    'song_id': row.get('song_id', '?'),
                    'song_name': row.get('song_name', '?'),
                    'expected_sec': duration_ms / 1000.0,
                    'actual_sec': youtube_duration,
                    'diff': check['diff'],
                    'percent_diff': check['percent_diff'],
                    'max_allowed': check['max_allowed'],
                    'youtube_title': row.get('youtube_title', '?'),
                    'confidence_score': row.get('confidence_score', '?')
                })
            else:
                stats['correct'] += 1
    
    return {
        'stats': stats,
        'mismatched_songs': mismatched_songs
    }


def print_report(results: Dict):
    """Print formatted report."""
    
    stats = results['stats']
    mismatched = results['mismatched_songs']
    
    # Overall Statistics
    print("\n📊 OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total downloaded songs:     {stats['total_downloaded']:,}")
    print(f"Correct duration matches:   {stats['correct']:,} ({stats['correct']/stats['total_downloaded']*100:.1f}%)")
    print(f"Duration MISMATCHES:        {stats['mismatched']:,} ({stats['mismatched']/stats['total_downloaded']*100:.1f}%)")
    print(f"Missing duration data:      {stats['missing_data']:,}")
    
    # Mismatch severity breakdown
    if mismatched:
        print("\n📈 MISMATCH SEVERITY BREAKDOWN")
        print("=" * 80)
        
        extreme = [m for m in mismatched if m['percent_diff'] > 50]
        high = [m for m in mismatched if 30 < m['percent_diff'] <= 50]
        moderate = [m for m in mismatched if m['percent_diff'] <= 30]
        
        print(f"Extreme (>50% off):         {len(extreme):,} songs")
        print(f"High (30-50% off):          {len(high):,} songs")
        print(f"Moderate (just over 30%):   {len(moderate):,} songs")
        
        # Top 20 worst mismatches
        print("\n🔴 TOP 20 WORST MISMATCHES")
        print("=" * 80)
        sorted_mismatches = sorted(mismatched, key=lambda x: x['diff'], reverse=True)
        
        print(f"{'Row':<8} {'Expected':<10} {'Actual':<10} {'Diff':<10} {'%Diff':<8} {'Song Name'}")
        print("-" * 80)
        
        for song in sorted_mismatches[:20]:
            print(f"{song['row_idx']:<8} "
                  f"{song['expected_sec']:<10.1f} "
                  f"{song['actual_sec']:<10.1f} "
                  f"{song['diff']:<10.1f} "
                  f"{song['percent_diff']:<8.1f} "
                  f"{song['song_name'][:40]}")
        
        # File paths for deletion
        print("\n📁 FILES WITH DURATION MISMATCHES")
        print("=" * 80)
        
        files_exist = 0
        files_missing = 0
        
        for song in mismatched:
            filename = f"{int(song['row_idx']):06d}_opus.webm"
            filepath = AUDIO_DIR / filename
            if filepath.exists():
                files_exist += 1
            else:
                files_missing += 1
        
        print(f"Files that exist:           {files_exist:,}")
        print(f"Files already deleted:      {files_missing:,}")
        
        # Export list
        export_file = PROJECT_ROOT / "data" / "logs" / "duration_mismatches.txt"
        with open(export_file, 'w') as f:
            for song in mismatched:
                f.write(f"{song['row_idx']}\n")
        
        print(f"\n✅ Mismatched row indices saved to: {export_file}")
    
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)
    
    if stats['mismatched'] > 0:
        print(f"⚠️  Found {stats['mismatched']:,} songs with incorrect durations")
        print(f"   These songs were downloaded with old lenient rules")
        print(f"   New stricter rules would reject them")
        print()
        print("Options:")
        print("1. Delete mismatched files and re-download with strict rules")
        print("   python analyze_duration_mismatches.py --delete-mismatched")
        print()
        print("2. Keep existing files (may have wrong songs)")
        print()
        print("3. Manual review - check top 20 worst cases above")
    else:
        print("✅ All downloaded songs have correct durations!")


def delete_mismatched_files(mismatched_songs: List[Dict]) -> Dict:
    """Delete audio files with duration mismatches and remove from log."""
    
    deleted = 0
    not_found = 0
    
    print("\n🗑️  DELETING MISMATCHED FILES")
    print("=" * 80)
    
    # Collect ALL row indices to remove (whether file exists or not)
    rows_to_remove = set()
    
    for song in mismatched_songs:
        filename = f"{int(song['row_idx']):06d}_opus.webm"
        filepath = AUDIO_DIR / filename
        
        # Always add to removal list
        rows_to_remove.add(song['row_idx'])
        
        if filepath.exists():
            filepath.unlink()
            deleted += 1
            print(f"Deleted: {filename} (diff: {song['diff']:.1f}s, {song['percent_diff']:.1f}%)")
        else:
            not_found += 1
            print(f"Already gone: {filename}")
    
    print(f"\n✅ Deleted: {deleted:,} files")
    print(f"⚠️  Already deleted: {not_found:,} files")
    
    # Remove rows from log file
    if rows_to_remove:
        print("\n📝 UPDATING LOG FILE")
        print("=" * 80)
        
        # Create backup
        backup_file = LOG_FILE.parent / f"{LOG_FILE.stem}_backup.csv"
        import shutil
        shutil.copy2(LOG_FILE, backup_file)
        print(f"Backup created: {backup_file}")
        
        # Read all rows except the ones to remove
        kept_rows = []
        removed_count = 0
        
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            for row in reader:
                if row['row_idx'] not in rows_to_remove:
                    kept_rows.append(row)
                else:
                    removed_count += 1
        
        # Write back the filtered rows
        with open(LOG_FILE, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)
        
        print(f"✅ Removed {removed_count:,} rows from log file")
        print(f"✅ Kept {len(kept_rows):,} rows in log file")
    
    return {'deleted': deleted, 'not_found': not_found, 'log_rows_removed': len(rows_to_remove)}


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze duration mismatches in downloaded songs')
    parser.add_argument('--delete-mismatched', action='store_true', 
                       help='Delete files with duration mismatches')
    
    args = parser.parse_args()
    
    # Analyze log
    results = analyze_log()
    
    # Print report
    print_report(results)
    
    # Delete if requested
    if args.delete_mismatched:
        if results['mismatched_songs']:
            confirm = input(f"\n⚠️  Delete {len(results['mismatched_songs']):,} files? (yes/no): ")
            if confirm.lower() == 'yes':
                delete_mismatched_files(results['mismatched_songs'])
            else:
                print("Cancelled.")
        else:
            print("\nNo mismatched files to delete.")


if __name__ == '__main__':
    main()
