#!/usr/bin/env python3
"""
Download Log Deduplicator

Removes duplicate rows from download_log_pilot.csv, keeping only the 
last occurrence of each row_idx. This is useful when retries or 
multiple runs have appended duplicate information for the same songs.

Usage:
    python deduplicate_log.py
"""
import csv
import shutil
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_FILE = PROJECT_ROOT / "data" / "logs" / "download_log_pilot.csv"

def deduplicate():
    if not LOG_FILE.exists():
        print(f"ERROR: Log file not found: {LOG_FILE}")
        return

    print(f"Reading: {LOG_FILE}")
    
    # Read all rows and keep the last occurrence of each row_idx
    rows_by_id = {}
    fieldnames = []
    total_count = 0
    
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            total_count += 1
            rows_by_id[row['row_idx']] = row
            
    unique_count = len(rows_by_id)
    duplicate_count = total_count - unique_count
    
    if duplicate_count == 0:
        print("✅ No duplicate rows found.")
        return

    print(f"Found {total_count} total rows.")
    print(f"Found {duplicate_count} duplicate rows.")
    print(f"Keeping {unique_count} unique (last occurrence) rows.")
    
    # Create backup
    backup_file = LOG_FILE.parent / f"{LOG_FILE.stem}_pre_dedup.csv"
    shutil.copy2(LOG_FILE, backup_file)
    print(f"Backup created: {backup_file}")
    
    # Sort by row_idx if they are numeric
    try:
        sorted_rows = sorted(rows_by_id.values(), key=lambda x: int(x['row_idx']))
    except ValueError:
        # Fallback to original order of first discovery if not numeric
        sorted_rows = list(rows_by_id.values())
        
    # Write back to file
    with open(LOG_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)
        
    print(f"✅ Successfully deduplicated and saved to: {LOG_FILE}")

if __name__ == '__main__':
    deduplicate()
