#!/usr/bin/env python3
"""
Merge artist data from multiple CSV files (artists.csv, artists_v2.csv, artists_v3.csv, artists_v4.csv).

This script:
1. Reads all artist CSV files
2. Combines them into one DataFrame
3. Removes duplicates based on spotify_id (keeping first occurrence)
4. Saves the merged result back to artists.csv (with backup)

Usage:
    python merge_artist_files.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'

INPUT_FILES = [
    DATA_DIR / 'artists.csv',
    DATA_DIR / 'artists_v2.csv',
    DATA_DIR / 'artists_v3.csv',
    DATA_DIR / 'artists_v4.csv',
]

OUTPUT_FILE = DATA_DIR / 'artists_merged.csv'
BACKUP_FILE = DATA_DIR / 'artists_backup.csv'
DUPLICATES_LOG_FILE = DATA_DIR / 'artists_merge_duplicates.log'

# Expected columns
EXPECTED_COLUMNS = ['searched_name', 'spotify_id', 'spotify_name', 'popularity', 'followers', 'genres', 'type']


def merge_duplicate_rows(group):
    """
    Intelligently merge duplicate rows for the same spotify_id.
    - followers/popularity: take max
    - searched_name: merge with semicolons
    - genres: take union
    - spotify_name/type: take first non-null
    """
    merged = {}
    
    # spotify_id: use the common value
    merged['spotify_id'] = group['spotify_id'].iloc[0]
    
    # searched_name: merge all unique names with semicolons
    unique_names = group['searched_name'].dropna().unique()
    merged['searched_name'] = ';'.join(sorted(unique_names))
    
    # spotify_name: take first non-null (should be same for same ID)
    merged['spotify_name'] = group['spotify_name'].dropna().iloc[0] if not group['spotify_name'].dropna().empty else None
    
    # popularity: take max
    merged['popularity'] = group['popularity'].max() if not group['popularity'].isna().all() else None
    
    # followers: take max
    merged['followers'] = group['followers'].max() if not group['followers'].isna().all() else None
    
    # genres: take union of all genre lists
    all_genres = set()
    for genres_str in group['genres'].dropna():
        try:
            genres_list = eval(genres_str) if isinstance(genres_str, str) else genres_str
            if isinstance(genres_list, list):
                all_genres.update(genres_list)
        except:
            pass
    merged['genres'] = json.dumps(sorted(list(all_genres)))
    
    # type: take first non-null (should be same)
    merged['type'] = group['type'].dropna().iloc[0] if not group['type'].dropna().empty else None
    
    return pd.Series(merged)


def log_duplicates(df, log_file):
    """
    Log detailed information about duplicate spotify_ids to a file.
    Groups duplicates and shows all columns for comparison.
    Only logs duplicates with differences (skips identical rows).
    """
    # Find rows with duplicate spotify_ids
    duplicate_mask = df['spotify_id'].duplicated(keep=False)
    duplicate_rows = df[duplicate_mask].copy()
    
    if len(duplicate_rows) == 0:
        return 0, 0
    
    # Sort by spotify_id so duplicates are grouped together
    duplicate_rows = duplicate_rows.sort_values('spotify_id')
    
    # Track spotify_name differences
    spotify_name_diffs = []
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DUPLICATE SPOTIFY_IDs LOG\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Group by spotify_id
        grouped = duplicate_rows.groupby('spotify_id', dropna=False)
        
        f.write(f"Total duplicate spotify_ids: {len(grouped)}\n")
        f.write(f"Total duplicate rows: {len(duplicate_rows)}\n\n")
        f.write("Note: Only duplicates with differences are logged below.\n")
        f.write("Identical duplicates are omitted (safe to remove).\n\n")
        f.write("=" * 80 + "\n\n")
        
        logged_count = 0
        identical_count = 0
        
        # Log each group of duplicates
        for spotify_id, group in grouped:
            # Check if all values are identical
            all_identical = len(group.drop_duplicates()) == 1
            
            if all_identical:
                identical_count += 1
                continue  # Skip logging identical duplicates
            
            logged_count += 1
            
            # Check for spotify_name differences (concerning)
            unique_spotify_names = group['spotify_name'].dropna().unique()
            if len(unique_spotify_names) > 1:
                spotify_name_diffs.append((spotify_id, list(unique_spotify_names)))
            
            f.write(f"Spotify ID: {spotify_id}\n")
            f.write(f"Number of occurrences: {len(group)}\n")
            f.write("-" * 80 + "\n")
            f.write("Status: DIFFERENCES FOUND (will be merged intelligently)\n\n")
            
            # Show each row with all columns
            for idx, (_, row) in enumerate(group.iterrows(), 1):
                f.write(f"  Occurrence {idx}:\n")
                for col in EXPECTED_COLUMNS:
                    value = row[col]
                    # Format value for better readability
                    if pd.isna(value):
                        value_str = "NULL"
                    elif col == 'genres':
                        # Pretty print genres
                        try:
                            genres_list = eval(value) if isinstance(value, str) else value
                            value_str = json.dumps(genres_list, ensure_ascii=False)
                        except:
                            value_str = str(value)
                    else:
                        value_str = str(value)
                    
                    f.write(f"    {col:20s}: {value_str}\n")
                f.write("\n")
            
            # Show differences
            f.write("  Differences:\n")
            for col in EXPECTED_COLUMNS:
                if col == 'spotify_id':
                    continue
                unique_values = group[col].dropna().unique()
                if len(unique_values) > 1:
                    f.write(f"    {col:20s}: {len(unique_values)} different values\n")
                    for val in unique_values:
                        if col == 'genres':
                            try:
                                val_str = json.dumps(eval(val) if isinstance(val, str) else val, ensure_ascii=False)
                            except:
                                val_str = str(val)
                        else:
                            val_str = str(val)
                        f.write(f"      - {val_str}\n")
            
            # Show merge strategy
            f.write("\n")
            
            f.write("=" * 80 + "\n\n")
        
        # Summary statistics
        f.write("\nSUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Duplicate IDs with identical data (not logged): {identical_count}\n")
        f.write(f"Duplicate IDs with differences (logged above): {logged_count}\n")
        f.write(f"\nAll duplicates merged intelligently using strategies above.\n")
    
    return logged_count, spotify_name_diffs


def main():
    print("=" * 60)
    print("Artist Data Merger")
    print("=" * 60)
    
    # Check which files exist
    existing_files = [f for f in INPUT_FILES if f.exists()]
    
    if not existing_files:
        print("ERROR: No artist CSV files found!")
        sys.exit(1)
    
    print(f"\nFound {len(existing_files)} files to merge:")
    for f in existing_files:
        print(f"  - {f.name}")
    
    # Read all files
    all_dfs = []
    total_rows = 0
    
    for file_path in existing_files:
        try:
            df = pd.read_csv(file_path)
            
            # Verify columns
            if list(df.columns) != EXPECTED_COLUMNS:
                print(f"\nWARNING: {file_path.name} has unexpected columns: {list(df.columns)}")
                print(f"Expected: {EXPECTED_COLUMNS}")
                continue
            
            rows = len(df)
            total_rows += rows
            all_dfs.append(df)
            print(f"  ✓ {file_path.name}: {rows:,} rows")
            
        except Exception as e:
            print(f"  ✗ {file_path.name}: Error reading file - {e}")
            continue
    
    if not all_dfs:
        print("\nERROR: No valid files to merge!")
        sys.exit(1)
    
    print(f"\nTotal rows before merge: {total_rows:,}")
    
    # Combine all DataFrames
    print("\nCombining DataFrames...")
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"Combined rows: {len(merged_df):,}")
    
    # Check for duplicates
    duplicates_before = merged_df['spotify_id'].duplicated().sum()
    print(f"Duplicate spotify_ids found: {duplicates_before:,}")
    
    if duplicates_before > 0:
        # Log detailed duplicate information to file
        print(f"\nLogging duplicate details to {DUPLICATES_LOG_FILE.name}...")
        try:
            logged_count, spotify_name_diffs = log_duplicates(merged_df, DUPLICATES_LOG_FILE)
            print(f"  ✓ Duplicate log saved successfully")
            print(f"  ✓ Logged {logged_count} duplicates with differences")
            print(f"  ✓ Skipped {len(merged_df[merged_df['spotify_id'].duplicated(keep=False)].groupby('spotify_id')) - logged_count} identical duplicates")
            
            # Report spotify_name differences to terminal (concerning)
            if spotify_name_diffs:
                print(f"\n⚠️  WARNING: {len(spotify_name_diffs)} spotify_ids have different spotify_names:")
                for spotify_id, names in spotify_name_diffs[:10]:  # Show first 10
                    print(f"     {spotify_id}: {names}")
                if len(spotify_name_diffs) > 10:
                    print(f"     ... and {len(spotify_name_diffs) - 10} more (see log file)")
        except Exception as e:
            print(f"  ✗ Failed to create duplicate log: {e}")
        
        # Show some examples of duplicates (brief)
        duplicate_ids = merged_df[merged_df['spotify_id'].duplicated(keep=False)]['spotify_id'].unique()[:5]
        print(f"\nExample duplicate spotify_ids: {list(duplicate_ids)}")
        
        # Merge duplicates intelligently
        print("\nMerging duplicates intelligently...")
        print("  - followers/popularity: taking max values")
        print("  - searched_names: merging with semicolons")
        print("  - genres: taking union of all genres")
        
        # Group by spotify_id and apply merge function
        duplicate_mask = merged_df['spotify_id'].duplicated(keep=False)
        duplicates = merged_df[duplicate_mask].groupby('spotify_id', as_index=False).apply(merge_duplicate_rows)
        non_duplicates = merged_df[~duplicate_mask]
        
        # Combine merged duplicates with non-duplicates
        merged_df = pd.concat([non_duplicates, duplicates], ignore_index=True)
        print(f"Rows after intelligent merge: {len(merged_df):,}")
    else:
        print("No duplicates found - skipping duplicate log file.")
    
    # Check for null spotify_ids
    null_ids = merged_df['spotify_id'].isnull().sum()
    if null_ids > 0:
        print(f"\nWARNING: Found {null_ids} rows with null spotify_id")
        print("These will be kept in the output (they represent failed lookups)")
    
    # Sort by searched_name for better readability
    print("\nSorting by searched_name...")
    merged_df = merged_df.sort_values('searched_name', na_position='last').reset_index(drop=True)
    
    # Backup existing artists.csv if it exists
    if OUTPUT_FILE.exists():
        print(f"\nBacking up existing {OUTPUT_FILE.name} to {BACKUP_FILE.name}...")
        try:
            # If backup already exists, add timestamp
            if BACKUP_FILE.exists():
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                timestamped_backup = DATA_DIR / f'artists_backup_{timestamp}.csv'
                OUTPUT_FILE.rename(timestamped_backup)
                print(f"  ✓ Existing file backed up to {timestamped_backup.name}")
            else:
                OUTPUT_FILE.rename(BACKUP_FILE)
                print(f"  ✓ Existing file backed up to {BACKUP_FILE.name}")
        except Exception as e:
            print(f"  ✗ Backup failed: {e}")
            print("  Continuing anyway...")
    
    # Save merged data
    print(f"\nSaving merged data to {OUTPUT_FILE.name}...")
    try:
        merged_df.to_csv(OUTPUT_FILE, index=False)
        print(f"  ✓ Successfully saved {len(merged_df):,} rows")
    except Exception as e:
        print(f"  ✗ Failed to save: {e}")
        sys.exit(1)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)
    print(f"Total unique artists: {len(merged_df):,}")
    print(f"Duplicates removed: {duplicates_before:,}")
    print(f"Artists with valid spotify_id: {merged_df['spotify_id'].notna().sum():,}")
    print(f"Artists with null spotify_id: {null_ids:,}")
    
    # Genre statistics
    non_empty_genres = merged_df['genres'].apply(lambda x: len(eval(x)) > 0 if pd.notna(x) and x != '[]' else False).sum()
    print(f"\nArtists with genres: {non_empty_genres:,}")
    
    # Follower statistics
    if merged_df['followers'].notna().any():
        print(f"\nFollower statistics:")
        print(f"  Min: {merged_df['followers'].min():,.0f}")
        print(f"  Max: {merged_df['followers'].max():,.0f}")
        print(f"  Mean: {merged_df['followers'].mean():,.0f}")
        print(f"  Median: {merged_df['followers'].median():,.0f}")
    
    print("\n✓ Merge completed successfully!")
    print(f"Output: {OUTPUT_FILE}")
    if duplicates_before > 0:
        print(f"Duplicate log: {DUPLICATES_LOG_FILE}")
    print("=" * 60)


if __name__ == '__main__':
    main()
