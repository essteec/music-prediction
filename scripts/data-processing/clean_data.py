"""
Data Cleaning Script
Based on validation results and user decisions

Actions:
1. Remove songs with tempo = 0 BPM (277 rows)
2. Clip loudness 0-1 dB to 0 dB
3. Remove songs with loudness > 1 dB
4. Remove songs with invalid year (3 rows)
5. Standardize key/mode encoding (43,893 rows have mixed formats)

Total expected removals: ~280 rows
Final dataset: ~733,114 rows
"""

import pandas as pd
from pathlib import Path

print("=" * 80)
print("DATA CLEANING SCRIPT")
print("=" * 80)

# Paths
input_file = Path('../../data/raw/songs_enhanced_full.csv')
output_file = Path('../../data/processed/songs_ml_ready.csv')
output_file.parent.mkdir(exist_ok=True)

# Read data
print(f"\nReading: {input_file}")
df = pd.read_csv(input_file)
print(f"Initial rows: {len(df):,}")

# Track what we remove
removed_log = []

# STEP 1: Remove tempo = 0
print("\n" + "-" * 80)
print("STEP 1: Remove songs with tempo = 0 BPM")
print("-" * 80)
tempo_zero = df['tempo'] == 0
removed_tempo = len(df[tempo_zero])
print(f"Rows with tempo = 0: {removed_tempo}")

df = df[~tempo_zero]
print(f"After removal: {len(df):,} rows")
removed_log.append(f"Tempo = 0: {removed_tempo} rows removed")

# STEP 2: Fix loudness
print("\n" + "-" * 80)
print("STEP 2: Fix loudness issues")
print("-" * 80)

# 2a: Clip 0 < loudness <= 1 to 0
clip_mask = (df['loudness'] > 0) & (df['loudness'] <= 1)
clipped_count = clip_mask.sum()
print(f"Rows with 0 < loudness <= 1 dB: {clipped_count}")
df.loc[clip_mask, 'loudness'] = 0.0
print(f"Clipped to 0.0 dB: {clipped_count} rows")
removed_log.append(f"Loudness clipped to 0: {clipped_count} rows")

# 2b: Remove loudness > 1
remove_mask = df['loudness'] > 1
removed_loud = remove_mask.sum()
print(f"Rows with loudness > 1 dB: {removed_loud}")
df = df[~remove_mask]
print(f"After removal: {len(df):,} rows")
removed_log.append(f"Loudness > 1 dB: {removed_loud} rows removed")

# STEP 3: Remove invalid years
print("\n" + "-" * 80)
print("STEP 3: Remove invalid years")
print("-" * 80)

year_invalid = (df['year'] == 0) | (df['year'] < 1900) | (df['year'] > 2025)
removed_year = year_invalid.sum()
print(f"Rows with invalid year: {removed_year}")
df = df[~year_invalid]
print(f"After removal: {len(df):,} rows")
removed_log.append(f"Invalid year: {removed_year} rows removed")

# STEP 4: Standardize key/mode encoding
print("\n" + "-" * 80)
print("STEP 4: Standardize key and mode encoding")
print("-" * 80)

# Key: Convert letter notation to numeric (0-11)
key_mapping = {
    'C': 0, 'C#': 1, 'Db': 1,
    'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4,
    'F': 5, 'F#': 6, 'Gb': 6,
    'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10,
    'B': 11
}

# Check current encoding
letter_keys = df['key'].astype(str).str.contains('[A-G]', regex=True, na=False)
text_modes = df['mode'].astype(str).str.contains('Major|Minor', regex=True, case=False, na=False)
print(f"Key - Letter format: {letter_keys.sum():,} rows")
print(f"Mode - Text format: {text_modes.sum():,} rows")

# Convert key: letter → numeric
def convert_key(value):
    """Convert key to numeric format (0-11, or -1 for unknown)"""
    if pd.isna(value):
        return -1
    
    # If already numeric, return as int
    try:
        num_val = float(value)
        if num_val == num_val:  # Check not NaN
            return int(num_val)
    except (ValueError, TypeError):
        pass
    
    # Convert letter notation
    value_str = str(value).strip()
    if value_str in key_mapping:
        return key_mapping[value_str]
    
    # Unknown/invalid
    return -1

# Convert mode: text → numeric
def convert_mode(value):
    """Convert mode to numeric format (0=minor, 1=major, -1=unknown)"""
    if pd.isna(value):
        return -1
    
    # If already numeric, return as int
    try:
        num_val = float(value)
        if num_val == num_val:  # Check not NaN
            return int(num_val)
    except (ValueError, TypeError):
        pass
    
    # Convert text notation
    value_str = str(value).strip().lower()
    if value_str in ['major', '1', '1.0']:
        return 1
    elif value_str in ['minor', '0', '0.0']:
        return 0
    
    # Unknown/invalid
    return -1

# Apply conversions
print("Converting key encoding...")
df['key'] = df['key'].apply(convert_key)
print("Converting mode encoding...")
df['mode'] = df['mode'].apply(convert_mode)

# Verify conversion
letter_keys_after = df['key'].astype(str).str.contains('[A-G]', regex=True, na=False)
text_modes_after = df['mode'].astype(str).str.contains('Major|Minor', regex=True, case=False, na=False)
print(f"After conversion - Letter keys: {letter_keys_after.sum()} (should be 0)")
print(f"After conversion - Text modes: {text_modes_after.sum()} (should be 0)")

standardized_count = letter_keys.sum() + text_modes.sum()
removed_log.append(f"Key/mode standardized: {standardized_count} rows converted to numeric")

# FINAL STATISTICS
print("\n" + "=" * 80)
print("CLEANING COMPLETE")
print("=" * 80)
print(f"\nFinal dataset size: {len(df):,} rows")
print(f"Total rows removed: {removed_tempo + removed_loud + removed_year}")
print(f"Data retention: {len(df)/733394*100:.2f}%")

print("\nCleaning summary:")
for log in removed_log:
    print(f"  - {log}")

# Save cleaned data
print(f"\nSaving to: {output_file}")
df.to_csv(output_file, index=False)
print("✅ Saved successfully!")

# Verification
print("\n" + "-" * 80)
print("VERIFICATION")
print("-" * 80)
print(f"Tempo = 0: {(df['tempo'] == 0).sum()} (should be 0)")
print(f"Loudness > 1: {(df['loudness'] > 1).sum()} (should be 0)")
print(f"Invalid year: {((df['year'] == 0) | (df['year'] < 1900) | (df['year'] > 2025)).sum()} (should be 0)")
print(f"Loudness range: [{df['loudness'].min():.2f}, {df['loudness'].max():.2f}] (should be [-60, 0])")
print(f"Tempo range: [{df['tempo'].min():.2f}, {df['tempo'].max():.2f}] (should be [20, 300])")
print(f"Year range: [{df['year'].min()}, {df['year'].max()}] (should be [1900, 2025])")
print(f"Key range: [{df['key'].min()}, {df['key'].max()}] (should be [-1, 11])")
print(f"Mode range: [{df['mode'].min()}, {df['mode'].max()}] (should be [-1, 1])")
print(f"Key encoding: {df['key'].dtype} (should be int)")
print(f"Mode encoding: {df['mode'].dtype} (should be int)")

print("\n" + "=" * 80)
print("✅ ALL DONE!")
print("=" * 80)
print(f"\nCleaned data saved to: {output_file}")
print("Next step: Run validation again to confirm!")
