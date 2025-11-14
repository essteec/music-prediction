"""
Investigate loudness, tempo, key, and mode outliers/inconsistencies
"""
import pandas as pd
import numpy as np

print("=" * 80)
print("INVESTIGATING LOUDNESS, TEMPO, KEY & MODE OUTLIERS")
print("=" * 80)

# Read the data
filepath = 'dataset/raw/songs_enhanced_full.csv'
print(f"\nReading: {filepath}")
print("This may take a moment...\n")

df = pd.read_csv(filepath)

# LOUDNESS INVESTIGATION
print("-" * 80)
print("LOUDNESS OUTLIERS (Expected: -60 to 0 dB)")
print("-" * 80)

loudness_outliers = df[(df['loudness'] < -60) | (df['loudness'] > 0)]
print(f"Total outliers: {len(loudness_outliers)}")

if len(loudness_outliers) > 0:
    print(f"\nLoudness statistics for outliers:")
    print(f"  Min:    {loudness_outliers['loudness'].min():.2f} dB")
    print(f"  Max:    {loudness_outliers['loudness'].max():.2f} dB")
    print(f"  Mean:   {loudness_outliers['loudness'].mean():.2f} dB")
    print(f"  Median: {loudness_outliers['loudness'].median():.2f} dB")
    
    print(f"\nBreakdown:")
    print(f"  Below -60: {len(loudness_outliers[loudness_outliers['loudness'] < -60])}")
    print(f"  Above 0:   {len(loudness_outliers[loudness_outliers['loudness'] > 0])}")
    
    print(f"\nSample of outlier values (first 20):")
    print(loudness_outliers['loudness'].head(20).tolist())
    
    # Check how far above 0
    above_zero = loudness_outliers[loudness_outliers['loudness'] > 0]
    if len(above_zero) > 0:
        print(f"\nValues above 0 dB:")
        print(f"  Min:  {above_zero['loudness'].min():.4f} dB")
        print(f"  Max:  {above_zero['loudness'].max():.4f} dB")
        print(f"  Mean: {above_zero['loudness'].mean():.4f} dB")

# TEMPO INVESTIGATION
print("\n" + "-" * 80)
print("TEMPO OUTLIERS (Expected: 20 to 300 BPM)")
print("-" * 80)

tempo_outliers = df[(df['tempo'] < 20) | (df['tempo'] > 300)]
print(f"Total outliers: {len(tempo_outliers)}")

if len(tempo_outliers) > 0:
    print(f"\nTempo statistics for outliers:")
    print(f"  Min:    {tempo_outliers['tempo'].min():.2f} BPM")
    print(f"  Max:    {tempo_outliers['tempo'].max():.2f} BPM")
    print(f"  Mean:   {tempo_outliers['tempo'].mean():.2f} BPM")
    print(f"  Median: {tempo_outliers['tempo'].median():.2f} BPM")
    
    print(f"\nBreakdown:")
    print(f"  Below 20:  {len(tempo_outliers[tempo_outliers['tempo'] < 20])}")
    print(f"  Above 300: {len(tempo_outliers[tempo_outliers['tempo'] > 300])}")
    
    print(f"\nSample of outlier values (first 20):")
    print(tempo_outliers['tempo'].head(20).tolist())
    
    # Check distribution above 300
    above_300 = tempo_outliers[tempo_outliers['tempo'] > 300]
    if len(above_300) > 0:
        print(f"\nValues above 300 BPM:")
        print(f"  Min:  {above_300['tempo'].min():.2f} BPM")
        print(f"  Max:  {above_300['tempo'].max():.2f} BPM")
        print(f"  Mean: {above_300['tempo'].mean():.2f} BPM")

# NORMAL DATA FOR COMPARISON
print("\n" + "-" * 80)
print("NORMAL VALUES FOR COMPARISON")
print("-" * 80)

normal_loudness = df[(df['loudness'] >= -60) & (df['loudness'] <= 0)]
print(f"\nNormal Loudness ({len(normal_loudness)} rows):")
print(f"  Min:    {normal_loudness['loudness'].min():.2f} dB")
print(f"  Max:    {normal_loudness['loudness'].max():.2f} dB")
print(f"  Mean:   {normal_loudness['loudness'].mean():.2f} dB")
print(f"  Median: {normal_loudness['loudness'].median():.2f} dB")

normal_tempo = df[(df['tempo'] >= 20) & (df['tempo'] <= 300)]
print(f"\nNormal Tempo ({len(normal_tempo)} rows):")
print(f"  Min:    {normal_tempo['tempo'].min():.2f} BPM")
print(f"  Max:    {normal_tempo['tempo'].max():.2f} BPM")
print(f"  Mean:   {normal_tempo['tempo'].mean():.2f} BPM")
print(f"  Median: {normal_tempo['tempo'].median():.2f} BPM")

# Normal key values
key_numeric = pd.to_numeric(df['key'], errors='coerce')
normal_key = key_numeric[(key_numeric >= -1) & (key_numeric <= 11)]
print(f"\nNormal Key ({len(normal_key)} rows, numeric -1 to 11):")
print(f"  Unique values: {sorted(normal_key.dropna().unique())}")
print(f"  Distribution:")
print(normal_key.value_counts().sort_index())

# Normal mode values
mode_numeric = pd.to_numeric(df['mode'], errors='coerce')
normal_mode = mode_numeric[(mode_numeric >= 0) & (mode_numeric <= 1)]
print(f"\nNormal Mode ({len(normal_mode)} rows, numeric 0-1):")
print(f"  Unique values: {sorted(normal_mode.dropna().unique())}")
print(f"  Distribution:")
print(normal_mode.value_counts().sort_index())

# KEY INVESTIGATION
print("\n" + "-" * 80)
print("KEY ENCODING INVESTIGATION (Expected: -1 to 11, all numeric)")
print("-" * 80)

print(f"\nKey column dtype: {df['key'].dtype}")
print(f"Total rows: {len(df)}")

# Check for non-numeric values
key_series = df['key'].astype(str)
non_numeric_key = ~key_series.str.match(r'^-?\d+\.?\d*$', na=False)
non_numeric_key_rows = df[non_numeric_key]

print(f"\nNon-numeric key values: {len(non_numeric_key_rows)}")
if len(non_numeric_key_rows) > 0:
    print(f"\nUnique non-numeric key values:")
    print(non_numeric_key_rows['key'].value_counts())

# Convert to numeric to check range
key_numeric = pd.to_numeric(df['key'], errors='coerce')
key_outliers = key_numeric[(key_numeric < -1) | (key_numeric > 11)]
print(f"\nNumeric key outliers (outside -1 to 11): {len(key_outliers)}")

if len(key_outliers) > 0:
    print(f"\nOutlier values:")
    print(key_outliers.value_counts().head(20))

# MODE INVESTIGATION
print("\n" + "-" * 80)
print("MODE ENCODING INVESTIGATION (Expected: 0 or 1, all numeric)")
print("-" * 80)

print(f"\nMode column dtype: {df['mode'].dtype}")

# Check for non-numeric values
mode_series = df['mode'].astype(str)
non_numeric_mode = ~mode_series.str.match(r'^-?\d+\.?\d*$', na=False)
non_numeric_mode_rows = df[non_numeric_mode]

print(f"\nNon-numeric mode values: {len(non_numeric_mode_rows)}")
if len(non_numeric_mode_rows) > 0:
    print(f"\nUnique non-numeric mode values:")
    print(non_numeric_mode_rows['mode'].value_counts())

# Convert to numeric to check range
mode_numeric = pd.to_numeric(df['mode'], errors='coerce')
mode_outliers = mode_numeric[(mode_numeric < 0) | (mode_numeric > 1)]
print(f"\nNumeric mode outliers (outside 0 to 1): {len(mode_outliers)}")

if len(mode_outliers) > 0:
    print(f"\nOutlier values:")
    print(mode_outliers.value_counts().head(20))

print("\n" + "=" * 80)
print("INVESTIGATION COMPLETE")
print("=" * 80)
