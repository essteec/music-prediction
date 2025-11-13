"""
Investigate loudness and tempo outliers
"""
import pandas as pd
import numpy as np

print("=" * 80)
print("INVESTIGATING LOUDNESS & TEMPO OUTLIERS")
print("=" * 80)

# Read the data
filepath = '../dataset/raw/songs_enhanced_full.csv'
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

print("\n" + "=" * 80)
print("INVESTIGATION COMPLETE")
print("=" * 80)
