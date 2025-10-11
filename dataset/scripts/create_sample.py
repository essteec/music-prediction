#!/usr/bin/env python3
"""
Create a stratified sample of the dataset for scraping.
This ensures we have good representation across all 4 target variables.
"""
import pandas as pd
import numpy as np

def create_stratified_sample(input_file, output_file, sample_size=50000, random_state=42):
    """
    Create a stratified sample ensuring good distribution of target variables
    """
    print("=" * 70)
    print(f"CREATING STRATIFIED SAMPLE ({sample_size:,} songs)")
    print("=" * 70)
    
    # Load full dataset
    print("\n📂 Loading full dataset...")
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df):,} songs")
    
    # Create bins for stratification (for each target variable)
    print("\n🎯 Creating stratification bins...")
    df['valence_bin'] = pd.cut(df['valence'], bins=5, labels=['very_low', 'low', 'mid', 'high', 'very_high'])
    df['energy_bin'] = pd.cut(df['energy'], bins=5, labels=['very_low', 'low', 'mid', 'high', 'very_high'])
    df['danceability_bin'] = pd.cut(df['danceability'], bins=5, labels=['very_low', 'low', 'mid', 'high', 'very_high'])
    
    # Combine bins for stratification
    df['strata'] = (df['valence_bin'].astype(str) + '_' + 
                    df['energy_bin'].astype(str) + '_' + 
                    df['danceability_bin'].astype(str))
    
    # Sample proportionally from each stratum
    print(f"✓ Created {df['strata'].nunique()} unique strata")
    
    print(f"\n📊 Sampling {sample_size:,} songs...")
    sample_df = df.groupby('strata', group_keys=False).apply(
        lambda x: x.sample(min(len(x), max(1, int(sample_size * len(x) / len(df)))), 
                          random_state=random_state)
    )
    
    # If we didn't get enough, fill with random samples
    if len(sample_df) < sample_size:
        remaining = sample_size - len(sample_df)
        remaining_df = df[~df.index.isin(sample_df.index)].sample(remaining, random_state=random_state)
        sample_df = pd.concat([sample_df, remaining_df])
    
    # Drop stratification columns
    sample_df = sample_df.drop(columns=['valence_bin', 'energy_bin', 'danceability_bin', 'strata'])
    
    # Save
    sample_df.to_csv(output_file, index=False)
    print(f"✓ Saved {len(sample_df):,} songs to {output_file}")
    
    # Show statistics
    print(f"\n📈 SAMPLE STATISTICS:")
    print(f"  Sample size: {len(sample_df):,} ({len(sample_df)/len(df)*100:.1f}% of dataset)")
    print(f"\n  Target distributions:")
    for target in ['valence', 'energy', 'danceability']:
        orig_mean = df[target].mean()
        sample_mean = sample_df[target].mean()
        print(f"    {target.capitalize()}: {sample_mean:.3f} (original: {orig_mean:.3f})")
    
    # Estimate scraping time
    hours = (len(sample_df) * 7) / 3600
    print(f"\n⏱️  SCRAPING ESTIMATE:")
    print(f"  Time: ~{hours:.1f} hours ({hours/24:.1f} days)")
    
    print("\n" + "=" * 70)
    return sample_df

def create_smaller_test_sample(input_file, output_file, sample_size=1000, random_state=42):
    """Create a small test sample"""
    print(f"\nCreating small test sample ({sample_size} songs)...")
    df = pd.read_csv(input_file)
    sample = df.sample(sample_size, random_state=random_state)
    sample.to_csv(output_file, index=False)
    print(f"✓ Saved to {output_file}")
    return sample

if __name__ == "__main__":
    INPUT_FILE = 'songs_with_attributes_and_lyrics.csv'
    
    print("\n🎵 DATASET SAMPLING FOR THESIS")
    print("\nOptions:")
    print("1. Create 50,000 song sample (RECOMMENDED for thesis)")
    print("2. Create 10,000 song sample (faster, still good)")
    print("3. Create 1,000 song test sample (for testing scraper)")
    print("4. Use full dataset (955K songs - not recommended)")
    
    choice = input("\nChoose option (1-4): ").strip()
    
    if choice == '1':
        sample_df = create_stratified_sample(INPUT_FILE, 'songs_sample_50k.csv', 50000)
        print("\n✓ Ready to scrape! Use: songs_sample_50k.csv")
        
    elif choice == '2':
        sample_df = create_stratified_sample(INPUT_FILE, 'songs_sample_10k.csv', 10000)
        print("\n✓ Ready to scrape! Use: songs_sample_10k.csv")
        
    elif choice == '3':
        sample_df = create_smaller_test_sample(INPUT_FILE, 'songs_sample_1k.csv', 1000)
        print("\n✓ Ready to test! Use: songs_sample_1k.csv")
        
    elif choice == '4':
        print("\n⚠️  Warning: This will take ~77 days of continuous scraping!")
        confirm = input("Are you sure? (yes/no): ").strip().lower()
        if confirm == 'yes':
            print("Using full dataset: songs_with_attributes_and_lyrics.csv")
        else:
            print("Cancelled.")
    else:
        print("\nInvalid choice.")
