#!/usr/bin/env python3
"""
Filter English songs from songs_ml_ready.csv based on lyrics language detection.
Outputs to english_ml_ready.csv
"""

import pandas as pd
from langdetect import detect, LangDetectException
from tqdm import tqdm
import os

def detect_language(text):
    """
    Detect the language of the given text.
    Returns 'en' for English, 'unknown' if detection fails or text is empty.
    """
    if pd.isna(text) or text == '' or not isinstance(text, str):
        return 'unknown'
    
    try:
        # langdetect works better with longer text
        # For very short text, it might be unreliable
        if len(text.strip()) < 20:
            return 'unknown'
        
        language = detect(text)
        return language
    except LangDetectException:
        return 'unknown'

def filter_english_songs(input_file, output_file):
    """
    Read songs_ml_ready.csv, detect English songs, and save to english_ml_ready.csv
    """
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Total songs: {len(df)}")
    print(f"Songs with lyrics: {df['lyrics'].notna().sum()}")
    print(f"Songs without lyrics: {df['lyrics'].isna().sum()}")
    
    # Detect language for each song
    print("\nDetecting languages...")
    tqdm.pandas(desc="Language detection")
    df['detected_language'] = df['lyrics'].progress_apply(detect_language)
    
    # Show language distribution
    print("\nLanguage distribution:")
    print(df['detected_language'].value_counts())
    
    # Filter only English songs
    english_df = df[df['detected_language'] == 'en'].copy()
    
    # Drop the temporary language column
    english_df = english_df.drop(columns=['detected_language'])
    
    print(f"\nEnglish songs found: {len(english_df)}")
    print(f"Percentage: {len(english_df)/len(df)*100:.2f}%")
    
    # Save to output file
    english_df.to_csv(output_file, index=False)
    print(f"\nSaved English songs to {output_file}")
    
    # Show sample statistics
    print("\n=== English Songs Statistics ===")
    print(f"Total English songs: {len(english_df)}")
    print(f"Genres: {english_df['genre'].nunique()}")
    print(f"Artists: {english_df['artists'].nunique()}")
    print(f"\nTop genres:")
    print(english_df['genre'].value_counts().head(10))

def main():
    # File paths
    input_file = '../dataset/processed/songs_ml_ready.csv'
    output_file = '../dataset/processed/english_ml_ready.csv'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        print("Please run this script from the 'scripts' directory or adjust the path.")
        return
    
    # Filter English songs
    filter_english_songs(input_file, output_file)

if __name__ == "__main__":
    main()
