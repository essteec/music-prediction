#!/usr/bin/env python3
"""
Test script to verify scraper setup and run a small test
"""
import sys
import os

# Add parent directory to path so we can import from scripts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chosic_scraper import ChosicScraper
import pandas as pd

def test_scraper():
    """Test the scraper with a single track"""
    print("=" * 60)
    print("Testing Chosic Scraper Setup")
    print("=" * 60)
    
    # Check if main dataset exists
    dataset_path = '../songs_with_attributes_and_lyrics.csv'
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset not found at {dataset_path}")
        return False
    
    print(f"✓ Found main dataset")
    
    # Load a sample track
    df = pd.read_csv(dataset_path, nrows=1)
    test_track_id = df['id'].iloc[0]
    test_track_name = df['name'].iloc[0]
    
    print(f"✓ Test track: {test_track_name} (ID: {test_track_id})")
    
    # Try to scrape
    print("\n" + "=" * 60)
    print("Attempting to scrape metadata...")
    print("=" * 60)
    
    try:
        scraper = ChosicScraper(headless=True)
        metadata = scraper.scrape_track_metadata(test_track_id)
        
        print("\n✓ Scraping successful!")
        print(f"  Year: {metadata.get('year')}")
        print(f"  Genre: {metadata.get('genre')}")
        print(f"  Explicit: {metadata.get('explicit')}")
        print(f"  Popularity: {metadata.get('popularity')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during scraping: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\nThis will test the scraper with ONE track.")
    print("Make sure Chrome/Chromium browser is installed!\n")
    
    success = test_scraper()
    
    if success:
        print("\n" + "=" * 60)
        print("✓ Setup looks good!")
        print("=" * 60)
        print("\nYou can now run the full scraper:")
        print("  cd /home/esstee/documents/bitirme/dataset/scripts")
        print("  python chosic_scraper.py")
    else:
        print("\n" + "=" * 60)
        print("❌ Setup test failed")
        print("=" * 60)
        print("Please check the error messages above.")
