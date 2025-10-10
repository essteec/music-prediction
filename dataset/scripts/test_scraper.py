"""
Quick test of the ChosicScraper with updated selectors
"""
from chosic_scraper import ChosicScraper

# Test with single track
TRACK_ID = "0Prct5TDjAnEgIqbxcldY9"

print("Testing ChosicScraper with updated selectors...")
print(f"Track ID: {TRACK_ID}")
print("-" * 50)

scraper = ChosicScraper(headless=False)  # Set to False to see browser

try:
    metadata = scraper.scrape_track_metadata(TRACK_ID)
    
    print("\n" + "=" * 50)
    print("RESULTS:")
    print("=" * 50)
    print(f"Year: {metadata['year']}")
    print(f"Genre: {metadata['genre']}")
    print(f"Explicit: {metadata['explicit']}")
    print(f"Popularity: {metadata['popularity']}")
    print("=" * 50)
    
finally:
    del scraper
    print("\nTest completed!")
