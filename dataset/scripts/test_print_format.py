"""
Quick test to show the improved print output
"""
from chosic_scraper_http import ChosicScraperHTTP

print("=" * 60)
print("Testing HTTP Scraper with New Print Format")
print("=" * 60)

scraper = ChosicScraperHTTP()
scraper.do_handshake()

# Test tracks
test_tracks = [
    "3BovdzfaX4jb5KFQwoPfAw",  # Beat It - Michael Jackson
    "7qiZfU4dY1lWllzX7mPBI",   # Shape of You - Ed Sheeran (example)
]

for idx, track_id in enumerate(test_tracks, 1):
    print(f"\n[{idx}/{len(test_tracks)}] Scraping: {track_id}")
    
    metadata = scraper.scrape_track_metadata(track_id)
    
    if metadata:
        print(f"  ✓ Year: {metadata['year']}, Pop: {metadata['popularity']}, Genre: {metadata['genre']}")
    else:
        print(f"  ❌ Failed to scrape")

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)
