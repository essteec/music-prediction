"""
Test single track scraping to verify genre collection
"""
from chosic_scraper_http import ChosicScraperHTTP

scraper = ChosicScraperHTTP()
scraper.do_handshake()

# Test with Michael Jackson - Beat It
track_id = "3QOio6kiBn9nRIU5WkFCFe"

print("Testing full scrape for track:", track_id)
print("=" * 60)

metadata = scraper.scrape_track_metadata(track_id)

print("\n" + "=" * 60)
print("RESULTS:")
print("=" * 60)
print(f"Year: {metadata['year']}")
print(f"Genre: {metadata['genre']}")
print(f"Explicit: {metadata['explicit']}")
print(f"Popularity: {metadata['popularity']}")
