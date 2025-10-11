"""
Test to find where explicit flag is located
"""
import requests
import json
from bs4 import BeautifulSoup

# Test with a known explicit track
# Let's try an explicit track - you can replace this with one you know is explicit
track_id = "3BovdzfaX4jb5KFQwoPfAw"  # Beat It - probably not explicit

session = requests.Session()

headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:143.0) Gecko/20100101 Firefox/143.0',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Accept-Language': 'en-US,en;q=0.5',
    'app': 'genre_finder',
    'X-Requested-With': 'XMLHttpRequest',
    'Referer': f'https://www.chosic.com/music-genre-finder/?track={track_id}',
}

session.cookies.set('pll_language', 'en')
session.cookies.set('r_c1062550', '1760111056%7Ce4bd8b1f86d75d37%7C6952e8658357b9d55e73e9eb25b01241ab5dbbf4f673a2d227d8975c5b902417')

# Test: Get track metadata and look for explicit field
print("=" * 60)
print("Checking Track Metadata for 'explicit' field")
print("=" * 60)
url = f'https://www.chosic.com/api/tools/tracks/{track_id}'
response = session.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    print(json.dumps(data, indent=2))
    
    if 'explicit' in data:
        print(f"\n✓ Found 'explicit' field: {data['explicit']}")
    else:
        print("\n⚠ 'explicit' field NOT in API response")
        print("\nLet me check the HTML page instead...")
        
        # Get the HTML page
        page_url = f"https://www.chosic.com/music-genre-finder/?track={track_id}"
        page_response = session.get(page_url)
        
        if page_response.status_code == 200:
            soup = BeautifulSoup(page_response.text, 'html.parser')
            
            # Search for "explicit" in the page
            page_text = soup.get_text().lower()
            if 'explicit' in page_text:
                print("\n✓ Found 'explicit' in HTML page text")
                
                # Try to find the specific element
                explicit_span = soup.find('span', class_='span-explicit')
                if explicit_span:
                    print(f"✓ Found span.span-explicit: {explicit_span.get_text()}")
                
                # Search for any element containing "explicit"
                explicit_elements = soup.find_all(string=lambda text: text and 'explicit' in text.lower())
                print(f"\nFound {len(explicit_elements)} elements containing 'explicit':")
                for elem in explicit_elements[:5]:  # Show first 5
                    print(f"  - {elem.strip()}")
            else:
                print("\n⚠ 'explicit' NOT found in HTML page either")
                print("\nMaybe this track is not explicit. Try an explicit track ID.")

print("\n" + "=" * 60)
print("Please provide an explicit track ID if you have one")
print("=" * 60)
