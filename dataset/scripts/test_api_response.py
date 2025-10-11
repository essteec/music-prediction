"""
Quick test to see what the Chosic API returns
"""
import requests
import json

# Test track ID (Michael Jackson - Billie Jean)
track_id = "3BovdzfaX4jb5KFQwoPfAw"

session = requests.Session()

# Set up headers
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

# Test 1: Track metadata
print("=" * 60)
print("TEST 1: Track Metadata API")
print("=" * 60)
url = f'https://www.chosic.com/api/tools/tracks/{track_id}'
response = session.get(url, headers=headers)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(json.dumps(data, indent=2))
    
    # Extract artist IDs
    if 'artists' in data:
        artist_ids = [artist['id'] for artist in data['artists']]
        print(f"\nArtist IDs: {artist_ids}")
        
        # Test 2: Artist genres
        print("\n" + "=" * 60)
        print("TEST 2: Artist Genres API")
        print("=" * 60)
        artist_ids_str = ','.join(artist_ids)
        url2 = f'https://www.chosic.com/api/tools/artists?ids={artist_ids_str}'
        response2 = session.get(url2, headers=headers)
        print(f"Status: {response2.status_code}")
        if response2.status_code == 200:
            data2 = response2.json()
            print(json.dumps(data2, indent=2))
else:
    print(f"Error: {response.text}")
