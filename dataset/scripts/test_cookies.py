#!/usr/bin/env python3
"""
Quick test script to verify cookies work with Chosic API
"""

import requests

# Test configuration
TEST_TRACK_ID = '5G0o5l3ifaEZHSi7FeCgHc'  # Known working track from your curl
COOKIES_FILE = 'cookies.txt'

def test_with_cookies(cookies_string):
    """Test API calls with provided cookies"""
    
    # Create session
    session = requests.Session()
    
    # Add headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:143.0) Gecko/20100101 Firefox/143.0',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'app': 'genre_finder',
        'X-Requested-With': 'XMLHttpRequest',
        'DNT': '1',
        'Sec-GPC': '1',
        'Connection': 'keep-alive',
        'Referer': f'https://www.chosic.com/music-genre-finder/?track={TEST_TRACK_ID}',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
    }
    
    session.headers.update(headers)
    
    # Parse and add cookies
    for cookie in cookies_string.split(';'):
        cookie = cookie.strip()
        if '=' in cookie:
            key, value = cookie.split('=', 1)
            session.cookies.set(key.strip(), value.strip(), domain='.chosic.com')
    
    print(f"✓ Loaded cookies: {list(session.cookies.keys())}")
    print()
    
    # Test 1: Track metadata
    print("=" * 60)
    print("TEST 1: Track Metadata API")
    print("=" * 60)
    url = f'https://www.chosic.com/api/tools/tracks/{TEST_TRACK_ID}'
    
    try:
        response = session.get(url, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✓ SUCCESS!")
            print(f"Track: {data.get('name', 'N/A')}")
            print(f"Artist: {data.get('artists', [{}])[0].get('name', 'N/A')}")
            print(f"Popularity: {data.get('popularity', 'N/A')}")
            print(f"Album: {data.get('album', {}).get('name', 'N/A')}")
            print(f"Release: {data.get('album', {}).get('release_date', 'N/A')}")
            return True
        elif response.status_code == 403:
            print("❌ FAILED: 403 Forbidden")
            print("Your cookies may be expired or invalid.")
            print("\nGet fresh cookies:")
            print("1. Open browser to https://www.chosic.com/music-genre-finder/")
            print("2. Open DevTools (F12) > Network tab")
            print("3. Refresh page and click any request")
            print("4. Copy Cookie header value")
            return False
        else:
            print(f"❌ FAILED: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    print("=" * 60)
    print("Chosic Cookie Test")
    print("=" * 60)
    print()
    
    # Try to load from file
    try:
        with open(COOKIES_FILE, 'r') as f:
            cookies_string = f.read().strip()
        print(f"✓ Loaded cookies from {COOKIES_FILE}")
        print()
    except FileNotFoundError:
        print(f"⚠ No {COOKIES_FILE} found")
        print()
        print("Please enter your cookie string:")
        print("(Get from browser: DevTools > Network > Cookie header)")
        print()
        cookies_string = input("Cookies: ").strip()
        
        if not cookies_string:
            print("\n❌ No cookies provided. Exiting.")
            return
        
        # Save for future use
        save = input("\nSave to cookies.txt? (y/n): ").strip().lower()
        if save == 'y':
            with open(COOKIES_FILE, 'w') as f:
                f.write(cookies_string)
            print(f"✓ Saved to {COOKIES_FILE}")
        print()
    
    # Run test
    success = test_with_cookies(cookies_string)
    
    print()
    print("=" * 60)
    if success:
        print("✓ ALL TESTS PASSED!")
        print("You can now run chosic_scraper_http.py")
    else:
        print("❌ TESTS FAILED")
        print("Fix the issues above before running the scraper")
    print("=" * 60)

if __name__ == "__main__":
    main()
