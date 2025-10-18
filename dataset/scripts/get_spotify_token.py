#!/usr/bin/env python3
"""
Spotify Access Token Helper
Gets a fresh access token using Client Credentials flow
Token is valid for 1 hour
"""

import requests
import base64
import json
import os

def get_access_token_client_credentials(client_id, client_secret):
    """
    Get access token using Client Credentials flow
    This token lasts 1 hour and doesn't require user authentication
    
    Args:
        client_id: Your Spotify app's Client ID
        client_secret: Your Spotify app's Client Secret
        
    Returns:
        Access token string
    """
    # Encode credentials
    auth_string = f"{client_id}:{client_secret}"
    auth_bytes = auth_string.encode('utf-8')
    auth_base64 = base64.b64encode(auth_bytes).decode('utf-8')
    
    # Request token
    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": f"Basic {auth_base64}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "client_credentials"
    }
    
    response = requests.post(url, headers=headers, data=data)
    
    if response.status_code == 200:
        json_result = response.json()
        access_token = json_result['access_token']
        expires_in = json_result['expires_in']
        
        print("✓ Access token retrieved successfully!")
        print(f"  Token expires in: {expires_in} seconds ({expires_in/3600:.1f} hours)")
        return access_token
    else:
        print(f"❌ Failed to get token: {response.status_code}")
        print(f"   Response: {response.text}")
        return None

def main():
    """
    Interactive token retrieval
    """
    print("=" * 60)
    print("Spotify Access Token Generator")
    print("=" * 60)
    print()
    print("You need a Spotify App (free):")
    print("1. Go to: https://developer.spotify.com/dashboard")
    print("2. Create an app (or use existing)")
    print("3. Copy Client ID and Client Secret from Settings")
    print()
    print("=" * 60)
    print()
    
    # Try to get from environment first
    client_id = os.environ.get('SPOTIFY_CLIENT_ID')
    client_secret = os.environ.get('SPOTIFY_CLIENT_SECRET')
    
    if not client_id:
        client_id = input("Enter your Client ID: ").strip()
    else:
        print(f"Using Client ID from environment: {client_id[:10]}...")
    
    if not client_secret:
        client_secret = input("Enter your Client Secret: ").strip()
    else:
        print(f"Using Client Secret from environment")
    
    print()
    print("Getting access token...")
    print()
    
    access_token = get_access_token_client_credentials(client_id, client_secret)
    
    if access_token:
        print()
        print("=" * 60)
        print("Your Access Token:")
        print("=" * 60)
        print(access_token)
        print("=" * 60)
        print()
        print("To use it:")
        print()
        print("Option 1 - Export to environment:")
        print(f"  export SPOTIFY_ACCESS_TOKEN='{access_token}'")
        print()
        print("Option 2 - Copy and paste when prompted by the scraper")
        print()
        print("⏰ Remember: This token expires in 1 hour!")
        print()
        
        # Save to a file (optional)
        save = input("Save to .env file? (y/n): ").strip().lower()
        if save == 'y':
            with open('../.env', 'w') as f:
                f.write(f"SPOTIFY_ACCESS_TOKEN='{access_token}'\n")
                f.write(f"SPOTIFY_CLIENT_ID='{client_id}'\n")
                f.write(f"SPOTIFY_CLIENT_SECRET='{client_secret}'\n")
            print("✓ Saved to dataset/.env")
            print("  Run: source dataset/.env")

if __name__ == "__main__":
    main()
