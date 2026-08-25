import json
import logging
import time
from pathlib import Path
import pandas as pd
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ARTISTS_CSV_PATH = BASE_DIR / "data" / "processed" / "artists.csv"
GENRES_CSV_PATH = BASE_DIR / "data" / "processed" / "genres.csv"

# Headers for Chosic requests
CHOSIC_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:153.0) Gecko/20100101 Firefox/153.0',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.chosic.com/spotify-playlist-analyzer/',
    'app': 'playlist_analyzer',
    'X-Requested-With': 'XMLHttpRequest',
    'Origin': 'https://www.chosic.com'
}

CHOSIC_REFRESH_COOKIES = {
    'pll_language': 'en',
    'spotify_stats_refresh_token': 'AQDxfa_gENBgIysIDrqkE3PjZPsYLcjorOsMQ-ao9M44Prt1qWQUYRWh2I1PluQpkQk6o_ukGdPepFlSOHjQHtKTr0qxaJ4UK8FSi9A8Crk9jZJVr7lIagThvALmsJ03MLE',
    'r_34874064': '1787088334%7C2ca304e7f195dcdc%7C9626c9ff72c4569c598b48cd6fd1a9aea854cc560b0cd2d595fa2bc0597cac9b',
    'playlist_analyzer_refresh_token': 'AQBSp9yNWC-LBg8WngnzFRrLU_fL-tS4EWu724WzWEtFlhiHG-AJf1cP3HWWo5i9JRkHsf95LwF3qN9RcL5Ag6hIYquoZZny2v9o2Khuo5E3YQLM8wga2ZAAjkyPD29Qeys'
}

def fetch_spotify_token():
    logging.info("Fetching fresh Spotify token from Chosic...")
    res = requests.post(
        'https://www.chosic.com/api/tools/t/',
        headers=CHOSIC_HEADERS,
        data={'app': 'playlist_analyzer'},
        timeout=15
    )
    res.raise_for_status()
    data = res.json()
    if isinstance(data, str):
        data = json.loads(data)
    token = data.get('token')
    if not token:
        raise ValueError("Failed to retrieve token from Chosic token endpoint.")
    return token

def get_cookie_header(token):
    cookies_dict = CHOSIC_REFRESH_COOKIES.copy()
    cookies_dict['playlist_analyzer_spotify_token'] = token
    return '; '.join([f'{k}={v}' for k, v in cookies_dict.items()])

def fetch_spotify_artist_genres(artist_ids, token):
    url = f"https://api.spotify.com/v1/artists?ids={','.join(artist_ids)}"
    sp_headers = {
        'User-Agent': CHOSIC_HEADERS['User-Agent'],
        'Authorization': f'Bearer {token}'
    }
    res = requests.get(url, headers=sp_headers, timeout=15)
    if res.status_code == 401:
        logging.info("Token expired, refreshing...")
        token = fetch_spotify_token()
        sp_headers['Authorization'] = f'Bearer {token}'
        res = requests.get(url, headers=sp_headers, timeout=15)
    res.raise_for_status()
    data = res.json()
    artist_genres_map = {}
    for artist_obj in data.get('artists', []):
        if artist_obj and 'id' in artist_obj:
            artist_genres_map[artist_obj['id']] = artist_obj.get('genres', [])
    return artist_genres_map, token

def fetch_chosic_artist_genres(artist_ids, token):
    headers = CHOSIC_HEADERS.copy()
    headers['Cookie'] = get_cookie_header(token)
    headers['Content-Type'] = 'application/x-www-form-urlencoded; charset=UTF-8'
    
    res = requests.post(
        'https://www.chosic.com/api/tools/get-artists-genres/',
        headers=headers,
        data={'ids[]': artist_ids},
        timeout=15
    )
    if res.status_code != 200:
        logging.warning(f"Chosic get-artists-genres returned status {res.status_code}")
        return {}
    try:
        data = res.json()
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, list):
            return {}
        return data
    except Exception as e:
        logging.warning(f"Error parsing Chosic get-artists-genres response: {e}")
        return {}

def main():
    if not ARTISTS_CSV_PATH.exists():
        raise FileNotFoundError(f"Input file {ARTISTS_CSV_PATH} not found.")
    if not GENRES_CSV_PATH.exists():
        raise FileNotFoundError(f"Genre mapping file {GENRES_CSV_PATH} not found.")

    # 1. Load genres.csv subgenre -> parent genres mapping
    logging.info(f"Loading genre mappings from {GENRES_CSV_PATH}...")
    df_genres = pd.read_csv(GENRES_CSV_PATH)
    subgenre_to_parents = {}
    for _, row in df_genres.iterrows():
        sg = str(row['subgenre']).strip()
        mg = row['main_genre']
        if pd.notna(mg) and str(mg).strip():
            parents = [p.strip() for p in str(mg).split('|') if p.strip()]
        else:
            parents = []
        subgenre_to_parents[sg] = parents

    # 2. Load artists.csv
    logging.info(f"Loading artists from {ARTISTS_CSV_PATH}...")
    df_artists = pd.read_csv(ARTISTS_CSV_PATH)
    
    # Existing fallback genres map
    existing_genres_map = {}
    for _, row in df_artists.iterrows():
        aid = str(row['artist_id']).strip()
        raw_genres = row.get('genres')
        if pd.notna(raw_genres) and str(raw_genres).strip():
            existing_genres_map[aid] = [g.strip() for g in str(raw_genres).split(',') if g.strip()]
        else:
            existing_genres_map[aid] = []

    artist_ids = [str(aid).strip() for aid in df_artists['artist_id'].dropna().unique()]
    logging.info(f"Total artists to update: {len(artist_ids)}")

    # 3. Fetch dynamic Spotify token
    token = fetch_spotify_token()

    # 4. Fetch genres from Spotify API in batches of 50
    spotify_artist_genres = {}
    batch_size = 50
    logging.info("Fetching subgenres from Spotify API...")
    for i in range(0, len(artist_ids), batch_size):
        chunk = artist_ids[i:i + batch_size]
        try:
            genres_map, token = fetch_spotify_artist_genres(chunk, token)
            spotify_artist_genres.update(genres_map)
        except Exception as e:
            logging.error(f"Error fetching Spotify genres for batch starting at index {i}: {e}")
        time.sleep(0.2)

    # Identify artists with empty genre lists
    empty_genre_artist_ids = [aid for aid in artist_ids if not spotify_artist_genres.get(aid)]
    logging.info(f"Artists with empty genre list from Spotify API: {len(empty_genre_artist_ids)}")

    # 5. Fetch missing genres from Chosic get-artists-genres API
    chosic_artist_genres = {}
    if empty_genre_artist_ids:
        logging.info("Fetching subgenres from Chosic get-artists-genres API...")
        chosic_batch_size = 30
        for i in range(0, len(empty_genre_artist_ids), chosic_batch_size):
            chunk = empty_genre_artist_ids[i:i + chosic_batch_size]
            c_map = fetch_chosic_artist_genres(chunk, token)
            chosic_artist_genres.update(c_map)
            time.sleep(0.8)

    # 6. Build updated genres and main_genre per artist
    updated_genres_col = []
    updated_main_genre_col = []

    fallback_count = 0
    for idx, row in df_artists.iterrows():
        aid = str(row['artist_id']).strip()
        
        # Determine subgenres
        subgenres = spotify_artist_genres.get(aid, [])
        if not subgenres:
            subgenres = chosic_artist_genres.get(aid, [])
        if not subgenres:
            subgenres = existing_genres_map.get(aid, [])
            if subgenres:
                fallback_count += 1
                logging.warning(f"Warning: No genres found for artist ID {aid} from Spotify or Chosic API. Kept existing artists.csv genres.")
        
        # Clean subgenres
        clean_subgenres = [g.strip() for g in subgenres if g.strip()]
        genres_str = ", ".join(clean_subgenres)
        
        # Determine parent genres from genres.csv mapping
        parent_set = set()
        for sg in clean_subgenres:
            parents = subgenre_to_parents.get(sg, [])
            for p in parents:
                parent_set.add(p)
        
        main_genre_str = ", ".join(sorted(parent_set)) if parent_set else ""
        
        updated_genres_col.append(genres_str)
        updated_main_genre_col.append(main_genre_str)

    # Update DataFrame columns
    df_artists['genres'] = updated_genres_col
    df_artists['main_genre'] = updated_main_genre_col

    # Write back to artists.csv
    df_artists.to_csv(ARTISTS_CSV_PATH, index=False)
    logging.info(f"Successfully updated {ARTISTS_CSV_PATH} with updated subgenres and parent genres ({fallback_count} fallbacks used).")

if __name__ == "__main__":
    main()
