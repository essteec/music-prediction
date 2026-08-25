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

def fetch_parent_genres(genres_batch, retries=5, initial_delay=3.0):
    headers = CHOSIC_HEADERS.copy()
    headers['Content-Type'] = 'application/x-www-form-urlencoded; charset=UTF-8'
    
    for attempt in range(retries):
        try:
            res = requests.post(
                'https://www.chosic.com/api/tools/parent-genres/',
                headers=headers,
                data={
                    'genres': json.dumps(genres_batch),
                    'version': '2'
                },
                timeout=15
            )
            if res.status_code == 429:
                wait_time = initial_delay * (2 ** attempt)
                logging.warning(f"Rate limited (429) on parent-genres API. Waiting {wait_time:.1f}s (attempt {attempt+1}/{retries})...")
                time.sleep(wait_time)
                continue
            res.raise_for_status()
            data = res.json()
            if isinstance(data, str):
                data = json.loads(data)
            return data
        except Exception as e:
            if attempt == retries - 1:
                logging.error(f"Failed parent-genres batch after {retries} attempts: {e}")
                return []
            wait_time = initial_delay * (2 ** attempt)
            logging.warning(f"Error on parent-genres API ({e}). Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
    return []

def main():
    if not ARTISTS_CSV_PATH.exists():
        raise FileNotFoundError(f"Input file {ARTISTS_CSV_PATH} not found.")

    logging.info(f"Loading artists from {ARTISTS_CSV_PATH}...")
    df_artists = pd.read_csv(ARTISTS_CSV_PATH)
    
    # Store fallback genres from artists.csv
    fallback_map = {}
    for _, row in df_artists.iterrows():
        aid = str(row['artist_id']).strip()
        raw_genres = row.get('genres')
        if pd.notna(raw_genres) and raw_genres:
            fallback_genres = [g.strip() for g in str(raw_genres).split(',') if g.strip()]
        else:
            fallback_genres = []
        fallback_map[aid] = fallback_genres

    artist_ids = [str(aid).strip() for aid in df_artists['artist_id'].dropna().unique()]
    logging.info(f"Total unique artists to process: {len(artist_ids)}")

    # 1. Fetch token
    token = fetch_spotify_token()

    # 2. Fetch genres from Spotify API in batches of 50
    spotify_artist_genres = {}
    batch_size = 50
    logging.info("Requesting Spotify API for artist genres in batches of 50...")
    for i in range(0, len(artist_ids), batch_size):
        chunk = artist_ids[i:i + batch_size]
        try:
            genres_map, token = fetch_spotify_artist_genres(chunk, token)
            spotify_artist_genres.update(genres_map)
        except Exception as e:
            logging.error(f"Error fetching Spotify genres for batch starting at index {i}: {e}")
        time.sleep(0.05)

    # Identify artists with empty genre lists
    empty_genre_artist_ids = [aid for aid in artist_ids if not spotify_artist_genres.get(aid)]
    logging.info(f"Artists with empty genre list from Spotify API: {len(empty_genre_artist_ids)}")

    # 3. Fallback to Chosic get-artists-genres API
    chosic_artist_genres = {}
    if empty_genre_artist_ids:
        logging.info("Requesting Chosic get-artists-genres API for artists with missing genres...")
        chosic_batch_size = 30
        for i in range(0, len(empty_genre_artist_ids), chosic_batch_size):
            chunk = empty_genre_artist_ids[i:i + chosic_batch_size]
            c_map = fetch_chosic_artist_genres(chunk, token)
            chosic_artist_genres.update(c_map)
            time.sleep(0.3)

    # Consolidate artist genres
    final_artist_genres = {}
    for aid in artist_ids:
        genres = spotify_artist_genres.get(aid, [])
        if not genres:
            genres = chosic_artist_genres.get(aid, [])
        if not genres:
            logging.warning(f"Warning: No genres found for artist ID {aid} from Spotify or Chosic API. Falling back to artists.csv genres column.")
            genres = fallback_map.get(aid, [])
        final_artist_genres[aid] = genres

    # Collect all unique subgenres
    all_subgenres = set()
    for g_list in final_artist_genres.values():
        for g in g_list:
            clean_g = g.strip()
            if clean_g:
                all_subgenres.add(clean_g)

    sorted_subgenres = sorted(list(all_subgenres))
    logging.info(f"Total unique subgenres collected: {len(sorted_subgenres)}")

    # 4. Fetch parent genres from Chosic API in batches
    logging.info("Fetching parent genres from Chosic parent-genres API...")
    parent_map = {}  # subgenre -> set of parents
    parent_batch_size = 30
    for i in range(0, len(sorted_subgenres), parent_batch_size):
        chunk = sorted_subgenres[i:i + parent_batch_size]
        try:
            results = fetch_parent_genres(chunk)
            for item in results:
                g = item.get('genre')
                p = item.get('parent')
                if g and p:
                    if g not in parent_map:
                        parent_map[g] = set()
                    parent_map[g].add(p)
        except Exception as e:
            logging.error(f"Error fetching parent genres for batch {i}: {e}")
        time.sleep(1.0)

    # 5. Create final mapping dataframe and write to CSV
    output_rows = []
    for g in sorted_subgenres:
        parents = parent_map.get(g, set())
        main_genre_str = "|".join(sorted(parents)) if parents else ""
        output_rows.append({
            'subgenre': g,
            'main_genre': main_genre_str
        })

    df_out = pd.DataFrame(output_rows)
    GENRES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(GENRES_CSV_PATH, index=False)
    logging.info(f"Successfully generated {GENRES_CSV_PATH} with {len(df_out)} genres.")

if __name__ == "__main__":
    main()