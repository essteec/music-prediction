import logging
from pathlib import Path
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ARTISTS_CSV_PATH = BASE_DIR / "data" / "processed" / "artists.csv"
SONGS_CSV_PATH = BASE_DIR / "data" / "processed" / "songs.csv"

def main():
    if not ARTISTS_CSV_PATH.exists():
        raise FileNotFoundError(f"Input file {ARTISTS_CSV_PATH} not found.")
    if not SONGS_CSV_PATH.exists():
        raise FileNotFoundError(f"Input file {SONGS_CSV_PATH} not found.")

    # 1. Load artists.csv into lookup maps
    logging.info(f"Loading artists lookup from {ARTISTS_CSV_PATH}...")
    df_artists = pd.read_csv(ARTISTS_CSV_PATH)
    
    artist_subgenres_map = {}
    artist_main_genres_map = {}

    for _, row in df_artists.iterrows():
        aid = str(row['artist_id']).strip()
        raw_sub = row.get('genres')
        raw_main = row.get('main_genre')

        if pd.notna(raw_sub) and str(raw_sub).strip():
            subgenres = [g.strip() for g in str(raw_sub).split(',') if g.strip()]
        else:
            subgenres = []

        if pd.notna(raw_main) and str(raw_main).strip():
            main_genres = [g.strip() for g in str(raw_main).split(',') if g.strip()]
        else:
            main_genres = []

        artist_subgenres_map[aid] = subgenres
        artist_main_genres_map[aid] = main_genres

    # 2. Load songs.csv
    logging.info(f"Loading songs from {SONGS_CSV_PATH}...")
    df_songs = pd.read_csv(SONGS_CSV_PATH)
    logging.info(f"Total songs to process: {len(df_songs)}")

    # 3. Update artist_genres and main_genres for each song
    updated_artist_genres = []
    updated_main_genres = []

    for _, row in df_songs.iterrows():
        raw_ids = str(row.get('artist_ids')) if pd.notna(row.get('artist_ids')) else ""
        a_ids = [a.strip() for a in raw_ids.split('|') if a.strip()]

        song_subgenres_set = set()
        song_main_genres_set = set()

        for aid in a_ids:
            for sg in artist_subgenres_map.get(aid, []):
                song_subgenres_set.add(sg)
            for mg in artist_main_genres_map.get(aid, []):
                song_main_genres_set.add(mg)

        subgenres_str = ", ".join(sorted(song_subgenres_set)) if song_subgenres_set else ""
        main_genres_str = ", ".join(sorted(song_main_genres_set)) if song_main_genres_set else ""

        updated_artist_genres.append(subgenres_str)
        updated_main_genres.append(main_genres_str)

    df_songs['artist_genres'] = updated_artist_genres
    df_songs['main_genres'] = updated_main_genres

    # 4. Save updated songs.csv
    logging.info(f"Writing updated songs to {SONGS_CSV_PATH}...")
    df_songs.to_csv(SONGS_CSV_PATH, index=False)
    logging.info("Successfully updated songs.csv!")

if __name__ == "__main__":
    main()
