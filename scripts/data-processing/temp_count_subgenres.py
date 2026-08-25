from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
ARTISTS_CSV = BASE_DIR / "data" / "processed" / "artists.csv"


def main() -> None:
    if not ARTISTS_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {ARTISTS_CSV}")

    df = pd.read_csv(ARTISTS_CSV)
    if "genres" not in df.columns:
        raise KeyError("artists.csv does not contain a 'genres' column")

    subgenres = set()
    for raw_value in df["genres"].dropna():
        for genre in str(raw_value).split(","):
            cleaned = genre.strip()
            if cleaned:
                subgenres.add(cleaned)

    print(f"Total unique subgenres: {len(subgenres)}")


if __name__ == "__main__":
    main()