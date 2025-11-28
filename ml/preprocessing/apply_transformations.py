"""Comprehensive preprocessing pipeline for ML-ready features.

Responsibilities
----------------
1. Load artist-aware splits from ``data/processed``.
2. Drop rows with missing/empty lyrics (these rows should not reach NLP steps).
3. Apply transformations recommended by EDA:
   * Yeo-Johnson (PowerTransformer) for highly skewed audio features
     (``acousticness``, ``instrumentalness``, ``speechiness``).
   * Standard scaling for wide-range continuous features
     (``loudness``, ``tempo``, ``duration_ms``).
   * Min/Max scaling for ``year`` to keep temporal values in [0, 1].
   * Cyclical encoding for ``key`` plus passthrough ``mode``.
   * One-hot encoding for ``genre`` with ``handle_unknown='ignore'``.
4. Log-transform the popularity target (``log1p``) before saving.
5. Scale text statistics + sentiment outputs so the downstream models
   can combine audio/text blocks without manual tweaks.
6. Persist every fitted transformer/scaler for reproducibility.

The script focuses on deterministic, train-fit / val&test-transform behaviour
so that any algorithm (linear, tree-based, neural) consumes the exact same
feature tensors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    OneHotEncoder,
    PowerTransformer,
    StandardScaler,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FEATURES_DIR = REPO_ROOT / "ml" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_SKEWED = ["acousticness", "instrumentalness", "speechiness"]
AUDIO_CONTINUOUS = ["loudness", "tempo", "duration_ms"]
AUDIO_METADATA = ["liveness"]  # already ~[0, 1] but kept for completeness
AUDIO_YEAR_COL = "year"
AUDIO_CATEGORICAL = ["mode"]
AUDIO_KEY_COL = "key"
AUDIO_GENRE_COL = "genre"
TARGETS = ["valence", "energy", "danceability", "popularity"]

TEXT_STAT_COLUMNS = [
    "word_count",
    "unique_word_count",
    "unique_ratio",
    "avg_word_length",
    "char_count",
]
SENTIMENT_COLUMNS = ["sentiment_polarity", "sentiment_subjectivity"]

# ---------------------------------------------------------------------------
# Helper data structures
# ---------------------------------------------------------------------------
@dataclass
class SplitData:
    name: str
    frame: pd.DataFrame


def _load_split(name: str) -> SplitData:
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected split file missing: {path}")
    return SplitData(name=name, frame=pd.read_csv(path))


def _drop_rows_without_lyrics(split: SplitData) -> Tuple[SplitData, int]:
    df = split.frame
    mask = df["lyrics"].astype(str).str.strip().ne("")
    filtered = df[mask].reset_index(drop=True)
    dropped = len(df) - len(filtered)
    return SplitData(name=split.name, frame=filtered), dropped


def _fit_power_transformer(train: pd.DataFrame) -> PowerTransformer:
    transformer = PowerTransformer(method="yeo-johnson")
    transformer.fit(train[AUDIO_SKEWED])
    return transformer


def _apply_power_transform(
    transformer: PowerTransformer,
    split_frames: Dict[str, pd.DataFrame],
) -> None:
    for df in split_frames.values():
        df[AUDIO_SKEWED] = transformer.transform(df[AUDIO_SKEWED])


def _encode_key(df: pd.DataFrame) -> None:
    key = pd.to_numeric(df[AUDIO_KEY_COL], errors="coerce")
    radians = 2 * np.pi * key / 12.0
    df["key_sin"] = np.sin(radians)
    df["key_cos"] = np.cos(radians)
    df.loc[key.isna() | (key == -1), ["key_sin", "key_cos"]] = 0.0


def _normalize_year(train: pd.DataFrame) -> Tuple[float, float]:
    year_min = float(train[AUDIO_YEAR_COL].min())
    year_max = float(train[AUDIO_YEAR_COL].max())
    if year_max == year_min:
        raise ValueError("year column has no variance; cannot normalize.")
    return year_min, year_max


def _apply_year_normalization(
    year_bounds: Tuple[float, float],
    split_frames: Dict[str, pd.DataFrame],
) -> None:
    year_min, year_max = year_bounds
    denom = year_max - year_min
    for df in split_frames.values():
        df["year_normalized"] = (df[AUDIO_YEAR_COL] - year_min) / denom


def _fit_continuous_scaler(train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train[AUDIO_CONTINUOUS])
    return scaler


def _apply_continuous_scaler(
    scaler: StandardScaler,
    split_frames: Dict[str, pd.DataFrame],
) -> None:
    for df in split_frames.values():
        df[AUDIO_CONTINUOUS] = scaler.transform(df[AUDIO_CONTINUOUS])


def _fit_genre_encoder(train: pd.DataFrame) -> OneHotEncoder:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(train[[AUDIO_GENRE_COL]])
    return encoder


def _build_audio_matrix(
    df: pd.DataFrame,
    encoder: OneHotEncoder,
) -> np.ndarray:
    normalized_cols = AUDIO_METADATA + AUDIO_SKEWED  # now transformed
    components = [df[normalized_cols].to_numpy(copy=False)]
    components.append(df[AUDIO_CONTINUOUS].to_numpy(copy=False))
    components.append(df[AUDIO_CATEGORICAL].to_numpy(copy=False))
    components.append(df[["key_sin", "key_cos"]].to_numpy(copy=False))
    components.append(df[["year_normalized"]].to_numpy(copy=False))
    genre_block = encoder.transform(df[[AUDIO_GENRE_COL]])
    components.append(genre_block)
    return np.hstack(components)


def _compute_text_stats(lyrics: pd.Series) -> pd.DataFrame:
    def extract(row: str) -> Dict[str, float]:
        text = row if isinstance(row, str) else ""
        text = text.strip()
        if not text:
            return dict.fromkeys(TEXT_STAT_COLUMNS, 0.0)
        words = text.split()
        unique = {w.lower() for w in words}
        counts = {
            "word_count": len(words),
            "unique_word_count": len(unique),
            "unique_ratio": len(unique) / max(len(words), 1),
            "avg_word_length": float(np.mean([len(w) for w in words])),
            "char_count": len(text),
        }
        return counts

    records = [extract(text) for text in lyrics]
    return pd.DataFrame(records, columns=TEXT_STAT_COLUMNS)


def _scale_text_statistics(stats_train: pd.DataFrame, splits: Dict[str, pd.DataFrame]) -> Tuple[StandardScaler, Dict[str, np.ndarray]]:
    stats_scaler = StandardScaler()
    log_cols = ["word_count", "unique_word_count", "char_count"]
    stats_train = stats_train.copy()
    stats_train[log_cols] = np.log1p(stats_train[log_cols])
    stats_scaler.fit(stats_train)

    scaled = {}
    for name, stats_df in splits.items():
        copy_df = stats_df.copy()
        copy_df[log_cols] = np.log1p(copy_df[log_cols])
        scaled[name] = stats_scaler.transform(copy_df)
    return stats_scaler, scaled


def _scale_sentiment(sentiment_frames: Dict[str, pd.DataFrame]) -> Tuple[StandardScaler, Dict[str, np.ndarray]]:
    scaler = StandardScaler()
    scaler.fit(sentiment_frames["train"])
    scaled = {name: scaler.transform(df) for name, df in sentiment_frames.items()}
    return scaler, scaled


def _extract_sentiment(lyrics: pd.Series) -> pd.DataFrame:
    # Local import keeps TextBlob overhead away when sentiment is precomputed.
    from textblob import TextBlob  # type: ignore

    def compute(text: str) -> Dict[str, float]:
        if not isinstance(text, str) or not text.strip():
            return {
                "sentiment_polarity": 0.0,
                "sentiment_subjectivity": 0.0,
            }
        blob = TextBlob(text)
        return {
            "sentiment_polarity": blob.sentiment.polarity,
            "sentiment_subjectivity": blob.sentiment.subjectivity,
        }

    records = [compute(line) for line in lyrics]
    return pd.DataFrame(records, columns=SENTIMENT_COLUMNS)


def _transform_targets(splits: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, np.ndarray]]:
    target_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for target in TARGETS:
        target_arrays[target] = {}
        for name, df in splits.items():
            series = df[target].to_numpy(dtype=np.float32)
            if target == "popularity":
                series = np.log1p(series)
            target_arrays[target][name] = series
    return target_arrays


def _save_np_arrays(data: Dict[str, np.ndarray], prefix: str) -> None:
    for split_name, arr in data.items():
        path = FEATURES_DIR / f"{prefix}_{split_name}.npy"
        np.save(path, arr)


def main() -> None:
    print("Preparing splits from data/processed ...")
    splits = {name: _load_split(name).frame for name in ("train", "val", "test")}

    # Drop rows without lyrics for every split independently.
    lyric_drops = {}
    for name in list(splits.keys()):
        split = SplitData(name=name, frame=splits[name])
        filtered, dropped = _drop_rows_without_lyrics(split)
        splits[name] = filtered.frame
        lyric_drops[name] = dropped
        if dropped:
            print(f"Dropped {dropped} rows without lyrics from {name} split")

    # Fit/transform audio blocks
    power_transformer = _fit_power_transformer(splits["train"])
    _apply_power_transform(power_transformer, splits)

    year_bounds = _normalize_year(splits["train"])
    _apply_year_normalization(year_bounds, splits)

    continuous_scaler = _fit_continuous_scaler(splits["train"])
    _apply_continuous_scaler(continuous_scaler, splits)

    for df in splits.values():
        _encode_key(df)

    genre_encoder = _fit_genre_encoder(splits["train"])

    audio_arrays = {}
    for name, df in splits.items():
        audio_arrays[name] = _build_audio_matrix(df, genre_encoder)

    for split_name, arr in audio_arrays.items():
        np.save(FEATURES_DIR / f"X_{split_name}_audio.npy", arr)

    # Textual features
    text_stats_frames = {name: _compute_text_stats(df["lyrics"]) for name, df in splits.items()}
    stats_scaler, scaled_text_stats = _scale_text_statistics(
        text_stats_frames["train"],
        text_stats_frames,
    )
    for split_name, arr in scaled_text_stats.items():
        np.save(FEATURES_DIR / f"X_{split_name}_text_stats.npy", arr)

    sentiment_frames = {name: _extract_sentiment(df["lyrics"]) for name, df in splits.items()}
    sentiment_scaler, scaled_sentiment = _scale_sentiment(sentiment_frames)
    for split_name, arr in scaled_sentiment.items():
        np.save(FEATURES_DIR / f"X_{split_name}_sentiment.npy", arr)

    # Targets
    target_arrays = _transform_targets(splits)
    for target, split_arrays in target_arrays.items():
        for split_name, arr in split_arrays.items():
            np.save(FEATURES_DIR / f"y_{split_name}_{target}.npy", arr)

    # Persist transformers/metadata
    import joblib

    joblib.dump(power_transformer, FEATURES_DIR / "audio_power_transformer.pkl")
    joblib.dump(continuous_scaler, FEATURES_DIR / "audio_continuous_scaler.pkl")
    joblib.dump(genre_encoder, FEATURES_DIR / "genre_encoder.pkl")
    joblib.dump(stats_scaler, FEATURES_DIR / "text_stats_scaler.pkl")
    joblib.dump(sentiment_scaler, FEATURES_DIR / "sentiment_scaler.pkl")

    metadata = {
        "audio_skewed": AUDIO_SKEWED,
        "audio_continuous": AUDIO_CONTINUOUS,
        "audio_metadata": AUDIO_METADATA,
        "audio_categorical": AUDIO_CATEGORICAL,
        "genre_categories": genre_encoder.categories_[0].tolist(),
        "year_min": year_bounds[0],
        "year_max": year_bounds[1],
        "lyric_rows_dropped": lyric_drops,
        "targets": TARGETS,
    }
    with open(FEATURES_DIR / "preprocessing_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Preprocessing complete. Feature arrays saved under ml/features.")


if __name__ == "__main__":
    main()
