"""
BERTopic Topic Modeling on Full 10,000 Track Lyrics.
Uses pre-computed BGE-M3 embeddings (1024-D) to cluster lyrics into 32 coherent topic themes.
Outputs: data/features/lyric/bertopic_topics.parquet
"""

import os
import json
import unicodedata
import re
from pathlib import Path
import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
LANG_PARQUET = DATA_DIR / "features" / "lyric" / "language_id.parquet"
BGE_M3_NPY = DATA_DIR / "embeddings" / "lyrics" / "bge_m3_1024d.npy"
OUTPUT_DIR = DATA_DIR / "features" / "lyric"
OUTPUT_FILE = OUTPUT_DIR / "bertopic_topics.parquet"
TOPICS_JSON = OUTPUT_DIR / "bertopic_topic_labels.json"

def clean_lyric(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\[.*?\]', '', text)
    lines = text.split('\n')
    lines = [l for l in lines if not re.match(
        r'^(Contributors?|Lyrics?\s*by|Source|Embed|You might also like|\d+Embed)',
        l.strip(), re.IGNORECASE)]
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading songs and language metadata...")
    df = pd.read_csv(SONGS_CSV)
    df_lang = pd.read_parquet(LANG_PARQUET)
    n_songs = len(df)

    print(f"Loading BGE-M3 embeddings from {BGE_M3_NPY}...")
    embeddings = np.load(BGE_M3_NPY)

    # Clean lyrics for topic representation
    cleaned_docs = [clean_lyric(t) if (isinstance(t, str) and len(t) > 20) else "instrumental track" for t in df['lyrics']]

    print("Fitting BERTopic with precomputed BGE-M3 embeddings...")
    vectorizer = CountVectorizer(stop_words="english", min_df=5, max_df=0.6, ngram_range=(1, 2))
    hdbscan_model = HDBSCAN(min_cluster_size=20, min_samples=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

    topic_model = BERTopic(
        vectorizer_model=vectorizer,
        hdbscan_model=hdbscan_model,
        nr_topics=32,
        calculate_probabilities=True,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(cleaned_docs, embeddings)

    # Get topic information
    topic_info = topic_model.get_topic_info()
    print("\n--- Discovered Topics (Top 10) ---")
    print(topic_info.head(10)[['Topic', 'Count', 'Name']])

    # Save topic labels dictionary
    topic_labels = {}
    for _, row in topic_info.iterrows():
        t_id = int(row['Topic'])
        t_name = str(row['Name'])
        t_words = [w[0] for w in topic_model.get_topic(t_id)] if topic_model.get_topic(t_id) else []
        topic_labels[str(t_id)] = {
            'name': t_name,
            'count': int(row['Count']),
            'top_words': t_words
        }

    with open(TOPICS_JSON, 'w') as f:
        json.dump(topic_labels, f, indent=2, ensure_ascii=False)
    print(f"Saved topic labels to: {TOPICS_JSON}")

    # Build output dataframe
    max_prob = np.max(probs, axis=1) if (probs is not None and len(probs.shape) > 1) else np.zeros(n_songs, dtype=np.float32)
    out_df = pd.DataFrame({
        'row_idx': np.arange(n_songs, dtype=np.int32),
        'track_id': df['track_id'].values,
        'topic_id': np.array(topics, dtype=np.int32),
        'topic_confidence': np.round(max_prob.astype(np.float32), 4)
    })

    out_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved BERTopic features to: {OUTPUT_FILE}")
    print(f"Shape: {out_df.shape}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()
