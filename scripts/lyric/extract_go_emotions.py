"""
GoEmotions 28-Class Emotion Probabilities Extraction Script.
Extracts 28 emotion probabilities for all English song lyrics using RoBERTa-base GoEmotions.
Model: SamLowe/roberta-base-go_emotions (Apache-2.0)
Outputs: data/features/lyric/go_emotions.parquet
"""

import os
import gc
import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
LANG_PARQUET = DATA_DIR / "features" / "lyric" / "language_id.parquet"
OUTPUT_DIR = DATA_DIR / "features" / "lyric"
OUTPUT_FILE = OUTPUT_DIR / "go_emotions.parquet"

MODEL_ID = "SamLowe/roberta-base-go_emotions"

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading GoEmotions model ({MODEL_ID}) on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device)
    model.eval()

    labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    print(f"Detected {len(labels)} GoEmotions labels: {labels[:5]}...")

    emotion_matrix = np.zeros((n_songs, len(labels)), dtype=np.float32)

    # Process English tracks with lyrics
    english_mask = df_lang['is_english'] & df_lang['has_lyrics']
    english_indices = np.where(english_mask)[0]
    print(f"Found {len(english_indices)} English tracks with lyrics to extract emotions...")

    batch_size = 32
    for b_start in tqdm(range(0, len(english_indices), batch_size)):
        b_end = min(b_start + batch_size, len(english_indices))
        batch_idx = english_indices[b_start:b_end]

        batch_texts = []
        for idx in batch_idx:
            txt = clean_lyric(df.iloc[idx]['lyrics'])
            batch_texts.append(txt[:1500] if txt else "")

        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()

        for local_i, global_i in enumerate(batch_idx):
            emotion_matrix[global_i] = probs[local_i]

        if b_start % 500 == 0 and device == "cuda":
            torch.cuda.empty_cache()

    # Build output dataframe
    out_dict = {
        'row_idx': np.arange(n_songs, dtype=np.int32),
        'track_id': df['track_id'].values,
        'is_english_annotated': english_mask.values
    }
    for i, label in enumerate(labels):
        out_dict[f'emotion_{label}'] = np.round(emotion_matrix[:, i], 4)

    out_df = pd.DataFrame(out_dict)
    out_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved GoEmotions features to: {OUTPUT_FILE}")
    print(f"Shape: {out_df.shape}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()
