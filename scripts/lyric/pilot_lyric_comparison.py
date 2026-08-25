"""
Phase 1.5: 500-Song Lyric Model Pilot Benchmark (Isolated Process per Model).
Compares top candidate multilingual embedding models on a stratified 500-song subset:
1. BAAI/bge-m3
2. Alibaba-NLP/gte-multilingual-base
3. intfloat/multilingual-e5-large
4. ibm-granite/granite-embedding-311m-multilingual-r2
5. Baseline: sentence-transformers/all-mpnet-base-v2
Outputs: docs/lyric_pilot_results.md
"""

import os
import re
import time
import json
import subprocess
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
LANG_PARQUET = DATA_DIR / "features" / "lyric" / "language_id.parquet"
OUTPUT_REPORT = PROJECT_ROOT / "docs" / "lyric_pilot_results.md"

MODELS_CONFIG = [
    {"name": "BGE-M3", "id": "BAAI/bge-m3", "license": "MIT", "dim": 1024, "max_tok": 4096},
    {"name": "GTE-multilingual-base", "id": "Alibaba-NLP/gte-multilingual-base", "license": "Apache-2.0", "dim": 768, "max_tok": 4096, "trust_remote": True},
    {"name": "multilingual-E5-large", "id": "intfloat/multilingual-e5-large", "license": "MIT", "dim": 1024, "max_tok": 512},
    {"name": "granite-311m-multilingual", "id": "ibm-granite/granite-embedding-311m-multilingual-r2", "license": "Apache-2.0", "dim": 768, "max_tok": 2048},
    {"name": "Baseline: MPNet", "id": "sentence-transformers/all-mpnet-base-v2", "license": "Apache-2.0", "dim": 768, "max_tok": 512}
]

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

def compute_ndcg_at_k(sim_matrix, relevant_mask, k=10):
    n_queries = sim_matrix.shape[0]
    ndcgs = []
    
    for i in range(n_queries):
        rel = relevant_mask[i].copy()
        rel[i] = 0
        if rel.sum() == 0:
            continue
            
        scores = sim_matrix[i].copy()
        scores[i] = -999.0
        
        ranked_idx = np.argsort(scores)[::-1][:k]
        actual_rel = rel[ranked_idx]
        
        dcg = np.sum(actual_rel / np.log2(np.arange(2, len(actual_rel) + 2)))
        ideal_rel = np.sort(rel)[::-1][:k]
        idcg = np.sum(ideal_rel / np.log2(np.arange(2, len(ideal_rel) + 2)))
        
        ndcg = dcg / (idcg + 1e-8)
        ndcgs.append(ndcg)
        
    return float(np.mean(ndcgs)) if ndcgs else 0.0

def evaluate_model_isolated(model_cfg, cleaned_lyrics, genre_mask, artist_mask, sample_lang):
    import torch
    from sentence_transformers import SentenceTransformer
    
    model_name = model_cfg['name']
    model_id = model_cfg['id']
    trust_remote = model_cfg.get('trust_remote', False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    t0 = time.time()
    model = SentenceTransformer(model_id, device=device, trust_remote_code=trust_remote)
    if hasattr(model, 'max_seq_length'):
        model.max_seq_length = min(model_cfg['max_tok'], 4096)
        
    t1 = time.time()
    embeddings = model.encode(cleaned_lyrics, batch_size=16, show_progress_bar=False, normalize_embeddings=True)
    t_encode = time.time() - t1
    time_per_song = (t_encode / len(cleaned_lyrics)) * 1000.0
    
    sim_matrix = cosine_similarity(embeddings)
    
    ndcg_genre = compute_ndcg_at_k(sim_matrix, genre_mask, k=10)
    ndcg_artist = compute_ndcg_at_k(sim_matrix, artist_mask, k=5)
    
    non_en_mask = (~sample_lang['is_english']).values
    non_en_idx = np.where(non_en_mask)[0]
    if len(non_en_idx) > 10:
        ndcg_non_en = compute_ndcg_at_k(sim_matrix[non_en_idx][:, non_en_idx], genre_mask[non_en_idx][:, non_en_idx], k=10)
    else:
        ndcg_non_en = 0.0

    hi_mask = sample_lang['is_hindi_any'].values
    hi_idx = np.where(hi_mask)[0]
    hi_coherence = float(np.mean(sim_matrix[hi_idx][:, hi_idx])) if len(hi_idx) > 1 else 0.0

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'Model': model_name,
        'License': model_cfg['license'],
        'Dim': model_cfg['dim'],
        'Max Tokens': model_cfg['max_tok'],
        'Time/Song (ms)': round(time_per_song, 1),
        'Total 500 Time (s)': round(t_encode, 1),
        'Genre nDCG@10 (Overall)': round(ndcg_genre, 4),
        'Genre nDCG@10 (Non-English)': round(ndcg_non_en, 4),
        'Artist nDCG@5': round(ndcg_artist, 4),
        'Hindi/Hinglish Coherence': round(hi_coherence, 4),
        'Status': 'Success'
    }

def main():
    print(f"Loading songs and language metadata...")
    df_songs = pd.read_csv(SONGS_CSV)
    df_lang = pd.read_parquet(LANG_PARQUET)

    valid_mask = df_lang['has_lyrics'] & (df_songs['lyrics'].str.len() > 50)
    df_valid = df_songs[valid_mask].copy().reset_index(drop=True)
    df_lang_valid = df_lang[valid_mask].copy().reset_index(drop=True)

    rng = np.random.RandomState(42)
    idx_en = df_lang_valid[df_lang_valid['is_english']].index.values
    idx_es = df_lang_valid[df_lang_valid['is_spanish']].index.values
    idx_hi = df_lang_valid[df_lang_valid['is_hindi_any']].index.values
    idx_pt = df_lang_valid[df_lang_valid['is_portuguese']].index.values
    idx_asian = df_lang_valid[df_lang_valid['is_korean'] | df_lang_valid['is_japanese'] | df_lang_valid['is_chinese']].index.values
    idx_other = df_lang_valid[~df_lang_valid.index.isin(np.concatenate([idx_en, idx_es, idx_hi, idx_pt, idx_asian]))].index.values

    sample_en = rng.choice(idx_en, min(250, len(idx_en)), replace=False)
    sample_es = rng.choice(idx_es, min(100, len(idx_es)), replace=False)
    sample_hi = rng.choice(idx_hi, min(40, len(idx_hi)), replace=False)
    sample_pt = rng.choice(idx_pt, min(30, len(idx_pt)), replace=False)
    sample_asian = rng.choice(idx_asian, min(30, len(idx_asian)), replace=False)
    sample_other = rng.choice(idx_other, min(50, len(idx_other)), replace=False)

    sampled_indices = np.unique(np.concatenate([sample_en, sample_es, sample_hi, sample_pt, sample_asian, sample_other]))
    sample_df = df_valid.iloc[sampled_indices].copy().reset_index(drop=True)
    sample_lang = df_lang_valid.iloc[sampled_indices].copy().reset_index(drop=True)
    
    cleaned_lyrics = [clean_lyric(t) for t in sample_df['lyrics']]

    artist_mask = np.zeros((len(sample_df), len(sample_df)), dtype=np.int32)
    for i, a1 in enumerate(sample_df['artist_names']):
        for j, a2 in enumerate(sample_df['artist_names']):
            if i != j and a1 == a2:
                artist_mask[i, j] = 1

    genre_mask = np.zeros((len(sample_df), len(sample_df)), dtype=np.int32)
    for i, g1 in enumerate(sample_df['main_genres']):
        for j, g2 in enumerate(sample_df['main_genres']):
            if i != j and isinstance(g1, str) and isinstance(g2, str) and g1 == g2:
                genre_mask[i, j] = 1

    results_table = []

    for cfg in MODELS_CONFIG:
        print(f"--- Evaluating {cfg['name']} ---")
        try:
            res = evaluate_model_isolated(cfg, cleaned_lyrics, genre_mask, artist_mask, sample_lang)
            results_table.append(res)
            print(f"-> Success: Genre nDCG={res['Genre nDCG@10 (Overall)']}, Non-EN nDCG={res['Genre nDCG@10 (Non-English)']}, Time={res['Time/Song (ms)']}ms\n")
        except Exception as e:
            print(f"-> Failed {cfg['name']}: {e}\n")
            results_table.append({
                'Model': cfg['name'], 'License': cfg['license'], 'Dim': cfg['dim'],
                'Max Tokens': cfg['max_tok'], 'Time/Song (ms)': 0, 'Total 500 Time (s)': 0,
                'Genre nDCG@10 (Overall)': 0, 'Genre nDCG@10 (Non-English)': 0,
                'Artist nDCG@5': 0, 'Hindi/Hinglish Coherence': 0, 'Status': f'Failed: {str(e)[:50]}'
            })

    res_df = pd.DataFrame(results_table)
    
    with open(OUTPUT_REPORT, 'w') as f:
        f.write("# Phase 1.5: 500-Song Lyric Model Pilot Benchmark Report\n\n")
        f.write(f"> **Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"> **Sample Size:** {len(sample_df)} stratified tracks ({sum(sample_lang['is_english'])} English, {sum(~sample_lang['is_english'])} Non-English)\n\n")
        f.write("## Benchmark Results Table\n\n")
        f.write(res_df.to_markdown(index=False))
        f.write("\n\n## Recommendations & Decisions\n")
        
        success_df = res_df[res_df['Status'] == 'Success'].sort_values(by='Genre nDCG@10 (Non-English)', ascending=False)
        if len(success_df) > 0:
            top_model = success_df.iloc[0]
            f.write(f"- **Top Winner:** `{top_model['Model']}` (Non-English nDCG: {top_model['Genre nDCG@10 (Non-English)']}, Dim: {top_model['Dim']}, License: {top_model['License']})\n")
        if len(success_df) > 1:
            second_model = success_df.iloc[1]
            f.write(f"- **Second Place / Backup:** `{second_model['Model']}` (Non-English nDCG: {second_model['Genre nDCG@10 (Non-English)']}, Dim: {second_model['Dim']}, License: {second_model['License']})\n")

    print(f"\n=======================================================")
    print(f"PILOT BENCHMARK COMPLETED! Saved to {OUTPUT_REPORT}")
    print(f"=======================================================")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()
