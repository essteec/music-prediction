"""
Leave-One-Group-Out (LOGO) Ablation Study for Lyric Similarity Fusion.

Models Evaluated:
  1. Harrier-OSS-v1-0.6B (1024-D)
  2. Multilingual E5-Large (1024-D)
  3. BGE-M3 (1024-D)

Methodology:
  - Each embedding is separately L2-normalized.
  - Baseline: All 3 groups concatenated and re-normalized (3072-D).
  - For each group: Rebuild fused representation without that group.
  - Compute Top-100 nearest neighbors (cosine similarity) on GPU.
  - Benchmark Metrics:
      * Neighbor Overlap @ 10, 50, 100 vs Baseline
      * Genre Agreement Rate @ 10 & Delta (Δ)
      * Artist Agreement Rate @ 10 & Delta (Δ)
Outputs:
  - CLI formatted summary table
  - docs/lyric_similarity_ablation_report.md
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EMB_DIR = DATA_DIR / "embeddings" / "lyric"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
REPORT_MD = PROJECT_ROOT / "docs" / "lyric_similarity_ablation_report.md"

MODELS = {
    "Harrier-0.6B": {"path": EMB_DIR / "harrier_embeddings_1024d.npy", "dim": 1024, "desc": "Microsoft Harrier-OSS-v1-0.6B (1024-D)"},
    "E5-Large": {"path": EMB_DIR / "multilingual_e5_large_1024d.npy", "dim": 1024, "desc": "Multilingual E5-Large (1024-D)"},
    "BGE-M3": {"path": EMB_DIR / "bge_m3_1024d.npy", "dim": 1024, "desc": "BAAI BGE-M3 Multilingual (1024-D)"},
}

def load_and_norm(path: Path) -> torch.Tensor:
    arr = np.load(path).astype(np.float32)
    t = torch.from_numpy(arr)
    # Mask zero vectors
    nonzero = (torch.norm(t, p=2, dim=1, keepdim=True) > 1e-6).float()
    norms = torch.norm(t, p=2, dim=1, keepdim=True).clamp(min=1e-12)
    return (t / norms) * nonzero

def get_fused_top_k(tensors: list, k: int = 100, device: str = "cuda") -> torch.Tensor:
    concat = torch.cat(tensors, dim=1).to(device)
    nonzero = (torch.norm(concat, p=2, dim=1, keepdim=True) > 1e-6).float()
    norms = torch.norm(concat, p=2, dim=1, keepdim=True).clamp(min=1e-12)
    normed = (concat / norms) * nonzero
    
    n = normed.shape[0]
    top_indices = torch.zeros((n, k), dtype=torch.int64, device=device)
    
    block_size = 2000
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        sim_block = torch.matmul(normed[start:end], normed.T)
        for i in range(start, end):
            sim_block[i - start, i] = -1e9
        _, top_idx = torch.topk(sim_block, k=k, dim=1, largest=True)
        top_indices[start:end] = top_idx
        
    return top_indices.cpu()

def compute_overlap_at_k(base_topk: torch.Tensor, ablated_topk: torch.Tensor, valid_indices: list, k: int) -> float:
    overlaps = []
    for i in valid_indices:
        s_base = set(base_topk[i, :k].numpy())
        s_abl = set(ablated_topk[i, :k].numpy())
        overlaps.append(len(s_base & s_abl) / k)
    return float(np.mean(overlaps)) * 100.0 if overlaps else 0.0

def parse_set(val):
    if pd.isna(val) or not str(val).strip():
        return set()
    s = str(val).replace('|', ',').lower()
    return {x.strip() for x in s.split(',') if x.strip()}

def evaluate_agreement(top_k_indices: torch.Tensor, ground_truth_sets: list, valid_indices: list, k: int = 10) -> float:
    agreements = []
    for i in valid_indices:
        gt_i = ground_truth_sets[i]
        if not gt_i:
            continue
        neighbors = top_k_indices[i, :k].numpy()
        matches = sum(1 for nb in neighbors if bool(gt_i & ground_truth_sets[nb]))
        agreements.append(matches / k)
    return float(np.mean(agreements)) * 100.0 if agreements else 0.0

def classify_impact(ov10: float, g_delta: float, a_delta: float) -> str:
    if g_delta <= -0.5 or a_delta <= -0.5:
        return "Essential Signal (Keep - High Quality Drop)"
    elif (g_delta < -0.1 or a_delta < -0.1) and ov10 < 85.0:
        return "Beneficial Signal (Keep - Quality Drop)"
    elif (g_delta < -0.05 or a_delta < -0.05) and ov10 < 85.0:
        return "Moderate Signal (Keep)"
    elif ov10 >= 95.0 and abs(g_delta) < 0.05 and abs(a_delta) < 0.05:
        return "Redundant (Negligible Unique Value)"
    elif ov10 < 85.0 and (g_delta >= 0.05 or a_delta >= 0.05):
        return "Distinct but Harmful / Noisy (Drop Candidate)"
    else:
        return "Marginal / Neutral Contribution"

def main():
    print("\n" + "="*75)
    print("LEAVE-ONE-GROUP-OUT (LOGO) ABLATION STUDY: LYRIC SIMILARITY FUSION")
    print("="*75 + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load metadata
    df = pd.read_csv(SONGS_CSV)
    n_songs = len(df)
    
    # Identify valid tracks with lyrics
    valid_lyric_mask = df['lyrics'].notna() & (df['lyrics'].str.strip() != "")
    valid_indices = df[valid_lyric_mask].index.tolist()
    print(f"Loaded {n_songs:,} songs ({len(valid_indices):,} with valid lyrics).")

    # Ground-truth sets
    artist_sets = [parse_set(r['artist_names']) for _, r in df.iterrows()]
    genre_sets = []
    for _, r in df.iterrows():
        g1 = parse_set(r['main_genres'])
        g2 = parse_set(r['artist_genres'])
        genre_sets.append(g1 | g2)

    # Load tensors
    loaded_tensors = {}
    for name, meta in MODELS.items():
        print(f"Loading {name} ({meta['dim']}-D)...")
        loaded_tensors[name] = load_and_norm(meta['path'])

    # Baseline
    print("\n[1/4] Computing Full 3-Model Baseline Lyric Fusion (Harrier + E5 + BGE-M3)...")
    all_names = list(MODELS.keys())
    baseline_tensors = [loaded_tensors[m] for m in all_names]
    baseline_dim = sum(MODELS[m]['dim'] for m in all_names)
    
    t0 = time.time()
    baseline_top100 = get_fused_top_k(baseline_tensors, k=100, device=device)
    baseline_time = time.time() - t0
    
    base_genre_agr = evaluate_agreement(baseline_top100, genre_sets, valid_indices, k=10)
    base_artist_agr = evaluate_agreement(baseline_top100, artist_sets, valid_indices, k=10)
    
    print(f"  Baseline ({baseline_dim}-D) computed in {baseline_time:.2f}s")
    print(f"  Baseline Genre Agreement @10:  {base_genre_agr:.2f}%")
    print(f"  Baseline Artist Agreement @10: {base_artist_agr:.2f}%")

    results = []
    results.append({
        "group": "None (Full Baseline)",
        "dim": baseline_dim,
        "overlap10": 100.0,
        "overlap50": 100.0,
        "overlap100": 100.0,
        "genre_agr": base_genre_agr,
        "genre_delta": 0.0,
        "artist_agr": base_artist_agr,
        "artist_delta": 0.0,
        "impact": "Reference (Full 3-Model Ensemble)"
    })

    # Leave-One-Out
    for idx, excluded in enumerate(all_names, start=2):
        included = [m for m in all_names if m != excluded]
        inc_tensors = [loaded_tensors[m] for m in included]
        rem_dim = sum(MODELS[m]['dim'] for m in included)

        print(f"\n[{idx}/4] Ablating '{excluded}' (Remaining: {rem_dim}-D)...")
        abl_top100 = get_fused_top_k(inc_tensors, k=100, device=device)

        ov10 = compute_overlap_at_k(baseline_top100, abl_top100, valid_indices, k=10)
        ov50 = compute_overlap_at_k(baseline_top100, abl_top100, valid_indices, k=50)
        ov100 = compute_overlap_at_k(baseline_top100, abl_top100, valid_indices, k=100)

        g_agr = evaluate_agreement(abl_top100, genre_sets, valid_indices, k=10)
        g_delta = g_agr - base_genre_agr

        a_agr = evaluate_agreement(abl_top100, artist_sets, valid_indices, k=10)
        a_delta = a_agr - base_artist_agr

        verdict = classify_impact(ov10, g_delta, a_delta)

        results.append({
            "group": f"Without {excluded}",
            "dim": rem_dim,
            "overlap10": ov10,
            "overlap50": ov50,
            "overlap100": ov100,
            "genre_agr": g_agr,
            "genre_delta": g_delta,
            "artist_agr": a_agr,
            "artist_delta": a_delta,
            "impact": verdict
        })
        print(f"  Overlap@10: {ov10:.1f}% | Overlap@50: {ov50:.1f}% | Overlap@100: {ov100:.1f}%")
        print(f"  Genre Agr: {g_agr:.2f}% (Δ {g_delta:+.2f}%) | Artist Agr: {a_agr:.2f}% (Δ {a_delta:+.2f}%) | Verdict: {verdict}")

    # Display Table
    print("\n" + "="*125)
    print("FINAL LYRIC ABLATION RESULTS SUMMARY")
    print("="*125)
    header = f"{'Configuration':<25} | {'Dim':<6} | {'Ov@10':<7} | {'Ov@50':<7} | {'Ov@100':<7} | {'Genre@10':<9} | {'Genre Δ':<8} | {'Artist@10':<9} | {'Artist Δ':<8} | {'Impact / Verdict'}"
    print(header)
    print("-" * 125)
    for r in results:
        g_d_str = f"{r['genre_delta']:+.2f}%" if r['group'] != "None (Full Baseline)" else "0.00%"
        a_d_str = f"{r['artist_delta']:+.2f}%" if r['group'] != "None (Full Baseline)" else "0.00%"
        print(f"{r['group']:<25} | {r['dim']:<6} | {r['overlap10']:>6.1f}% | {r['overlap50']:>6.1f}% | {r['overlap100']:>6.1f}% | {r['genre_agr']:>8.2f}% | {g_d_str:>8} | {r['artist_agr']:>8.2f}% | {a_d_str:>8} | {r['impact']}")
    print("="*125 + "\n")

    generate_markdown_report(results, baseline_dim)

def generate_markdown_report(results: list, baseline_dim: int):
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_MD, 'w') as f:
        f.write("# Lyric Similarity Fusion: Leave-One-Group-Out Ablation Study\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This study evaluates the individual contribution of each multilingual lyric embedding model (`Harrier-OSS-v1-0.6B`, `Multilingual E5-Large`, `BGE-M3`) within the fused Top-100 kNN graph for the 10,000 Spotify tracks dataset.\n\n")
        
        f.write("### Ablation Results Table\n\n")
        f.write("| Configuration | Remaining Dim | Overlap @10 | Overlap @50 | Overlap @100 | Genre Agr @10 | Genre Δ | Artist Agr @10 | Artist Δ | Impact Verdict |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results:
            g_d_str = f"{r['genre_delta']:+.2f}%" if r['group'] != "None (Full Baseline)" else "0.00%"
            a_d_str = f"{r['artist_delta']:+.2f}%" if r['group'] != "None (Full Baseline)" else "0.00%"
            f.write(f"| **{r['group']}** | {r['dim']} | {r['overlap10']:.1f}% | {r['overlap50']:.1f}% | {r['overlap100']:.1f}% | {r['genre_agr']:.2f}% | `{g_d_str}` | {r['artist_agr']:.2f}% | `{a_d_str}` | **{r['impact']}** |\n")
        
        f.write("\n## Metric Definitions\n\n")
        f.write("1. **Neighbor Overlap @ K (Jaccard Rank Overlap):** Percentage of Top-K nearest neighbors shared with the full 3-model baseline. Lower overlap indicates the removed model provides a **distinct semantic representation**.\n")
        f.write("2. **Genre Agreement Rate @ 10 & Delta (Δ):** Percentage of Top-10 neighbors that share at least one genre with the query song. Negative Δ indicates removing the model degrades genre consistency.\n")
        f.write("3. **Artist Agreement Rate @ 10 & Delta (Δ):** Percentage of Top-10 neighbors by the same artist/collaborator.\n\n")

    print(f"Saved full markdown report to: {REPORT_MD}")

if __name__ == "__main__":
    main()
