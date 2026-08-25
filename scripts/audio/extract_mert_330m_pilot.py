"""
Phase 4: MERT-v1-330M Pilot Extraction (200 Songs).
Compares MERT-330M (1024-D) vs existing MERT-95M (768-D) on 200 tracks.
Outputs: data/embeddings/audio/pilot/mert_330m_pilot_200.npy & comparison metrics.
"""

import os
import gc
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm
from scipy.stats import pearsonr
from transformers import AutoModel, AutoFeatureExtractor

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
AUDIO_DIR = DATA_DIR / "audio" / "pilot"
MERT_95M_NPY = DATA_DIR / "embeddings" / "audio" / "mert_embeddings_768d.npy"
PILOT_DIR = DATA_DIR / "embeddings" / "audio" / "pilot"
OUTPUT_FILE = PILOT_DIR / "mert_330m_pilot_200.npy"
OUTPUT_REPORT = PROJECT_ROOT / "docs" / "mert_330m_pilot_results.md"

MODEL_ID = "m-a-p/MERT-v1-330M"
SAMPLE_RATE = 24000
CHUNK_SEC = 30
SAMPLE_SIZE = 200

def main():
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading MERT-95M baseline...")
    m95 = np.load(MERT_95M_NPY)[:SAMPLE_SIZE]
    df = pd.read_csv(SONGS_CSV).iloc[:SAMPLE_SIZE]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device} (fp16)...")
    
    processor = AutoFeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
    if device == "cuda":
        model = model.half()
    model.eval()

    embeddings_330m = np.zeros((SAMPLE_SIZE, 1024), dtype=np.float32)

    print(f"Extracting MERT-330M embeddings for {SAMPLE_SIZE} tracks (chunked 30s)...")
    t0 = time.time()
    
    for i in tqdm(range(SAMPLE_SIZE)):
        audio_path = AUDIO_DIR / f"{i:06d}_opus.webm"
        if not audio_path.exists():
            continue

        try:
            # Load first 30s for pilot
            audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, duration=CHUNK_SEC, mono=True)
            inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            input_values = inputs["input_values"].to(device)
            if device == "cuda":
                input_values = input_values.half()

            with torch.no_grad():
                outputs = model(input_values, output_hidden_states=True)
                # Mean pool last hidden state
                last_hidden = outputs.last_hidden_state.squeeze(0)  # (T, 1024)
                pooled = torch.mean(last_hidden, dim=0).cpu().float().numpy()
                embeddings_330m[i] = pooled
        except Exception as e:
            print(f"Error track {i}: {e}")

        if i % 25 == 0 and device == "cuda":
            torch.cuda.empty_cache()

    t_total = time.time() - t0
    np.save(OUTPUT_FILE, embeddings_330m)
    print(f"\nSaved MERT-330M pilot embeddings to: {OUTPUT_FILE}")
    print(f"Extraction time for 200 tracks: {t_total:.1f}s ({t_total/SAMPLE_SIZE*1000:.1f} ms/song)")

    # Correlation Analysis
    # Compare first 768 dims of 330M with 95M
    r_vals = []
    for i in range(SAMPLE_SIZE):
        if np.any(embeddings_330m[i] != 0) and np.any(m95[i] != 0):
            r, _ = pearsonr(m95[i], embeddings_330m[i, :768])
            r_vals.append(r)

    mean_r = float(np.mean(r_vals)) if r_vals else 0.0
    print(f"\nMean Pearson correlation with MERT-95M: {mean_r:.4f}")

    # Write report
    with open(OUTPUT_REPORT, 'w') as f:
        f.write("# MERT-v1-330M Pilot Benchmark Report\n\n")
        f.write(f"- **Sample Size:** {SAMPLE_SIZE} tracks\n")
        f.write(f"- **Mean Correlation vs MERT-95M:** `{mean_r:.4f}`\n")
        f.write(f"- **Inference Speed:** `{t_total/SAMPLE_SIZE*1000:.1f} ms / song`\n")
        f.write(f"- **Projected Full 10k Time:** `~{t_total/SAMPLE_SIZE*10000/3600:.2f} hours`\n\n")
        if mean_r > 0.90:
            f.write("### Recommendation: **SKIP Full 10k MERT-330M Extraction**\n")
            f.write(f"Correlation is extremely high ({mean_r:.4f} > 0.90) indicating massive feature redundancy with the already extracted MERT-95M.\n")
        else:
            f.write("### Recommendation: **PROCEED with Full 10k MERT-330M Extraction**\n")
            f.write(f"Correlation is moderate ({mean_r:.4f} <= 0.90), showing complementary acoustic information.\n")

    print(f"Report saved to: {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()
