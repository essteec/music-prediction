"""
Full-Song PANNs (2048-D) and AudioSet Tags (527-D) Extraction.
Model: Cnn14 (panns_inference, 32kHz)

Feeds 100% of each song waveform directly to Cnn14 (global average pooling
over time) to produce penultimate 2048-D embedding and 527-D class tag probabilities.

Outputs:
  - data/embeddings/audio/panns_embeddings_2048d.npy (10000, 2048) float32
  - data/embeddings/audio/panns_tags_527d.npy (10000, 527) float32
"""

import os
import gc
import json
import subprocess
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from panns_inference import AudioTagging, labels

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
AUDIO_DIR = DATA_DIR / "audio" / "pilot"
OUTPUT_DIR = DATA_DIR / "embeddings" / "audio"
EMB_OUTPUT_FILE = OUTPUT_DIR / "panns_embeddings_2048d.npy"
TAGS_OUTPUT_FILE = OUTPUT_DIR / "panns_tags_527d.npy"
LABELS_FILE = OUTPUT_DIR / "panns_tags_labels.json"
CHECKPOINT_DIR = DATA_DIR / "embeddings" / "checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "panns_full_checkpoint.npz"

SAMPLE_RATE = 32000

def decode_audio_full(path: Path) -> np.ndarray:
    """Decode full audio to 32kHz float32 mono via ffmpeg pipe."""
    try:
        cmd = [
            "ffmpeg", "-i", str(path),
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-"
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return np.frombuffer(res.stdout, dtype=np.float32)
    except Exception:
        return np.zeros(0, dtype=np.float32)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Save labels JSON
    with open(LABELS_FILE, 'w') as f:
        json.dump(labels, f, indent=2)

    df = pd.read_csv(SONGS_CSV)
    n_songs = len(df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f"Full-Song PANNs (2048-D) & Tags (527-D) Extraction ({device})")
    print(f"Sampling Rate: {SAMPLE_RATE} Hz | Tracks: {n_songs}")
    print(f"Target Embeddings: {EMB_OUTPUT_FILE}")
    print(f"Target Tags:       {TAGS_OUTPUT_FILE}")
    print(f"=======================================================\n")

    if CHECKPOINT_FILE.exists():
        ckpt = np.load(CHECKPOINT_FILE)
        embeddings = ckpt['embeddings']
        tags = ckpt['tags']
        done = np.where(np.any(embeddings != 0, axis=1))[0]
        start_idx = int(done[-1]) + 1 if len(done) > 0 else 0
        print(f"Resuming from checkpoint at track {start_idx}/{n_songs}...")
    else:
        embeddings = np.zeros((n_songs, 2048), dtype=np.float32)
        tags = np.zeros((n_songs, 527), dtype=np.float32)
        start_idx = 0

    print("Loading PANNs AudioTagging model...")
    model = AudioTagging(checkpoint_path=None, device=device)

    pbar = tqdm(total=n_songs, initial=start_idx, desc="Extracting PANNs")

    for i in range(start_idx, n_songs):
        audio_path = AUDIO_DIR / f"{i:06d}_opus.webm"
        if not audio_path.exists():
            pbar.update(1)
            continue

        try:
            audio = decode_audio_full(audio_path)
            if len(audio) >= SAMPLE_RATE * 2:
                # Direct full waveform inference
                clipwise_output, emb = model.inference(audio[np.newaxis, :])
                tags[i] = clipwise_output[0].astype(np.float32)
                embeddings[i] = emb[0].astype(np.float32)
        except Exception as e:
            # If OOM on extreme audio length, slice into 30s chunks as fallback
            try:
                torch.cuda.empty_cache()
                chunk_samples = SAMPLE_RATE * 30
                c_embs, c_tags = [], []
                for s in range(0, len(audio), chunk_samples):
                    c = audio[s : s + chunk_samples]
                    if len(c) >= SAMPLE_RATE * 2:
                        cw, em = model.inference(c[np.newaxis, :])
                        c_tags.append(cw[0])
                        c_embs.append(em[0])
                if c_embs:
                    embeddings[i] = np.mean(c_embs, axis=0).astype(np.float32)
                    tags[i] = np.mean(c_tags, axis=0).astype(np.float32)
            except Exception:
                pass

        pbar.update(1)

        if (i + 1) % 50 == 0 and device == "cuda":
            torch.cuda.empty_cache()

        if (i + 1) % 250 == 0 or (i + 1) == n_songs:
            np.savez_compressed(CHECKPOINT_FILE, embeddings=embeddings, tags=tags)
            gc.collect()

    pbar.close()

    np.save(EMB_OUTPUT_FILE, embeddings)
    np.save(TAGS_OUTPUT_FILE, tags)

    filled = int(np.any(embeddings != 0, axis=1).sum())
    print(f"\n=======================================================")
    print(f"PANNs Extraction Complete!")
    print(f"Saved Embeddings: {EMB_OUTPUT_FILE} ({EMB_OUTPUT_FILE.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Saved Tags:       {TAGS_OUTPUT_FILE} ({TAGS_OUTPUT_FILE.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Filled: {filled}/{n_songs} | NaNs: {np.isnan(embeddings).sum()}")
    print(f"=======================================================\n")

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

if __name__ == "__main__":
    main()
