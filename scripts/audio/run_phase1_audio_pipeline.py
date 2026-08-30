"""
Phase 1 Audio Extraction Orchestration Pipeline.
Sequentially runs and validates:
  1. LAION-CLAP (512-D) - 10s chunk mean-pool over full track
  2. PANNs Cnn14 (2048-D) + AudioSet Tags (527-D) - Full waveform single pass
  3. Google VGGish (128-D) - Full waveform frame-wise average
  4. Mel Spectrogram Stats (512-D) - Full waveform global statistics
  5. Consolidates lyrics/ -> lyric/ and audits all embeddings
"""

import sys
import time
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable

def run_step(step_name: str, script_path: str):
    print(f"\n{'='*70}")
    print(f"STARTING STEP: {step_name}")
    print(f"Script: {script_path}")
    print(f"{'='*70}\n")
    t0 = time.time()
    res = subprocess.run([PYTHON, script_path], cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with return code {res.returncode}!")
        sys.exit(res.returncode)
    elapsed = time.time() - t0
    print(f"\n[SUCCESS] {step_name} completed in {elapsed/60:.2f} minutes.")

def main():
    pipeline_start = time.time()
    print("\n" + "#"*70)
    print("#  PHASE 1 FULL-SONG AUDIO EXTRACTION PIPELINE INITIALIZED  #")
    print("#"*70 + "\n")

    # Step 1: CLAP
    run_step("1. LAION-CLAP (512-D)", "scripts/audio/extract_clap_full_song.py")

    # Step 2: PANNs & Tags
    run_step("2. PANNs (2048-D) & AudioSet Tags (527-D)", "scripts/audio/extract_panns_full_song.py")

    # Step 3: VGGish
    run_step("3. Google VGGish (128-D)", "scripts/audio/extract_vggish_full_song.py")

    # Step 4: Mel Stats
    run_step("4. Mel Spectrogram Statistics (512-D)", "scripts/audio/extract_mel_stats_full_song.py")

    # Step 5: Consolidate lyric directory naming
    print("\n" + "="*70)
    print("STEP 5: Consolidating lyrics/ into lyric/")
    print("="*70)
    lyric_dir = PROJECT_ROOT / "data" / "embeddings" / "lyric"
    lyrics_dir = PROJECT_ROOT / "data" / "embeddings" / "lyrics"
    lyric_dir.mkdir(parents=True, exist_ok=True)
    if lyrics_dir.exists():
        for f in lyrics_dir.iterdir():
            target = lyric_dir / f.name
            print(f"  Moving {f.name} -> data/embeddings/lyric/")
            shutil.move(str(f), str(target))
        try:
            lyrics_dir.rmdir()
            print("  Removed redundant data/embeddings/lyrics/ directory.")
        except Exception:
            pass

    # Step 6: Full Integrity Audit
    print("\n" + "="*70)
    print("STEP 6: Full Integrity Audit Across All Embeddings")
    print("="*70)

    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "songs.csv")
    n_songs = len(df)

    audit_files = [
        ("CLAP", PROJECT_ROOT / "data" / "embeddings" / "audio" / "clap_512d.npy", 512),
        ("PANNs", PROJECT_ROOT / "data" / "embeddings" / "audio" / "panns_embeddings_2048d.npy", 2048),
        ("PANNs Tags", PROJECT_ROOT / "data" / "embeddings" / "audio" / "panns_tags_527d.npy", 527),
        ("MERT-330M Mean", PROJECT_ROOT / "data" / "embeddings" / "audio" / "mert_330m_embeddings_1024d.npy", 1024),
        ("VGGish", PROJECT_ROOT / "data" / "embeddings" / "audio" / "vggish_embeddings_128d.npy", 128),
        ("Mel Stats", PROJECT_ROOT / "data" / "embeddings" / "audio" / "mel_stats_embeddings_512d.npy", 512),
        ("Harrier", PROJECT_ROOT / "data" / "embeddings" / "lyric" / "harrier_embeddings_1024d.npy", 1024),
        ("E5-Large", PROJECT_ROOT / "data" / "embeddings" / "lyric" / "multilingual_e5_large_1024d.npy", 1024),
        ("BGE-M3", PROJECT_ROOT / "data" / "embeddings" / "lyric" / "bge_m3_1024d.npy", 1024),
    ]

    all_passed = True
    print(f"\n{'Name':<18} | {'Shape':<14} | {'Dtype':<8} | {'Filled':<12} | {'NaNs':<6} | {'Status'}")
    print("-" * 75)
    for name, path, exp_dim in audit_files:
        if not path.exists():
            print(f"{name:<18} | {'MISSING':<14} | {'-':<8} | {'-':<12} | {'-':<6} | ❌ FAILED")
            all_passed = False
            continue
        arr = np.load(path)
        filled = int(np.any(arr != 0, axis=1).sum())
        nans = int(np.isnan(arr).sum())
        infs = int(np.isinf(arr).sum())
        shape_ok = (arr.shape == (n_songs, exp_dim))
        clean_ok = (nans == 0 and infs == 0)
        status = "✅ PASS" if (shape_ok and clean_ok) else "❌ FAILED"
        if not (shape_ok and clean_ok):
            all_passed = False
        print(f"{name:<18} | {str(arr.shape):<14} | {str(arr.dtype):<8} | {f'{filled}/{n_songs}':<12} | {nans:<6} | {status}")

    total_time = (time.time() - pipeline_start) / 60
    print("\n" + "="*70)
    if all_passed:
        print(f"PIPELINE COMPLETED SUCCESSFULLY in {total_time:.2f} minutes!")
    else:
        print(f"PIPELINE FINISHED WITH WARNINGS in {total_time:.2f} minutes.")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
