"""
Mel Spectrogram Statistics Extractor

Extracts 512-dimensional feature vector from mel spectrograms:
- 128 mel frequency bands
- 4 statistics per band: mean, std, max, min

This captures low-level frequency information that complements
high-level embeddings from neural networks (MERT, VGGish, PANNs).

Input: Audio at 22050Hz (librosa default)
Output: 512-d feature vector per song

Usage:
    python extract_mel_stats.py                 # Full extraction
    python extract_mel_stats.py --test 10      # Test on 10 songs
    python extract_mel_stats.py --resume       # Resume from checkpoint

Author: Music Prediction Project
Date: April 2026
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import librosa
import warnings

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
import utils

# Model constants
MODEL_NAME = 'mel_stats'
EMBEDDING_DIM = 512  # 128 bands × 4 statistics
SAMPLE_RATE = 22050  # librosa default
N_MELS = 128         # Number of mel frequency bands
N_FFT = 2048         # FFT window size
HOP_LENGTH = 512     # Hop between frames


def extract_mel_stats(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract mel spectrogram statistics from audio.
    
    Args:
        audio: Audio waveform, shape (samples,)
        sr: Sample rate
        
    Returns:
        Feature vector of shape (512,) containing:
        - [0:128]: Mean of each mel band over time
        - [128:256]: Std of each mel band over time
        - [256:384]: Max of each mel band over time
        - [384:512]: Min of each mel band over time
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    
    # Convert to dB scale (log scale, more perceptually relevant)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Compute statistics over time axis (axis=1)
    # mel_spec_db shape: (n_mels, time_frames)
    mel_mean = mel_spec_db.mean(axis=1)   # (128,)
    mel_std = mel_spec_db.std(axis=1)     # (128,)
    mel_max = mel_spec_db.max(axis=1)     # (128,)
    mel_min = mel_spec_db.min(axis=1)     # (128,)
    
    # Concatenate: (128 × 4) = 512 features
    features = np.concatenate([mel_mean, mel_std, mel_max, mel_min])
    
    return features


def run_extraction(
    test_count: Optional[int] = None,
    resume: bool = False,
    checkpoint_interval: int = 1000
):
    """
    Run mel spectrogram extraction on all downloaded audio files.
    
    Args:
        test_count: If set, only process this many songs (for testing)
        resume: If True, resume from checkpoint
        checkpoint_interval: Save checkpoint every N songs
    """
    print("="*60)
    print("Mel Spectrogram Statistics Extraction")
    print("="*60)
    
    # Get downloads
    downloads = utils.get_successful_downloads()
    total_songs = len(downloads)
    print(f"[INFO] Found {total_songs:,} audio files to process")
    
    if test_count:
        downloads = downloads[:test_count]
        print(f"[INFO] TEST MODE: Processing only {test_count} songs")
    
    # Initialize state
    embeddings_dict: Dict[str, np.ndarray] = {}
    processed_ids: set = set()
    error_log = []
    
    # Get current IDs as a set for fast lookup
    current_ids = {d[0] for d in downloads}
    all_target_ids = utils.get_all_successful_ids()
    
    # Resume from checkpoint if requested
    if resume:
        checkpoint = utils.load_checkpoint(MODEL_NAME)
        if checkpoint:
            # Load EVERYTHING from checkpoint (cumulative)
            embeddings_dict = checkpoint.get('embeddings', {})
            processed_ids = set(checkpoint.get('processed_ids', []))
            error_log = checkpoint.get('errors', [])
            
            # Count how many of the CURRENT batch are already done
            done_current = sum(1 for sid in current_ids if sid in processed_ids)
            if done_current > 0:
                print(f"[INFO] Resuming: {done_current} of {len(downloads)} current songs already done")
            
            # ROBUST EARLY EXIT: Only exit if all CURRENT IDs are accounted for
            error_ids = {e['spotify_id'] for e in error_log}
            if current_ids.issubset(processed_ids.union(error_ids)):
                print(f"[INFO] All {len(downloads)} current songs already processed according to checkpoint.")
                # Still save embeddings to be safe if file doesn't exist
                output_path = utils.EMBEDDINGS_DIR / f"{MODEL_NAME}_embeddings_{EMBEDDING_DIM}d.npy"
                if not output_path.exists():
                    utils.save_embeddings_npy(embeddings_dict, output_path, all_target_ids)
                return embeddings_dict

    # Track progress
    successful = len(embeddings_dict)
    failed = len(error_log)
    error_ids = {e['spotify_id'] for e in error_log} # For fast lookup
    start_time = time.time()
    
    print(f"[INFO] Starting extraction...")
    print(f"[INFO] Total embeddings currently in memory: {len(embeddings_dict):,}")
    print(f"[INFO] Using CPU (librosa-based)")
    print(f"[INFO] Checkpoint every {checkpoint_interval} songs")
    print("-"*60)
    
    # Process songs
    for idx, (spotify_id, filepath, row_idx) in enumerate(downloads):
        # Skip already processed (from ANY batch)
        if spotify_id in processed_ids:
            continue
        
        # Skip if already in error log too
        if spotify_id in error_ids:
            continue

        song_start = time.time()
        
        try:
            # Load audio at 22050Hz (librosa default)
            audio, sr = utils.load_audio_file(filepath, target_sr=SAMPLE_RATE)
            
            # Extract mel stats
            features = extract_mel_stats(audio, sr)
            
            # Validate
            if not utils.validate_embeddings(features, EMBEDDING_DIM, spotify_id):
                raise ValueError("Feature validation failed")
            
            # Store
            embeddings_dict[spotify_id] = features
            processed_ids.add(spotify_id)
            successful += 1
            
            time_taken = time.time() - song_start
            
            # Log progress
            utils.log_extraction_progress(
                MODEL_NAME, spotify_id, True, time_taken,
                embedding_shape=features.shape
            )
            
            # Progress update every 100 songs
            if successful % 100 == 0:
                elapsed = time.time() - start_time
                avg_time = (time.time() - start_time) / (idx + 1)
                remaining = (len(downloads) - idx) * avg_time
                print(f"[{idx+1:,}/{len(downloads):,}] {spotify_id} - "
                      f"{time_taken:.2f}s/song, ETA: {remaining/3600:.1f}h")
            
        except Exception as e:
            error_msg = str(e)
            error_log.append({
                'spotify_id': spotify_id,
                'row_idx': row_idx,
                'error': error_msg
            })
            error_ids.add(spotify_id) # Update lookup set
            failed += 1
            
            utils.log_extraction_progress(
                MODEL_NAME, spotify_id, False, time.time() - song_start,
                error_msg=error_msg
            )
            
            if failed <= 10:  # Only print first 10 errors
                print(f"[ERROR] {spotify_id}: {error_msg[:100]}")
        
        # Checkpoint
        if (idx + 1) % checkpoint_interval == 0:
            utils.create_checkpoint(
                MODEL_NAME,
                idx,
                list(processed_ids),
                embeddings_dict,
                error_log
            )
            print(f"[CHECKPOINT] Saved at index {idx+1}")
    
    # Final checkpoint after loop finishes
    utils.create_checkpoint(
        MODEL_NAME,
        len(downloads) - 1,
        list(processed_ids),
        embeddings_dict,
        error_log
    )
    print(f"[CHECKPOINT] Final checkpoint saved at {len(processed_ids)} total embeddings")

    # Final save
    total_time = time.time() - start_time
    
    # Save embeddings (use consistent naming with other extractors: model_embeddings_dimd.npy)
    output_path = utils.EMBEDDINGS_DIR / f"{MODEL_NAME}_embeddings_{EMBEDDING_DIM}d.npy"
    utils.save_embeddings_npy(embeddings_dict, output_path, all_target_ids)
    
    # Print summary
    utils.print_extraction_summary(
        MODEL_NAME.upper(),
        successful + failed,
        successful,
        failed,
        total_time
    )
    
    return embeddings_dict


def main():
    parser = argparse.ArgumentParser(description="Extract mel spectrogram statistics from audio files")
    parser.add_argument('--test', type=int, help="Test mode: process only N songs")
    parser.add_argument('--resume', action='store_true', help="Resume from checkpoint")
    parser.add_argument('--checkpoint-interval', type=int, default=1000,
                        help="Save checkpoint every N songs (default: 1000)")
    
    args = parser.parse_args()
    
    run_extraction(
        test_count=args.test,
        resume=args.resume,
        checkpoint_interval=args.checkpoint_interval
    )


if __name__ == "__main__":
    main()
