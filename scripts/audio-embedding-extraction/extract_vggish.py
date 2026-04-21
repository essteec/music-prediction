"""
VGGish Audio Embedding Extractor

Extracts 128-dimensional embeddings using Google's VGGish model.
VGGish was trained on YouTube-8M dataset for audio classification.

Model: TensorFlow Hub VGGish
Input: Audio at 16kHz, mono
Output: 128-d embedding vector per song (temporal mean pooled)

Usage:
    python extract_vggish.py                    # Full extraction
    python extract_vggish.py --test 10         # Test on 10 songs
    python extract_vggish.py --resume          # Resume from checkpoint

Author: Music Prediction Project
Date: April 2026
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
import utils

# TensorFlow setup - limit GPU memory growth
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info messages

import tensorflow as tf
import tensorflow_hub as hub

# Configure TensorFlow to use limited GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] TensorFlow GPU memory growth enabled")
    except RuntimeError as e:
        print(f"[WARN] Could not set GPU memory growth: {e}")


# Model constants
MODEL_NAME = 'vggish'
EMBEDDING_DIM = 128
SAMPLE_RATE = 16000  # VGGish requires 16kHz

# VGGish model URL
VGGISH_MODEL_URL = "https://tfhub.dev/google/vggish/1"


def load_vggish_model():
    """
    Load VGGish model from TensorFlow Hub.
    
    Returns:
        Loaded VGGish model
    """
    print("[INFO] Loading VGGish model from TensorFlow Hub...")
    start = time.time()
    model = hub.load(VGGISH_MODEL_URL)
    print(f"[INFO] VGGish loaded in {time.time()-start:.1f}s")
    return model


def extract_vggish_embedding(model, audio: np.ndarray) -> np.ndarray:
    """
    Extract VGGish embedding from audio waveform.
    
    Args:
        model: Loaded VGGish model
        audio: Audio waveform at 16kHz, shape (samples,)
        
    Returns:
        Embedding of shape (128,) after temporal mean pooling
    """
    # VGGish expects float32 waveform
    audio = audio.astype(np.float32)
    
    # Wrap in tf.constant for safer input handling
    audio_tensor = tf.constant(audio, dtype=tf.float32)
    
    # Run model - outputs shape (time_frames, 128)
    embeddings = model(audio_tensor)
    
    # Convert to numpy and mean pool over time
    embeddings_np = embeddings.numpy()
    
    # Mean pool: (time_frames, 128) -> (128,)
    pooled = embeddings_np.mean(axis=0)
    
    return pooled


def run_extraction(
    test_count: Optional[int] = None,
    resume: bool = False,
    checkpoint_interval: int = 1000
):
    """
    Run VGGish extraction on all downloaded audio files.
    
    Args:
        test_count: If set, only process this many songs (for testing)
        resume: If True, resume from checkpoint
        checkpoint_interval: Save checkpoint every N songs
    """
    print("="*60)
    print("VGGish Embedding Extraction")
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

    # Load model
    model = load_vggish_model()
    
    # Track progress
    successful = len(embeddings_dict)
    failed = len(error_log)
    error_ids = {e['spotify_id'] for e in error_log} # For fast lookup
    start_time = time.time()
    
    print(f"[INFO] Starting extraction...")
    print(f"[INFO] Total embeddings currently in memory: {len(embeddings_dict):,}")
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
            # Load audio at 16kHz
            audio, sr = utils.load_audio_file(filepath, target_sr=SAMPLE_RATE)
            
            # Extract embedding
            embedding = extract_vggish_embedding(model, audio)
            
            # Validate
            if not utils.validate_embeddings(embedding, EMBEDDING_DIM, spotify_id):
                raise ValueError("Embedding validation failed")
            
            # Store
            embeddings_dict[spotify_id] = embedding
            processed_ids.add(spotify_id)
            successful += 1
            
            time_taken = time.time() - song_start
            
            # Log progress
            utils.log_extraction_progress(
                MODEL_NAME, spotify_id, True, time_taken, 
                embedding_shape=embedding.shape
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
    
    # Save embeddings using the FULL ID list to preserve order
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
    parser = argparse.ArgumentParser(description="Extract VGGish embeddings from audio files")
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
