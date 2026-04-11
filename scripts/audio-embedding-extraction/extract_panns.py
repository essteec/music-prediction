"""
PANNs Audio Embedding Extractor

Extracts 2048-dimensional embeddings using PANNs (Pre-trained Audio Neural Networks).
PANNs is trained on AudioSet (2M+ audio clips) for audio tagging/classification.

Model: Cnn14 (14-layer CNN pretrained on AudioSet)
Input: Audio at 32kHz, mono
Output: 2048-d embedding vector per song (from penultimate layer)

Usage:
    python extract_panns.py                    # Full extraction
    python extract_panns.py --test 10         # Test on 10 songs
    python extract_panns.py --resume          # Resume from checkpoint

Author: Music Prediction Project
Date: April 2026
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
import utils

import torch
from panns_inference import AudioTagging

# Model constants
MODEL_NAME = 'panns'
EMBEDDING_DIM = 2048
SAMPLE_RATE = 32000  # PANNs uses 32kHz


def load_panns_model(device: torch.device):
    """
    Load PANNs AudioTagging model.
    
    Args:
        device: torch device (cuda or cpu)
        
    Returns:
        PANNs AudioTagging model
    """
    print("[INFO] Loading PANNs (Cnn14) model...")
    start = time.time()
    
    # Load model (will download weights automatically if not cached)
    model = AudioTagging(
        checkpoint_path=None,  # Will download default Cnn14
        device=device.type
    )
    
    print(f"[INFO] PANNs loaded in {time.time()-start:.1f}s")
    
    return model


def extract_panns_embedding(
    model: AudioTagging, 
    audio: np.ndarray
) -> np.ndarray:
    """
    Extract PANNs embedding from audio waveform.
    
    Args:
        model: PANNs AudioTagging model
        audio: Audio waveform at 32kHz, shape (samples,)
        
    Returns:
        Embedding of shape (2048,) from penultimate layer
    """
    # PANNs expects (batch, samples) input
    audio = audio[np.newaxis, :]
    
    # Run inference
    clipwise_output, embedding = model.inference(audio)
    
    # embedding shape: (1, 2048) -> (2048,)
    embedding = embedding.squeeze(0)
    
    return embedding


def run_extraction(
    test_count: Optional[int] = None,
    resume: bool = False,
    checkpoint_interval: int = 1000
):
    """
    Run PANNs extraction on all downloaded audio files.
    
    Args:
        test_count: If set, only process this many songs (for testing)
        resume: If True, resume from checkpoint
        checkpoint_interval: Save checkpoint every N songs
    """
    print("="*60)
    print("PANNs Embedding Extraction (AudioSet Features)")
    print("="*60)
    
    # Get device
    device = utils.get_device()
    
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
    start_idx = 0
    
    # Resume from checkpoint if requested
    if resume:
        checkpoint = utils.load_checkpoint(MODEL_NAME)
        if checkpoint:
            print(f"[INFO] Resuming from checkpoint: {checkpoint['processed_count']} songs already done")
            embeddings_dict = checkpoint.get('embeddings', {})
            processed_ids = set(checkpoint.get('processed_ids', []))
            start_idx = checkpoint['last_processed_idx'] + 1
            error_log = checkpoint.get('errors', [])
            
            # Check if already finished
            if len(processed_ids) + len(error_log) >= len(downloads):
                print(f"[INFO] All {len(downloads)} songs already processed according to checkpoint.")
                # Still save embeddings to be safe if file doesn't exist
                output_path = utils.EMBEDDINGS_DIR / f"{MODEL_NAME}_embeddings_{EMBEDDING_DIM}d.npy"
                if not output_path.exists():
                    utils.save_embeddings_npy(embeddings_dict, output_path)
                return embeddings_dict

    # Load model
    model = load_panns_model(device)
    
    # Track progress
    successful = len(embeddings_dict)
    failed = len(error_log)
    start_time = time.time()
    
    print(f"[INFO] Starting extraction from index {start_idx}")
    print(f"[INFO] Checkpoint every {checkpoint_interval} songs")
    print("-"*60)
    
    # Process songs
    for idx, (spotify_id, filepath, row_idx) in enumerate(downloads):
        # Skip already processed
        if spotify_id in processed_ids:
            continue
        
        # Skip if already in error log too
        if any(e['spotify_id'] == spotify_id for e in error_log):
            continue

        song_start = time.time()
        
        try:
            # Load audio at 32kHz
            audio, sr = utils.load_audio_file(filepath, target_sr=SAMPLE_RATE)
            
            # Extract embedding
            embedding = extract_panns_embedding(model, audio)
            
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
                avg_time = elapsed / successful
                remaining = (len(downloads) - idx) * avg_time
                print(f"[{successful:,}/{len(downloads):,}] {spotify_id} - "
                      f"{time_taken:.2f}s/song, ETA: {remaining/3600:.1f}h")
                
                # Clear GPU cache periodically
                utils.clear_gpu_memory()
            
        except Exception as e:
            error_msg = str(e)
            error_log.append({
                'spotify_id': spotify_id,
                'row_idx': row_idx,
                'error': error_msg
            })
            failed += 1
            
            utils.log_extraction_progress(
                MODEL_NAME, spotify_id, False, time.time() - song_start,
                error_msg=error_msg
            )
            
            if failed <= 10:  # Only print first 10 errors
                print(f"[ERROR] {spotify_id}: {error_msg[:100]}")
            
            # Clear GPU cache on error
            utils.clear_gpu_memory()
        
        # Checkpoint
        if (successful + failed) % checkpoint_interval == 0:
            utils.create_checkpoint(
                MODEL_NAME,
                idx,
                list(processed_ids),
                embeddings_dict,
                error_log
            )
            print(f"[CHECKPOINT] Saved at {successful + failed} songs")
            utils.clear_gpu_memory()
    
    # Final checkpoint after loop finishes
    utils.create_checkpoint(
        MODEL_NAME,
        len(downloads) - 1,
        list(processed_ids),
        embeddings_dict,
        error_log
    )
    print(f"[CHECKPOINT] Final checkpoint saved at {successful + failed} songs")

    # Final save
    total_time = time.time() - start_time
    
    # Save embeddings
    output_path = utils.EMBEDDINGS_DIR / f"{MODEL_NAME}_embeddings_{EMBEDDING_DIM}d.npy"
    utils.save_embeddings_npy(embeddings_dict, output_path)
    
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
    parser = argparse.ArgumentParser(description="Extract PANNs embeddings from audio files")
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
