"""
MERT Audio Embedding Extractor

Extracts 768-dimensional embeddings using MERT (Music undERstanding Transformer).
MERT is specifically designed for music understanding tasks and is 
trained on 160K hours of music data.

Model: m-a-p/MERT-v1-95M (95 million parameters)
Input: Audio at 24kHz, mono
Output: 768-d embedding vector per song (temporal mean pooled)

Memory optimization:
- FP16 mixed precision (halves VRAM usage)
- Batch size 1 (safest for 6GB VRAM)
- Processes audio in chunks for long songs

Usage:
    python extract_mert.py                    # Full extraction
    python extract_mert.py --test 10         # Test on 10 songs
    python extract_mert.py --resume          # Resume from checkpoint

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
from transformers import AutoProcessor, AutoModel

# Model constants
MODEL_NAME = 'mert'
MODEL_ID = 'm-a-p/MERT-v1-95M'
EMBEDDING_DIM = 768
SAMPLE_RATE = 24000  # MERT requires 24kHz
MAX_AUDIO_LENGTH = 30 * SAMPLE_RATE  # 30 seconds max to avoid OOM


def load_mert_model(device: torch.device):
    """
    Load MERT model from HuggingFace.
    
    Args:
        device: torch device (cuda or cpu)
        
    Returns:
        Tuple of (processor, model)
    """
    print(f"[INFO] Loading MERT model: {MODEL_ID}")
    start = time.time()
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Move to device (keep FP32 for stability - FP16 causes NaN values)
    model = model.to(device)
    
    # Set to eval mode
    model.eval()
    
    print(f"[INFO] MERT loaded in {time.time()-start:.1f}s")
    print(f"[INFO] Using FP32 (FP16 causes NaN with this model)")
    
    return processor, model


def extract_mert_embedding(
    processor, 
    model, 
    audio: np.ndarray, 
    device: torch.device,
    use_fp16: bool = False
) -> np.ndarray:
    """
    Extract MERT embedding from audio waveform.
    
    Args:
        processor: MERT processor
        model: MERT model
        audio: Audio waveform at 24kHz, shape (samples,)
        device: torch device
        use_fp16: Whether to use FP16 (may cause NaN on some inputs)
        
    Returns:
        Embedding of shape (768,) after temporal mean pooling
    """
    # Process audio (already truncated at load time via max_duration parameter)
    inputs = processor(
        audio, 
        sampling_rate=SAMPLE_RATE, 
        return_tensors="pt"
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Extract embeddings (use FP32 for stability - FP16 causes NaN)
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=use_fp16):
            outputs = model(**inputs, output_hidden_states=True)
    
    # Get last hidden state: (batch=1, time_frames, hidden_dim=768)
    hidden_states = outputs.last_hidden_state
    
    # Mean pool over time dimension: (1, time, 768) -> (768,)
    pooled = hidden_states.mean(dim=1).squeeze(0)
    
    # Convert to numpy
    embedding = pooled.cpu().float().numpy()
    
    return embedding


def run_extraction(
    test_count: Optional[int] = None,
    resume: bool = False,
    checkpoint_interval: int = 500
):
    """
    Run MERT extraction on all downloaded audio files.
    
    Args:
        test_count: If set, only process this many songs (for testing)
        resume: If True, resume from checkpoint
        checkpoint_interval: Save checkpoint every N songs
    """
    print("="*60)
    print("MERT Embedding Extraction (Music Understanding)")
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
    processor, model = load_mert_model(device)
    
    # Track progress
    successful = len(embeddings_dict)
    failed = len(error_log)
    start_time = time.time()
    
    print(f"[INFO] Starting extraction from index {start_idx}")
    print(f"[INFO] Max audio length: {MAX_AUDIO_LENGTH/SAMPLE_RATE:.0f} seconds")
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
            # Load audio at 24kHz with 30s limit to avoid RAM spike from long songs
            audio, sr = utils.load_audio_file(
                filepath, 
                target_sr=SAMPLE_RATE,
                max_duration=MAX_AUDIO_LENGTH / SAMPLE_RATE  # 30 seconds
            )
            
            # Extract embedding
            embedding = extract_mert_embedding(processor, model, audio, device)
            
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
            
            # Progress update every 50 songs (MERT is slow)
            if successful % 50 == 0:
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
    parser = argparse.ArgumentParser(description="Extract MERT embeddings from audio files")
    parser.add_argument('--test', type=int, help="Test mode: process only N songs")
    parser.add_argument('--resume', action='store_true', help="Resume from checkpoint")
    parser.add_argument('--checkpoint-interval', type=int, default=500,
                        help="Save checkpoint every N songs (default: 500)")
    
    args = parser.parse_args()
    
    run_extraction(
        test_count=args.test,
        resume=args.resume,
        checkpoint_interval=args.checkpoint_interval
    )


if __name__ == "__main__":
    main()
