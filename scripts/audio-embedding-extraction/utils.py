"""
Shared utilities for audio embedding extraction pipeline.

Provides common functions for:
- Audio file loading and preprocessing
- Checkpoint management (resume capability)
- Embedding validation and saving
- Progress logging and error tracking

Author: Music Prediction Project
Date: April 2026
"""

import json
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from tqdm import tqdm
import warnings

# Suppress noisy warnings from librosa/audioread (WebM files use ffmpeg fallback - expected)
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
warnings.filterwarnings('ignore', category=UserWarning, message='PySoundFile failed')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio" / "pilot"
EMBEDDINGS_DIR = DATA_DIR / "embeddings" / "audio"
LOGS_DIR = DATA_DIR / "embeddings" / "extraction_logs"
CHECKPOINTS_DIR = DATA_DIR / "embeddings" / "checkpoints"
DOWNLOAD_LOG = DATA_DIR / "logs" / "download_log_pilot.csv"


def get_successful_downloads() -> List[Tuple[str, Path, int]]:
    """
    Get list of successfully downloaded audio files from download log.
    
    Returns:
        List of (spotify_id, audio_filepath, row_idx) tuples for successful downloads
        
    Note:
        Audio files are named by row_idx (e.g., 000001_opus.webm), not spotify_id.
        The row_idx is included for debugging and file identification.
        
    Example:
        >>> downloads = get_successful_downloads()
        >>> len(downloads)
        45321
        >>> downloads[0]
        ('2ASl4wirkeYm3OWZxXKYuq', PosixPath('.../data/audio/pilot/000001_opus.webm'), 1)
    """
    if not DOWNLOAD_LOG.exists():
        raise FileNotFoundError(f"Download log not found: {DOWNLOAD_LOG}")
    
    # Read download log
    df = pd.read_csv(DOWNLOAD_LOG)
    
    # Filter successful downloads
    successful = df[df['download_success'] == True].copy()
    
    # Build list of (spotify_id, filepath, row_idx) tuples
    results = []
    for _, row in successful.iterrows():
        spotify_id = row['song_id']
        row_idx = row['row_idx']
        
        # Files are named by row_idx with zero-padding
        filepath = AUDIO_DIR / f"{row_idx:06d}_opus.webm"
        
        # Only include if file actually exists
        if filepath.exists():
            results.append((spotify_id, filepath, row_idx))
    
    return results


def load_audio_file(
    filepath: Union[str, Path], 
    target_sr: int = 16000,
    mono: bool = True,
    max_duration: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        filepath: Path to audio file (WebM, MP3, WAV, etc.)
        target_sr: Target sample rate in Hz
        mono: If True, convert to mono
        max_duration: If set, truncate audio to this many seconds
        
    Returns:
        Tuple of (audio_waveform, sample_rate)
        - audio_waveform: numpy array of shape (samples,) for mono or (channels, samples) for stereo
        - sample_rate: actual sample rate (should equal target_sr)
        
    Raises:
        RuntimeError: If audio loading fails
        
    Example:
        >>> audio, sr = load_audio_file("song.webm", target_sr=16000)
        >>> audio.shape
        (480000,)  # 30 seconds at 16kHz
        >>> sr
        16000
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")
    
    try:
        # Load with librosa (handles WebM, MP3, etc.)
        audio, sr = librosa.load(
            filepath, 
            sr=target_sr, 
            mono=mono,
            duration=max_duration
        )
        
        return audio, sr
        
    except Exception as e:
        raise RuntimeError(f"Failed to load audio from {filepath}: {str(e)}")


def load_audio_torch(
    filepath: Union[str, Path],
    target_sr: int = 16000,
    mono: bool = True,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Load audio file as PyTorch tensor for GPU processing.
    
    Args:
        filepath: Path to audio file
        target_sr: Target sample rate
        mono: If True, convert to mono
        device: Device to load tensor to ('cpu' or 'cuda')
        
    Returns:
        PyTorch tensor of shape (1, samples) for model input
        
    Example:
        >>> audio = load_audio_torch("song.webm", device='cuda')
        >>> audio.shape
        torch.Size([1, 480000])
    """
    audio, sr = load_audio_file(filepath, target_sr=target_sr, mono=mono)
    
    # Convert to torch tensor and add batch dimension
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    
    return audio_tensor.to(device)


def create_checkpoint(
    model_name: str,
    last_processed_idx: int,
    processed_ids: List[str],
    embeddings_dict: Dict[str, np.ndarray],
    error_log: List[Dict],
    checkpoint_dir: Optional[Path] = None
) -> Path:
    """
    Save extraction checkpoint for resume capability.
    
    Args:
        model_name: Name of the model (e.g., 'mert', 'vggish')
        last_processed_idx: Index of last processed song in the list
        processed_ids: List of spotify_ids already processed
        embeddings_dict: Dictionary mapping spotify_id -> embedding array
        error_log: List of error records
        checkpoint_dir: Directory to save checkpoint (default: CHECKPOINTS_DIR)
        
    Returns:
        Path to saved checkpoint file
        
    Notes:
        - Saves both JSON metadata and NPY embeddings
        - Can be resumed with load_checkpoint()
    """
    checkpoint_dir = checkpoint_dir or CHECKPOINTS_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metadata JSON
    metadata = {
        'model_name': model_name,
        'last_processed_idx': last_processed_idx,
        'processed_count': len(processed_ids),
        'processed_ids': processed_ids,
        'error_count': len(error_log),
        'errors': error_log,
        'timestamp': timestamp
    }
    
    json_path = checkpoint_dir / f"{model_name}_checkpoint.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save embeddings as NPY (incremental) using pickle instead of kwargs expansion
    # to avoid scalability issues with 45k+ items (kwargs expansion is slow and memory-intensive)
    if embeddings_dict:
        npy_path = checkpoint_dir / f"{model_name}_checkpoint_embeddings.pkl"
        import pickle
        with open(npy_path, 'wb') as pf:
            pickle.dump(embeddings_dict, pf, protocol=pickle.HIGHEST_PROTOCOL)
    
    return json_path


def load_checkpoint(model_name: str, checkpoint_dir: Optional[Path] = None) -> Optional[Dict]:
    """
    Load existing checkpoint for resume.
    
    Args:
        model_name: Name of the model
        checkpoint_dir: Directory containing checkpoint
        
    Returns:
        Dictionary with checkpoint data, or None if no checkpoint exists
        
    Example:
        >>> checkpoint = load_checkpoint('mert')
        >>> if checkpoint:
        ...     start_idx = checkpoint['last_processed_idx'] + 1
        ...     processed = set(checkpoint['processed_ids'])
    """
    checkpoint_dir = checkpoint_dir or CHECKPOINTS_DIR
    json_path = checkpoint_dir / f"{model_name}_checkpoint.json"
    
    if not json_path.exists():
        return None
    
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    # Load embeddings if they exist (check both new pickle format and legacy npz)
    pkl_path = checkpoint_dir / f"{model_name}_checkpoint_embeddings.pkl"
    npz_path = checkpoint_dir / f"{model_name}_checkpoint_embeddings.npz"
    
    if pkl_path.exists():
        import pickle
        with open(pkl_path, 'rb') as pf:
            metadata['embeddings'] = pickle.load(pf)
    elif npz_path.exists():
        # Legacy npz format support
        npz_data = np.load(npz_path)
        metadata['embeddings'] = {key: npz_data[key] for key in npz_data.files}
    else:
        metadata['embeddings'] = {}
    
    return metadata


def validate_embeddings(
    embeddings: np.ndarray, 
    expected_dim: int,
    spotify_id: str = "unknown"
) -> bool:
    """
    Validate embedding array for quality issues.
    
    Args:
        embeddings: Embedding array to validate
        expected_dim: Expected dimensionality
        spotify_id: Song ID for error reporting
        
    Returns:
        True if valid, False otherwise
        
    Checks:
        - Correct dimensionality
        - No NaN values
        - No Inf values
        - Non-zero (not all zeros)
    """
    # Check shape
    if embeddings.shape[-1] != expected_dim:
        print(f"[WARN] {spotify_id}: Wrong dimension {embeddings.shape[-1]}, expected {expected_dim}")
        return False
    
    # Check for NaN
    if np.isnan(embeddings).any():
        print(f"[WARN] {spotify_id}: Contains NaN values")
        return False
    
    # Check for Inf
    if np.isinf(embeddings).any():
        print(f"[WARN] {spotify_id}: Contains Inf values")
        return False
    
    # Check for all zeros (suspicious)
    if np.allclose(embeddings, 0):
        print(f"[WARN] {spotify_id}: All zeros (suspicious)")
        return False
    
    return True


def save_embeddings_npy(
    embeddings_dict: Dict[str, np.ndarray],
    output_path: Union[str, Path],
    spotify_ids_order: Optional[List[str]] = None
) -> Tuple[Path, Path]:
    """
    Save embeddings dictionary to NPY file with matching ID file.
    
    Args:
        embeddings_dict: Dictionary mapping spotify_id -> embedding array
        output_path: Path for output NPY file
        spotify_ids_order: Optional ordered list of IDs (uses dict order if None)
        
    Returns:
        Tuple of (embeddings_path, ids_path)
        
    Output files:
        - {name}.npy: Embeddings array of shape (n_songs, embedding_dim)
        - {name}_ids.npy: Array of spotify_ids in same order as embeddings
        
    Example:
        >>> save_embeddings_npy({'id1': emb1, 'id2': emb2}, 'mert_embeddings_768d.npy')
        (PosixPath('.../mert_embeddings_768d.npy'), PosixPath('.../mert_embeddings_768d_ids.npy'))
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get ordered IDs - filter to only IDs that exist in embeddings_dict
    if spotify_ids_order is None:
        filtered_ids = list(embeddings_dict.keys())
    else:
        # Critical fix: ensure IDs match embeddings to prevent misalignment
        filtered_ids = [sid for sid in spotify_ids_order if sid in embeddings_dict]
    
    # Stack embeddings in order
    embeddings_list = [embeddings_dict[sid] for sid in filtered_ids]
    embeddings_array = np.stack(embeddings_list, axis=0)
    
    # Save embeddings
    np.save(output_path, embeddings_array)
    
    # Save IDs in same order (matching filtered list, not original)
    ids_path = output_path.parent / f"{output_path.stem}_ids.npy"
    np.save(ids_path, np.array(filtered_ids))
    
    print(f"[INFO] Saved embeddings: {output_path}")
    print(f"       Shape: {embeddings_array.shape}")
    print(f"       IDs file: {ids_path}")
    
    return output_path, ids_path


def log_extraction_progress(
    model_name: str,
    spotify_id: str,
    success: bool,
    time_taken: float,
    error_msg: Optional[str] = None,
    embedding_shape: Optional[Tuple] = None,
    log_dir: Optional[Path] = None
):
    """
    Log extraction result to CSV for debugging and analysis.
    
    Args:
        model_name: Name of the model
        spotify_id: Spotify track ID
        success: Whether extraction succeeded
        time_taken: Time taken in seconds
        error_msg: Error message if failed
        embedding_shape: Shape of extracted embedding
        log_dir: Directory for log file
    """
    log_dir = log_dir or LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = log_dir / f"{model_name}_extraction_log.csv"
    
    # Create header if file doesn't exist
    write_header = not log_path.exists()
    
    with open(log_path, 'a') as f:
        if write_header:
            f.write("timestamp,spotify_id,success,time_taken_sec,embedding_shape,error_msg\n")
        
        timestamp = datetime.now().isoformat()
        shape_str = str(embedding_shape) if embedding_shape else ""
        # Sanitize error message: replace newlines and commas to preserve CSV structure
        error_str = error_msg.replace("\n", " ").replace("\r", " ").replace(",", ";") if error_msg else ""
        
        f.write(f"{timestamp},{spotify_id},{success},{time_taken:.2f},{shape_str},{error_str}\n")


def get_device() -> torch.device:
    """
    Get best available device (CUDA if available, else CPU).
    
    Returns:
        torch.device for computation
        
    Example:
        >>> device = get_device()
        >>> print(device)
        cuda
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"       VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("[INFO] Using CPU (GPU not available)")
    
    return device


def clear_gpu_memory():
    """Clear GPU memory cache to prevent OOM errors."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def temporal_mean_pool(embeddings: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Mean pool embeddings over time dimension.
    
    Args:
        embeddings: Array of shape (time_steps, embedding_dim) or (batch, time, dim)
        
    Returns:
        Pooled embedding of shape (embedding_dim,) or (batch, dim)
        
    Example:
        >>> emb = np.random.randn(300, 768)  # 300 time steps, 768 dims
        >>> pooled = temporal_mean_pool(emb)
        >>> pooled.shape
        (768,)
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    # Handle different input shapes
    if embeddings.ndim == 2:
        # (time, dim) -> (dim,)
        return embeddings.mean(axis=0)
    elif embeddings.ndim == 3:
        # (batch, time, dim) -> (batch, dim)
        return embeddings.mean(axis=1)
    else:
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")


def print_extraction_summary(
    model_name: str,
    total: int,
    successful: int,
    failed: int,
    total_time: float
):
    """Print formatted extraction summary."""
    print("\n" + "="*60)
    print(f"EXTRACTION COMPLETE: {model_name}")
    print("="*60)
    print(f"Total songs:     {total:,}")
    print(f"Successful:      {successful:,} ({100*successful/total:.1f}%)")
    print(f"Failed:          {failed:,} ({100*failed/total:.1f}%)")
    print(f"Total time:      {total_time/3600:.2f} hours")
    print(f"Avg per song:    {total_time/total:.2f} seconds")
    print("="*60 + "\n")


# Convenience function to get embedding dimension for each model
MODEL_EMBEDDING_DIMS = {
    'mert': 768,          # MERT-v1-95M
    'mert_large': 1024,   # MERT-v1-330M
    'vggish': 128,
    'panns': 2048,
    'wav2vec2': 768,
    'clap': 512,
    'mel_stats': 512,     # 128 bands × 4 stats (mean, std, max, min)
}


def get_embedding_dim(model_name: str) -> int:
    """Get expected embedding dimension for a model."""
    model_name = model_name.lower()
    if model_name not in MODEL_EMBEDDING_DIMS:
        raise ValueError(f"Unknown model: {model_name}. Known: {list(MODEL_EMBEDDING_DIMS.keys())}")
    return MODEL_EMBEDDING_DIMS[model_name]
