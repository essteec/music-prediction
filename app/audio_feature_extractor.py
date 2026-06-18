"""
Audio Feature Extraction Module

Estimates the Spotify-like metadata fields used as model inputs from an audio
file. These are waveform heuristics, not Spotify API values.
"""

import librosa
import numpy as np
from typing import Dict


def _to_scalar(value) -> float:
    """Coerce librosa/numpy scalar-or-array outputs to a plain float."""
    arr = np.asarray(value)
    if arr.size == 0:
        return 0.0
    return float(arr.reshape(-1)[0])


def extract_audio_features(audio_path: str) -> Dict[str, float]:
    """
    Estimate base metadata features from an audio file (MP3, WAV, etc.).

    This intentionally returns only fields that are valid model inputs. Target
    variables such as valence, energy, and danceability are predicted by the
    trained models and must not be injected as input features.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with Spotify-like metadata fields required by the app
    """
    # Load audio file
    y, sr = librosa.load(audio_path, duration=30)  # Analyze first 30 seconds
    
    # Tempo and beats
    tempo, _beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = _to_scalar(tempo)
    
    # Spectral features used by the heuristic estimators
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    
    # Zero Crossing Rate (proxy for speechiness)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    
    # Chroma features (for key detection)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Duration (get full file duration, not just the 30s snippet)
    try:
        full_duration = librosa.get_duration(path=audio_path)
        duration_ms = full_duration * 1000
    except Exception:
        # Fallback if file reading fails
        duration_ms = len(y) / sr * 1000
    
    # Loudness (approximate in dB)
    loudness = 20 * np.log10(np.mean(np.abs(y)) + 1e-10)
    
    # Estimate only the base metadata inputs expected by the trained models.
    features = {
        'acousticness': estimate_acousticness(y_harmonic, y_percussive),
        'instrumentalness': estimate_instrumentalness(y, sr),
        'liveness': estimate_liveness(spectral_bandwidth),
        'loudness': float(loudness),
        'speechiness': normalize_speechiness(np.mean(zcr)),
        'tempo': tempo,
        'duration_ms': float(duration_ms),
        'key': estimate_key(chroma),
        'mode': estimate_mode(chroma),
    }
    
    return features


def estimate_acousticness(harmonic: np.ndarray, percussive: np.ndarray) -> float:
    """Estimate acousticness from harmonic/percussive separation"""
    harmonic_energy = np.sum(harmonic**2)
    percussive_energy = np.sum(percussive**2)
    total_energy = harmonic_energy + percussive_energy
    
    if total_energy == 0:
        return 0.5
    
    acousticness = harmonic_energy / total_energy
    return float(np.clip(acousticness, 0, 1))


def estimate_instrumentalness(y: np.ndarray, sr: int) -> float:
    """Estimate instrumentalness (inverse of vocal presence)"""
    # Use spectral contrast as proxy for vocal presence
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    vocal_range_contrast = _to_scalar(np.mean(contrast[2:5]))  # Mid frequencies where vocals are
    
    # Higher contrast in vocal range = more vocals = less instrumental
    instrumentalness = 1.0 - min(vocal_range_contrast / 30, 1.0)
    
    return float(np.clip(instrumentalness, 0, 1))


def estimate_liveness(bandwidth: np.ndarray) -> float:
    """Estimate liveness from spectral bandwidth variance"""
    # Live recordings have more variance in spectral characteristics
    bandwidth_var = _to_scalar(np.var(bandwidth))
    liveness = min(bandwidth_var / 1e6, 1.0)
    
    return float(np.clip(liveness, 0, 1))


def normalize_speechiness(zcr: float) -> float:
    """Normalize zero crossing rate to speechiness scale"""
    # Typical ZCR range is 0.0 to 0.3
    speechiness = _to_scalar(zcr) / 0.3
    return float(np.clip(speechiness, 0, 1))


def estimate_key(chroma: np.ndarray) -> int:
    """Estimate musical key from chroma features"""
    # Average chroma across time
    chroma_mean = np.mean(chroma, axis=1)
    
    # Key is the pitch class with highest average energy
    key = int(np.argmax(chroma_mean))
    
    return key  # 0-11 (C, C#, D, ..., B)


def estimate_mode(chroma: np.ndarray) -> int:
    """Estimate mode (major=1, minor=0) from chroma features"""
    # Simplified heuristic: check if major third (4 semitones) is stronger than minor third (3 semitones)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Get root note
    root = np.argmax(chroma_mean)
    
    # Check major vs minor third
    major_third = chroma_mean[(root + 4) % 12]
    minor_third = chroma_mean[(root + 3) % 12]
    
    mode = 1 if major_third > minor_third else 0
    
    return mode


def print_feature_summary(features: Dict[str, float]):
    """Print extracted features in a readable format"""
    print("\n" + "="*50)
    print("EXTRACTED AUDIO FEATURES")
    print("="*50)
    
    print("\nCore Metadata Inputs:")
    print(f"  Acousticness:     {features['acousticness']:.3f}")
    print(f"  Instrumentalness: {features['instrumentalness']:.3f}")
    print(f"  Liveness:         {features['liveness']:.3f}")
    print(f"  Speechiness:      {features['speechiness']:.3f}")
    print(f"  Loudness:         {features['loudness']:.2f} dB")
    
    print("\nMusical Metadata:")
    print(f"  Tempo:            {features['tempo']:.1f} BPM")
    print(f"  Key:              {features['key']} ({['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][features['key']]})")
    print(f"  Mode:             {'Major' if features['mode'] == 1 else 'Minor'}")
    print(f"  Duration:         {features['duration_ms']/1000:.1f} seconds")
    
    print("="*50 + "\n")


if __name__ == "__main__":
    # Test the feature extractor
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python audio_feature_extractor.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    features = extract_audio_features(audio_file)
    print_feature_summary(features)
