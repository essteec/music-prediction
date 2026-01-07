"""
Audio Feature Extraction Module
Extracts Spotify-like features from audio files using librosa
"""

import librosa
import numpy as np
from typing import Dict


def extract_audio_features(audio_path: str) -> Dict[str, float]:
    """
    Extract audio features from an audio file (MP3, WAV, etc.)
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with extracted features matching Spotify API format
    """
    # Load audio file
    y, sr = librosa.load(audio_path, duration=30)  # Analyze first 30 seconds
    
    # Tempo and beats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    
    # RMS Energy
    rms = librosa.feature.rms(y=y)[0]
    energy = np.mean(rms)
    
    # Zero Crossing Rate (proxy for speechiness)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    
    # Chroma features (for key detection)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # MFCC (timbre)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
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
    
    # Normalize features to match Spotify's 0-1 scale
    features = {
        # Core audio features
        'acousticness': estimate_acousticness(y_harmonic, y_percussive),
        'danceability': estimate_danceability(tempo, beats, y, sr),
        'energy': normalize_energy(energy),
        'instrumentalness': estimate_instrumentalness(y, sr),
        'liveness': estimate_liveness(spectral_bandwidth),
        'loudness': loudness,
        'speechiness': normalize_speechiness(np.mean(zcr)),
        'tempo': float(tempo),
        
        # Musical features
        'duration_ms': duration_ms,
        'key': estimate_key(chroma),
        'mode': estimate_mode(chroma),
        'time_signature': estimate_time_signature(beats),
        
        # Metadata (will be provided by user)
        'valence': 0.5,  # Placeholder - this is what we're predicting!
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


def estimate_danceability(tempo: float, beats: np.ndarray, y: np.ndarray, sr: int) -> float:
    """Estimate danceability from tempo and beat strength"""
    # Beat strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    beat_strength = np.mean(onset_env)
    
    # Tempo score (optimal dance tempo around 120 BPM)
    tempo_score = 1.0 - abs(tempo - 120) / 120
    tempo_score = max(0, tempo_score)
    
    # Regularity of beats
    if len(beats) > 1:
        beat_intervals = np.diff(beats)
        beat_regularity = 1.0 - np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-10)
        beat_regularity = np.clip(beat_regularity, 0, 1)
    else:
        beat_regularity = 0.5
    
    # Combine factors
    danceability = (tempo_score * 0.4 + beat_regularity * 0.3 + 
                   min(beat_strength / 10, 1.0) * 0.3)
    
    return float(np.clip(danceability, 0, 1))


def normalize_energy(rms_energy: float) -> float:
    """Normalize RMS energy to 0-1 scale"""
    # Typical RMS range is 0.0 to 0.3
    normalized = rms_energy / 0.3
    return float(np.clip(normalized, 0, 1))


def estimate_instrumentalness(y: np.ndarray, sr: int) -> float:
    """Estimate instrumentalness (inverse of vocal presence)"""
    # Use spectral contrast as proxy for vocal presence
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    vocal_range_contrast = np.mean(contrast[2:5])  # Mid frequencies where vocals are
    
    # Higher contrast in vocal range = more vocals = less instrumental
    instrumentalness = 1.0 - min(vocal_range_contrast / 30, 1.0)
    
    return float(np.clip(instrumentalness, 0, 1))


def estimate_liveness(bandwidth: np.ndarray) -> float:
    """Estimate liveness from spectral bandwidth variance"""
    # Live recordings have more variance in spectral characteristics
    bandwidth_var = np.var(bandwidth)
    liveness = min(bandwidth_var / 1e6, 1.0)
    
    return float(np.clip(liveness, 0, 1))


def normalize_speechiness(zcr: float) -> float:
    """Normalize zero crossing rate to speechiness scale"""
    # Typical ZCR range is 0.0 to 0.3
    speechiness = zcr / 0.3
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


def estimate_time_signature(beats: np.ndarray) -> int:
    """Estimate time signature from beat intervals"""
    if len(beats) < 4:
        return 4  # Default to 4/4
    
    # Most common music is 4/4 or 3/4
    # This is a simplified estimation
    return 4  # Default to 4/4 for simplicity


def print_feature_summary(features: Dict[str, float]):
    """Print extracted features in a readable format"""
    print("\n" + "="*50)
    print("EXTRACTED AUDIO FEATURES")
    print("="*50)
    
    print("\n📊 Core Audio Features:")
    print(f"  Energy:           {features['energy']:.3f}")
    print(f"  Danceability:     {features['danceability']:.3f}")
    print(f"  Acousticness:     {features['acousticness']:.3f}")
    print(f"  Instrumentalness: {features['instrumentalness']:.3f}")
    print(f"  Liveness:         {features['liveness']:.3f}")
    print(f"  Speechiness:      {features['speechiness']:.3f}")
    print(f"  Loudness:         {features['loudness']:.2f} dB")
    
    print("\n🎵 Musical Features:")
    print(f"  Tempo:            {features['tempo']:.1f} BPM")
    print(f"  Key:              {features['key']} ({['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][features['key']]})")
    print(f"  Mode:             {'Major' if features['mode'] == 1 else 'Minor'}")
    print(f"  Time Signature:   {features['time_signature']}/4")
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
