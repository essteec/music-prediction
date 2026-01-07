"""
Music Prediction Gradio App
Upload audio + provide metadata → Get predictions for valence, energy, danceability, popularity
"""

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
import sys
import os

# Add parent directory to path to import audio feature extractor
sys.path.insert(0, str(Path(__file__).parent))
from audio_feature_extractor import extract_audio_features, print_feature_summary


# ========================
# Load Models and Resources
# ========================

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "ml" / "models" / "saved" / "experiment2_with_artist" / "rfe_best"
FEATURES_DIR = BASE_DIR / "ml" / "features"

# Load best models (RFE models with optimal features)
MODELS = {
    'valence': ('XGBoost_tuned_valence_iter23.pkl', 'optimal_features_valence_iter23_20260101_192219.csv'),
    'energy': ('CatBoost_tuned_energy_iter38.pkl', 'optimal_features_energy_iter38_20260101_192219.csv'),
    'danceability': ('CatBoost_tuned_danceability_iter34.pkl', 'optimal_features_danceability_iter34_20260101_192219.csv'),
    'popularity': ('CatBoost_tuned_popularity_iter2.pkl', 'optimal_features_popularity_iter2_20260101_192219.csv')
}

# Load models and their optimal feature indices
loaded_models = {}
optimal_features = {}

for target, (model_file, features_file) in MODELS.items():
    # Load model
    model_path = MODELS_DIR / model_file
    if model_path.exists():
        loaded_models[target] = joblib.load(model_path)
        print(f"✓ Loaded {target} model: {model_file}")
    else:
        print(f"⚠ Model not found: {model_path}")
        loaded_models[target] = None
    
    # Load optimal feature indices
    features_path = MODELS_DIR / features_file
    if features_path.exists():
        features_df = pd.read_csv(features_path)
        optimal_features[target] = features_df['feature_index'].values
        print(f"  → {len(optimal_features[target])} features selected")
    else:
        print(f"⚠ Features file not found: {features_path}")
        optimal_features[target] = None

# Load scalers and transformers
audio_scaler = joblib.load(FEATURES_DIR / "audio_scaler.pkl")  # StandardScaler for 4 features
audio_power_transformer = joblib.load(FEATURES_DIR / "audio_power_transformer.pkl")  # PowerTransformer for 3 features
genre_encoder = joblib.load(FEATURES_DIR / "genre_encoder.pkl")  # OneHotEncoder for genre
text_scaler = joblib.load(FEATURES_DIR / "text_stats_scaler.pkl")
sentiment_scaler = joblib.load(FEATURES_DIR / "sentiment_scaler.pkl")

# Load embedding model
print("Loading embedding model (this may take a moment)...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✓ Embedding model loaded")

# Genre list (must match training)
GENRES = [
    'pop', 'rock', 'hip hop', 'r&b', 'country', 
    'electronic', 'jazz', 'latin', 'indie', 'other'
]


# ========================
# Feature Processing Functions
# ========================

def process_lyrics(lyrics: str):
    """Extract text statistics and sentiment from lyrics"""
    if not lyrics or len(lyrics.strip()) == 0:
        lyrics = "no lyrics available"
    
    # Text statistics
    words = lyrics.lower().split()
    word_count = len(words)
    unique_words = len(set(words))
    unique_ratio = unique_words / word_count if word_count > 0 else 0
    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    char_count = len(lyrics)
    
    # Apply log1p transformation to count features (matching training)
    # The scaler expects these to be log-transformed
    text_stats = np.array([[
        np.log1p(word_count),
        np.log1p(unique_words),
        unique_ratio,
        avg_word_length,
        np.log1p(char_count)
    ]])
    text_stats_scaled = text_scaler.transform(text_stats)
    
    # Sentiment
    blob = TextBlob(lyrics)
    sentiment = np.array([[blob.sentiment.polarity, blob.sentiment.subjectivity]])
    sentiment_scaled = sentiment_scaler.transform(sentiment)
    
    # Embeddings
    embeddings = embedding_model.encode([lyrics])
    
    return text_stats_scaled, sentiment_scaled, embeddings


def process_metadata(genre: str, year: int, artist_followers: int, artist_popularity: int):
    """Process metadata features
    
    Note: Year gets normalized here but is then passed to scaler in build_feature_vector.
    The scaler expects raw year values, so we pass the actual year, not normalized.
    """
    # Genre one-hot encoding using the trained encoder
    # Map user genre to encoder categories
    genre_mapping = {
        'pop': 'Pop',
        'rock': 'Rock',
        'hip hop': 'Hip-Hop',
        'r&b': 'R&B',
        'country': 'Country',
        'electronic': 'Electronic',
        'jazz': 'Jazz',
        'latin': 'Folk',  # Approximate mapping
        'indie': 'Pop',    # Approximate mapping
        'other': 'Pop'     # Default to Pop
    }
    
    genre_capitalized = genre_mapping.get(genre.lower(), 'Pop')
    genre_encoded = genre_encoder.transform([[genre_capitalized]])[0]
    
    # Year - pass raw value (will be scaled in build_feature_vector)
    year_raw = year
    
    # Artist features (log transform for followers, keep popularity on 0-100 scale)
    log_followers = np.log1p(artist_followers)
    artist_popularity_raw = float(artist_popularity)
    
    return genre_encoded, year_raw, log_followers, artist_popularity_raw


def build_feature_vector(audio_features, text_stats_scaled, sentiment_scaled, 
                         embeddings, genre_encoded, year_raw, log_followers, artist_popularity):
    """Combine all features into single vector matching training format
    
    Feature order (23 audio features):
    1. Power-transformed (3): acousticness, instrumentalness, speechiness
    2. Normalized (1): liveness
    3. Standard scaled (4): loudness, tempo, duration_ms, year
    4. Categorical (1): mode
    5. Cyclical (2): key_sin, key_cos
    6. Genre one-hot (10): genre encodings
    7. Artist (2): log_total_artist_followers, avg_artist_popularity
    """
    
    # 1. Power-transformed features (3)
    skewed_features = np.array([[
        audio_features['acousticness'],
        audio_features['instrumentalness'],
        audio_features['speechiness']
    ]])
    X_power = audio_power_transformer.transform(skewed_features)
    
    # 2. Normalized feature (1) - liveness (already 0-1, no scaling)
    X_normalized = np.array([[audio_features['liveness']]])
    
    # 3. Standard scaled features (4): loudness, tempo, duration_ms, year
    scaled_features = np.array([[
        audio_features['loudness'],
        audio_features['tempo'],
        audio_features['duration_ms'],
        year_raw  # year will be scaled by audio_scaler
    ]])
    X_scaled = audio_scaler.transform(scaled_features)
    
    # 4. Categorical (1) - mode (0 or 1, no scaling)
    X_categorical = np.array([[audio_features['mode']]])
    
    # 5. Cyclical key encoding (2)
    key = audio_features['key']
    key_sin = np.sin(2 * np.pi * key / 12)
    key_cos = np.cos(2 * np.pi * key / 12)
    X_cyclical = np.array([[key_sin, key_cos]])
    
    # 6. Genre one-hot (10)
    X_genre = genre_encoded.reshape(1, -1)
    
    # 7. Artist features (2) - already processed
    X_artist = np.array([[log_followers, artist_popularity]])
    
    # Combine audio features in correct order (23 total)
    audio_complete = np.hstack([
        X_power,        # 3
        X_normalized,   # 1
        X_scaled,       # 4
        X_categorical,  # 1
        X_cyclical,     # 2
        X_genre,        # 10
        X_artist        # 2
    ])
    
    # Combine all features (414 total)
    # Order: audio (23) + text_stats (5) + sentiment (2) + embeddings (384)
    full_features = np.hstack([
        audio_complete,      # 23
        text_stats_scaled,   # 5
        sentiment_scaled,    # 2
        embeddings           # 384
    ])
    
    print(f"\n✓ Feature vector shape: {full_features.shape}")
    print(f"  - Audio features: {audio_complete.shape[1]}")
    print(f"    * Power-transformed: 3")
    print(f"    * Normalized: 1")
    print(f"    * Standard scaled: 4")
    print(f"    * Categorical: 1")
    print(f"    * Cyclical: 2")
    print(f"    * Genre: 10")
    print(f"    * Artist: 2")
    print(f"  - Text statistics: {text_stats_scaled.shape[1]}")
    print(f"  - Sentiment: {sentiment_scaled.shape[1]}")
    print(f"  - Embeddings: {embeddings.shape[1]}")
    print(f"  - Total: {full_features.shape[1]}")
    
    return full_features


# ========================
# Prediction Function
# ========================

def predict(audio_file, lyrics, genre, year, artist_name, artist_followers, artist_popularity):
    """Main prediction function"""
    
    try:
        # 1. Extract audio features
        print(f"\n{'='*60}")
        print("🎵 PROCESSING AUDIO FILE")
        print(f"{'='*60}")
        audio_features = extract_audio_features(audio_file)
        print_feature_summary(audio_features)
        
        # 2. Process lyrics
        print("📝 PROCESSING LYRICS")
        print(f"{'='*60}")
        print(f"Lyrics length: {len(lyrics)} characters")
        text_stats_scaled, sentiment_scaled, embeddings = process_lyrics(lyrics)
        print(f"✓ Text features extracted")
        
        # 3. Process metadata
        print("\n🎤 PROCESSING METADATA")
        print(f"{'='*60}")
        print(f"Genre: {genre}")
        print(f"Year: {year}")
        print(f"Artist: {artist_name}")
        print(f"Followers: {artist_followers:,}")
        print(f"Popularity: {artist_popularity}/100")
        genre_encoded, year_raw, log_followers, artist_pop_raw = process_metadata(
            genre, year, artist_followers, artist_popularity
        )
        print(f"✓ Metadata processed")
        
        # 4. Build complete feature vector
        print(f"\n{'='*60}")
        print("🔧 BUILDING FEATURE VECTOR")
        print(f"{'='*60}")
        features = build_feature_vector(
            audio_features, text_stats_scaled, sentiment_scaled, embeddings,
            genre_encoded, year_raw, log_followers, artist_pop_raw
        )
        
        # 5. Make predictions with RFE feature selection
        print(f"\n{'='*60}")
        print("🎯 MAKING PREDICTIONS (RFE Models)")
        print(f"{'='*60}")
        predictions = {}
        
        for target, model in loaded_models.items():
            if model is not None and optimal_features[target] is not None:
                # Select only the optimal features for this model
                target_features = features[:, optimal_features[target]]
                pred = model.predict(target_features)[0]
                
                # De-transform popularity (trained on log1p)
                if target == 'popularity':
                    pred = np.expm1(pred)
                    # Clip to 0-100 range
                    pred = np.clip(pred, 0, 100)
                else:
                    # Clip other targets to 0-1 range
                    pred = np.clip(pred, 0, 1)
                
                predictions[target] = float(pred)
                print(f"  {target.capitalize():15s}: {pred:.4f} (using {len(optimal_features[target])} features)")
            else:
                predictions[target] = 0.5
                print(f"  {target.capitalize():15s}: [MODEL OR FEATURES NOT FOUND]")
        
        print(f"{'='*60}\n")
        
        # 6. Format output
        result_text = f"""
## 🎵 Prediction Results

**Artist:** {artist_name}  
**Genre:** {genre.capitalize()} | **Year:** {year}

### 📊 Predicted Attributes:

- **🎭 Valence (Positivity):** {predictions['valence']:.3f} / 1.0
- **⚡ Energy (Intensity):** {predictions['energy']:.3f} / 1.0
- **💃 Danceability:** {predictions['danceability']:.3f} / 1.0
- **🔥 Popularity:** {predictions['popularity']:.1f} / 100

### 📈 Interpretation:

- **Valence:** {"😊 Happy/Positive" if predictions['valence'] > 0.5 else "😔 Sad/Negative"} ({predictions['valence']:.1%} positivity)
- **Energy:** {"🔥 High energy" if predictions['energy'] > 0.7 else "😌 Low energy" if predictions['energy'] < 0.3 else "⚖️ Medium energy"}
- **Danceability:** {"💃 Very danceable" if predictions['danceability'] > 0.7 else "🚶 Not very danceable" if predictions['danceability'] < 0.3 else "👍 Moderately danceable"}
- **Popularity:** {"🌟 Likely popular" if predictions['popularity'] > 60 else "🎵 Niche appeal" if predictions['popularity'] < 30 else "📊 Average appeal"}

### 🎼 Extracted Audio Features:

- **Tempo:** {audio_features['tempo']:.1f} BPM
- **Key:** {['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][audio_features['key']]} {['Minor', 'Major'][audio_features['mode']]}
- **Loudness:** {audio_features['loudness']:.2f} dB
- **Acousticness:** {audio_features['acousticness']:.3f}
- **Instrumentalness:** {audio_features['instrumentalness']:.3f}
"""
        
        return result_text
        
    except Exception as e:
        import traceback
        error_msg = f"""
## ❌ Error

An error occurred during prediction:

```
{str(e)}
```

**Stack trace:**
```
{traceback.format_exc()}
```

Please check:
1. Audio file is valid (MP3, WAV, etc.)
2. All fields are filled in
3. Models are loaded correctly
"""
        return error_msg


# ========================
# Gradio Interface
# ========================

def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="🎵 Music Prediction App", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 🎵 Music Attribute Prediction
        
        Upload an audio file and provide metadata to predict musical attributes using trained ML models.
        
        **Predictions:**
        - 🎭 **Valence:** Emotional positivity (0-1)
        - ⚡ **Energy:** Intensity and activity (0-1)
        - 💃 **Danceability:** How suitable for dancing (0-1)
        - 🔥 **Popularity:** Expected popularity (0-100)
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Inputs
                gr.Markdown("### 📁 Upload & Input")
                
                audio_input = gr.Audio(
                    label="Audio File (MP3, WAV, etc.)",
                    type="filepath"
                )
                
                lyrics_input = gr.Textbox(
                    label="Lyrics",
                    placeholder="Paste the song lyrics here...",
                    lines=8
                )
                
                with gr.Row():
                    genre_input = gr.Dropdown(
                        choices=GENRES,
                        label="Genre",
                        value="pop"
                    )
                    year_input = gr.Slider(
                        minimum=1960,
                        maximum=2025,
                        value=2020,
                        step=1,
                        label="Release Year"
                    )
                
                artist_input = gr.Textbox(
                    label="Artist Name",
                    placeholder="e.g., Taylor Swift"
                )
                
                with gr.Row():
                    followers_input = gr.Number(
                        label="Artist Followers (approx)",
                        value=1000000,
                        minimum=0
                    )
                    popularity_input = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Artist Popularity (0-100)"
                    )
                
                predict_btn = gr.Button("🎯 Predict", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                # Output
                gr.Markdown("### 📊 Results")
                output = gr.Markdown(label="Predictions")
        
        # Examples
        gr.Markdown("""
        ### 💡 Tips:
        - For best results, use clear audio files (not too compressed)
        - The model analyzes the first 30 seconds of audio
        - If you don't know exact artist followers, estimate or use 0
        - Artist popularity is typically 0-100 (50 = moderate popularity)
        """)
        
        # Connect button
        predict_btn.click(
            fn=predict,
            inputs=[
                audio_input,
                lyrics_input,
                genre_input,
                year_input,
                artist_input,
                followers_input,
                popularity_input
            ],
            outputs=output
        )
    
    return app


# ========================
# Main
# ========================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎵 MUSIC PREDICTION APP")
    print("="*60)
    print(f"Models loaded: {len([m for m in loaded_models.values() if m is not None])}/4")
    print("="*60 + "\n")
    
    # Create and launch app
    app = create_interface()
    app.launch(
        share=False,  # Set to True to create public link
        server_name="127.0.0.1",
        server_port=7860
    )
