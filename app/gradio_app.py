"""
Thesis Music Prediction Gradio App

Runs both final tuned CatBoost models and the tuned retrained
AttentionTaskGatedFusionMLP checkpoint on the 4,254-feature thesis input space.
"""

from __future__ import annotations

import importlib
import os
import pickle
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from textblob import TextBlob


BASE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = BASE_DIR / "app"
FEATURES_DIR = BASE_DIR / "ml" / "features"
SCALER_DIR = FEATURES_DIR / "scalers"
ML_MODELS_DIR = BASE_DIR / "ml" / "models" / "saved" / "thesis_ml_test"
DL_CHECKPOINT_PATH = BASE_DIR / "models" / "checkpoints" / "thesis_final_tuned" / "AttentionTaskGatedFusionMLP_retrained.pt"
DOWNLOAD_DIR = BASE_DIR / "app" / "downloads"

TARGETS = ["valence", "energy", "danceability", "popularity"]
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
GENRES = ["Blues", "Classical", "Country", "Electronic", "Folk", "Hip-Hop", "Jazz", "Pop", "R&B", "Rock"]
AUDIO_SKEWED_COLUMNS = ["acousticness", "instrumentalness", "speechiness"]
AUDIO_SCALER_COLUMNS = ["loudness", "tempo", "duration_ms", "year"]
TEXT_STAT_COLUMNS = ["word_count", "unique_word_count", "unique_ratio", "avg_word_length", "char_count"]
SENTIMENT_COLUMNS = ["sentiment_polarity", "sentiment_subjectivity"]
FEATURE_PARTS = [
    ("metadata", 30),
    ("mpnet", 768),
    ("vggish", 128),
    ("mert", 768),
    ("panns", 2048),
    ("mel_stats", 512),
]


sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(APP_DIR))
from audio_feature_extractor import extract_audio_features  # noqa: E402


def _optional_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency '{module_name}'. Install app requirements first.") from exc


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


@lru_cache(maxsize=1)
def load_resources() -> dict[str, Any]:
    models = {}
    for target in TARGETS:
        path = ML_MODELS_DIR / f"CatBoost_tuned_{target}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Missing ML model: {path}")
        models[target] = joblib.load(path)

    resources = {
        "ml_models": models,
        "audio_scaler": joblib.load(FEATURES_DIR / "audio_scaler.pkl"),
        "audio_power_transformer": joblib.load(FEATURES_DIR / "audio_power_transformer.pkl"),
        "genre_encoder": joblib.load(FEATURES_DIR / "genre_encoder.pkl"),
        "text_stats_scaler": joblib.load(FEATURES_DIR / "text_stats_scaler.pkl"),
        "sentiment_scaler": joblib.load(FEATURES_DIR / "sentiment_scaler.pkl"),
        "modal_scalers": {
            name: _load_pickle(SCALER_DIR / f"modal_scaler_{name}.pkl")
            for name in ["mpnet", "vggish", "mert", "panns", "mel_stats"]
        },
    }

    return resources


@lru_cache(maxsize=1)
def load_mpnet_model():
    sentence_transformers = _optional_import("sentence_transformers")
    model = sentence_transformers.SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    model.max_seq_length = 512
    return model


@lru_cache(maxsize=1)
def load_mert():
    transformers = _optional_import("transformers")
    torch = _optional_import("torch")
    processor = transformers.AutoProcessor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    model = transformers.AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    model.eval()
    return processor, model, torch


@lru_cache(maxsize=1)
def load_panns():
    torch = _optional_import("torch")
    panns_inference = _optional_import("panns_inference")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return panns_inference.AudioTagging(checkpoint_path=None, device=device), device


@lru_cache(maxsize=1)
def load_vggish():
    tensorflow_hub = _optional_import("tensorflow_hub")
    return tensorflow_hub.load("https://tfhub.dev/google/vggish/1")


@lru_cache(maxsize=1)
def load_dl_model():
    torch = _optional_import("torch")
    from dl.utils.thesis_models import AttentionTaskGatedFusionMLP

    if not DL_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing DL checkpoint: {DL_CHECKPOINT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionTaskGatedFusionMLP(
        dropout_enc=0.21061863181632923,
        dropout_fusion=0.2071847502459279,
        metadata_dim=30,
    ).to(device)
    checkpoint = torch.load(DL_CHECKPOINT_PATH, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, torch, device


def resolve_audio_source(uploaded_audio: str | None, youtube_url: str | None) -> tuple[str, str]:
    if uploaded_audio:
        return uploaded_audio, "Uploaded audio"

    if not youtube_url or not youtube_url.strip():
        raise ValueError("Provide either an audio file or a YouTube URL.")

    try:
        yt_dlp = _optional_import("yt_dlp")
    except RuntimeError:
        pip = shutil.which("pip") or shutil.which("pip3")
        if pip is None:
            raise RuntimeError("yt-dlp is not installed and pip is not available.")
        subprocess.check_call([pip, "install", "yt-dlp"])
        yt_dlp = _optional_import("yt_dlp")

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    output_template = str(DOWNLOAD_DIR / "%(id)s.%(ext)s")
    options = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(youtube_url.strip(), download=True)
        source = f"YouTube: {info.get('title', youtube_url.strip())}"
        audio_path = Path(downloader.prepare_filename(info)).with_suffix(".wav")

    if not audio_path.exists():
        matches = sorted(DOWNLOAD_DIR.glob(f"{info['id']}.*"))
        if not matches:
            raise FileNotFoundError("yt-dlp finished but no downloaded audio file was found.")
        audio_path = matches[0]

    return str(audio_path), source


def process_text(lyrics: str, resources: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text = lyrics.strip() if lyrics and lyrics.strip() else ""
    embedding_text = text[:3000] if len(text) > 3000 else text
    words = text.split()
    unique = {word.lower() for word in words}
    raw_stats = np.array([[
        np.log1p(len(words)),
        np.log1p(len(unique)),
        len(unique) / max(len(words), 1),
        float(np.mean([len(word) for word in words])) if words else 0.0,
        np.log1p(len(text)),
    ]], dtype=np.float32)
    text_stats_df = pd.DataFrame(raw_stats, columns=TEXT_STAT_COLUMNS)
    text_stats = resources["text_stats_scaler"].transform(text_stats_df).astype(np.float32)

    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
    except Exception:
        polarity = 0.0
        subjectivity = 0.0
    sentiment_raw = np.array([[polarity, subjectivity]], dtype=np.float32)
    sentiment_df = pd.DataFrame(sentiment_raw, columns=SENTIMENT_COLUMNS)
    sentiment = resources["sentiment_scaler"].transform(sentiment_df).astype(np.float32)

    mpnet = load_mpnet_model().encode(
        [embedding_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    return text_stats, sentiment, mpnet


def build_base_audio_features(
    extracted: dict[str, float] | None,
    base_audio_mode: str,
    manual: dict[str, float],
    genre: str,
    year: int,
    artist_followers: int,
    artist_popularity: int,
    resources: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float], str]:
    if base_audio_mode == "Manual":
        raw = dict(manual)
        source = "manual"
    elif extracted is not None:
        raw = dict(extracted)
        source = "automatic"
    else:
        raise ValueError("Automatic base audio features require an uploaded or downloaded audio file.")

    skewed = pd.DataFrame(
        [[raw["acousticness"], raw["instrumentalness"], raw["speechiness"]]],
        columns=AUDIO_SKEWED_COLUMNS,
    )
    power = resources["audio_power_transformer"].transform(skewed)
    normalized = np.array([[raw["liveness"]]], dtype=np.float32)
    scaled_raw = pd.DataFrame([[
        raw["loudness"],
        raw["tempo"],
        raw["duration_ms"],
        float(year),
    ]], columns=AUDIO_SCALER_COLUMNS)
    scaled = resources["audio_scaler"].transform(scaled_raw)
    mode = np.array([[int(raw["mode"])]], dtype=np.float32)
    key = int(np.clip(raw["key"], 0, 11))
    key_block = np.array([[np.sin(2 * np.pi * key / 12), np.cos(2 * np.pi * key / 12)]], dtype=np.float32)
    genre_block = resources["genre_encoder"].transform(pd.DataFrame([[genre]], columns=["genre"])).astype(np.float32)
    artist_block = np.array([[np.log1p(max(artist_followers, 0)), float(artist_popularity)]], dtype=np.float32)

    audio = np.hstack([power, normalized, scaled, mode, key_block, genre_block, artist_block]).astype(np.float32)
    if audio.shape[1] != 23:
        raise ValueError(f"Expected 23 base audio features, got {audio.shape[1]}")

    raw["key"] = key
    raw["mode"] = int(raw["mode"])
    raw["duration_ms"] = float(raw["duration_ms"])
    return audio, raw, source


def build_base_audio_features_from_training_row(row: pd.Series, resources: dict[str, Any]) -> np.ndarray:
    row = row.copy()
    row["log_total_artist_followers"] = np.log1p(max(float(row.get("total_artist_followers", 0) or 0), 0))
    row["avg_artist_popularity"] = float(row.get("avg_artist_popularity", 0) or 0)

    skewed = pd.DataFrame([[row[col] for col in AUDIO_SKEWED_COLUMNS]], columns=AUDIO_SKEWED_COLUMNS)
    power = resources["audio_power_transformer"].transform(skewed)
    normalized = pd.DataFrame([[row["liveness"]]], columns=["liveness"]).to_numpy(copy=False)
    scaled_raw = pd.DataFrame([[row[col] for col in AUDIO_SCALER_COLUMNS]], columns=AUDIO_SCALER_COLUMNS)
    scaled = resources["audio_scaler"].transform(scaled_raw)
    mode = pd.DataFrame([[row["mode"]]], columns=["mode"]).to_numpy(copy=False)

    key = pd.to_numeric(row["key"], errors="coerce")
    if pd.isna(key) or key == -1:
        key_block = np.array([[0.0, 0.0]])
    else:
        key_block = np.array([[np.sin(2 * np.pi * key / 12), np.cos(2 * np.pi * key / 12)]])

    genre_block = resources["genre_encoder"].transform(pd.DataFrame([[row["genre"]]], columns=["genre"]))
    artist_block = pd.DataFrame(
        [[row["log_total_artist_followers"], row["avg_artist_popularity"]]],
        columns=["log_total_artist_followers", "avg_artist_popularity"],
    ).to_numpy(copy=False)

    return np.hstack([power, normalized, scaled, mode, key_block, genre_block, artist_block]).astype(np.float32)


def validate_preprocessing_against_saved(split: str = "train", row_index: int = 0) -> dict[str, float]:
    resources = load_resources()
    row = pd.read_csv(BASE_DIR / "data" / "processed" / f"{split}.csv").iloc[row_index]
    lyrics = str(row.get("lyrics", ""))
    text = lyrics.strip() if lyrics and lyrics.strip() else ""
    words = text.split()
    unique = {word.lower() for word in words}
    text_stats_raw = pd.DataFrame([{
        "word_count": np.log1p(len(words)),
        "unique_word_count": np.log1p(len(unique)),
        "unique_ratio": len(unique) / max(len(words), 1),
        "avg_word_length": float(np.mean([len(word) for word in words])) if words else 0.0,
        "char_count": np.log1p(len(text)),
    }], columns=TEXT_STAT_COLUMNS)
    text_stats = resources["text_stats_scaler"].transform(text_stats_raw).astype(np.float32)

    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
    except Exception:
        polarity = 0.0
        subjectivity = 0.0
    sentiment_raw = pd.DataFrame([{
        "sentiment_polarity": polarity,
        "sentiment_subjectivity": subjectivity,
    }], columns=SENTIMENT_COLUMNS)
    sentiment = resources["sentiment_scaler"].transform(sentiment_raw).astype(np.float32)
    audio = build_base_audio_features_from_training_row(row, resources)

    expected_audio = np.load(FEATURES_DIR / f"X_{split}_audio.npy", mmap_mode="r")[row_index].astype(np.float32)
    expected_text = np.load(FEATURES_DIR / f"X_{split}_text_stats.npy", mmap_mode="r")[row_index].astype(np.float32)
    expected_sentiment = np.load(FEATURES_DIR / f"X_{split}_sentiment.npy", mmap_mode="r")[row_index].astype(np.float32)

    return {
        "audio_max_abs_diff": float(np.max(np.abs(audio[0] - expected_audio))),
        "text_stats_max_abs_diff": float(np.max(np.abs(text_stats[0] - expected_text))),
        "sentiment_max_abs_diff": float(np.max(np.abs(sentiment[0] - expected_sentiment))),
    }


def _scale_embedding(name: str, arr: np.ndarray, resources: dict[str, Any]) -> np.ndarray:
    scaler = resources["modal_scalers"][name]
    zero_rows = np.abs(arr).sum(axis=1) == 0
    scaled = ((arr - scaler["mean"]) / scaler["std"]).astype(np.float32)
    scaled[zero_rows] = 0.0
    return scaled


def extract_mel_stats(audio_path: str) -> np.ndarray:
    librosa = _optional_import("librosa")
    y, sr = librosa.load(audio_path, sr=16000, mono=True, duration=30)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    stats = np.concatenate([
        log_mel.mean(axis=1),
        log_mel.std(axis=1),
        log_mel.min(axis=1),
        log_mel.max(axis=1),
    ])
    return stats.reshape(1, -1).astype(np.float32)


def extract_mert(audio_path: str) -> np.ndarray:
    librosa = _optional_import("librosa")
    processor, model, torch = load_mert()
    y, sr = librosa.load(audio_path, sr=24000, mono=True, duration=30)
    inputs = processor(y, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return emb.astype(np.float32)


def extract_panns(audio_path: str) -> np.ndarray:
    librosa = _optional_import("librosa")
    audio_tagger, _device = load_panns()
    y, _sr = librosa.load(audio_path, sr=32000, mono=True, duration=30)
    _clipwise, embedding = audio_tagger.inference(y[None, :])
    return np.asarray(embedding, dtype=np.float32)


def extract_vggish(audio_path: str) -> np.ndarray:
    librosa = _optional_import("librosa")
    model = load_vggish()
    y, _sr = librosa.load(audio_path, sr=16000, mono=True, duration=30)
    emb = model(y.astype(np.float32))
    arr = np.asarray(emb)
    if arr.ndim == 2:
        arr = arr.mean(axis=0, keepdims=True)
    return arr.reshape(1, -1).astype(np.float32)


def extract_audio_embeddings(audio_path: str, allow_zero_fallback: bool) -> tuple[dict[str, np.ndarray], list[str]]:
    extractors = {
        "vggish": (extract_vggish, 128),
        "mert": (extract_mert, 768),
        "panns": (extract_panns, 2048),
        "mel_stats": (extract_mel_stats, 512),
    }
    outputs: dict[str, np.ndarray] = {}
    notes: list[str] = []
    for name, (extractor, dim) in extractors.items():
        try:
            arr = extractor(audio_path)
            if arr.shape != (1, dim):
                raise ValueError(f"{name} extractor returned {arr.shape}, expected (1, {dim})")
        except Exception as exc:
            if not allow_zero_fallback:
                raise RuntimeError(f"{name} extraction failed: {exc}") from exc
            arr = np.zeros((1, dim), dtype=np.float32)
            notes.append(f"{name}: zero fallback ({exc})")
        outputs[name] = arr.astype(np.float32)
    return outputs, notes


def build_feature_blocks(
    audio_path: str | None,
    lyrics: str,
    genre: str,
    year: int,
    artist_followers: int,
    artist_popularity: int,
    base_audio_mode: str,
    allow_zero_fallback: bool,
    manual_audio: dict[str, float],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    resources = load_resources()
    text_stats, sentiment, mpnet_raw = process_text(lyrics, resources)
    mpnet_dl = _scale_embedding("mpnet", mpnet_raw, resources)

    extracted_audio = extract_audio_features(audio_path) if audio_path and base_audio_mode == "Automatic" else None
    base_audio, raw_audio, base_source = build_base_audio_features(
        extracted_audio,
        base_audio_mode,
        manual_audio,
        genre,
        year,
        artist_followers,
        artist_popularity,
        resources,
    )

    if audio_path:
        audio_embeddings, extraction_notes = extract_audio_embeddings(audio_path, allow_zero_fallback)
    elif allow_zero_fallback:
        audio_embeddings = {
            name: np.zeros((1, dim), dtype=np.float32)
            for name, dim in [("vggish", 128), ("mert", 768), ("panns", 2048), ("mel_stats", 512)]
        }
        extraction_notes = ["No audio file: audio embedding branches used zero fallback."]
    else:
        raise ValueError("DL audio embeddings require an uploaded/downloaded audio file unless zero fallback is enabled.")

    metadata = np.hstack([base_audio, text_stats, sentiment]).astype(np.float32)
    ml_blocks = {
        "metadata": metadata,
        "mpnet": mpnet_raw,
        **audio_embeddings,
    }
    dl_blocks = {
        "metadata": metadata,
        "mpnet": mpnet_dl,
        **{
            name: _scale_embedding(name, arr, resources)
            for name, arr in audio_embeddings.items()
        },
    }

    feature_count = sum(block.shape[1] for block in ml_blocks.values())
    if feature_count != 4254:
        raise ValueError(f"Expected 4,254 features, got {feature_count}")

    context = {
        "raw_audio": raw_audio,
        "base_audio_source": base_source,
        "extraction_notes": extraction_notes,
    }
    return ml_blocks, dl_blocks, context


def predict_ml(blocks: dict[str, np.ndarray]) -> dict[str, float]:
    resources = load_resources()
    X = np.hstack([
        blocks["metadata"][:, :23],
        blocks["metadata"][:, 23:28],
        blocks["metadata"][:, 28:30],
        blocks["mpnet"],
        blocks["vggish"],
        blocks["mert"],
        blocks["panns"],
        blocks["mel_stats"],
    ]).astype(np.float32)

    preds = {}
    for target, model in resources["ml_models"].items():
        value = float(model.predict(X)[0])
        if target == "popularity":
            value = float(np.clip(np.expm1(value), 0, 100))
        else:
            value = float(np.clip(value, 0, 1))
        preds[target] = value
    return preds


def predict_dl(blocks: dict[str, np.ndarray]) -> dict[str, float]:
    model, torch, device = load_dl_model()
    tensors = [
        torch.from_numpy(blocks[name].astype(np.float32)).to(device)
        for name, _dim in FEATURE_PARTS
    ]
    with torch.no_grad():
        pred = model(*tensors).detach().cpu().numpy()[0]

    return {
        "valence": float(np.clip(pred[0], 0, 1)),
        "energy": float(np.clip(pred[1], 0, 1)),
        "danceability": float(np.clip(pred[2], 0, 1)),
        "popularity": float(np.clip(np.expm1(pred[3]), 0, 100)),
    }


def format_results(source: str, ml_preds: dict[str, float], dl_preds: dict[str, float], context: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    rows = []
    for target in TARGETS:
        rows.append({
            "Target": target.capitalize(),
            "CatBoost tuned ML": round(ml_preds[target], 3 if target != "popularity" else 1),
            "Attention DL tuned": round(dl_preds[target], 3 if target != "popularity" else 1),
        })

    audio = context["raw_audio"]
    notes = context["extraction_notes"]
    notes_text = "\n".join(f"- {note}" for note in notes) if notes else "- All audio embedding branches extracted."
    summary = f"""
### Source
{source}

### Base Audio Features
- Mode: {context["base_audio_source"]}
- Tempo: {audio["tempo"]:.1f} BPM
- Key: {KEY_NAMES[int(audio["key"])]} {"Major" if int(audio["mode"]) == 1 else "Minor"}
- Loudness: {audio["loudness"]:.2f} dB
- Duration: {audio["duration_ms"] / 1000:.1f} seconds
- Acousticness: {audio["acousticness"]:.3f}
- Instrumentalness: {audio["instrumentalness"]:.3f}
- Speechiness: {audio["speechiness"]:.3f}
- Liveness: {audio["liveness"]:.3f}

### Extraction Notes
{notes_text}
"""
    return pd.DataFrame(rows), summary


def predict(
    uploaded_audio,
    youtube_url,
    lyrics,
    genre,
    year,
    artist_followers,
    artist_popularity,
    base_audio_mode,
    acousticness,
    instrumentalness,
    speechiness,
    liveness,
    loudness,
    tempo,
    duration_ms,
    key,
    mode,
    allow_zero_fallback,
):
    try:
        audio_path, source = resolve_audio_source(uploaded_audio, youtube_url)
        manual_audio = {
            "acousticness": float(acousticness),
            "instrumentalness": float(instrumentalness),
            "speechiness": float(speechiness),
            "liveness": float(liveness),
            "loudness": float(loudness),
            "tempo": float(tempo),
            "duration_ms": float(duration_ms),
            "key": int(key),
            "mode": 1 if mode == "Major" else 0,
        }
        ml_blocks, dl_blocks, context = build_feature_blocks(
            audio_path=audio_path,
            lyrics=lyrics or "",
            genre=genre,
            year=int(year),
            artist_followers=int(artist_followers or 0),
            artist_popularity=int(artist_popularity or 0),
            base_audio_mode=base_audio_mode,
            allow_zero_fallback=bool(allow_zero_fallback),
            manual_audio=manual_audio,
        )
        ml_preds = predict_ml(ml_blocks)
        dl_preds = predict_dl(dl_blocks)
        return (*format_results(source, ml_preds, dl_preds, context), "")
    except Exception as exc:
        import traceback

        return (
            pd.DataFrame(columns=["Target", "CatBoost tuned ML", "Attention DL tuned"]),
            "",
            f"```\n{exc}\n\n{traceback.format_exc()}\n```",
        )


def create_interface():
    with gr.Blocks(title="Music Attribute Prediction") as app:
        gr.Markdown("# Music Attribute Prediction")

        with gr.Row():
            with gr.Column(scale=1):
                uploaded_audio = gr.Audio(label="Upload MP3/WAV", type="filepath")
                youtube_url = gr.Textbox(label="YouTube URL", placeholder="Paste a video URL when not uploading a file")
                lyrics = gr.Textbox(label="Lyrics", lines=8)

                with gr.Row():
                    genre = gr.Dropdown(GENRES, value="Pop", label="Genre")
                    year = gr.Slider(1960, 2026, value=2020, step=1, label="Release Year")

                with gr.Row():
                    artist_followers = gr.Number(label="Artist Followers", value=1_000_000, minimum=0)
                    artist_popularity = gr.Slider(0, 100, value=50, step=1, label="Artist Popularity")

                base_audio_mode = gr.Radio(["Automatic", "Manual"], value="Automatic", label="Base Audio Features")
                allow_zero_fallback = gr.Checkbox(
                    label="Allow zero fallback for unavailable audio embedding extractors",
                    value=False,
                )

                with gr.Accordion("Manual Base Audio Features", open=False):
                    acousticness = gr.Slider(0, 1, value=0.3, step=0.001, label="Acousticness")
                    instrumentalness = gr.Slider(0, 1, value=0.0, step=0.001, label="Instrumentalness")
                    speechiness = gr.Slider(0, 1, value=0.05, step=0.001, label="Speechiness")
                    liveness = gr.Slider(0, 1, value=0.1, step=0.001, label="Liveness")
                    loudness = gr.Number(label="Loudness dB", value=-8.0)
                    tempo = gr.Number(label="Tempo BPM", value=120.0)
                    duration_ms = gr.Number(label="Duration ms", value=180000)
                    key = gr.Slider(0, 11, value=0, step=1, label="Key")
                    mode = gr.Radio(["Major", "Minor"], value="Major", label="Mode")

                predict_btn = gr.Button("Predict", variant="primary")

            with gr.Column(scale=1):
                result_table = gr.Dataframe(label="Predictions", interactive=False)
                detail_output = gr.Markdown()
                error_output = gr.Markdown()

        predict_btn.click(
            fn=predict,
            inputs=[
                uploaded_audio,
                youtube_url,
                lyrics,
                genre,
                year,
                artist_followers,
                artist_popularity,
                base_audio_mode,
                acousticness,
                instrumentalness,
                speechiness,
                liveness,
                loudness,
                tempo,
                duration_ms,
                key,
                mode,
                allow_zero_fallback,
            ],
            outputs=[result_table, detail_output, error_output],
        )

    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(
        share=False,
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Soft(),
    )
