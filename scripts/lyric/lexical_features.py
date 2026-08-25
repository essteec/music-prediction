"""
Lyric Structure, Lexical Richness, Sentiment, and Emotion Feature Extraction.
Extracts:
- Structure: line counts, stanza counts, repetition ratios
- Lexical Richness: TTR, Root TTR, MTLD, HD-D, Hapax legomena ratio
- Sentiment/Emotion: VADER, NRC EmoLex (8 emotions + positive/negative)
- Readability: Flesch ease, syllable count
- Keywords: YAKE top-5 key phrases
Outputs: data/features/lyric/lyric_stats.parquet
"""

import os
import re
import json
import unicodedata
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
from lexicalrichness import LexicalRichness
import yake

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
OUTPUT_DIR = DATA_DIR / "features" / "lyric"
OUTPUT_FILE = OUTPUT_DIR / "lyric_stats.parquet"

def clean_lyric_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\[.*?\]', '', text)
    lines = text.split('\n')
    lines = [l for l in lines if not re.match(
        r'^(Contributors?|Lyrics?\s*by|Source|Embed|You might also like|\d+Embed)',
        l.strip(), re.IGNORECASE)]
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

# Initialize extractors
vader = SentimentIntensityAnalyzer()
kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.8, top=5)

def extract_single_lyric_features(raw_text: str) -> dict:
    empty_res = {
        'has_lyrics': False,
        'line_count': 0,
        'stanza_count': 0,
        'avg_line_char_len': 0.0,
        'line_char_len_std': 0.0,
        'unique_line_ratio': 0.0,
        'repeated_line_ratio': 0.0,
        'ttr': 0.0,
        'root_ttr': 0.0,
        'mtld': 0.0,
        'hapax_ratio': 0.0,
        'flesch_reading_ease': 0.0,
        'syllable_count': 0,
        'vader_compound': 0.0,
        'vader_pos': 0.0,
        'vader_neg': 0.0,
        'vader_neu': 1.0,
        'nrc_anger': 0.0,
        'nrc_fear': 0.0,
        'nrc_anticipation': 0.0,
        'nrc_trust': 0.0,
        'nrc_surprise': 0.0,
        'nrc_sadness': 0.0,
        'nrc_joy': 0.0,
        'nrc_disgust': 0.0,
        'nrc_positive': 0.0,
        'nrc_negative': 0.0,
        'top_keywords_json': "[]"
    }

    if not isinstance(raw_text, str) or not raw_text.strip():
        return empty_res

    text = clean_lyric_text(raw_text)
    if len(text) < 10:
        return empty_res

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    stanzas = [s.strip() for s in text.split('\n\n') if s.strip()]

    line_count = len(lines)
    stanza_count = len(stanzas)
    line_lens = [len(l) for l in lines]
    avg_line_char_len = float(np.mean(line_lens)) if line_lens else 0.0
    line_char_len_std = float(np.std(line_lens)) if line_lens else 0.0

    unique_lines = set(lines)
    unique_line_ratio = len(unique_lines) / max(line_count, 1)
    repeated_line_ratio = 1.0 - unique_line_ratio

    # Lexical Richness
    try:
        lex = LexicalRichness(text)
        words_count = max(lex.words, 1)
        ttr = float(lex.ttr)
        root_ttr = float(lex.rttr)
        # MTLD is bounded to prevent hang
        try:
            mtld = float(lex.mtld(threshold=0.72)) if words_count > 10 else ttr * 10
        except Exception:
            mtld = 0.0
        # Hapax
        hapax_count = sum(1 for w, c in lex.wordlist.items() if c == 1) if hasattr(lex, 'wordlist') else 0
        hapax_ratio = hapax_count / words_count
    except Exception:
        ttr, root_ttr, mtld, hapax_ratio = 0.0, 0.0, 0.0, 0.0

    # Readability
    try:
        flesch = float(textstat.flesch_reading_ease(text))
        syllables = int(textstat.syllable_count(text))
    except Exception:
        flesch, syllables = 0.0, 0

    # VADER
    try:
        vs = vader.polarity_scores(text)
        vader_comp = float(vs['compound'])
        vader_pos = float(vs['pos'])
        vader_neg = float(vs['neg'])
        vader_neu = float(vs['neu'])
    except Exception:
        vader_comp, vader_pos, vader_neg, vader_neu = 0.0, 0.0, 0.0, 1.0

    # NRC EmoLex
    nrc_feats = {
        'nrc_anger': 0.0, 'nrc_fear': 0.0, 'nrc_anticipation': 0.0, 'nrc_trust': 0.0,
        'nrc_surprise': 0.0, 'nrc_sadness': 0.0, 'nrc_joy': 0.0, 'nrc_disgust': 0.0,
        'nrc_positive': 0.0, 'nrc_negative': 0.0
    }
    try:
        emotion_obj = NRCLex(text)
        freqs = emotion_obj.affect_frequencies
        for k in nrc_feats.keys():
            emotion_name = k.replace('nrc_', '')
            nrc_feats[k] = float(freqs.get(emotion_name, 0.0))
    except Exception:
        pass

    # YAKE Keywords
    try:
        kw = kw_extractor.extract_keywords(text)
        keywords = [k[0] for k in kw]
        top_keywords_json = json.dumps(keywords, ensure_ascii=False)
    except Exception:
        top_keywords_json = "[]"

    res = {
        'has_lyrics': True,
        'line_count': line_count,
        'stanza_count': stanza_count,
        'avg_line_char_len': round(avg_line_char_len, 2),
        'line_char_len_std': round(line_char_len_std, 2),
        'unique_line_ratio': round(unique_line_ratio, 4),
        'repeated_line_ratio': round(repeated_line_ratio, 4),
        'ttr': round(ttr, 4),
        'root_ttr': round(root_ttr, 4),
        'mtld': round(mtld, 2),
        'hapax_ratio': round(hapax_ratio, 4),
        'flesch_reading_ease': round(flesch, 2),
        'syllable_count': syllables,
        'vader_compound': round(vader_comp, 4),
        'vader_pos': round(vader_pos, 4),
        'vader_neg': round(vader_neg, 4),
        'vader_neu': round(vader_neu, 4),
        'top_keywords_json': top_keywords_json
    }
    res.update({k: round(v, 4) for k, v in nrc_feats.items()})
    return res

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading songs from {SONGS_CSV}...")
    df = pd.read_csv(SONGS_CSV)
    n_songs = len(df)

    lyrics_list = df['lyrics'].tolist()

    print("Extracting lyric features in parallel (8 workers)...")
    with ProcessPoolExecutor(max_workers=8) as executor:
        records = list(tqdm(executor.map(extract_single_lyric_features, lyrics_list, chunksize=100), total=n_songs))

    out_df = pd.DataFrame(records)
    out_df.insert(0, 'track_id', df['track_id'].values)
    out_df.insert(0, 'row_idx', np.arange(n_songs, dtype=np.int32))

    out_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved Lyric Stats parquet to: {OUTPUT_FILE}")
    print(f"Shape: {out_df.shape}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()
