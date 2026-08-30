"""
Comprehensive Language and Script Identification Script for Song Lyrics.
Detects:
- FastText 176 ISO language codes + confidence scores
- Native Script detection (Latin, Devanagari, Hangul, CJK/Kana, Cyrillic, Arabic, Thai)
- Specific detection for:
  * Hindi (both Devanagari alphabetical and Romanized/Hinglish)
  * Indonesian, Japanese, Chinese, Dutch, German, Russian, Italian, French, Arabic
  * Scandinavian languages (Swedish, Norwegian, Danish, Finnish)
  * Turkish, Korean, Portuguese, Tagalog, Punjabi, Tamil, Telugu, Polish
Outputs: data/features/lyric/language_id.parquet
"""

import os
import re
import unicodedata
from pathlib import Path
import pandas as pd
import numpy as np
import fasttext
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "lid.176.ftz"
OUTPUT_DIR = DATA_DIR / "features" / "lyric"
OUTPUT_FILE = OUTPUT_DIR / "language_id.parquet"

# Unambiguous Hindi/Urdu Romanized words (excluding English/Spanish collision words)
HINDI_ROMAN_WORDS = {
    'tum', 'meri', 'mera', 'tere', 'tera', 'tujhe', 'mujhe', 'kabhi', 'zindagi', 'rabba',
    'pyar', 'pyaar', 'ishq', 'kyun', 'aankhon', 'deewana', 'sanam', 'saath', 'duniya',
    'chahiye', 'nachle', 'jaana', 'raha', 'rahi', 'kuch', 'hona', 'gaya', 'gayi',
    'dekha', 'dekho', 'kahan', 'batao', 'aashiq', 'mohabbat', 'khuda', 'naina', 'yaara',
    'chhod', 'karti', 'karta', 'paas', 'humein', 'tumhe', 'unhe', 'apna', 'apni',
    'dil', 'pehla', 'pehli', 'khushi', 'bewafa', 'raaton', 'baatein', 'dilwale',
    'dhadkan', 'sajna', 'dhola', 'mahi', 'jiyen', 'jeena', 'marna', 'sath',
    'bhula', 'chaha', 'khoya', 'milega', 'samjha', 'hoga', 'hogi', 'aaye', 'aayega',
    'socha', 'kahin', 'wahi', 'sabse', 'accha', 'achha', 'pyari', 'pyare'
}

def analyze_track_language(text, model):
    empty_res = {
        'has_lyrics': False, 'primary_language': 'none', 'secondary_language': 'none',
        'lang_confidence': 0.0, 'primary_script': 'none',
        'is_english': False, 'is_spanish': False,
        'is_hindi_devanagari': False, 'is_hindi_romanized': False, 'is_hindi_any': False,
        'is_indonesian': False, 'is_japanese': False, 'is_chinese': False,
        'is_dutch': False, 'is_german': False, 'is_russian': False,
        'is_italian': False, 'is_french': False, 'is_arabic': False,
        'is_portuguese': False, 'is_turkish': False, 'is_korean': False,
        'is_tagalog': False, 'is_swedish': False, 'is_norwegian': False,
        'is_danish': False, 'is_finnish': False, 'is_scandinavian': False,
        'is_punjabi': False, 'is_tamil': False, 'is_telugu': False,
        'is_polish': False, 'is_multilingual': False
    }
    if not isinstance(text, str) or not text.strip():
        return empty_res
    
    total_chars = max(len(text), 1)
    
    # 1. Script detection
    devanagari_cnt = len(re.findall(r'[\u0900-\u097F]', text))
    hangul_cnt = len(re.findall(r'[\uAC00-\uD7AF\u1100-\u11FF]', text))
    kana_cnt = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF]', text))
    cjk_cnt = len(re.findall(r'[\u4E00-\u9FFF]', text))
    cyrillic_cnt = len(re.findall(r'[\u0400-\u04FF]', text))
    arabic_cnt = len(re.findall(r'[\u0600-\u06FF]', text))
    latin_cnt = len(re.findall(r'[a-zA-Z]', text))
    
    scripts = {
        'devanagari': devanagari_cnt, 'hangul': hangul_cnt, 'japanese_kana': kana_cnt,
        'cjk': cjk_cnt, 'cyrillic': cyrillic_cnt, 'arabic': arabic_cnt, 'latin': latin_cnt
    }
    dom_script = max(scripts, key=scripts.get)
    if scripts[dom_script] == 0:
        dom_script = 'other'
    
    clean_text = unicodedata.normalize('NFC', text)
    clean_text = re.sub(r'\[.*?\]', ' ', clean_text)
    clean_line = re.sub(r'\s+', ' ', clean_text).strip()
    
    if len(clean_line) < 10:
        empty_res['has_lyrics'] = True
        empty_res['primary_language'] = 'too_short'
        empty_res['primary_script'] = dom_script
        return empty_res

    # FastText predictions (top 2)
    labels, probs = model.predict(clean_line, k=2)
    top_lang = labels[0].replace('__label__', '')
    top_conf = float(probs[0])
    sec_lang = labels[1].replace('__label__', '') if len(labels) > 1 else 'none'
    
    # Script overrides
    if devanagari_cnt / total_chars > 0.10:
        top_lang = 'hi'
        top_conf = 1.0
        dom_script = 'devanagari'
    elif hangul_cnt / total_chars > 0.08:
        top_lang = 'ko'
        top_conf = 1.0
        dom_script = 'hangul'
    elif kana_cnt / total_chars > 0.05:
        top_lang = 'ja'
        top_conf = 1.0
        dom_script = 'japanese_kana'
    elif cjk_cnt / total_chars > 0.10 and kana_cnt == 0:
        top_lang = 'zh'
        top_conf = 0.95
        dom_script = 'cjk'
    elif cyrillic_cnt / total_chars > 0.10:
        top_lang = 'ru' if top_lang != 'uk' else 'uk'
        top_conf = 1.0
        dom_script = 'cyrillic'
    elif arabic_cnt / total_chars > 0.10:
        top_lang = 'ar'
        top_conf = 1.0
        dom_script = 'arabic'
        
    # Check Romanized Hindi in Latin text
    words = re.findall(r'\b[a-zA-Z]+\b', clean_text.lower())
    n_words = max(len(words), 1)
    hindi_matches = sum(1 for w in words if w in HINDI_ROMAN_WORDS)
    hindi_density = hindi_matches / n_words
    
    is_hi_devanagari = (dom_script == 'devanagari') or (top_lang == 'hi' and dom_script != 'latin')
    is_hi_roman = (dom_script == 'latin') and ((hindi_matches >= 6 and hindi_density >= 0.035) or (hindi_matches >= 12))
    
    if is_hi_roman:
        sec_lang = top_lang
        top_lang = 'hi_romanized'
        top_conf = round(min(1.0, float(hindi_density * 8)), 2)

    is_hi_any = is_hi_devanagari or is_hi_roman
    is_en = (top_lang == 'en') and not is_hi_roman
    is_es = (top_lang == 'es')
    is_id = (top_lang == 'id')
    is_ja = (top_lang == 'ja') or (dom_script == 'japanese_kana')
    is_zh = (top_lang == 'zh') or (dom_script == 'cjk' and dom_script != 'japanese_kana')
    is_nl = (top_lang == 'nl')
    is_de = (top_lang == 'de')
    is_ru = (top_lang == 'ru') or (dom_script == 'cyrillic')
    is_it = (top_lang == 'it')
    is_fr = (top_lang == 'fr')
    is_ar = (top_lang == 'ar') or (dom_script == 'arabic')
    is_pt = (top_lang == 'pt')
    is_tr = (top_lang == 'tr')
    is_ko = (top_lang == 'ko') or (dom_script == 'hangul')
    is_tl = (top_lang == 'tl')
    is_sv = (top_lang == 'sv')
    is_no = (top_lang == 'no')
    is_da = (top_lang == 'da')
    is_fi = (top_lang == 'fi')
    is_scandi = is_sv or is_no or is_da or is_fi
    is_pa = (top_lang == 'pa')
    is_ta = (top_lang == 'ta')
    is_te = (top_lang == 'te')
    is_pl = (top_lang == 'pl')
    
    return {
        'has_lyrics': True,
        'primary_language': top_lang,
        'secondary_language': sec_lang,
        'lang_confidence': round(top_conf, 3),
        'primary_script': dom_script,
        'is_english': is_en,
        'is_spanish': is_es,
        'is_hindi_devanagari': is_hi_devanagari,
        'is_hindi_romanized': is_hi_roman,
        'is_hindi_any': is_hi_any,
        'is_indonesian': is_id,
        'is_japanese': is_ja,
        'is_chinese': is_zh,
        'is_dutch': is_nl,
        'is_german': is_de,
        'is_russian': is_ru,
        'is_italian': is_it,
        'is_french': is_fr,
        'is_arabic': is_ar,
        'is_portuguese': is_pt,
        'is_turkish': is_tr,
        'is_korean': is_ko,
        'is_tagalog': is_tl,
        'is_swedish': is_sv,
                'is_danish': is_da,
        'is_finnish': is_fi,
        'is_scandinavian': is_scandi,
        'is_punjabi': is_pa,
        'is_tamil': is_ta,
        'is_telugu': is_te,
        'is_polish': is_pl,
        'is_multilingual': not is_en
    }

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading FastText model from {MODEL_PATH}...")
    model = fasttext.load_model(str(MODEL_PATH))

    print(f"Loading songs from {SONGS_CSV}...")
    df = pd.read_csv(SONGS_CSV)
    n_songs = len(df)

    print("Analyzing languages and scripts for 10,000 tracks...")
    records = [analyze_track_language(t, model) for t in tqdm(df['lyrics'], total=n_songs)]
    out_df = pd.DataFrame(records)
    out_df.insert(0, 'track_id', df['track_id'].values)
    out_df.insert(0, 'row_idx', np.arange(n_songs, dtype=np.int32))

    out_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved Comprehensive Language ID parquet to: {OUTPUT_FILE}")
    print(f"Shape: {out_df.shape}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

    print("\n--- Detailed Language Summary ---")
    cols_to_print = [
        'is_english', 'is_spanish', 'is_hindi_any', 'is_hindi_devanagari', 'is_hindi_romanized',
        'is_indonesian', 'is_portuguese', 'is_korean', 'is_japanese', 'is_turkish',
        'is_tagalog', 'is_french', 'is_italian', 'is_german', 'is_chinese',
        'is_dutch', 'is_russian', 'is_arabic', 'is_scandinavian', 'is_swedish',
        'is_danish', 'is_finnish', 'is_punjabi', 'is_tamil', 'is_telugu', 'is_polish'
    ]
    for c in cols_to_print:
        print(f"  {c:<24}: {out_df[c].sum():>5} tracks")

if __name__ == "__main__":
    main()
