"""
Generates notebooks/02_global_multilingual_lyric_analysis.ipynb
Global Cultural & Multilingual Lyric Analysis (Option D).
"""

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("""# 🌍 Global Cultural & Multilingual Lyric Analysis
### Analyzing Emotional Expression, Lexical Diversity, and Thematic Clusters Across 10,000 Global Songs

This notebook explores the rich **multilingual and NLP descriptors** included in the Spotify 10k dataset, focusing on cross-lingual emotional patterns, vocabulary richness, and topic themes across **35 language/script categories** (including Devanagari Hindi, Romanized Hinglish, Spanish, Korean, Japanese, Portuguese, Indonesian, Turkish, etc.).
"""))

# Section 1: Environment & Setup
cells.append(nbf.v4.new_markdown_cell("""## 1. Setup & Data Loading"""))
cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

# Robust path resolution
if Path('data/metadata/songs.parquet').exists():
    DATA_DIR = Path('data')
elif Path('../data/metadata/songs.parquet').exists():
    DATA_DIR = Path('../data')
elif Path('/kaggle/input/spotify-10k-music-features').exists():
    DATA_DIR = Path('/kaggle/input/spotify-10k-music-features')
else:
    raise FileNotFoundError("Dataset path not found.")

songs = pd.read_parquet(DATA_DIR / 'metadata' / 'songs.parquet')
lang_df = pd.read_parquet(DATA_DIR / 'features' / 'lyric' / 'language_id.parquet')
lyric_stats = pd.read_parquet(DATA_DIR / 'features' / 'lyric' / 'lyric_stats.parquet')
emotions = pd.read_parquet(DATA_DIR / 'features' / 'lyric' / 'go_emotions.parquet')
topics = pd.read_parquet(DATA_DIR / 'features' / 'lyric' / 'bertopic_topics.parquet')

with open(DATA_DIR / 'features' / 'lyric' / 'bertopic_topic_labels.json') as f:
    topic_labels = json.load(f)

print(f"Loaded {len(songs):,} songs with multilingual annotations.")
"""))

# Section 2: Global Language Breakdown
cells.append(nbf.v4.new_markdown_cell("""## 2. Global Music Language & Script Distribution
We inspect the distribution across major global music markets, highlighting both standard scripts and Romanized variants (such as Latin-script Hinglish).
"""))
cells.append(nbf.v4.new_code_cell("""# Summarize key language counts
lang_counts = {
    'English': lang_df['is_english'].sum(),
    'Spanish': lang_df['is_spanish'].sum(),
    'Portuguese': lang_df['is_portuguese'].sum(),
    'Hindi (Devanagari)': lang_df['is_hindi_devanagari'].sum(),
    'Hindi (Romanized/Hinglish)': lang_df['is_hindi_romanized'].sum(),
    'Indonesian': lang_df['is_indonesian'].sum(),
    'Korean': lang_df['is_korean'].sum(),
    'Turkish': lang_df['is_turkish'].sum(),
    'Japanese': lang_df['is_japanese'].sum(),
    'French': lang_df['is_french'].sum(),
    'Italian': lang_df['is_italian'].sum(),
    'German': lang_df['is_german'].sum()
}

s_lang = pd.Series(lang_counts).sort_values(ascending=True)

plt.figure(figsize=(9, 5))
s_lang.plot(kind='barh', color='teal', edgecolor='black', alpha=0.8)
plt.title("Top Languages & Script Categories in Spotify Top-10,000")
plt.xlabel("Number of Tracks")
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
"""))

# Section 3: Lexical Complexity & Richness
cells.append(nbf.v4.new_markdown_cell("""## 3. Lexical Diversity & Vocabulary Richness Across Languages
We analyze **Type-Token Ratio (TTR)**, **Measure of Textual Lexical Diversity (MTLD)**, and **Hapax Legomena Ratio** (ratio of single-use unique words).
"""))
cells.append(nbf.v4.new_code_cell("""# Combine language and lexical richness features
df_lex = pd.DataFrame({
    'language': lang_df['primary_language'],
    'ttr': lyric_stats['ttr'],
    'root_ttr': lyric_stats['root_ttr'],
    'mtld': lyric_stats['mtld'],
    'hapax_ratio': lyric_stats['hapax_ratio'],
    'line_count': lyric_stats['line_count']
})

# Filter top languages with > 50 songs
top_langs = ['en', 'es', 'pt', 'hi', 'id', 'ko', 'tr', 'ja', 'fr']
df_lex_filtered = df_lex[df_lex['language'].isin(top_langs) & (df_lex['line_count'] > 5)]

lex_summary = df_lex_filtered.groupby('language')[['ttr', 'root_ttr', 'mtld', 'hapax_ratio']].median()
print("Median Lexical Diversity by Language:")
lex_summary.sort_values(by='ttr', ascending=False)
"""))

# Section 4: Emotion Profiles Across Top Genres
cells.append(nbf.v4.new_markdown_cell("""## 4. Emotional Profiles Across Genres (NRC & GoEmotions)
We examine the dominant emotional nuances (Joy, Sadness, Anger, Fear, Love, Optimism) across different genres.
"""))
cells.append(nbf.v4.new_code_cell("""# Calculate sentiment & emotion averages for popular genres
df_genres_emot = pd.DataFrame({
    'genre': songs['main_genres'],
    'vader_compound': lyric_stats['vader_compound'],
    'nrc_joy': lyric_stats['nrc_joy'],
    'nrc_sadness': lyric_stats['nrc_sadness'],
    'nrc_anger': lyric_stats['nrc_anger'],
    'nrc_anticipation': lyric_stats['nrc_anticipation']
})

top_genre_list = df_genres_emot['genre'].value_counts().head(8).index
genre_emot_avg = df_genres_emot[df_genres_emot['genre'].isin(top_genre_list)].groupby('genre').mean()

plt.figure(figsize=(10, 5))
for col in ['nrc_joy', 'nrc_sadness', 'nrc_anger']:
    plt.plot(genre_emot_avg.index, genre_emot_avg[col], marker='o', label=col.replace('nrc_', '').capitalize())

plt.title("NRC Emotional Intensity Across Top Genres")
plt.xticks(rotation=35, ha='right')
plt.ylabel("Mean Emotion Word Ratio")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""))

# Section 5: Thematic Clusters via BERTopic
cells.append(nbf.v4.new_markdown_cell("""## 5. Discovering Thematic Clusters via BERTopic
Explore the 32 discovered lyric themes and their top c-TF-IDF keyword signatures.
"""))
cells.append(nbf.v4.new_code_cell("""# Display top 10 lyric topics with their counts and keywords
topic_summary = []
for t_id, data in topic_labels.items():
    topic_summary.append({
        'topic_id': int(t_id),
        'theme_name': data['name'],
        'song_count': data['count'],
        'top_keywords': ", ".join(data['top_words'][:5])
    })

df_topics = pd.DataFrame(topic_summary).sort_values(by='song_count', ascending=False)
df_topics.head(10)
"""))

# Section 6: 2D Semantic Map
cells.append(nbf.v4.new_markdown_cell("""## 6. 2D Semantic Lyric Map Exploration
Visualizing the global 10,000 lyric semantic space colored by language category.
"""))
cells.append(nbf.v4.new_code_cell("""umap_lyric = pd.read_parquet(DATA_DIR / 'similarity' / 'umap_2d_lyric.parquet')

plt.figure(figsize=(8, 5))
plt.scatter(
    umap_lyric['proj_x'], 
    umap_lyric['proj_y'], 
    c=lang_df['is_english'].map({True: 'dodgerblue', False: 'crimson'}), 
    alpha=0.25, 
    s=8
)
plt.title("2D Semantic Lyric Space (Blue: English | Red: Non-English / Multilingual)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True, alpha=0.2)
plt.show()
"""))

nb['cells'] = cells

out_file = Path("notebooks/02_global_multilingual_lyric_analysis.ipynb")
with open(out_file, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Created notebook: {out_file}")
