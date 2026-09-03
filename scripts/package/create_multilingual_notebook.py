"""
Generates notebooks/02_global_multilingual_lyric_analysis.ipynb
Global Cultural & Multilingual Lyric Analysis with Harrier-0.6B, E5, and NLP descriptors.
"""

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("""# 🌍 Global Cultural & Multilingual Lyric Analysis
### Cross-Lingual Semantic Space, Emotional Patterns, Lexical Diversity, and Thematic Clusters Across 10,000 Songs

This notebook explores the rich **multilingual NLP descriptors** and **deep neural embeddings** included in the Spotify 10k dataset. We analyze lyric storytelling patterns across **35 language/script categories** (including English, Spanish, Hindi Devanagari & Romanized Hinglish, Korean, Japanese, Portuguese, Indonesian, Turkish, French, etc.) using:

- **Microsoft Harrier-OSS-v1-0.6B (1024-D):** 32k context multilingual decoder embeddings.
- **Multilingual E5-Large (1024-D):** Dense contrastive sentence representations.
- **Language ID & Script Flags (`language_id.parquet`):** 34 primary/secondary language and script indicators.
- **Lexical Statistics (`lyric_stats.parquet`):** TTR, Root-TTR, MTLD, Hapax ratio, VADER sentiment, and NRC EmoLex.
- **GoEmotions (`go_emotions.parquet`):** 28 fine-grained RoBERTa emotion probability scores.
- **BERTopic (`bertopic_topics.parquet`):** 32 discovered thematic lyric clusters.
"""))

# Section 1: Environment & Setup
cells.append(nbf.v4.new_markdown_cell("""## 1. Setup & Data Loading"""))
cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# Robust path resolution for Local Repo or Kaggle environment
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

# Load deep lyric embeddings
harrier = np.load(DATA_DIR / 'embeddings' / 'lyric' / 'harrier_embeddings_1024d.npy')

with open(DATA_DIR / 'features' / 'lyric' / 'bertopic_topic_labels.json') as f:
    topic_labels = json.load(f)

print(f"Loaded {len(songs):,} songs with multilingual annotations.")
"""))

# Section 2: Global Language Breakdown
cells.append(nbf.v4.new_markdown_cell("""## 2. Global Music Language & Script Distribution
We inspect the distribution across major global music markets, highlighting both native scripts and Romanized variants (such as Latin-script Hinglish).
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
plt.title("Top Languages & Script Categories in Spotify Top-10,000", fontsize=12, fontweight='bold')
plt.xlabel("Number of Tracks")
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
"""))

# Section 3: 2D Semantic Lyric Space
cells.append(nbf.v4.new_markdown_cell("""## 3. High-Resolution 2D Semantic Lyric Space (`umap_2d_lyric.parquet`)
We visualize the **2D UMAP projection** of the fused `Harrier-0.6B + Multilingual E5-Large` representation, colored across global languages with dedicated cluster callouts.
"""))
cells.append(nbf.v4.new_code_cell("""umap_lyric = pd.read_parquet(DATA_DIR / 'similarity' / 'umap_2d_lyric.parquet')

plt.figure(figsize=(10, 6.5))

# Map known languages, then fill unknowns with 'lime'
colors = lang_df['primary_language'].map({
    'en': 'crimson',
    'es': 'gold',
    'pt': 'darkorange',
    'hi': 'darksalmon',
    'id': 'mediumseagreen',
    'ko': 'mediumpurple',
    'tr': 'turquoise',
    'ja': 'hotpink',
    'fr': 'royalblue',
    'none': 'gray'
}).fillna('lime')

plt.scatter(
    umap_lyric['proj_x'], 
    umap_lyric['proj_y'],
    c=colors,
    alpha=0.35, 
    s=10,
    edgecolors='none'
)

plt.title("2D Semantic Lyric Space (Harrier-0.6B + Multilingual E5)", fontsize=13, fontweight='bold')
plt.xlabel("UMAP Dimension 1", fontsize=10)
plt.ylabel("UMAP Dimension 2", fontsize=10)
plt.grid(True, alpha=0.2)

# Rich legend with colored patches
legend_elements = [
    Patch(facecolor='crimson', label='English'),
    Patch(facecolor='gold', label='Spanish'),
    Patch(facecolor='darkorange', label='Portuguese'),
    Patch(facecolor='darksalmon', label='Hindi (Devanagari & Hinglish)'),
    Patch(facecolor='mediumseagreen', label='Indonesian'),
    Patch(facecolor='mediumpurple', label='Korean'),
    Patch(facecolor='turquoise', label='Turkish'),
    Patch(facecolor='hotpink', label='Japanese'),
    Patch(facecolor='royalblue', label='French'),
    Patch(facecolor='gray', label='Instrumental / None'),
    Patch(facecolor='lime', label='Other Languages')
]
plt.legend(handles=legend_elements, loc='lower left', fontsize=8, framealpha=0.95)

plt.tight_layout()
plt.show()
"""))

# Section 4: Cross-Lingual Semantic Similarity Matrix
cells.append(nbf.v4.new_markdown_cell("""## 4. Cross-Lingual Semantic Similarity Geometry
Using the normalized **Harrier-0.6B (1024-D)** embeddings, we compute the average pairwise cosine similarity between different global language corpora to see which languages cluster closest in narrative style.
"""))
cells.append(nbf.v4.new_code_cell("""# Compute language centroids using Harrier embeddings
top_langs_eval = ['en', 'es', 'pt', 'hi', 'id', 'ko', 'tr', 'ja', 'fr']
lang_labels = {
    'en': 'English', 'es': 'Spanish', 'pt': 'Portuguese', 'hi': 'Hindi',
    'id': 'Indonesian', 'ko': 'Korean', 'tr': 'Turkish', 'ja': 'Japanese', 'fr': 'French'
}

# L2 Normalize Harrier vectors
harrier_norm = harrier / np.maximum(np.linalg.norm(harrier, axis=1, keepdims=True), 1e-8)

centroids = {}
for lang in top_langs_eval:
    mask = (lang_df['primary_language'] == lang) & np.any(harrier != 0, axis=1)
    if mask.sum() > 0:
        c = harrier_norm[mask].mean(axis=0)
        centroids[lang_labels[lang]] = c / np.linalg.norm(c)

df_centroids = pd.DataFrame(centroids)
sim_matrix = df_centroids.T @ df_centroids

plt.figure(figsize=(8, 6.5))
sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
plt.title("Cross-Lingual Semantic Similarity Matrix (Harrier-0.6B)", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

# Section 5: Lexical Diversity Across Languages
cells.append(nbf.v4.new_markdown_cell("""## 5. Lexical Diversity & Vocabulary Richness Across Languages
We analyze **Type-Token Ratio (TTR)**, **Measure of Textual Lexical Diversity (MTLD)**, and **Hapax Legomena Ratio** (ratio of unique single-use words).
"""))
cells.append(nbf.v4.new_code_cell("""df_lex = pd.DataFrame({
    'language': lang_df['primary_language'].map(lang_labels),
    'ttr': lyric_stats['ttr'],
    'root_ttr': lyric_stats['root_ttr'],
    'mtld': lyric_stats['mtld'],
    'hapax_ratio': lyric_stats['hapax_ratio'],
    'line_count': lyric_stats['line_count']
})

df_lex_filtered = df_lex[df_lex['language'].notna() & (df_lex['line_count'] > 5)]

lex_summary = df_lex_filtered.groupby('language')[['ttr', 'root_ttr', 'mtld', 'hapax_ratio']].median()
print("Median Lexical Diversity by Language:")
lex_summary.sort_values(by='ttr', ascending=False)
"""))

# Section 6: Emotion Profiles Across Top Genres
cells.append(nbf.v4.new_markdown_cell("""## 6. Emotional Profiles Across Genres (NRC & GoEmotions)
We examine the dominant emotional nuances (Joy, Sadness, Anger, Anticipation, Fear) across musical genres.
"""))
cells.append(nbf.v4.new_code_cell("""df_genres_emot = pd.DataFrame({
    'genre': songs['main_genres'],
    'nrc_joy': lyric_stats['nrc_joy'],
    'nrc_sadness': lyric_stats['nrc_sadness'],
    'nrc_anger': lyric_stats['nrc_anger'],
    'nrc_anticipation': lyric_stats['nrc_anticipation'],
    'nrc_fear': lyric_stats['nrc_fear']
})

top_genre_list = df_genres_emot['genre'].value_counts().head(7).index
genre_emot_avg = df_genres_emot[df_genres_emot['genre'].isin(top_genre_list)].groupby('genre').mean()

plt.figure(figsize=(10, 5))
for col in ['nrc_joy', 'nrc_sadness', 'nrc_anger', 'nrc_anticipation']:
    plt.plot(genre_emot_avg.index, genre_emot_avg[col], marker='o', linewidth=2, label=col.replace('nrc_', '').capitalize())

plt.title("NRC Emotional Intensity Across Global Genres", fontsize=12, fontweight='bold')
plt.xticks(rotation=30, ha='right')
plt.ylabel("Mean Emotion Word Ratio")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""))

# Section 7: Thematic Clusters via BERTopic
cells.append(nbf.v4.new_markdown_cell("""## 7. Discovering Thematic Clusters via BERTopic
Explore the 32 discovered lyric themes and their top c-TF-IDF keyword signatures.
"""))
cells.append(nbf.v4.new_code_cell("""topic_summary = []
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

nb['cells'] = cells

out_file = Path("notebooks/02_global_multilingual_lyric_analysis.ipynb")
with open(out_file, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Created notebook: {out_file}")
