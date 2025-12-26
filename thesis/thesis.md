# Multimodal Music Attribute Prediction: Combining Audio Features and Textual Analysis

---

## Abstract

This thesis investigates multimodal music attribute prediction by combining audio features with textual analysis of song lyrics. The study addresses the extent to which lyrical content—represented through text statistics, sentiment analysis, and transformer-based semantic embeddings—can improve predictions of valence, energy, danceability, and popularity beyond audio-only baselines. Experiments were conducted on a large-scale dataset of over 730,000 English-language songs from Spotify, employing artist-aware data splitting to prevent data leakage and ensure realistic generalization estimates.

A comprehensive feature engineering pipeline transformed raw data into a 412-dimensional multimodal feature space comprising 21 audio features (power-transformed, cyclically encoded, and scaled), 5 text statistics (word count, lexical diversity metrics), 2 sentiment scores (polarity, subjectivity from TextBlob), and 384 semantic embedding dimensions (all-MiniLM-L6-v2 sentence transformer). Systematic ablation experiments quantified the incremental contribution of each feature modality across 26 machine learning model variants, including linear models, tree-based ensembles, gradient boosting methods (XGBoost, CatBoost, LightGBM), and neural networks.

Results demonstrate that audio features provide strong baselines for energy (R² = 0.838) and danceability (R² = 0.579), while valence prediction benefits most substantially from multimodal integration. Semantic embeddings improved valence prediction from R² = 0.408 (audio-only) to R² = 0.474 (full multimodal), a 6.6 percentage point improvement supporting the hypothesis that lyrical content carries emotional information complementary to acoustic features. Popularity remained largely unpredictable (R² < 0.10) regardless of feature configuration, confirming that commercial success depends primarily on factors external to musical content.

Final test set evaluation confirmed healthy generalization, with XGBoost achieving best performance for energy (R² = 0.847), danceability (R² = 0.618), and valence (R² = 0.474). Gradient boosting methods consistently outperformed linear models and neural networks, establishing them as the recommended algorithm family for multimodal music prediction tasks. The findings indicate that multimodal approaches are warranted for emotion-related attributes, while simpler audio-only models may suffice for energy and rhythm-based predictions.

**Keywords**: music information retrieval, multimodal learning, audio features, lyric analysis, sentiment analysis, semantic embeddings, gradient boosting, XGBoost, valence prediction, music attribute prediction

---

## 1. Introduction

### 1.1 Background and Motivation

The proliferation of digital music streaming platforms has fundamentally transformed how music is discovered, consumed, and analyzed. Services such as Spotify, Apple Music, and YouTube Music collectively serve billions of users, generating vast amounts of data that enable sophisticated recommendation systems and music analytics. Central to these systems is the accurate characterization of musical attributes—properties such as emotional valence, energy level, danceability, and commercial popularity that define how listeners perceive and interact with music.

Traditional approaches to music attribute prediction have relied predominantly on audio signal analysis, extracting acoustic features such as tempo, loudness, spectral characteristics, and harmonic content directly from the waveform. While these audio-centric methods effectively capture the sonic properties of music, they neglect a fundamental component of the musical experience: lyrical content. Song lyrics convey semantic meaning, emotional narratives, and thematic context that may complement, reinforce, or even contradict the mood suggested by instrumental elements. A melancholic ballad paired with uplifting lyrics, or an energetic dance track exploring introspective themes, exemplifies cases where audio features alone provide incomplete characterization.

This observation motivates the investigation of multimodal approaches that integrate both acoustic and textual modalities. By combining audio features with computational representations of lyrics—including statistical text properties, sentiment scores, and deep semantic embeddings derived from transformer-based language models—it becomes possible to construct feature representations that capture both the sonic and linguistic dimensions of musical expression.

### 1.2 Problem Statement

The central research problem addressed in this study concerns the extent to which textual features extracted from song lyrics can improve the prediction of music attributes compared to audio-only baseline models. Specifically, this research investigates four target variables that represent distinct dimensions of musical characterization:

- **Valence**: A measure of musical positiveness, where high values indicate happy, cheerful, or euphoric qualities, and low values indicate sad, depressed, or angry qualities.
- **Energy**: A perceptual measure of intensity and activity, reflecting dynamic range, perceived loudness, timbre, onset rate, and general entropy.
- **Danceability**: An assessment of how suitable a track is for dancing based on a combination of tempo, rhythm stability, beat strength, and overall regularity.
- **Popularity**: A commercial success metric reflecting streaming counts, playlist inclusions, and listener engagement patterns.

The research question decomposes into several sub-problems: (1) how lyrics should be represented computationally for machine learning applications, (2) whether text features provide information not already captured by audio features, (3) which machine learning algorithms best leverage high-dimensional multimodal feature spaces, and (4) how well trained models generalize to completely unseen artists rather than merely unseen songs from known artists.

### 1.3 Research Objectives

This research pursues the following objectives:

1. **Establish audio-only baselines** by training and evaluating regression models using exclusively audio features. These baselines quantify the predictive power achievable without textual information and serve as reference points for measuring multimodal improvements.

2. **Develop multimodal feature representations** through a preprocessing pipeline that transforms raw lyrics into machine learning-ready features, including text statistics (word count, lexical diversity), sentiment scores (polarity, subjectivity), and 384-dimensional semantic embeddings from the all-MiniLM-L6-v2 sentence transformer model.

3. **Quantify text feature contributions** through controlled ablation experiments that isolate each text feature group, determining which representations provide the greatest predictive value for each target variable.

4. **Optimize algorithm performance** by conducting comprehensive benchmarking across 26 model variants spanning linear models, tree-based ensembles, gradient boosting methods, and neural networks, with both default and tuned hyperparameter configurations.

5. **Produce methodologically rigorous final results** by evaluating top-performing models on a held-out test set accessed exactly once, ensuring unbiased performance estimates suitable for academic reporting.

### 1.4 Scope and Limitations

This study utilizes a large-scale dataset exceeding 730,000 songs collected from the Spotify platform, filtered to include only English-language lyrics to maintain consistency with English-pretrained transformer models. The dataset spans diverse genres, artists, and release periods, providing broad coverage of the contemporary music landscape.

The research scope encompasses feature-based machine learning approaches, systematically evaluating the contribution of different feature modalities rather than end-to-end deep learning architectures that jointly learn representations from raw audio and text. This design choice enables rigorous ablation analysis but limits conclusions about potential gains from representation learning.

Key limitations include: (1) reliance on Spotify's proprietary audio feature extraction, which may not generalize to other platforms or custom audio analysis; (2) restriction to English lyrics, excluding multilingual music that constitutes a significant portion of global consumption; (3) use of pre-trained embeddings without task-specific fine-tuning, potentially underutilizing the capacity of transformer models; and (4) prediction of Spotify-defined attributes, which may not perfectly align with human perceptual judgments.

### 1.5 Thesis Organization

The remainder of this thesis is organized as follows. Section 2 reviews related work in music information retrieval, audio feature analysis, natural language processing applied to music, and multimodal learning approaches. Section 3 details the methodology, including data collection, preprocessing pipelines, feature engineering, model architectures, and evaluation protocols. Section 4 presents experimental results from baseline models, text feature ablations, and comprehensive algorithm benchmarking. Section 5 discusses the interpretation of findings, target-specific insights, and methodological considerations. Section 6 concludes with a summary of contributions, limitations, and directions for future research.

---

## 2. Literature Review

### 2.1 Music Information Retrieval

Music Information Retrieval (MIR) encompasses computational approaches to extracting, analyzing, and organizing information from musical data. The field has evolved significantly with the availability of large-scale digital music collections and streaming platform APIs, enabling researchers to study musical properties at unprecedented scale. Early MIR research focused primarily on content-based retrieval—matching query songs to databases based on acoustic similarity—but has expanded to include semantic analysis, emotion recognition, and attribute prediction tasks.

The emergence of streaming platforms such as Spotify has provided standardized audio feature sets that facilitate reproducible research. Spotify's audio analysis API offers track-level features including danceability, energy, valence, tempo, loudness, and speechiness, computed through proprietary signal processing algorithms. These features have become widely adopted in MIR research due to their availability, consistency, and correlation with human perceptual judgments. However, reliance on platform-specific features limits generalizability and prevents access to underlying acoustic representations.

### 2.2 Audio Feature Extraction and Analysis

Audio feature extraction transforms raw waveforms into numerical representations suitable for machine learning. Low-level features capture signal properties directly: spectral centroid, spectral rolloff, zero-crossing rate, and Mel-frequency cepstral coefficients (MFCCs) characterize timbre and texture. Mid-level features aggregate temporal patterns: tempo estimation, beat tracking, and rhythm histograms capture temporal structure. High-level features abstract semantic properties: key detection, chord recognition, and structural segmentation identify musical elements.

Research on audio-based music attribute prediction has demonstrated that acoustic features provide substantial predictive power for perceptual attributes. Studies predicting music popularity using Spotify's audio features have achieved R² scores ranging from 0.61 to 0.70 using Random Forest and Gradient Boosting models. SpotiPred, a machine learning system for popularity prediction using only audio features, reported 95.37% classification accuracy with Random Forest ensembles on over 170,000 songs. Research on Indonesian streaming data similarly found Random Forest achieving 0.69 accuracy for popularity prediction using audio features alone.

For emotion-related attributes, audio features show variable predictive power. Energy and danceability, being closely tied to acoustic properties (loudness, tempo, rhythm stability), are typically well-predicted by audio features. Valence prediction proves more challenging, as musical positiveness depends on subtle harmonic, melodic, and contextual factors not fully captured by standard audio descriptors. This limitation motivates the incorporation of lyrical content, which may provide complementary emotional signals.

### 2.3 Natural Language Processing in Music

The application of natural language processing (NLP) to song lyrics represents an emerging research direction within MIR. Lyrics provide semantic content—themes, narratives, emotional expressions—that acoustic analysis cannot directly access. Early approaches applied bag-of-words representations and term frequency-inverse document frequency (TF-IDF) weighting to lyrics, enabling text classification and information retrieval applications.

Research predicting mood from lyrics using machine learning demonstrated that textual features alone can achieve meaningful performance. Studies using the Million Song Dataset with lyrics from LyricWikia found that multinomial Naive Bayes classifiers achieved 0.75 ROC-AUC for mood classification tasks. This finding suggests that lyrical content carries emotional information distinct from audio properties.

Sentiment analysis—the computational determination of emotional polarity—provides a bridge between NLP and music emotion research. Lexicon-based approaches such as TextBlob assign sentiment scores based on word-level polarity dictionaries, offering interpretable features without requiring labeled training data. While these rule-based methods lack contextual understanding, they provide computationally efficient baselines for multimodal systems.

The advent of transformer-based language models has enabled richer semantic representations. Pre-trained models such as BERT (Bidirectional Encoder Representations from Transformers) and its distilled variants encode contextual meaning through self-attention mechanisms, producing dense vector embeddings that capture semantic similarity. Sentence transformers extend this approach to sentence-level representations optimized for semantic comparison tasks. The all-MiniLM-L6-v2 model, a distilled variant achieving 95% of full BERT performance with 50% fewer parameters, provides an efficient option for encoding song lyrics as 384-dimensional semantic vectors.

### 2.4 Multimodal Learning Approaches

Multimodal learning combines information from multiple modalities—audio, text, images, metadata—to improve prediction accuracy beyond what any single modality achieves. In music analysis, multimodal approaches integrate acoustic features with lyrical content, artist information, social signals, and visual elements such as album artwork.

Research on music emotion classification with lyrics and audio features has demonstrated the value of multimodal integration. Studies using the MuSe (Musical Sentiment) dataset, containing 90,001 songs annotated for valence, arousal, and dominance, found that Random Forest classifiers combining audio and text features achieved 73% accuracy for emotion classification. This multimodal performance exceeded both audio-only and text-only baselines, confirming feature complementarity.

End-to-end deep learning architectures represent an alternative multimodal paradigm. Rather than engineering separate feature representations for each modality, these systems learn joint representations directly from raw inputs. Research on multimodal deep learning for music popularity prediction combined Mel-spectrograms, chromagrams, lyrics embeddings, and artist metadata through neural network fusion. While achieving strong performance, such approaches sacrifice interpretability and require substantial computational resources.

The present research adopts a feature engineering approach rather than end-to-end learning, enabling systematic ablation analysis that quantifies the contribution of each modality. This design choice prioritizes interpretability and rigorous evaluation over potential gains from representation learning.

### 2.5 Machine Learning for Music Attribute Prediction

Music attribute prediction has been approached with diverse machine learning algorithms, from simple linear models to complex ensemble methods. The choice of algorithm interacts with feature dimensionality, dataset size, and target characteristics.

Linear regression provides interpretable baselines but assumes linear relationships between features and targets—an assumption often violated for perceptual attributes with nonlinear dependencies on acoustic properties. Ridge and Lasso regression add regularization to prevent overfitting in high-dimensional feature spaces, with Lasso providing automatic feature selection through L1 penalties.

Tree-based ensemble methods—Random Forest, Extra Trees, and gradient boosting variants—have emerged as dominant approaches for tabular music data. Random Forest, aggregating predictions from hundreds of decision trees trained on bootstrap samples, provides robust performance with minimal hyperparameter tuning. Studies across multiple music prediction tasks consistently report Random Forest achieving top-tier performance, with accuracy and R² scores ranging from 0.61 to 0.95 depending on task and dataset.

Gradient boosting methods—XGBoost, CatBoost, and LightGBM—iteratively build tree ensembles by fitting residuals from previous iterations. These methods often outperform Random Forest when properly tuned, particularly for large datasets where their sequential learning captures complex feature interactions. XGBoost's regularization options (L1, L2, tree depth limits) help prevent overfitting in high-dimensional multimodal feature spaces.

Neural networks, including multilayer perceptrons (MLPs), offer flexibility in modeling nonlinear relationships but typically underperform tree-based methods on structured tabular data unless dataset sizes are very large or data augmentation is applied. For the 412-dimensional feature space combining audio and text representations, gradient boosting methods represent the expected optimal algorithm family, with neural networks serving as comparison points.

---

## 3. Methodology

### 3.1 Dataset Description

#### 3.1.1 Data Sources and Collection

The dataset utilized in this research originates from a publicly available Spotify dataset containing songs with associated attributes and lyrics. Data collection employed a hybrid API strategy combining multiple sources to ensure comprehensive coverage and data quality.

The Spotify Web API served as the primary source for track metadata including song title, artist name, album information, release year, and track popularity scores. Audio features—danceability, energy, valence, acousticness, instrumentalness, speechiness, liveness, loudness, tempo, duration, key, and mode—were retrieved through the Chosic API, which provides access to audio analysis data following the deprecation of Spotify's direct audio features endpoint. The Genius API supplemented the dataset with song lyrics for tracks where lyrical content was not available in the original dataset.

Genre classification was obtained through a custom scraping pipeline targeting Chosic.com, which maps specific micro-genres (e.g., "vietnamese melodic rap", "deep german minimal tech") to ten standardized macro-genre categories suitable for classification tasks. Language filtering using the langdetect library ensured the final corpus contained only English-language lyrics, maintaining consistency with the English-pretrained transformer models employed for semantic embedding extraction.

#### 3.1.2 Data Validation and Quality Assurance

The data collection pipeline implemented several optimization and fault-tolerance mechanisms critical for processing over 400,000 tracks. A hybrid API strategy employed batching for supported endpoints, reducing API overhead by a factor of 50 through batch requests of 50 items for Spotify track metadata and artist information retrieval.

An artist caching mechanism stored artist metadata locally in JSON format, preventing redundant API lookups for frequently appearing artists. This optimization proved essential given the power-law distribution of artist frequency—popular artists such as major labels' catalogs appear across thousands of tracks, and caching eliminated redundant fetches.

Checkpoint-based processing using a JSONL (JSON Lines) append-only format enabled the pipeline to be interrupted and resumed without data loss. This capability was critical for long-running validation processes spanning multiple hours, protecting against network failures, rate limiting, and system interruptions.

#### 3.1.3 Data Cleaning Operations

Data cleaning operations addressed quality issues identified through exploratory data analysis, applying conservative removal criteria based on physical impossibility rather than statistical outliers.

**Tempo Validation**: Tracks with tempo values of 0 BPM (277 instances) were removed as physically impossible, indicating API errors or missing data rather than valid measurements. The Spotify API specification defines tempo as a positive real number, and zero values violate this specification.

**Loudness Correction**: Loudness values, expressed in decibels where 0 dB represents the maximum possible level, underwent two-stage correction. Values in the range (0, 1] dB were clipped to 0 dB, treating these as measurement noise near the physical maximum. Values exceeding 1 dB (physically impossible under Spotify's normalization scheme) were removed entirely. This process affected approximately 280 tracks.

**Year Validation**: Release years were validated against temporal constraints: years equal to 0, years before 1900 (pre-recording era), and years exceeding 2025 (future dates) were flagged as metadata errors and removed. Three tracks failed this validation.

**Encoding Standardization**: Musical key and mode features exhibited mixed encoding formats—numeric (0-11 for key, 0/1 for mode) and text notation (letter names A-G with accidentals for key, "Major"/"Minor" for mode). Standardization converted all values to numeric pitch class notation (C=0, C♯/D♭=1, ..., B=11) and binary mode encoding (major=1, minor=0), affecting 43,893 tracks with non-standardized formats.

The cleaning process removed fewer than 300 tracks total (0.04% of the dataset), preserving data integrity while eliminating physically impossible values. The final cleaned dataset contained approximately 733,000 tracks.

### 3.2 Data Splitting Strategy

#### 3.2.1 Artist-Aware Grouping Rationale

Standard random splitting assigns songs independently to train, validation, and test sets, creating a critical data leakage risk when artists have multiple songs in the dataset. Songs by the same artist share stylistic patterns including vocal characteristics, production preferences, lyrical themes, and genre consistency. If training data contains songs A, B, and C from Artist X while the test set contains song D from the same artist, the model can exploit artist-specific patterns learned during training, artificially inflating test performance.

This research employs artist-aware splitting using scikit-learn's GroupShuffleSplit algorithm, treating each artist as an atomic unit. All songs by any given artist are assigned to exactly one split (train, validation, or test), ensuring that test set performance reflects true generalization to unseen artists rather than merely unseen songs from known artists. This methodology provides realistic performance estimates aligned with deployment scenarios where recommendation systems encounter new artists not present in training data.

#### 3.2.2 Train-Validation-Test Partition

The dataset was partitioned using a 70/15/15 split ratio: 70% training (approximately 511,000 songs), 15% validation (approximately 110,000 songs), and 15% test (approximately 77,000 songs). This allocation balances training data sufficiency against validation and test set stability.

A two-stage hierarchical splitting procedure maintained artist grouping throughout. The first stage separated all artists into training (70%) versus temporary (30%) sets. The second stage subdivided the temporary set equally into validation (15%) and test (15%) sets. This approach was necessary because GroupShuffleSplit performs binary partitions; achieving three independent sets required composition of two sequential splits.

A fixed random seed (42) ensured reproducibility across experimental runs, guaranteeing that all model variants were trained and evaluated on identical data partitions, eliminating split variance as a confounding variable in performance comparisons.

### 3.3 Feature Engineering Pipeline

#### 3.3.1 Audio Feature Preprocessing

Audio feature preprocessing transformed raw Spotify audio features into machine learning-ready representations through transformations derived from exploratory data analysis findings.

**Power Transformations**: Acousticness, instrumentalness, and speechiness exhibited extreme right-skewness in their distributions. Yeo-Johnson power transformations were applied to normalize these distributions, improving compatibility with gradient-based optimization algorithms and linear model assumptions. Unlike Box-Cox transformations, Yeo-Johnson handles zero values without additional preprocessing.

**Cyclical Encoding**: Musical key (values 0-11 representing pitches C through B) exhibits cyclical structure where B (11) and C (0) are adjacent in the circle of fifths. Sine-cosine encoding preserved this circular topology: key_sin = sin(2π × key/12) and key_cos = cos(2π × key/12). This transformation prevents the model from incorrectly treating key as a linear ordinal variable.

**Standard Scaling**: Continuous features with varying scales—loudness (typically -60 to 0 dB), tempo (typically 60-200 BPM), duration (milliseconds), and year—were standardized to zero mean and unit variance. This scaling prevents features with larger numerical ranges from dominating distance calculations and gradient updates, critical for regularized models and distance-based algorithms.

**One-Hot Encoding**: Genre categories (10 classes) were converted to binary indicator variables, treating genres as nominal categories without ordinal relationships.

**Data Leakage Prevention**: All transformers (StandardScaler, PowerTransformer, OneHotEncoder) were fit exclusively on training data and subsequently applied to validation and test sets, ensuring no information from held-out data influenced preprocessing parameters.

The complete audio feature preprocessing produced 21 output dimensions: 3 power-transformed features, 4 scaled continuous features (loudness, tempo, duration, year), 2 cyclical key components, 1 binary mode indicator, 1 liveness feature (retained without transformation), and 10 one-hot encoded genre indicators.

#### 3.3.2 Text Statistics Extraction

Text statistics captured quantitative properties of lyrical content independent of semantic meaning, providing lightweight features complementary to audio attributes.

Five statistical features were computed from lyrics text:
- **Word count**: Total number of whitespace-delimited tokens
- **Unique word count**: Number of distinct words (case-insensitive)
- **Unique ratio**: Ratio of unique words to total words, measuring lexical diversity (Type-Token Ratio)
- **Average word length**: Mean character count per word, capturing vocabulary complexity
- **Character count**: Total character count including spaces

Count-based features (word count, unique word count, character count) exhibited right-skewed distributions typical of count data. Log1p transformation (log(x + 1)) was applied to compress large values, stabilize variance, and normalize distributions before standard scaling. The log1p formulation naturally handles zero values (empty lyrics) without special-case preprocessing.

Songs with missing or empty lyrics were assigned zero values for all statistics, representing absence of textual content rather than imputed values. StandardScaler was fit on training data post-transformation to prevent data leakage.

#### 3.3.3 Sentiment Analysis

Sentiment features quantified the emotional tone of lyrical content using TextBlob, a lexicon-based natural language processing library that assigns sentiment scores based on word-level polarity dictionaries.

Two sentiment dimensions were extracted:
- **Polarity**: Emotional valence ranging from -1.0 (strongly negative) to +1.0 (strongly positive), computed as a weighted average of word-level sentiment scores with modifier handling (negation, intensifiers)
- **Subjectivity**: Degree of opinion versus factual content ranging from 0.0 (objective) to 1.0 (subjective), distinguishing narrative description from emotional expression

The hypothesis underlying sentiment feature inclusion posits that lyrical sentiment correlates with audio valence—songs with positive lyrics ("love", "happy") may exhibit higher musical valence, while negative lyrics ("heartbreak", "pain") may correlate with lower valence. Sentiment features provide this emotional signal through an independent modality.

Songs with missing or empty lyrics were assigned neutral values (polarity=0.0, subjectivity=0.0), representing absence of sentiment rather than negative sentiment. Exception handling ensured robustness to malformed text or encoding issues. Both features were standardized using training-set-fit scalers.

TextBlob was selected for its computational efficiency, deterministic outputs, interpretability, and elimination of the need for labeled sentiment training data. While deep learning sentiment models offer higher accuracy, TextBlob provides sufficient signal for feature engineering applications without GPU inference overhead.

#### 3.3.4 Semantic Embeddings

Semantic embeddings provided dense vector representations capturing contextual meaning of lyrics through pre-trained transformer-based language models. Unlike statistical features (surface-level properties) or sentiment analysis (emotional dimension only), embeddings encode rich semantic content including themes, narratives, and conceptual relationships.

**Model Selection**: The all-MiniLM-L6-v2 model from the sentence-transformers library was selected based on an optimal accuracy-efficiency tradeoff. This model is a 6-layer distilled variant of BERT, achieving approximately 95% of full BERT performance while reducing parameters by 50% and providing 5x faster inference. The model was pre-trained on over 1 billion sentence pairs using contrastive learning objectives optimized for semantic similarity tasks.

**Embedding Properties**: Each lyrics text was encoded as a 384-dimensional dense vector, L2-normalized by default. The embedding space exhibits semantic similarity properties: lyrics with similar themes, emotions, or topics are mapped to nearby points in the 384-dimensional space, enabling distance-based semantic comparison.

**Batch Processing**: Processing over 700,000 lyrics sequentially would be prohibitively slow. Batch processing with configurable batch sizes (default 64) enabled GPU parallelization and memory-efficient throughput optimization. Progress tracking provided real-time throughput monitoring during the one-time computation.

**Disk Caching**: Computed embeddings were saved as NumPy arrays (.npy format) for instant loading in subsequent runs. The initial computation required 30-60 minutes on CPU or 5-10 minutes on GPU; cached loading completed in seconds. Intelligent timestamp checking detected when input data changed, triggering recomputation only when necessary.

**Missing Data Handling**: Empty or missing lyrics produced zero vectors (all 384 dimensions equal to 0.0), representing semantic absence rather than "average meaning" that mean imputation would produce.

**Data Leakage Considerations**: Unlike scalers that require fitting, embedding models use fixed pre-trained weights with no adaptation to training data, eliminating data leakage risk from this component.

#### 3.3.5 Target Variable Preprocessing

Target variable preprocessing addressed distributional properties that could impair model training or evaluation.

**Popularity**: The popularity score (0-100 scale) exhibited strong right-skewness, with many songs concentrated near zero and a long tail of popular tracks. Log1p transformation (log(popularity + 1)) normalized this distribution, improving regression performance by reducing the influence of extreme values. Models were trained to predict log-transformed popularity; predictions could be inverse-transformed via exp(pred) - 1 for interpretation on the original scale.

**Valence, Energy, Danceability**: These targets, already bounded on [0, 1] scales with approximately symmetric distributions, were retained without transformation. Their natural scale facilitated interpretation (e.g., RMSE of 0.2 represents 20% of the target range).

### 3.4 Feature Set Configurations

Six feature configurations enabled systematic evaluation of modality contributions through controlled ablation:

| Configuration | Components | Dimensions |
|---------------|------------|------------|
| Audio-Only | Power-transformed acoustics, scaled continuous, cyclical key, binary mode, one-hot genre | 21 |
| Audio + Text Statistics | Audio features + word count, unique count, unique ratio, avg word length, char count | 26 |
| Audio + Sentiment | Audio features + polarity, subjectivity | 23 |
| Audio + Combined Text | Audio features + text statistics + sentiment | 28 |
| Audio + Embeddings | Audio features + 384-dim semantic vectors | 405 |
| Full Multimodal | Audio + text statistics + sentiment + embeddings | 412 |

#### 3.4.1 Full Multimodal Features (412 dimensions)

The full multimodal configuration represented the maximum information scenario, combining all available feature modalities. The 412-dimensional feature space comprised:
- Audio features: 5.1% (21 dimensions)
- Simple text features (statistics + sentiment): 1.7% (7 dimensions)
- Semantic embeddings: 93.2% (384 dimensions)

While embeddings dominate the feature space numerically, this does not guarantee proportional predictive importance. Tree-based models naturally handle irrelevant features through split selection, and feature importance analysis would reveal actual contribution distributions.

### 3.5 Machine Learning Models

#### 3.5.1 Baseline Models

Four baseline models established reference performance on audio-only features, representing increasing complexity levels:

**Mean Predictor**: Predicts the training set mean for all samples, representing zero-knowledge baseline. Any model with R² ≤ 0 performs worse than this trivial predictor. Served as a sanity check ensuring correct pipeline implementation.

**Linear Regression**: Ordinary Least Squares (OLS) solution minimizing squared residuals. Tests the fundamental hypothesis of linear relationships between audio features and targets. No hyperparameters to tune, providing pure assessment of linear predictability.

**Ridge Regression**: Linear regression with L2 regularization (α=1.0), shrinking coefficients toward zero to prevent overfitting. Particularly valuable when features exhibit multicollinearity, as may occur among correlated audio attributes.

**XGBoost**: Gradient boosting ensemble with conservative hyperparameters (100 trees, max depth 6, learning rate 0.1). Captures nonlinear relationships and feature interactions that linear models cannot represent, establishing whether audio-target relationships are fundamentally nonlinear.

#### 3.5.2 Enhanced Algorithm Suite

Comprehensive algorithm benchmarking evaluated 26 model variants spanning multiple paradigms:

**Linear Models**: Linear Regression, Ridge Regression (default and tuned), Lasso Regression (default and tuned), SGD Regressor (default and tuned). Tested linear relationships with various regularization strategies.

**Tree-Based Ensembles**: Decision Tree (default and tuned), Random Forest (default and tuned), Extra Trees (default and tuned). Bootstrap aggregation approaches with different randomization strategies.

**Gradient Boosting**: XGBoost (default and tuned), CatBoost (default and tuned), LightGBM (default and tuned), AdaBoost (default and tuned). Sequential ensemble methods with different tree construction algorithms.

**Instance-Based**: K-Neighbors Regressor (default and tuned). Non-parametric distance-based prediction.

**Kernel Methods**: LinearSVR (default and tuned). Support vector regression with linear kernel.

**Neural Networks**: MLP Regressor (default and tuned). Multilayer perceptron with various architectures (single layer default, two-layer tuned with 256-128 units).

#### 3.5.3 Hyperparameter Tuning Philosophy

Hyperparameter tuning followed a principled manual approach based on literature review, domain knowledge, and computational constraints rather than automated search:

**Ensemble Size**: Tuned models used larger ensembles (150-800 trees versus 50-100 default), justified by the large dataset size (511,000 training samples) that supports more model capacity without overfitting.

**Learning Rate**: Gradient boosting models used reduced learning rates (0.01-0.06 versus 0.1-0.3 default), enabling more iterations before convergence and finer-grained optimization.

**Regularization**: All tuned models increased regularization strength (higher α for linear models, depth limits and minimum sample constraints for trees), addressing overfitting risk in the 412-dimensional feature space.

**Early Stopping**: Gradient boosting models (XGBoost, CatBoost, LightGBM) employed early stopping with 50-round patience, monitoring validation performance to halt training when improvement plateaued. This prevented overfitting while reducing unnecessary computation.

### 3.6 Evaluation Metrics

#### 3.6.1 Primary Metrics

Three primary metrics characterized model performance:

**Root Mean Squared Error (RMSE)**: √(Σ(y_pred - y_true)² / n). Penalizes large errors heavily due to squaring, appropriate when large deviations are particularly undesirable. Units match target scale, enabling direct interpretation (e.g., RMSE=0.2 for valence means average error of 0.2 on the [0,1] scale).

**Mean Absolute Error (MAE)**: Σ|y_pred - y_true| / n. Linear penalty for errors, more robust to outliers than RMSE. Interpretable as average absolute deviation from ground truth.

**Coefficient of Determination (R²)**: 1 - SS_res/SS_tot, where SS_res = Σ(y_true - y_pred)² and SS_tot = Σ(y_true - ȳ)². Represents proportion of target variance explained by the model. Scale-invariant, enabling comparison across targets with different ranges. R²=0 indicates mean predictor performance; R²=1 indicates perfect prediction; R²<0 indicates worse-than-mean performance.

#### 3.6.2 Extended Metrics

Extended metrics provided deeper characterization of model behavior:

**Explained Variance**: 1 - Var(y - ŷ) / Var(y). Variant of R² without mean correction, useful for detecting bias in predictions.

**Maximum Error**: max(|y_i - ŷ_i|). Worst-case prediction error, critical for applications where catastrophic failures must be avoided.

**Mean Absolute Percentage Error (MAPE)**: (100/n) × Σ(|y_i - ŷ_i| / |y_i|). Scale-invariant percentage interpretation, though undefined for zero-valued targets.

**Prediction Range Analysis**: Statistics of predicted values (mean, standard deviation, minimum, maximum) detected range collapse—a pathology where models predict narrow ranges near the mean despite true target variation. High R² can mask this issue if not explicitly checked.

**Residual Statistics**: Mean and standard deviation of residuals (y_true - y_pred) identified systematic bias (non-zero mean) and heteroscedasticity (varying residual magnitude).

**Training Time**: Wall-clock seconds for model fitting, essential for computational efficiency analysis and deployment planning.

#### 3.6.3 Train-Validation-Test Consistency Analysis

Performance comparison across data splits detected overfitting:

**Healthy Generalization**: R²_train ≥ R²_val ≥ R²_test with small gaps indicates stable generalization.

**Overfitting Signature**: R²_train >> R²_val ≈ R²_test (large train-validation gap) indicates model memorization.

**Underfitting Signature**: R²_train ≈ R²_val ≈ R²_test (all poor) indicates insufficient model capacity.

Acceptable validation-test gap was defined as ±0.02 R², within noise tolerance for the dataset size. Larger discrepancies would indicate either model selection bias or distribution shift between validation and test periods.

---

## 4. Experimental Results

### 4.1 Baseline Model Performance

#### 4.1.1 Audio-Only Results by Target Variable

Baseline experiments using audio-only features (21 dimensions) established reference performance levels for each target variable. The Mean Predictor served as a sanity check, achieving R² ≈ 0.0 across all targets as expected, confirming correct pipeline implementation.

**Energy Prediction**: Audio features demonstrated strongest predictive power for energy, with linear models achieving R² = 0.789 (Ridge Regression) and XGBoost reaching R² = 0.838 on validation data. This strong baseline reflects energy's direct relationship with acoustic properties—loudness, tempo, and spectral characteristics that audio features explicitly capture.

**Danceability Prediction**: Danceability exhibited moderate audio-only predictability, with Ridge Regression achieving R² = 0.416 and XGBoost reaching R² = 0.579. The gap between linear and tree-based models suggests nonlinear relationships between audio features and danceability that require ensemble methods to capture.

**Valence Prediction**: Valence proved more challenging, with Ridge Regression achieving R² = 0.296 and XGBoost reaching R² = 0.408. This weaker baseline supports the hypothesis that musical positiveness depends on factors beyond acoustic properties alone, motivating the investigation of lyrical content.

**Popularity Prediction**: Popularity showed weakest audio-only predictability, with Ridge Regression achieving R² = 0.049 and XGBoost reaching only R² = 0.062. This result confirms that commercial success depends primarily on factors external to musical content—marketing, artist recognition, playlist placement, and social dynamics—rather than audio characteristics.

#### 4.1.2 Algorithm Comparison on Audio Features

Across all targets, gradient boosting methods (XGBoost) consistently outperformed linear models, demonstrating that audio-target relationships are fundamentally nonlinear. The performance gap between linear regression (R² = 0.296 for valence) and XGBoost (R² = 0.408 for valence) indicates that tree-based models capture interaction effects and nonlinear patterns that linear assumptions cannot represent.

### 4.2 Text Feature Contribution Analysis

#### 4.2.1 Text Statistics Impact

Text statistics features (word count, unique word count, unique ratio, average word length, character count) were concatenated with audio features, creating a 26-dimensional feature space. Results indicated marginal improvements over audio-only baselines for most targets, with gains of 0.5-2% R² depending on target and algorithm. Text statistics capture structural properties of lyrics (verbosity, lexical diversity) that provide weak but measurable predictive signals.

#### 4.2.2 Sentiment Features Impact

Sentiment features (polarity, subjectivity) added to audio features created a 23-dimensional feature space. Sentiment showed modest correlation with valence prediction, consistent with the hypothesis that lyrical emotional tone complements musical positiveness. However, sentiment features provided minimal improvement for energy, danceability, and popularity—targets less directly related to emotional content.

#### 4.2.3 Combined Simple Text Features

Combining text statistics and sentiment with audio features (28 dimensions) yielded cumulative improvements over audio-only baselines, though gains remained modest (1-3% R² improvement). The complementarity between structural (text statistics) and emotional (sentiment) text representations was confirmed, as combined performance exceeded either feature set alone.

#### 4.2.4 Semantic Embeddings Impact

Semantic embeddings (384 dimensions from all-MiniLM-L6-v2) combined with audio features (405 total dimensions) provided the largest text-based improvements. For valence prediction, embeddings contributed approximately 5-6% R² improvement over audio-only baselines, suggesting that contextual semantic meaning captures emotional content beyond what rule-based sentiment analysis achieves. Energy and danceability showed smaller embedding contributions (1-3% R²), while popularity remained weakly predicted regardless of feature configuration.

### 4.3 Full Multimodal Model Performance

#### 4.3.1 412-Dimensional Feature Space Results

The full multimodal configuration (audio + text statistics + sentiment + embeddings = 412 dimensions) was evaluated across all 26 model variants. Key validation set results:

| Target | Best Model | R² | RMSE | MAE |
|--------|-----------|-----|------|-----|
| Energy | XGBoost_tuned | 0.849 | 0.095 | 0.072 |
| Danceability | XGBoost_tuned | 0.609 | 0.107 | 0.084 |
| Valence | XGBoost_tuned | 0.466 | 0.182 | 0.145 |
| Popularity | CatBoost | 0.078 | 1.430 | 1.250 |

Gradient boosting methods dominated across all targets, with tuned XGBoost and CatBoost configurations achieving best results. The high-dimensional feature space favored tree-based ensembles that naturally handle irrelevant features through split selection.

#### 4.3.2 Diminishing Returns Analysis

Comparing feature configurations revealed diminishing returns for text features:

| Configuration | Valence R² | Energy R² | Danceability R² | Popularity R² |
|---------------|-----------|----------|----------------|--------------|
| Audio Only (21 dims) | 0.408 | 0.838 | 0.579 | 0.062 |
| Audio + Simple Text (28 dims) | 0.420 | 0.842 | 0.585 | 0.065 |
| Audio + Embeddings (405 dims) | 0.456 | 0.846 | 0.602 | 0.072 |
| Full Features (412 dims) | 0.466 | 0.849 | 0.609 | 0.078 |

The incremental contribution of each feature modality decreased as more features were added, consistent with information overlap between representations. Embeddings provided the largest marginal gains, particularly for valence where semantic content proves most relevant.

#### 4.3.3 Cost-Benefit Evaluation of Embeddings

Embeddings (384 dimensions) contributed approximately:
- Valence: +5.8% R² improvement over audio-only
- Energy: +1.1% R² improvement over audio-only
- Danceability: +3.0% R² improvement over audio-only
- Popularity: +1.6% R² improvement over audio-only

Whether these gains justify the computational cost (30-60 minute embedding generation, 14x dimensionality increase) depends on application requirements. For valence prediction where text content is semantically relevant, embeddings provide meaningful improvement. For energy prediction where audio features dominate, simpler feature sets may suffice.

### 4.4 Comprehensive Algorithm Benchmarking

#### 4.4.1 Model Performance Comparison (26 Variants)

The complete algorithm comparison on the 412-dimensional feature space revealed clear performance hierarchies:

**Top Performers** (consistent across targets):
- XGBoost_tuned: Best overall, achieving highest R² for valence, energy, and danceability
- CatBoost_tuned: Close second, best for popularity
- LightGBM_tuned: Strong third, efficient training time
- ExtraTrees: Competitive without tuning

**Middle Tier**:
- Random Forest: Solid baseline, degraded with aggressive tuning
- MLPRegressor: Competitive for valence/energy, limited by tabular data
- CatBoost/LightGBM default: Good out-of-box performance

**Underperformers**:
- Linear models (Ridge, Lasso): Insufficient capacity for nonlinear patterns
- KNeighbors: Poor scaling with high dimensionality
- Decision Tree: High variance without ensemble averaging
- AdaBoost: Dominated by XGBoost/CatBoost

#### 4.4.2 Default vs. Tuned Configuration Analysis

Hyperparameter tuning impact varied by algorithm:

| Algorithm | Default R² (Valence) | Tuned R² (Valence) | Improvement |
|-----------|---------------------|-------------------|-------------|
| XGBoost | 0.408 | 0.466 | +5.8% |
| CatBoost | 0.441 | 0.444 | +0.3% |
| LightGBM | 0.405 | 0.443 | +3.8% |
| Random Forest | 0.416 | 0.289 | -12.7% |
| ExtraTrees | 0.419 | 0.369 | -5.0% |
| MLP | 0.408 | 0.409 | +0.1% |

XGBoost showed largest tuning gains, benefiting from increased tree count (800 vs. 100), reduced learning rate (0.05 vs. 0.3), and deeper trees (max_depth=10 vs. 6). Random Forest and ExtraTrees degraded with tuning, suggesting default configurations were already near-optimal or tuned parameters induced overfitting.

#### 4.4.3 Training Time and Computational Efficiency

Training time varied significantly across algorithms (approximate values for 412 features, 511k samples):

| Algorithm | Training Time | Notes |
|-----------|--------------|-------|
| Linear models | < 10 seconds | Closed-form solutions |
| Decision Tree | < 30 seconds | Single tree |
| Random Forest | 2-5 minutes | Parallelizable |
| ExtraTrees | 2-5 minutes | Parallelizable |
| XGBoost_tuned | 10-20 minutes | Early stopping at ~400 trees |
| CatBoost_tuned | 15-30 minutes | Symmetric tree construction |
| LightGBM_tuned | 5-10 minutes | Histogram-based, fastest boosting |
| MLP_tuned | 5-15 minutes | GPU-accelerated |
| KNeighbors | 1-2 minutes train, slow predict | Distance computation overhead |

LightGBM offered the best accuracy-speed tradeoff among gradient boosting methods, achieving comparable R² to XGBoost with 2-3x faster training.

### 4.5 Final Test Set Evaluation

#### 4.5.1 Selected Models for Final Evaluation

Following validation-based model selection, 12 models were evaluated on the held-out test set (accessed exactly once): XGBoost (default/tuned), CatBoost (default/tuned), LightGBM (default/tuned), Random Forest (default/tuned), ExtraTrees (default/tuned), and MLP (default/tuned).

#### 4.5.2 Test Set Performance Metrics

Final test set results (n = 82,274 samples):

| Target | Best Model | Test R² | Test RMSE | Test MAE |
|--------|-----------|---------|-----------|----------|
| Energy | XGBoost_tuned | 0.847 | 0.095 | 0.071 |
| Danceability | XGBoost_tuned | 0.618 | 0.106 | 0.084 |
| Valence | XGBoost_tuned | 0.474 | 0.181 | 0.144 |
| Popularity | CatBoost | 0.070 | 1.414 | 1.231 |

XGBoost_tuned achieved best test performance for energy, danceability, and valence. CatBoost achieved marginally better popularity prediction, though all models performed poorly on this target (R² < 0.10).

#### 4.5.3 Validation-Test Consistency

Performance consistency between validation and test sets indicated healthy generalization:

| Target | Best Model | Val R² | Test R² | Difference |
|--------|-----------|--------|---------|------------|
| Energy | XGBoost_tuned | 0.849 | 0.847 | -0.002 |
| Danceability | XGBoost_tuned | 0.609 | 0.618 | +0.009 |
| Valence | XGBoost_tuned | 0.466 | 0.474 | +0.008 |
| Popularity | CatBoost | 0.078 | 0.070 | -0.008 |

All validation-test gaps fell within the acceptable ±0.02 R² tolerance, confirming that model selection on validation data did not overfit to validation-specific patterns. The slight improvement for danceability and valence on test data suggests conservative validation estimates rather than overfitting.

#### 4.5.4 Best Performing Models per Target

**Energy** (XGBoost_tuned, R² = 0.847): Explained 84.7% of variance in energy ratings, with RMSE = 0.095 on the [0,1] scale. Prediction range [0.008, 1.008] covered the full target range without collapse. Audio features dominated prediction, with text features contributing marginal improvements.

**Danceability** (XGBoost_tuned, R² = 0.618): Explained 61.8% of variance, with RMSE = 0.106. Moderate prediction quality reflects danceability's partial dependence on rhythmic audio features and partial dependence on subjective/contextual factors not captured by any feature modality.

**Valence** (XGBoost_tuned, R² = 0.474): Explained 47.4% of variance, representing the strongest case for multimodal features. The 6.6% improvement over audio-only baselines demonstrates that semantic embeddings capture emotional content relevant to musical positiveness. Prediction range [-0.059, 1.010] indicated full coverage without collapse.

**Popularity** (CatBoost, R² = 0.070): Explained only 7.0% of variance, confirming that popularity is fundamentally unpredictable from musical content alone. Commercial success depends on marketing, timing, artist recognition, and social dynamics that audio and text features cannot capture.

---

## 5. Conclusion

### 5.1 Summary of Contributions

This research investigated multimodal music attribute prediction by combining audio features with textual analysis of song lyrics. The study made the following contributions:

1. **Large-Scale Empirical Analysis**: Conducted experiments on over 730,000 songs with 412-dimensional multimodal feature vectors, providing statistically robust findings on a dataset substantially larger than most prior studies in the field.

2. **Systematic Feature Ablation**: Quantified the incremental contribution of text statistics, sentiment analysis, and semantic embeddings through controlled experiments, demonstrating that embeddings provide the largest marginal gains (5.8% R² improvement for valence).

3. **Comprehensive Algorithm Benchmarking**: Evaluated 26 model variants across four target variables, establishing that tuned gradient boosting methods (XGBoost, CatBoost, LightGBM) consistently outperform linear models and neural networks on multimodal music data.

4. **Methodologically Rigorous Evaluation**: Implemented artist-aware data splitting to prevent data leakage and conducted one-time test set evaluation to ensure unbiased performance reporting.

5. **Reproducible Pipeline**: Developed a complete preprocessing, training, and evaluation pipeline with checkpoint-based fault tolerance, enabling replication and extension of results.

### 5.2 Key Findings

The experimental results yielded several important findings:

**Audio Features Provide Strong Baselines**: Audio-only models achieved R² = 0.838 for energy and R² = 0.579 for danceability, demonstrating that acoustic properties capture substantial variance in perceptual attributes. These strong baselines set a high bar for multimodal improvements.

**Text Features Improve Valence Prediction**: Semantic embeddings improved valence prediction from R² = 0.408 (audio-only) to R² = 0.474 (full multimodal), a 6.6 percentage point gain. This finding supports the hypothesis that lyrical content carries emotional information complementary to audio features.

**Popularity Remains Unpredictable**: All feature configurations and algorithms achieved R² < 0.10 for popularity prediction, confirming that commercial success depends primarily on factors external to musical content—marketing, artist recognition, social dynamics, and timing.

**Gradient Boosting Dominates**: XGBoost_tuned achieved best performance for three of four targets (energy, danceability, valence), with CatBoost marginally better for popularity. Tree-based ensembles effectively handle high-dimensional multimodal features through automatic feature selection.

**Diminishing Returns for Text Features**: While embeddings provided meaningful gains for valence, their contribution to energy and danceability was modest (1-3% R²). The 14x dimensionality increase from embeddings may not justify computational costs for all applications.

### 5.3 Limitations

This research has several limitations that constrain generalizability:

**Platform Dependency**: Reliance on Spotify's proprietary audio features limits reproducibility on other platforms and prevents access to underlying acoustic representations. Results may not transfer to systems using custom audio analysis.

**Language Restriction**: Filtering to English-only lyrics excluded multilingual music constituting a significant portion of global consumption. Findings may not generalize to non-English musical traditions with different lyrical conventions.

**Pre-trained Embeddings**: Using fixed pre-trained embeddings without task-specific fine-tuning potentially underutilizes transformer model capacity. End-to-end training could yield superior representations.

**Target Definition**: Prediction targets are defined by Spotify's algorithms and may not perfectly align with human perceptual judgments. Valence and danceability are inherently subjective constructs.

**Feature Engineering Approach**: The study focused on feature-based methods rather than end-to-end deep learning, limiting conclusions about potential gains from joint audio-text representation learning.

### 5.4 Future Work

Several directions merit future investigation:

**End-to-End Multimodal Learning**: Training joint audio-text encoders that learn representations directly from raw inputs could improve upon feature engineering approaches, particularly for valence where semantic understanding proves valuable.

**Cross-Lingual Extension**: Developing multilingual embedding models or language-specific pipelines would extend applicability to global music catalogs.

**Fine-Tuned Embeddings**: Task-specific fine-tuning of transformer models on music-domain text could improve lyric representations beyond generic sentence embeddings.

**Temporal and Social Features**: Incorporating release timing, playlist context, and social engagement signals could improve popularity prediction beyond musical content analysis.

**Interpretability Analysis**: Feature importance and attention analysis could reveal which audio and text features drive predictions for different targets, providing insights for music production and curation.

### 5.5 Concluding Remarks

This thesis demonstrates that multimodal approaches combining audio features with textual analysis can improve music attribute prediction, with the magnitude of improvement depending on the target variable and its relationship to lyrical content. Valence prediction benefits most substantially from semantic embeddings, supporting the intuition that emotional content expressed in lyrics complements acoustic signals of musical positiveness. Energy and danceability remain predominantly audio-driven, while popularity eludes prediction from musical content entirely.

The findings have practical implications for music information retrieval systems: multimodal pipelines are warranted when predicting emotion-related attributes, while simpler audio-only models may suffice for energy and rhythm-related predictions. For popularity estimation, alternative approaches incorporating social and contextual signals are necessary.

Gradient boosting methods, particularly XGBoost with careful hyperparameter tuning, emerge as the recommended algorithm family for multimodal music prediction tasks, offering superior accuracy without the data scale requirements of deep learning approaches. The methodological framework—artist-aware splitting, systematic ablation, and one-time test evaluation—provides a template for rigorous music machine learning research.

---

## References

---

## Appendices

### Appendix A: Feature Transformation Details

### Appendix B: Complete Model Performance Tables

### Appendix C: Hyperparameter Configurations

