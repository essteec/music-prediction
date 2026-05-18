# Script execution order

## Dataset Preparation
`ml/preprocessing/`

- python data_splitting.py
- python run_preprocessing.py

## Machine learning model training
`ml/models/`

*Main sequence:*
- python baseline_models.py  # Audio only (23 features)
- python full_features_models.py  # All features (414 features)
- python enhanced_models.py  # Comprehensive 14+ algorithm comparison (414 features)
- python feature_selection_rfe.py  # Recursive Feature Elimination
- python retrain_rfe_best_iterations.py  # Retrain using optimal RFE features
- python test_evaluation_final.py  # One-time final test set evaluation

*Alternatives/Experiments:*
- python text_stats_models.py  # Audio + Text Statistics
- python sentiment_models.py  # Audio + Sentiment
- python embedding_models.py  # Audio + 384-dim Embeddings
- python combined_text_models.py  # Audio + Text Stats + Sentiment
- python compare_text_approaches.py  # Compare above text approaches

## Deep learning experiments
`dl/`

- python 01_xor_network.py
- python 02_train_mlp.py
- python 03_extract_better_embeddings.py
- python 04_train_mlp_with_mpnet.py
- python 05_train_mlp_with_audio.py

## Web application
`app/`

- python gradio_app.py
