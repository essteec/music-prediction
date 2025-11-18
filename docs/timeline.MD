### Timeline of what we did

# Data Collection

- scraped for popularity, genre and year. in dataset/scripts

    - songs_enhanced_full.csv: all successful scrapes
    - failed_tracks.csv: all failed scrapes
    - unknown_tracks.csv: successful scrapes with undetected genres

# Data Validation and Cleaning

- comprehensive validation script written in dataset/scripts/comprehensive_validation.py

    - before: outputs detailed report in reports/comprehensive_validation_report.txt 
    - after: outputs detailed report in reports/comprehensive_validation_report2.txt

- script to analyze outliers in numeric columns investigate_outliers.py 

    - Loudness: All 406 are slightly above 0 dB (range: 0.002 to 4.88 dB) 
    - Tempo: 277 rows with tempo=0 (invalid)
    - year: 3 invalid entries (negative or future years)

- Accepted and decided on cleaning actions for outliers clean_data.py
    - Remove songs with tempo = 0 BPM (277 rows)
    - Clip loudness 0-1 dB to 0 dB
    - Remove songs with loudness > 1 dB
    - Remove songs with invalid year (3 rows)

# Baseline Models

- data_splitting.py: CREATING SPLITS 
    - 70% train, 15% val, 15% test
    - Artists - Unique: 108,096 - 732,988 songs:
        - Train artists: 75,667 - 514,123 songs
        - Val artists:   16,214 - 109,343 Songs
        - Test artists:  16,215 - 109,522 Songs
    - outputs: train.csv, val.csv, test.csv

- audio_features.py: EXTRACTING AUDIO FEATURESUsing 9 audio features:
    - used features:
        1. acousticness
        2. instrumentalness
        3. liveness
        4. speechiness
        5. loudness
        6. tempo
        7. duration_ms
        8. key
        9. mode
    - ⚠️ Missing values found in training set:
    key     31423
    mode    31423
    dtype: int64
    - Scaling statistics:
    Feature              Mean         Std         
    acousticness         0.2674       0.3094      
    instrumentalness     0.0915       0.2240      
    liveness             0.2254       0.1997      
    speechiness          0.0844       0.0921      
    loudness             -7.7784      3.8302      
    tempo                122.7979     29.4877     
    duration_ms          236323.3635  93716.7066  
    key                  5.2832       3.5538      
    mode                 0.6628       0.4727   
    - saved
    ✅ Scaled features saved to: ../features/X_*_audio.npy
    ✅ Targets saved to: ../features/y_*_*.npy
    ✅ Scaler saved to: ../features/audio_scaler.pkl
    ✅ Feature names saved to: ../features/audio_feature_names.txt

- baseline_model.py: TRAINING BASELINE MODELS
    - Models trained:
        - Mean
        - Linear Regression
        - Ridge
        - XGBoost
    - results saved to results/metrics/baseline_results_20251114_021007.csv

- we repeated above after realizing inconsistencies in key and mode columns

