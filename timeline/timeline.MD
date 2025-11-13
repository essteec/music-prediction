### Timeline of what we did

# Data Collection

- scraped for popularity, genre and year. in dataset/scripts

    - songs_enhanced_full.csv: all successful scrapes
    - failed_tracks.csv: all failed scrapes
    - unknown_tracks.csv: successful scrapes with undetected genres

# Data Validation

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

