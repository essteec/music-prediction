"""
Comprehensive Data Validation Script
Phase 1: Complete column-by-column analysis

Validates ALL 20 columns according to specifications:
- id, name, album_name, artists (identity columns)
- danceability, energy, key, loudness, mode, speechiness, acousticness,
  instrumentalness, liveness, valence, tempo, duration_ms (audio features)
- lyrics (text feature)
- year, genre, popularity (metadata, including targets)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Column specifications
COLUMN_SPECS = {
    # Identity columns (no ML impact but must be valid)
    'id': {
        'type': 'string',
        'unique': True,
        'null_allowed': False,
        'ml_impact': None,
        'target': False
    },
    'name': {
        'type': 'string',
        'null_allowed': False,
        'ml_impact': None,
        'target': False
    },
    'album_name': {
        'type': 'string',
        'null_allowed': True,
        'ml_impact': None,
        'target': False
    },
    'artists': {
        'type': 'string',
        'null_allowed': False,
        'ml_impact': None,
        'target': False
    },
    
    # Audio features - Normalized [0-1]
    'danceability': {
        'type': 'float',
        'range': (0.0, 1.0),
        'null_allowed': False,
        'ml_impact': 'HIGH',
        'target': True  # TARGET 🎯
    },
    'energy': {
        'type': 'float',
        'range': (0.0, 1.0),
        'null_allowed': False,
        'ml_impact': 'HIGH',
        'target': True  # TARGET 🎯
    },
    'valence': {
        'type': 'float',
        'range': (0.0, 1.0),
        'null_allowed': False,  # CRITICAL
        'ml_impact': 'HIGH',
        'target': True  # TARGET 🎯
    },
    'speechiness': {
        'type': 'float',
        'range': (0.0, 1.0),
        'null_allowed': False,
        'ml_impact': 'MEDIUM',
        'target': False
    },
    'acousticness': {
        'type': 'float',
        'range': (0.0, 1.0),
        'null_allowed': False,
        'ml_impact': 'MEDIUM',
        'target': False
    },
    'instrumentalness': {
        'type': 'float',
        'range': (0.0, 1.0),
        'null_allowed': False,
        'ml_impact': 'MEDIUM',
        'target': False
    },
    'liveness': {
        'type': 'float',
        'range': (0.0, 1.0),
        'null_allowed': False,
        'ml_impact': 'LOW',
        'target': False
    },
    
    # Audio features - Specific ranges
    'key': {
        'type': 'int',
        'range': (-1, 11),  # -1 = no key detected, 0-11 = pitch classes
        'null_allowed': False,
        'ml_impact': 'LOW',
        'target': False
    },
    'mode': {
        'type': 'int',
        'range': (0, 1),
        'null_allowed': False,
        'ml_impact': 'LOW',
        'target': False
    },
    'loudness': {
        'type': 'float',
        'range': (-60.0, 0.0),
        'null_allowed': False,
        'ml_impact': 'MEDIUM',
        'target': False
    },
    'tempo': {
        'type': 'float',
        'range': (20.0, 300.0),
        'null_allowed': False,
        'ml_impact': 'MEDIUM',
        'target': False
    },
    'duration_ms': {
        'type': 'int',
        'range': (1000, 3600000),
        'null_allowed': False,
        'ml_impact': 'LOW',
        'target': False
    },
    
    # Text feature
    'lyrics': {
        'type': 'string',
        'null_allowed': True,  # Instrumentals OK
        'ml_impact': 'HIGH',
        'target': False
    },
    
    # Metadata
    'year': {
        'type': 'int',
        'range': (1900, 2025),
        'null_allowed': False,
        'ml_impact': 'MEDIUM',
        'target': False,
        'known_issues': ['year=0']
    },
    'genre': {
        'type': 'string',
        'null_allowed': False,
        'ml_impact': 'HIGH',
        'target': False,
        'known_issues': ['NaN values']
    },
    'popularity': {
        'type': 'int',
        'range': (0, 100),
        'null_allowed': True,  # Old songs may lack
        'ml_impact': 'HIGH',
        'target': True  # TARGET 🎯
    }
}


def analyze_dataset(filepath, chunksize=50000):
    """
    Comprehensive analysis of dataset with all column checks
    """
    
    stats = {
        'total_rows': 0,
        'unique_ids': set(),
        'duplicate_ids': [],
        'missing_values': {},
        'range_violations': {},
        'genre_issues': {'nan': 0, 'empty': 0, 'invalid': 0},
        'year_issues': {'zero': 0, 'negative': 0, 'future': 0, 'too_old': 0},
        'encoding_issues': {
            'key_letter_format': 0,  # C, D, E, F, G, A, B + sharps
            'mode_text_format': 0     # Major, Minor
        },
        'target_completeness': {
            'valence': {'complete': 0, 'missing': 0},
            'energy': {'complete': 0, 'missing': 0},
            'danceability': {'complete': 0, 'missing': 0},
            'popularity': {'complete': 0, 'missing': 0}
        }
    }
    
    # Initialize range violation counters
    for col, spec in COLUMN_SPECS.items():
        if 'range' in spec:
            stats['range_violations'][col] = 0
    
    print(f"Analyzing: {filepath}")
    print(f"Estimating rows (this may take a moment)...")
    
    # Get accurate row count using pandas (handles embedded newlines in CSV)
    total_rows = 0
    for chunk in pd.read_csv(filepath, chunksize=chunksize, low_memory=False):
        total_rows += len(chunk)
    
    print(f"Total rows: {total_rows:,}\n")
    
    print(f"Processing in chunks of {chunksize:,}...")
    with tqdm(total=total_rows, desc="Analyzing") as pbar:
        for chunk in pd.read_csv(filepath, chunksize=chunksize, low_memory=False):
            stats['total_rows'] += len(chunk)
            
            # ID uniqueness and duplicates
            if 'id' in chunk.columns:
                chunk_ids = chunk['id'].tolist()
                duplicates = [x for x in chunk_ids if x in stats['unique_ids']]
                stats['duplicate_ids'].extend(duplicates)
                stats['unique_ids'].update(chunk_ids)
            
            # Missing values for ALL columns
            for col in chunk.columns:
                if col not in stats['missing_values']:
                    stats['missing_values'][col] = 0
                stats['missing_values'][col] += chunk[col].isna().sum()
            
            # Range violations - [0, 1] normalized features
            for col in ['danceability', 'energy', 'valence', 'speechiness',
                       'acousticness', 'instrumentalness', 'liveness']:
                if col in chunk.columns:
                    # Convert to numeric, coercing errors to NaN
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                    stats['range_violations'][col] += (
                        (chunk[col] < 0) | (chunk[col] > 1)
                    ).sum()
            
            # Check for encoding issues BEFORE converting to numeric
            # Key: Check for letter-based notation (C, D, E, F, G, A, B + sharps)
            if 'key' in chunk.columns:
                # Detect letter-based keys
                letter_keys = chunk['key'].astype(str).str.contains('[A-G]', regex=True, na=False)
                stats['encoding_issues']['key_letter_format'] += letter_keys.sum()
                
                # Convert to numeric, coercing errors to NaN
                chunk['key'] = pd.to_numeric(chunk['key'], errors='coerce')
                stats['range_violations']['key'] += (
                    (chunk['key'] < -1) | (chunk['key'] > 11)
                ).sum()
            
            # Mode: Check for text notation (Major, Minor)
            if 'mode' in chunk.columns:
                # Detect text-based modes
                text_modes = chunk['mode'].astype(str).str.contains('Major|Minor', regex=True, case=False, na=False)
                stats['encoding_issues']['mode_text_format'] += text_modes.sum()
                
                # Convert to numeric, coercing errors to NaN
                chunk['mode'] = pd.to_numeric(chunk['mode'], errors='coerce')
                stats['range_violations']['mode'] += (
                    (chunk['mode'] < 0) | (chunk['mode'] > 1)
                ).sum()
            
            # Range violations - loudness (-60 to 0)
            if 'loudness' in chunk.columns:
                # Convert to numeric, coercing errors to NaN
                chunk['loudness'] = pd.to_numeric(chunk['loudness'], errors='coerce')
                stats['range_violations']['loudness'] += (
                    (chunk['loudness'] < -60) | (chunk['loudness'] > 0)
                ).sum()
            
            # Range violations - tempo (20-300)
            if 'tempo' in chunk.columns:
                # Convert to numeric, coercing errors to NaN
                chunk['tempo'] = pd.to_numeric(chunk['tempo'], errors='coerce')
                stats['range_violations']['tempo'] += (
                    (chunk['tempo'] < 20) | (chunk['tempo'] > 300)
                ).sum()
            
            # Range violations - duration (>0)
            if 'duration_ms' in chunk.columns:
                # Convert to numeric, coercing errors to NaN
                chunk['duration_ms'] = pd.to_numeric(chunk['duration_ms'], errors='coerce')
                stats['range_violations']['duration_ms'] += (
                    chunk['duration_ms'] <= 0
                ).sum()
            
            # Range violations - popularity (0-100)
            if 'popularity' in chunk.columns:
                # Convert to numeric, coercing errors to NaN
                chunk['popularity'] = pd.to_numeric(chunk['popularity'], errors='coerce')
                valid_pop = chunk['popularity'].dropna()
                if len(valid_pop) > 0:
                    stats['range_violations']['popularity'] += (
                        (valid_pop < 0) | (valid_pop > 100)
                    ).sum()
            
            # Genre issues
            if 'genre' in chunk.columns:
                stats['genre_issues']['nan'] += chunk['genre'].isna().sum()
                stats['genre_issues']['empty'] += (chunk['genre'] == '').sum()
                stats['genre_issues']['invalid'] += chunk['genre'].isin(
                    ['Unknown', 'nan', 'NaN', 'None', 'null']
                ).sum()
            
            # Year issues
            if 'year' in chunk.columns:
                # Convert to numeric, coercing errors to NaN
                chunk['year'] = pd.to_numeric(chunk['year'], errors='coerce')
                stats['year_issues']['zero'] += (chunk['year'] == 0).sum()
                stats['year_issues']['negative'] += (chunk['year'] < 0).sum()
                stats['year_issues']['future'] += (chunk['year'] > 2025).sum()
                stats['year_issues']['too_old'] += (
                    (chunk['year'] < 1900) & (chunk['year'] != 0)
                ).sum()
            
            # Target completeness (CRITICAL)
            for target in ['valence', 'energy', 'danceability', 'popularity']:
                if target in chunk.columns:
                    stats['target_completeness'][target]['complete'] += \
                        chunk[target].notna().sum()
                    stats['target_completeness'][target]['missing'] += \
                        chunk[target].isna().sum()
            
            pbar.update(len(chunk))
    
    stats['unique_tracks'] = len(stats['unique_ids'])
    stats['total_duplicates'] = len(stats['duplicate_ids'])
    del stats['unique_ids']  # Save memory
    
    return stats


def generate_report(stats, output_path):
    """
    Generate comprehensive validation report
    """
    with open(output_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("COMPREHENSIVE DATA VALIDATION REPORT\n")
        f.write("=" * 100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Rows Analyzed: {stats['total_rows']:,}\n")
        f.write(f"Unique Tracks: {stats['unique_tracks']:,}\n")
        f.write(f"Duplicate Rows: {stats['total_duplicates']:,}\n")
        f.write("\n")
        
        # Target Variables Status (CRITICAL)
        f.write("-" * 100 + "\n")
        f.write("🎯 TARGET VARIABLES COMPLETENESS (CRITICAL)\n")
        f.write("-" * 100 + "\n")
        for target in ['valence', 'energy', 'danceability', 'popularity']:
            complete = stats['target_completeness'][target]['complete']
            missing = stats['target_completeness'][target]['missing']
            total = complete + missing
            pct_complete = (complete / total * 100) if total > 0 else 0
            
            status = "✅ GOOD" if pct_complete >= 99 else "⚠️ NEEDS ATTENTION"
            if target == 'popularity' and pct_complete >= 80:
                status = "✅ OK (popularity can have missing)"
            
            f.write(f"\n{target.upper():20s} {status}\n")
            f.write(f"  Complete: {complete:>12,} ({pct_complete:>6.2f}%)\n")
            f.write(f"  Missing:  {missing:>12,} ({100-pct_complete:>6.2f}%)\n")
        
        # Missing Values
        f.write("\n" + "-" * 100 + "\n")
        f.write("MISSING VALUES BY COLUMN\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Column':<20s} {'Missing':>12s} {'Percentage':>12s} {'Acceptable?':>15s}\n")
        f.write("-" * 100 + "\n")
        
        for col in sorted(stats['missing_values'].keys()):
            missing = stats['missing_values'][col]
            pct = (missing / stats['total_rows'] * 100)
            
            # Determine if acceptable
            if col in COLUMN_SPECS:
                spec = COLUMN_SPECS[col]
                if spec.get('null_allowed', False):
                    acceptable = "✅ OK (nulls allowed)"
                elif missing == 0:
                    acceptable = "✅ PERFECT"
                elif pct < 1:
                    acceptable = "⚠️ MINOR (<1%)"
                elif pct < 5:
                    acceptable = "⚠️ MODERATE (<5%)"
                else:
                    acceptable = "❌ CRITICAL (>5%)"
            else:
                acceptable = "❓ Unknown column"
            
            f.write(f"{col:<20s} {missing:>12,} {pct:>11.2f}% {acceptable:>15s}\n")
        
        # Range Violations
        f.write("\n" + "-" * 100 + "\n")
        f.write("RANGE VIOLATIONS\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Column':<20s} {'Violations':>12s} {'Expected Range':>25s} {'Status':>15s}\n")
        f.write("-" * 100 + "\n")
        
        for col, violations in sorted(stats['range_violations'].items()):
            if col in COLUMN_SPECS and 'range' in COLUMN_SPECS[col]:
                range_val = COLUMN_SPECS[col]['range']
                range_str = f"[{range_val[0]}, {range_val[1]}]"
            else:
                range_str = "N/A"
            
            status = "✅ GOOD" if violations == 0 else "❌ NEEDS FIX"
            f.write(f"{col:<20s} {violations:>12,} {range_str:>25s} {status:>15s}\n")
        
        # Genre Issues
        f.write("\n" + "-" * 100 + "\n")
        f.write("GENRE ISSUES (KNOWN PROBLEM)\n")
        f.write("-" * 100 + "\n")
        f.write(f"NaN genres:     {stats['genre_issues']['nan']:>12,}\n")
        f.write(f"Empty genres:   {stats['genre_issues']['empty']:>12,}\n")
        f.write(f"Invalid genres: {stats['genre_issues']['invalid']:>12,}\n")
        total_genre_issues = sum(stats['genre_issues'].values())
        pct_genre = (total_genre_issues / stats['total_rows'] * 100)
        f.write(f"{'TOTAL:':<16s} {total_genre_issues:>12,} ({pct_genre:.2f}%)\n")
        f.write(f"\n⚠️ ACTION REQUIRED: Define strategy for handling genre issues\n")
        
        # Year Issues
        f.write("\n" + "-" * 100 + "\n")
        f.write("YEAR ISSUES (KNOWN PROBLEM)\n")
        f.write("-" * 100 + "\n")
        f.write(f"Year = 0:       {stats['year_issues']['zero']:>12,}\n")
        f.write(f"Year < 0:       {stats['year_issues']['negative']:>12,}\n")
        f.write(f"Year > 2025:    {stats['year_issues']['future']:>12,}\n")
        f.write(f"Year < 1900:    {stats['year_issues']['too_old']:>12,}\n")
        total_year_issues = sum(stats['year_issues'].values())
        pct_year = (total_year_issues / stats['total_rows'] * 100)
        f.write(f"{'TOTAL:':<16s} {total_year_issues:>12,} ({pct_year:.2f}%)\n")
        f.write(f"\n⚠️ ACTION REQUIRED: Define strategy for handling year issues\n")
        
        # Encoding Issues
        f.write("\n" + "-" * 100 + "\n")
        f.write("ENCODING INCONSISTENCIES (CRITICAL)\n")
        f.write("-" * 100 + "\n")
        f.write(f"Key (letter format):  {stats['encoding_issues']['key_letter_format']:>12,} rows\n")
        f.write(f"Mode (text format):   {stats['encoding_issues']['mode_text_format']:>12,} rows\n")
        total_encoding = sum(stats['encoding_issues'].values())
        pct_encoding = (stats['encoding_issues']['key_letter_format'] / stats['total_rows'] * 100)
        f.write(f"{'TOTAL:':<22s} {total_encoding:>12,} ({pct_encoding:.2f}%)\n")
        if total_encoding > 0:
            f.write(f"\n⚠️ CRITICAL: Mixed encoding formats detected!\n")
            f.write(f"   - Key column has both numeric (0-11) and letter (C, D, E, ...) formats\n")
            f.write(f"   - Mode column has both numeric (0, 1) and text (Major, Minor) formats\n")
            f.write(f"   - ACTION REQUIRED: Standardize all to numeric format before ML\n")
        
        # Recommendations
        f.write("\n" + "=" * 100 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 100 + "\n")
        f.write("1. Fix key/mode encoding inconsistencies - CRITICAL PRIORITY\n")
        f.write("2. Fix genre issues (NaN/empty values) - PRIORITY\n")
        f.write("3. Fix year = 0 values - PRIORITY\n")
        f.write("4. Remove duplicate rows if any\n")
        f.write("5. Validate all range violations\n")
        f.write("6. Ensure all target variables (valence, energy, danceability) are 100% complete\n")
        f.write("7. Document all cleaning decisions\n")
        f.write("=" * 100 + "\n")
    
    print(f"\n✅ Report saved to: {output_path}")


if __name__ == "__main__":
    # File paths
    data_dir = Path('dataset')
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)

    enhanced_path = data_dir / 'processed/english_ml_ready.csv' # 'raw/songs_enhanced_full.csv'
    report_path = reports_dir / 'comprehensive_validation_english_report_01.txt'
    
    # Run analysis
    print("Starting comprehensive validation analysis...\n")
    stats = analyze_dataset(enhanced_path, chunksize=50000)
    
    # Generate report
    generate_report(stats, report_path)
    
    print("\n" + "="*100)
    print("QUICK SUMMARY")
    print("="*100)
    print(f"Total Rows: {stats['total_rows']:,}")
    print(f"Unique Tracks: {stats['unique_tracks']:,}")
    print(f"Duplicates: {stats['total_duplicates']:,}")
    print(f"\nGenre Issues: {sum(stats['genre_issues'].values()):,}")
    print(f"Year Issues: {sum(stats['year_issues'].values()):,}")
    print(f"\nSee full report: {report_path}")
