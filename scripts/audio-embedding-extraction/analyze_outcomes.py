"""
Audio Embedding Extraction Outcome Analyzer (Improved)

Analyzes extraction logs to provide:
1. Success/Failure statistics per model.
2. List of failed songs across all models.
3. Summary of common error messages.

Handles cases where embedding_shape contains a comma (e.g., '(128,)').

Author: Music Prediction Project
Date: April 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import io

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = DATA_DIR / "embeddings" / "extraction_logs"
DOWNLOAD_LOG = DATA_DIR / "logs" / "download_log_pilot.csv"

MODELS = ['vggish', 'mert', 'panns', 'mel_stats']

def load_log_robust(file_path):
    """
    Load log file while handling the extra comma in (dim,) shapes.
    """
    lines = []
    try:
        with open(file_path, 'r') as f:
            header_line = f.readline().strip()
            if not header_line:
                return pd.DataFrame()
            header = header_line.split(',')
            
            for line in f:
                parts = line.strip().split(',')
                if not parts or len(parts) < 3:
                    continue
                
                # Normal: 6 parts
                # (dim,): 7 parts (comma inside shape like "(768,)")
                if len(parts) == 7:
                    # Merge parts 4 and 5 (shape) which were split by the comma in (dim,)
                    # parts: timestamp, spotify_id, success, time, (dim, ), error
                    # index: 0, 1, 2, 3, 4, 5, 6
                    merged_shape = f"{parts[4]},{parts[5]}"
                    new_parts = parts[:4] + [merged_shape] + [parts[6]]
                    lines.append(new_parts)
                elif len(parts) == 6:
                    lines.append(parts)
                else:
                    # For lines with even more commas (maybe in error message), 
                    # we assume the first 4 fields are fixed, and the shape is fields 4,5
                    # and the rest is error message
                    if len(parts) > 7:
                        merged_shape = f"{parts[4]},{parts[5]}"
                        merged_error = ",".join(parts[6:])
                        new_parts = parts[:4] + [merged_shape] + [merged_error]
                        lines.append(new_parts)
                    else:
                        # Too few fields, skip or pad
                        lines.append(parts + [""] * (6 - len(parts)))
        
        return pd.DataFrame(lines, columns=header)
    except Exception as e:
        print(f"[ERROR] Loading {file_path}: {e}")
        return pd.DataFrame()

def analyze_outcomes():
    print("="*60)
    print("AUDIO EMBEDDING EXTRACTION ANALYSIS")
    print("="*60)

    # 1. Load baseline
    if not DOWNLOAD_LOG.exists():
        print(f"[ERROR] Download log not found at {DOWNLOAD_LOG}")
        return

    dl_df = pd.read_csv(DOWNLOAD_LOG)
    successful_downloads = set(dl_df[dl_df['download_success'] == True]['song_id'].unique())
    total_baseline = len(successful_downloads)
    print(f"[INFO] Total successfully downloaded songs (baseline): {total_baseline:,}")
    print("-" * 60)

    # 2. Analyze each model
    model_stats = {}
    failed_per_model = {}
    all_failed_ids = set()

    for model in MODELS:
        log_path = LOGS_DIR / f"{model}_extraction_log.csv"
        if not log_path.exists():
            print(f"[WARN] Log file for {model} not found.")
            failed_per_model[model] = successful_downloads
            all_failed_ids.update(successful_downloads)
            continue

        log_df = load_log_robust(log_path)
        if log_df.empty:
            print(f"[WARN] Log for {model} is empty or could not be parsed.")
            failed_per_model[model] = successful_downloads
            all_failed_ids.update(successful_downloads)
            continue

        # Convert success to boolean
        log_df['success'] = log_df['success'].astype(str).str.lower() == 'true'
        
        # Determine successful and failed IDs
        success_map = log_df.groupby('spotify_id')['success'].any()
        successful_ids = set(success_map[success_map == True].index)
        
        # Calculate failures relative to baseline
        failed_ids = successful_downloads - successful_ids
        failed_per_model[model] = failed_ids
        all_failed_ids.update(failed_ids)
        
        print(f"Model: {model.upper()}")
        print(f"  Success:       {len(successful_ids):,}")
        print(f"  Failed/Missing: {len(failed_ids):,}")
        
        # Top error analysis
        if 'error_msg' in log_df.columns:
            failed_rows = log_df[log_df['success'] == False]
            if not failed_rows.empty:
                top_errors = failed_rows['error_msg'].value_counts().head(3)
                if not top_errors.empty:
                    print("  - Top Errors:")
                    for err, count in top_errors.items():
                        print(f"    * {str(err)[:60]}: {count}")
        print("-" * 30)

    # 3. Intersection analysis
    print("\n" + "="*60)
    print("CROSS-MODEL FAILURE ANALYSIS")
    print("="*60)
    
    # IDs that failed in ALL models
    failed_in_all = set.intersection(*failed_per_model.values()) if failed_per_model else set()
    
    print(f"Total unique songs with at least one failure: {len(all_failed_ids):,}")
    print(f"Songs that failed across ALL models:         {len(failed_in_all):,}")
    
    # 4. Save results
    output_path = DATA_DIR / "embeddings" / "failed_extractions_summary.csv"
    failed_summary = []
    for sid in all_failed_ids:
        row = {'spotify_id': sid}
        for model in MODELS:
            row[f'{model}_failed'] = sid in failed_per_model.get(model, set())
        failed_summary.append(row)
    
    if failed_summary:
        summary_df = pd.DataFrame(failed_summary)
        summary_df.to_csv(output_path, index=False)
        print(f"\n[INFO] Detailed failure summary saved to: {output_path}")

    print("="*60)

if __name__ == "__main__":
    analyze_outcomes()
