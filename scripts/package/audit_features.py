"""
Dataset Features Auditor.
Scans and audits all Parquet tables and NumPy feature arrays in data/.
Outputs:
- data/features_audit.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_JSON = DATA_DIR / "features_audit.json"

def audit_parquet(file_path: Path) -> dict:
    rel_path = str(file_path.relative_to(PROJECT_ROOT))
    df = pd.read_parquet(file_path)
    n_rows, n_cols = df.shape
    
    col_audits = []
    for col in df.columns:
        s = df[col]
        null_count = int(s.isna().sum())
        dtype_str = str(s.dtype)
        
        # Numeric checks
        if pd.api.types.is_numeric_dtype(s):
            zeros = int((s == 0).sum())
            zero_pct = round(float(zeros / n_rows * 100.0), 2)
            infs = int(np.isinf(s).sum()) if hasattr(s, 'values') else 0
            s_clean = s.dropna()
            s_finite = s_clean[~np.isinf(s_clean)] if infs > 0 else s_clean
            
            if len(s_finite) > 0:
                c_min = round(float(s_finite.min()), 4)
                c_mean = round(float(s_finite.mean()), 4)
                c_max = round(float(s_finite.max()), 4)
            else:
                c_min, c_mean, c_max = None, None, None
                
            all_zero = bool(zeros == n_rows)
        elif pd.api.types.is_bool_dtype(s):
            zeros = int((s == False).sum())
            zero_pct = round(float(zeros / n_rows * 100.0), 2)
            infs = 0
            c_min, c_mean, c_max = 0.0, round(float(s.mean()), 4), 1.0
            all_zero = bool(zeros == n_rows)
        else:
            dtype_str = "object/list"
            zeros = 0
            zero_pct = 0.0
            infs = 0
            c_min, c_mean, c_max = None, None, None
            all_zero = False
            
        col_audits.append({
            "col": col,
            "dtype": dtype_str,
            "nulls": null_count,
            "zeros": zeros,
            "zero_pct": zero_pct,
            "inf": infs,
            "min": c_min,
            "mean": c_mean,
            "max": c_max,
            "all_zero": all_zero
        })
        
    return {
        "file": rel_path,
        "shape": f"({n_rows}, {n_cols})",
        "columns_audit": col_audits
    }

def audit_numpy(file_path: Path) -> dict:
    rel_path = str(file_path.relative_to(PROJECT_ROOT))
    arr = np.load(file_path, allow_pickle=True)
    
    total_elements = int(arr.size)
    dtype_str = str(arr.dtype)
    
    if np.issubdtype(arr.dtype, np.number):
        nans = int(np.isnan(arr).sum())
        infs = int(np.isinf(arr).sum())
        zeros = int((arr == 0).sum())
        zero_pct = round(float(zeros / total_elements * 100.0), 2) if total_elements > 0 else 0.0
        
        # Check all zero rows if 2D
        if arr.ndim == 2:
            all_zero_rows = int((np.linalg.norm(arr, axis=1) < 1e-9).sum())
        else:
            all_zero_rows = 0
            
        arr_finite = arr[np.isfinite(arr)]
        if len(arr_finite) > 0:
            c_min = round(float(arr_finite.min()), 4)
            c_mean = round(float(arr_finite.mean()), 4)
            c_max = round(float(arr_finite.max()), 4)
        else:
            c_min, c_mean, c_max = None, None, None
    else:
        nans = 0
        infs = 0
        zeros = 0
        zero_pct = 0.0
        all_zero_rows = 0
        c_min, c_mean, c_max = None, None, None
        
    return {
        "file": rel_path,
        "shape": list(arr.shape),
        "dtype": dtype_str,
        "total_elements": total_elements,
        "nans": nans,
        "infs": infs,
        "zeros": zeros,
        "zero_pct": zero_pct,
        "all_zero_rows": all_zero_rows,
        "min": c_min,
        "mean": c_mean,
        "max": c_max
    }

def main():
    print("\n" + "="*80)
    print("AUDITING ALL DATASET FEATURES & EMBEDDINGS (PARQUET & NUMPY)")
    print("="*80 + "\n")
    
    p_files = sorted([p for p in DATA_DIR.glob("**/*.parquet") if "checkpoint" not in p.name and "pilot" not in str(p)])
    n_files = sorted([p for p in DATA_DIR.glob("**/*.npy") if "checkpoint" not in p.name and "pilot" not in str(p)])
    
    print(f"Found {len(p_files)} Parquet tables and {len(n_files)} NumPy feature files.\n")
    
    parquet_audits = []
    for idx, p in enumerate(p_files, 1):
        print(f"[{idx:2d}/{len(p_files)}] Auditing Parquet: {p.relative_to(PROJECT_ROOT)}...")
        parquet_audits.append(audit_parquet(p))
        
    print()
    numpy_audits = []
    for idx, n in enumerate(n_files, 1):
        print(f"[{idx:2d}/{len(n_files)}] Auditing NumPy:   {n.relative_to(PROJECT_ROOT)}...")
        numpy_audits.append(audit_numpy(n))
        
    audit_data = {
        "parquet": parquet_audits,
        "numpy": numpy_audits
    }
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(audit_data, f, indent=2)
        
    print(f"\n✓ Successfully updated: {OUTPUT_JSON}")
    print(f"  Total Parquet tables audited: {len(parquet_audits)}")
    print(f"  Total NumPy matrices audited: {len(numpy_audits)}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
