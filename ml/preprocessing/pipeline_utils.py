"""Utility functions for modular preprocessing pipeline with intelligent caching.

This module provides functions to check if preprocessing steps need to be run
based on file timestamps and hash-based change detection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = REPO_ROOT / "ml" / "features"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CACHE_FILE = FEATURES_DIR / ".preprocessing_cache.json"


def _compute_file_hash(filepath: Path) -> str:
    """Compute MD5 hash of a file's contents."""
    if not filepath.exists():
        return ""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _load_cache() -> Dict:
    """Load preprocessing cache from disk."""
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_cache(cache: Dict) -> None:
    """Save preprocessing cache to disk."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def check_if_step_needed(
    step_name: str,
    input_files: List[Path],
    output_files: List[Path],
    dependencies: Optional[List[str]] = None,
) -> bool:
    """Check if a preprocessing step needs to be run.
    
    Args:
        step_name: Name of the preprocessing step (e.g., 'audio_features')
        input_files: List of input file paths this step depends on
        output_files: List of output files this step produces
        dependencies: Optional list of other step names this step depends on
        
    Returns:
        True if the step needs to be run, False if it can be skipped
    """
    cache = _load_cache()
    
    # Check if any output files are missing
    for outfile in output_files:
        if not outfile.exists():
            print(f"  → Output file missing: {outfile.name}")
            return True
    
    # Check if step has been run before
    if step_name not in cache:
        print(f"  → Step '{step_name}' never run before")
        return True
    
    step_cache = cache[step_name]
    
    # Check if input files have changed
    for infile in input_files:
        file_key = str(infile.relative_to(REPO_ROOT))
        current_hash = _compute_file_hash(infile)
        
        if file_key not in step_cache.get("input_hashes", {}):
            print(f"  → New input file detected: {infile.name}")
            return True
            
        if current_hash != step_cache["input_hashes"][file_key]:
            print(f"  → Input file changed: {infile.name}")
            return True
    
    # Check if dependencies have been updated
    if dependencies:
        for dep_step in dependencies:
            if dep_step not in cache:
                print(f"  → Dependency '{dep_step}' not run yet")
                return True
            
            dep_timestamp = cache[dep_step].get("timestamp", 0)
            step_timestamp = step_cache.get("timestamp", 0)
            
            if dep_timestamp > step_timestamp:
                print(f"  → Dependency '{dep_step}' updated after this step")
                return True
    
    return False


def mark_step_complete(
    step_name: str,
    input_files: List[Path],
    output_files: List[Path],
    metadata: Optional[Dict] = None,
) -> None:
    """Mark a preprocessing step as complete and update cache.
    
    Args:
        step_name: Name of the preprocessing step
        input_files: List of input files used
        output_files: List of output files created
        metadata: Optional additional metadata to store
    """
    import time
    
    cache = _load_cache()
    
    # Compute hashes for all input files
    input_hashes = {}
    for infile in input_files:
        file_key = str(infile.relative_to(REPO_ROOT))
        input_hashes[file_key] = _compute_file_hash(infile)
    
    # Store step information
    cache[step_name] = {
        "timestamp": time.time(),
        "input_hashes": input_hashes,
        "output_files": [str(f.relative_to(REPO_ROOT)) for f in output_files],
        "metadata": metadata or {},
    }
    
    _save_cache(cache)
    print(f"✅ Step '{step_name}' marked complete in cache")


def get_step_metadata(step_name: str) -> Optional[Dict]:
    """Retrieve metadata for a completed preprocessing step.
    
    Args:
        step_name: Name of the preprocessing step
        
    Returns:
        Metadata dict if step exists in cache, None otherwise
    """
    cache = _load_cache()
    if step_name in cache:
        return cache[step_name].get("metadata")
    return None


def clear_cache(step_names: Optional[List[str]] = None) -> None:
    """Clear preprocessing cache for specific steps or all steps.
    
    Args:
        step_names: List of step names to clear, or None to clear all
    """
    if step_names is None:
        # Clear entire cache
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        print("🗑️  Entire preprocessing cache cleared")
    else:
        # Clear specific steps
        cache = _load_cache()
        for step_name in step_names:
            if step_name in cache:
                del cache[step_name]
                print(f"🗑️  Cache cleared for step '{step_name}'")
        _save_cache(cache)


def print_cache_status() -> None:
    """Print current cache status for all preprocessing steps."""
    cache = _load_cache()
    
    if not cache:
        print("No preprocessing steps cached yet")
        return
    
    print("\n" + "=" * 80)
    print("PREPROCESSING CACHE STATUS")
    print("=" * 80)
    
    import time
    for step_name, step_data in cache.items():
        timestamp = step_data.get("timestamp", 0)
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        output_count = len(step_data.get("output_files", []))
        
        print(f"\n{step_name}:")
        print(f"  Last run: {time_str}")
        print(f"  Output files: {output_count}")
        
        metadata = step_data.get("metadata", {})
        if metadata:
            print(f"  Metadata: {metadata}")
    
    print("\n" + "=" * 80)
