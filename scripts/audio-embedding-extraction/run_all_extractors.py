"""
Audio Embedding Extraction Orchestrator

Runs all embedding extractors in optimal order:
1. VGGish (fastest, ~12 hours for 45K songs)
2. Mel Stats (CPU-based, ~14 hours for 45K songs)
3. MERT (SOTA music embeddings, ~6 hours for 45K songs)
4. PANNs (AudioSet features, ~14 hours for 45K songs)

Total: ~145 hours (~6 days) for all 4 extractors

Usage:
    python run_all_extractors.py                # Run all extractors
    python run_all_extractors.py --model mert   # Run specific model
    python run_all_extractors.py --test 100     # Test mode (100 songs)

Author: Music Prediction Project
Date: April 2026
"""

import sys
import argparse
import subprocess
from pathlib import Path


# Extraction order (fastest first for quick validation)
EXTRACTORS = [
    ('vggish', 'extract_vggish.py', '~12h'),
    ('mel_stats', 'extract_mel_stats.py', '~14h'),
    ('mert', 'extract_mert.py', '~6h'),
    ('panns', 'extract_panns.py', '~14h'),
]


def run_extractor(script_name: str, test_count: int = None, resume: bool = False):
    """Run a single extractor script."""
    script_path = Path(__file__).parent / script_name
    
    cmd = [sys.executable, str(script_path)]
    if test_count:
        cmd.extend(['--test', str(test_count)])
    if resume:
        cmd.append('--resume')
    
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60 + '\n')
    
    result = subprocess.run(cmd, cwd=script_path.parent)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all audio embedding extractors")
    parser.add_argument('--model', choices=['vggish', 'mel_stats', 'mert', 'panns', 'all'],
                        default='all', help="Which model to run (default: all)")
    parser.add_argument('--test', type=int, help="Test mode: process only N songs")
    parser.add_argument('--resume', action='store_true', help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    print("="*60)
    print("AUDIO EMBEDDING EXTRACTION PIPELINE")
    print("="*60)
    print(f"Mode: {'TEST' if args.test else 'FULL'}")
    print(f"Resume: {args.resume}")
    print()
    
    # Filter extractors
    if args.model == 'all':
        to_run = EXTRACTORS
    else:
        to_run = [(name, script, time) for name, script, time in EXTRACTORS 
                  if name == args.model]
    
    print("Extractors to run:")
    for name, script, est_time in to_run:
        print(f"  - {name}: {est_time} (estimated)")
    print()
    
    # Run extractors
    results = {}
    for name, script, est_time in to_run:
        print(f"\n>>> Starting {name} extraction...")
        success = run_extractor(script, args.test, args.resume)
        results[name] = 'SUCCESS' if success else 'FAILED'
        
        if not success:
            print(f"[ERROR] {name} extraction failed!")
    
    # Summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    for name, status in results.items():
        print(f"  {name}: {status}")
    print("="*60)


if __name__ == "__main__":
    main()
