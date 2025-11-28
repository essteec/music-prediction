#!/usr/bin/env python3
"""Main Preprocessing Pipeline Orchestrator

This script intelligently runs only the preprocessing steps that are needed.
Each step is cached and skipped if inputs haven't changed.

EDA-Driven Transformations:
- Audio: PowerTransformer for acousticness, instrumentalness, speechiness
- Targets: Log1p transformation for popularity (heavy right skew)
- Text Stats: Log1p for count features (word_count, unique_word_count, char_count)
- Sentiment: StandardScaler for TextBlob features

Usage:
    # Run all preprocessing steps (skips cached steps automatically)
    python run_preprocessing.py
    
    # Run specific steps only
    python run_preprocessing.py --steps audio text_stats
    
    # Force re-run all steps (ignore cache)
    python run_preprocessing.py --force
    
    # Show cache status
    python run_preprocessing.py --status
    
    # Clear cache for specific steps
    python run_preprocessing.py --clear audio sentiment
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add preprocessing directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline_utils import clear_cache, print_cache_status
from process_audio import process_audio_features
from process_sentiment import process_sentiment
from process_targets import process_targets
from process_text_stats import process_text_statistics
from process_embeddings import process_embeddings


STEP_FUNCTIONS = {
    "audio": process_audio_features,
    "text_stats": process_text_statistics,
    "sentiment": process_sentiment,
    "embeddings": process_embeddings,
    "targets": process_targets,
}

STEP_DESCRIPTIONS = {
    "audio": "Audio features (power transform skewed, cyclical encoding, scaling, genre one-hot)",
    "text_stats": "Text statistics (word counts, vocabulary metrics, log transform)",
    "sentiment": "Sentiment analysis (TextBlob polarity & subjectivity)",
    "embeddings": "Lyric embeddings (all-MiniLM-L6-v2, 384-dim semantic vectors)",
    "targets": "Target variables (valence, energy, danceability, log-popularity)",
}


def run_pipeline(steps: list[str] | None = None, force: bool = False) -> None:
    """Run the preprocessing pipeline.
    
    Args:
        steps: List of step names to run, or None to run all
        force: If True, clear cache before running
    """
    if steps is None:
        steps = list(STEP_FUNCTIONS.keys())
    
    # Validate step names
    invalid = set(steps) - set(STEP_FUNCTIONS.keys())
    if invalid:
        print(f"❌ Invalid step names: {invalid}")
        print(f"   Valid steps: {list(STEP_FUNCTIONS.keys())}")
        return
    
    # Clear cache if force flag set
    if force:
        print("🗑️  Force mode: clearing cache for selected steps...")
        clear_cache(steps)
        print()
    
    print("=" * 80)
    print("PREPROCESSING PIPELINE")
    print("=" * 80)
    print(f"\nSteps to run: {', '.join(steps)}")
    print()
    
    # Run each step
    results = {}
    for step_name in steps:
        print(f"\n{'─' * 80}")
        print(f"Step: {step_name}")
        print(f"Description: {STEP_DESCRIPTIONS[step_name]}")
        print(f"{'─' * 80}")
        
        try:
            step_func = STEP_FUNCTIONS[step_name]
            results[step_name] = step_func(verbose=True)
            print(f"✅ {step_name} completed successfully")
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"\n✅ All {len(steps)} steps completed successfully")
    print(f"\nFeatures saved to: ml/features/")
    print("\nNext steps:")
    print("  1. Train baseline models: cd ml/models && python baseline_models.py")
    print("  2. Experiment with text models: python text_stats_models.py")
    print("  3. Try combined models: python combined_text_models.py")


def main():
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline with intelligent caching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=list(STEP_FUNCTIONS.keys()),
        help="Specific steps to run (default: all steps)",
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run steps, ignoring cache",
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show cache status and exit",
    )
    
    parser.add_argument(
        "--clear",
        nargs="*",
        metavar="STEP",
        help="Clear cache for specific steps (or all if no steps specified)",
    )
    
    args = parser.parse_args()
    
    # Handle status command
    if args.status:
        print_cache_status()
        return
    
    # Handle clear command
    if args.clear is not None:
        if len(args.clear) == 0:
            clear_cache(None)  # Clear all
        else:
            clear_cache(args.clear)
        return
    
    # Run pipeline
    run_pipeline(steps=args.steps, force=args.force)


if __name__ == "__main__":
    main()
