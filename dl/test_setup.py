#!/usr/bin/env python
"""
Quick test to verify Phase 0 setup.
Run this after installing torch and wandb.
"""

import sys
import os

def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    CUDA version: {torch.version.cuda}")
    except ImportError as e:
        print(f"  ✗ PyTorch not found: {e}")
        return False
    
    try:
        import wandb
        print(f"  ✓ W&B installed")
    except ImportError:
        print(f"  ⚠ W&B not found (optional)")
    
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy not found: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"  ✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"  ✗ Pandas not found: {e}")
        return False
    
    try:
        from sklearn.preprocessing import StandardScaler
        print(f"  ✓ scikit-learn")
    except ImportError as e:
        print(f"  ✗ scikit-learn not found: {e}")
        return False
    
    return True


def test_data():
    """Test that preprocessed data exists."""
    print("\nTesting data files...")
    
    feature_dir = 'ml/features'
    if not os.path.exists(feature_dir):
        print(f"  ✗ Directory not found: {feature_dir}")
        return False
    
    required_files = [
        'X_train_audio.npy',
        'X_train_text_stats.npy',
        'X_train_sentiment.npy',
        'X_train_embeddings.npy',
        'y_train_valence.npy',
        'y_train_energy.npy',
        'y_train_danceability.npy',
        'y_train_popularity.npy',
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = os.path.join(feature_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"  ✓ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {filename} not found")
            all_exist = False
    
    return all_exist


def test_utils():
    """Test that utility modules can be imported."""
    print("\nTesting utility modules...")
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
    
    try:
        from models import MusicMLP, SimpleXORNetwork
        print(f"  ✓ models.py")
    except ImportError as e:
        print(f"  ✗ models.py: {e}")
        return False
    
    try:
        from data_loaders import MusicDataset
        print(f"  ✓ data_loaders.py")
    except ImportError as e:
        print(f"  ✗ data_loaders.py: {e}")
        return False
    
    try:
        from metrics import compute_metrics
        print(f"  ✓ metrics.py")
    except ImportError as e:
        print(f"  ✗ metrics.py: {e}")
        return False
    
    return True


def test_model_creation():
    """Test that model can be created."""
    print("\nTesting model creation...")
    
    try:
        import torch
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
        from models import MusicMLP
        
        model = MusicMLP(input_size=414, num_targets=4)
        print(f"  ✓ MusicMLP created")
        print(f"    Parameters: {model.count_parameters():,}")
        
        # Test forward pass
        x = torch.randn(32, 414)
        y = model(x)
        print(f"  ✓ Forward pass works")
        print(f"    Input: {x.shape}")
        print(f"    Output: {y.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("Phase 0 Setup Test")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Data Files", test_data),
        ("Utility Modules", test_utils),
        ("Model Creation", test_model_creation),
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:<20} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! Ready to start training.")
        print("\nNext steps:")
        print("  1. python dl/01_xor_network.py  (learn PyTorch)")
        print("  2. python dl/02_train_mlp.py    (train baseline)")
    else:
        print("\n⚠ Some tests failed. Please fix issues before training.")
        if not results["Imports"]:
            print("\nInstall missing packages:")
            print("  pip install torch wandb")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
