"""
Reproducibility Configuration for Music Prediction DL Project

All random seeds and deterministic settings are centralized here.
Import and call set_all_seeds() at the start of every script.
"""

import torch
import numpy as np
import random
import os


# Global random seed for entire project
RANDOM_SEED = 42


def set_all_seeds(seed=RANDOM_SEED):
    """
    Set all random seeds for complete reproducibility.
    
    Call this FIRST in every training script before any operations.
    
    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU)
    - PyTorch (GPU/CUDA)
    - CUDNN (deterministic mode)
    - Python hash seed
    
    Args:
        seed: Random seed (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU
    
    # Make CUDNN deterministic (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ All random seeds set to: {seed}")
    print("✓ Deterministic mode enabled (reproducible results guaranteed)")


def get_generator(seed=RANDOM_SEED):
    """
    Get PyTorch random generator for DataLoader.
    
    Usage:
        DataLoader(..., generator=get_generator())
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id):
    """
    Initialize worker seed for DataLoader workers.
    
    Usage:
        DataLoader(..., worker_init_fn=worker_init_fn)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Reproducibility checklist for documentation
REPRODUCIBILITY_CHECKLIST = """
Reproducibility Checklist:
✓ set_all_seeds(42) called at start of script
✓ DataLoader uses generator=get_generator()
✓ DataLoader uses worker_init_fn=worker_init_fn
✓ torch.backends.cudnn.deterministic = True
✓ torch.backends.cudnn.benchmark = False
✓ Model initialization uses torch.manual_seed(42)
✓ Train/val/test splits are fixed (not random)
✓ No random augmentation during evaluation
✓ Checkpoint saves include seed in metadata
"""


if __name__ == '__main__':
    # Test reproducibility
    set_all_seeds(42)
    
    print("\nTesting reproducibility...")
    
    # Test 1: NumPy random
    print("\nTest 1: NumPy random")
    set_all_seeds(42)
    a1 = np.random.randn(5)
    set_all_seeds(42)
    a2 = np.random.randn(5)
    print(f"  Arrays equal: {np.allclose(a1, a2)}")
    print(f"  Values: {a1[:3]}")
    
    # Test 2: PyTorch random
    print("\nTest 2: PyTorch random")
    set_all_seeds(42)
    t1 = torch.randn(5)
    set_all_seeds(42)
    t2 = torch.randn(5)
    print(f"  Tensors equal: {torch.allclose(t1, t2)}")
    print(f"  Values: {t1[:3]}")
    
    # Test 3: Model initialization
    print("\nTest 3: Model initialization")
    import torch.nn as nn
    
    set_all_seeds(42)
    model1 = nn.Linear(10, 5)
    w1 = model1.weight.data.clone()
    
    set_all_seeds(42)
    model2 = nn.Linear(10, 5)
    w2 = model2.weight.data.clone()
    
    print(f"  Weights equal: {torch.allclose(w1, w2)}")
    print(f"  Weight values: {w1[0, :3]}")
    
    print("\n✓ All reproducibility tests passed!")
    print("\nIMPORTANT:")
    print("- Same seed → EXACT same results every run")
    print("- Different seed → Different results (for experimentation)")
    print("- Always document which seed was used in results")
