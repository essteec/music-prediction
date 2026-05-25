"""
Thesis-ready DL architectures for the ML vs DL comparison.

Architecture set:
1. FlatAllMLP                — Concatenated all-feature neural baseline
2. MultiModalFusionMLP       — Per-modality encoders + concat fusion (Exp A)
3. TaskGatedFusionMLP        — Per-target modality weighting (Exp C)
4. AttentionTaskGatedFusionMLP — Cross-modal attention (Exp G)
5. TaskGatedFusionMLP_FeatEng — Best-optimized variant (Exp F)

Usage:
    from utils.thesis_models import (
        FlatAllMLP, MultiModalFusionMLP, TaskGatedFusionMLP,
        AttentionTaskGatedFusionMLP, engineer_metadata,
        TARGET_NAMES, TOTAL_FLAT_DIM
    )
"""

import numpy as np
import torch
import torch.nn as nn

from .fusion import MultiModalFusionMLP, TaskGatedFusionMLP
from .fusion_attention import AttentionTaskGatedFusionMLP

TARGET_NAMES = ['valence', 'energy', 'danceability', 'popularity']
TOTAL_FLAT_DIM = 4254  # 30 + 768 + 128 + 768 + 2048 + 512


def engineer_metadata(arr: np.ndarray) -> np.ndarray:
    """
    Augment 30d metadata with pairwise interaction features (Exp F).

    audio branch layout:
        0: acousticness       1: instrumentalness    2: speechiness
        3: liveness           4: loudness            5: tempo
        6: duration_ms        7: year                8: mode
        9: key_sin           10: key_cos            11-20: genre one-hot
       21: log_artist_followers  22: avg_artist_popularity

    Interactions added:
        loudness * tempo        → strong energy predictor
        loudness * acousticness → energy anticorrelation
        tempo * mode            → danceability signal
        tempo * liveness        → live performance energy
        acousticness * instr    → acoustic genre
        year * tempo            → era-tempo trend
    """
    interactions = np.stack([
        arr[:, 4] * arr[:, 5],
        arr[:, 4] * arr[:, 0],
        arr[:, 5] * arr[:, 8],
        arr[:, 5] * arr[:, 3],
        arr[:, 0] * arr[:, 1],
        arr[:, 7] * arr[:, 5],
    ], axis=1)
    return np.concatenate([arr, interactions], axis=1)


class FlatAllMLP(nn.Module):
    """
    Baseline neural model: concatenated all-features → simple MLP.

    Uses the same MultiModalDataset interface as the fusion models
    (receives 6 modality tensors) but concatenates them internally
    and passes through a flat MLP.

    Args:
        input_dim: Total feature dimension (4254 regular, 4260 with feat eng).
        dropout: Dropout rate after each hidden layer.
    """

    def __init__(self, input_dim: int = TOTAL_FLAT_DIM, dropout: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 4),
        )

    def forward(self, metadata, mpnet, vggish, mert, panns, mel_stats):
        x = torch.cat([metadata, mpnet, vggish, mert, panns, mel_stats], dim=-1)
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
