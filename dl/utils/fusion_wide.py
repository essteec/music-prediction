"""
Wide Multi-Modal Fusion Architecture — Experiment E

Identical topology to fusion.py (TaskGatedFusionMLP) but with:
  - ENCODED_DIM: 128 → 256  (encoders compress to a richer shared space)
  - Proportionally wider encoder hidden dims
  - Deeper fusion MLP: 2 ResBlocks instead of 1, 1024→512 instead of 512→256
  - Slightly higher dropout (0.3 enc / 0.5 fusion) to compensate for extra capacity

Rationale:
  - Exp C/D peaked at val epoch ~4 with 128d encoders → model hits representational
    ceiling before overfitting kicks in. The PANNS encoder (2048→512→128) crushes
    2048 acoustic dimensions into 128 in two steps — too lossy for Energy/Danceability.
  - With 256d: PANNS → 1024 → 256 (4 intermediate dims, not 2), MERT → 512 → 256.
    Each encoder retains 2× more information before fusion.
  - Fused dim: 6 × 256 = 1536d → fusion MLP starts from a richer representation.
  - Total parameters: ~10M (vs 3.2M for Exp C). Still feasible for 344K samples
    with dropout=0.5 as implicit regularization.

Usage:
    from dl.utils.fusion_wide import WideTaskGatedFusionMLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MODALITY_DIMS = {
    "metadata":  30,
    "mpnet":    768,
    "vggish":   128,
    "mert":     768,
    "panns":   2048,
    "mel_stats": 512,
}
ENCODED_DIM   = 256        # 2× wider than Exp C (128)
NUM_MODALITIES = 6
FUSED_DIM     = ENCODED_DIM * NUM_MODALITIES  # 1536


# ──────────────────────────────────────────────────────────────────────────────
# Building Blocks (same as fusion.py — duplicated to avoid coupling)
# ──────────────────────────────────────────────────────────────────────────────

class ModalityEncoder(nn.Module):
    """
    Compresses one modality from its raw dimensionality to ENCODED_DIM.
    Architecture: Linear(in→h0) → LN → GELU → Drop → [more hidden layers]
                  → Linear(last_h→out) → LN → GELU
    """
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int = ENCODED_DIM,
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers += [nn.Linear(in_d, h_d), nn.LayerNorm(h_d), nn.GELU(), nn.Dropout(dropout)]
            in_d = h_d
        layers += [nn.Linear(in_d, output_dim), nn.LayerNorm(output_dim), nn.GELU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    """Pre-norm residual block: LN → Linear → GELU → Drop → Linear → + residual."""
    def __init__(self, dim: int, dropout: float = 0.5):
        super().__init__()
        self.norm    = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.act     = nn.GELU()
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear2(x)
        return x + r


def _build_wide_encoders(dropout_enc: float = 0.3) -> nn.ModuleDict:
    """Wide encoder factory: all encoders output ENCODED_DIM=256."""
    return nn.ModuleDict({
        # metadata 30d → 128 → 256  (wider bottleneck for tabular features)
        "metadata":  ModalityEncoder(30,   [128],        ENCODED_DIM, dropout_enc),
        # mpnet 768d → 512 → 256
        "mpnet":     ModalityEncoder(768,  [512],        ENCODED_DIM, dropout_enc),
        # vggish 128d → 256 → 256   (extra hidden to expand before compressing)
        "vggish":    ModalityEncoder(128,  [256],        ENCODED_DIM, dropout_enc),
        # mert 768d → 512 → 256
        "mert":      ModalityEncoder(768,  [512],        ENCODED_DIM, dropout_enc),
        # panns 2048d → 1024 → 256  (4x reduction then 4x → gives gradients room)
        "panns":     ModalityEncoder(2048, [1024],       ENCODED_DIM, dropout_enc),
        # mel_stats 512d → 512 → 256
        "mel_stats": ModalityEncoder(512,  [512],        ENCODED_DIM, dropout_enc),
    })


# ──────────────────────────────────────────────────────────────────────────────
# Experiment E — Wide Task-Gated Fusion MLP
# ──────────────────────────────────────────────────────────────────────────────

class WideTaskGatedFusionMLP(nn.Module):
    """
    Experiment E: Same topology as TaskGatedFusionMLP (Exp C) with:
      - ENCODED_DIM 128 → 256 (2× wider encoders)
      - Fusion MLP: 1536→1024 → 2 ResBlocks → 1024→512 → head
      - Dropout_enc=0.3, dropout_fusion=0.5 (tuned up to match extra capacity)

    Architecture:
        metadata  (30d)  → Encoder → 256d ─┐
        mpnet     (768d) → Encoder → 256d ─┤
        vggish    (128d) → Encoder → 256d ─┤─ cat → 1536d
        mert      (768d) → Encoder → 256d ─┤
        panns    (2048d) → Encoder → 256d ─┤
        mel_stats (512d) → Encoder → 256d ─┘
                                             ↓
                              Per-target gate (from mean of encodings)
                              → weighted 1536d
                                             ↓
                              Per-target fusion: 1536→1024 → ResBlock → ResBlock
                              → 1024→256 → head → scalar
    """

    def __init__(self, num_targets: int = 4,
                 dropout_enc: float = 0.3, dropout_fusion: float = 0.5):
        super().__init__()
        self.encoders    = _build_wide_encoders(dropout_enc)
        self.num_targets = num_targets

        # One gate network per target (same as Exp C but from 256d summary)
        self.task_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ENCODED_DIM, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, NUM_MODALITIES),
            )
            for _ in range(num_targets)
        ])

        # One fusion MLP per target — deeper than Exp C, but not bloated
        self.task_fusions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(FUSED_DIM, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout_fusion),
                ResidualBlock(512, dropout_fusion),
                ResidualBlock(512, dropout_fusion),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout_fusion),
            )
            for _ in range(num_targets)
        ])

        # One scalar output per target
        self.task_heads = nn.ModuleList([
            nn.Linear(256, 1) for _ in range(num_targets)
        ])

    def forward(self, metadata, mpnet, vggish, mert, panns, mel_stats):
        raw_inputs = [metadata, mpnet, vggish, mert, panns, mel_stats]
        encs = [self.encoders[k](v) for k, v in zip(MODALITY_DIMS.keys(), raw_inputs)]

        # Presence mask (for missing modality handling)
        presence = torch.stack([
            (x.abs().sum(dim=-1) > 0).float() for x in raw_inputs
        ], dim=-1)  # (B, 6)

        stacked = torch.stack(encs, dim=1)   # (B, 6, 256)
        summary = stacked.mean(dim=1)         # (B, 256)

        preds = []
        for gate_net, fusion, head in zip(self.task_gates, self.task_fusions, self.task_heads):
            logits = gate_net(summary)
            logits = logits + (1.0 - presence) * (-1e9)
            gates  = torch.softmax(logits, dim=-1).unsqueeze(-1)  # (B, 6, 1)
            gated  = (stacked * gates).view(-1, FUSED_DIM)         # (B, 1536)
            preds.append(head(fusion(gated)))                       # (B, 1)

        return torch.cat(preds, dim=-1)  # (B, 4)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
