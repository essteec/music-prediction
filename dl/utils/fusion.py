"""
Multi-Modal Fusion Architectures — Phase 4 (Music Prediction)

Three-experiment progression:
  A) MultiModalFusionMLP   — Per-modality encoders + simple concatenation fusion
  B) GatedFusionMLP        — Per-modality encoders + global learned modality gating
  C) TaskGatedFusionMLP    — Per-modality encoders + per-target learned gating

Modality input dimensions (all pre-aligned, zero-padded for missing songs):
  metadata   (audio + text_stats + sentiment): 30d
  mpnet      (MPNet-base-v2 lyrics):          768d
  vggish     (VGGish audio):                  128d
  mert       (MERT-v1-95M audio):             768d
  panns      (PANNs Cnn14 audio):            2048d
  mel_stats  (Mel spectrogram statistics):    512d

All encoders project to a shared ENCODED_DIM (default 128d).
Fused representation = 6 × 128 = 768d.
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
ENCODED_DIM = 128     # All encoders compress to this dimension
NUM_MODALITIES = 6    # Number of input branches
FUSED_DIM = ENCODED_DIM * NUM_MODALITIES  # 768


# ──────────────────────────────────────────────────────────────────────────────
# Building Blocks
# ──────────────────────────────────────────────────────────────────────────────

class ModalityEncoder(nn.Module):
    """
    Compresses one modality from its raw dimensionality to ENCODED_DIM.

    Architecture: Linear(in → h0) → LN → GELU → Dropout
                  [Linear(h0 → h1) → LN → GELU → Dropout] × n hidden layers
                  Linear(last_h → out) → LN → GELU

    The GELU at the end ensures the encoder output has nonlinearity before
    concatenation, giving the fusion layer a richer signal to work with.

    Args:
        input_dim:   Raw feature dimensionality.
        hidden_dims: List of intermediate layer widths (can be empty).
        output_dim:  Target dimensionality (ENCODED_DIM = 128).
        dropout:     Dropout rate applied after each hidden layer.
        l2_norm_input: If True, L2-normalize the raw input before encoding.
                       Use for embeddings with inconsistent scale (mel_stats).
    """

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int = ENCODED_DIM,
                 dropout: float = 0.3, l2_norm_input: bool = False):
        super().__init__()
        self.l2_norm_input = l2_norm_input
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers += [
                nn.Linear(in_d, h_d),
                nn.LayerNorm(h_d),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_d = h_d
        layers += [
            nn.Linear(in_d, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),   # ← activation so encoder output is non-trivially informative
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.l2_norm_input:
            x = F.normalize(x, p=2, dim=-1)
        return self.net(x)


class ResidualBlock(nn.Module):
    """
    Pre-norm residual block for the fusion MLP.
    Pre-norm (LN before linear) is more stable than post-norm at depth.

    Architecture: LN → Linear → GELU → Dropout → Linear → + residual
    """

    def __init__(self, dim: int, dropout: float = 0.5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x + residual


def _build_encoders(dropout_enc: float = 0.3,
                    metadata_dim: int = 30) -> nn.ModuleDict:
    """Shared encoder factory. metadata_dim can be overridden for engineered features."""
    return nn.ModuleDict({
        "metadata":  ModalityEncoder(metadata_dim, [64],  ENCODED_DIM, dropout_enc),
        "mpnet":     ModalityEncoder(768,  [256],       ENCODED_DIM, dropout_enc),
        "vggish":    ModalityEncoder(128,  [64],        ENCODED_DIM, dropout_enc),
        "mert":      ModalityEncoder(768,  [256],       ENCODED_DIM, dropout_enc),
        "panns":     ModalityEncoder(2048, [512],       ENCODED_DIM, dropout_enc),
        "mel_stats": ModalityEncoder(512,  [256],       ENCODED_DIM, dropout_enc),
    })


# ──────────────────────────────────────────────────────────────────────────────
# Experiment A — Multi-Branch Baseline (simple concat fusion)
# ──────────────────────────────────────────────────────────────────────────────

class MultiModalFusionMLP(nn.Module):
    """
    Experiment A: Six per-modality encoders → concat → residual fusion MLP.

    Answers: "Does per-modality compression alone eliminate the audio
    degradation problem observed with flat concatenation?"

    Architecture:
        metadata  (30d)  ─→ Encoder → 128d ─┐
        mpnet     (768d) ─→ Encoder → 128d ─┤
        vggish    (128d) ─→ Encoder → 128d ─┤─ cat → 768d
        mert      (768d) ─→ Encoder → 128d ─┤
        panns     (2048d)─→ Encoder → 128d ─┤
        mel_stats (512d) ─→ Encoder → 128d ─┘
                                              ↓
                               Linear(768→512) → LN → GELU → Drop
                                              ↓
                               ResidualBlock(512)
                                              ↓
                               Linear(512→256) → LN → GELU → Drop
                                              ↓
                               Linear(256→4)   [no activation]
    """

    def __init__(self, num_targets: int = 4, dropout_enc: float = 0.3, dropout_fusion: float = 0.5):
        super().__init__()
        self.encoders = _build_encoders(dropout_enc)

        self.fusion = nn.Sequential(
            nn.Linear(FUSED_DIM, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_fusion),
            ResidualBlock(512, dropout_fusion),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_fusion),
        )
        self.head = nn.Linear(256, num_targets)

    def forward(self, metadata, mpnet, vggish, mert, panns, mel_stats):
        enc = torch.cat([
            self.encoders["metadata"](metadata),
            self.encoders["mpnet"](mpnet),
            self.encoders["vggish"](vggish),
            self.encoders["mert"](mert),
            self.encoders["panns"](panns),
            self.encoders["mel_stats"](mel_stats),
        ], dim=-1)  # (B, 768)
        return self.head(self.fusion(enc))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Experiment B — Gated Fusion (global modality attention)
# ──────────────────────────────────────────────────────────────────────────────

class GatedFusionMLP(nn.Module):
    """
    Experiment B: Same encoders as A, but with a global gating network
    that learns per-sample modality importance weights.

    The gate produces 6 scalars (one per modality) via softmax. Each
    modality's 128d encoding is scaled by its gate weight before fusion.
    Missing modalities (zero-padded) are detected and masked out of the
    softmax so the gate redistributes mass only over present modalities.

    Answers: "Can the model learn which modalities to trust per sample?"

    Gate architecture:
        stack(6 encodings) → mean-pool to 128d
        → Linear(128 → 64) → GELU → Linear(64 → 6)
        → masked softmax → 6 weights
        → scale each 128d encoding → concat → 768d
    """

    def __init__(self, num_targets: int = 4, dropout_enc: float = 0.3, dropout_fusion: float = 0.5):
        super().__init__()
        self.encoders = _build_encoders(dropout_enc)

        # Gating network: summary of all encodings → modality weights
        self.gate_net = nn.Sequential(
            nn.Linear(ENCODED_DIM, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, NUM_MODALITIES),
        )

        self.fusion = nn.Sequential(
            nn.Linear(FUSED_DIM, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_fusion),
            ResidualBlock(512, dropout_fusion),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_fusion),
        )
        self.head = nn.Linear(256, num_targets)

    def forward(self, metadata, mpnet, vggish, mert, panns, mel_stats):
        raw_inputs = [metadata, mpnet, vggish, mert, panns, mel_stats]
        encs = [self.encoders[k](v) for k, v in zip(MODALITY_DIMS.keys(), raw_inputs)]
        # encs: list of 6 tensors, each (B, 128)

        # Build presence mask: modality is "present" if input is non-zero
        presence = torch.stack([
            (x.abs().sum(dim=-1) > 0).float() for x in raw_inputs
        ], dim=-1)  # (B, 6)

        # Global summary: mean of all present encodings
        stacked = torch.stack(encs, dim=1)          # (B, 6, 128)
        summary = stacked.mean(dim=1)                # (B, 128)

        # Gate logits + masked softmax
        logits = self.gate_net(summary)              # (B, 6)
        logits = logits + (1.0 - presence) * (-1e9)  # mask absent modalities
        gates = torch.softmax(logits, dim=-1)        # (B, 6)

        # Scale each modality encoding by its gate weight
        gates_exp = gates.unsqueeze(-1)              # (B, 6, 1)
        gated = (stacked * gates_exp).view(-1, FUSED_DIM)  # (B, 768)

        return self.head(self.fusion(gated))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Experiment C — Task-Gated Fusion (per-target modality attention)
# ──────────────────────────────────────────────────────────────────────────────

class TaskGatedFusionMLP(nn.Module):
    """
    Experiment C: Each prediction target gets its own gating network and
    its own fusion pathway. This allows the model to discover that:
        - Valence relies more on MPNet (lyrics/emotion)
        - Energy/Danceability rely more on MERT/PANNs (acoustics)
        - Popularity relies more on metadata (artist, genre)

    Architecture:
        Shared encoders (same as A & B) → 6 × 128d
        ↓
        Per-target gate  → weighted 768d fused repr
        Per-target head  → scalar prediction
        ↓
        Stack 4 targets → (B, 4)

    Answers: "Do different targets benefit from different modality mixtures?"
    """

    def __init__(self, num_targets: int = 4, dropout_enc: float = 0.3,
                 dropout_fusion: float = 0.5, metadata_dim: int = 30):
        super().__init__()
        self.encoders = _build_encoders(dropout_enc, metadata_dim=metadata_dim)
        self.num_targets = num_targets

        # One gate network per target
        self.task_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ENCODED_DIM, 64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, NUM_MODALITIES),
            )
            for _ in range(num_targets)
        ])

        # One fusion MLP per target
        self.task_fusions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(FUSED_DIM, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout_fusion),
                ResidualBlock(256, dropout_fusion),
                nn.Linear(256, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout_fusion),
            )
            for _ in range(num_targets)
        ])

        # One scalar output per target
        self.task_heads = nn.ModuleList([
            nn.Linear(64, 1) for _ in range(num_targets)
        ])

    def forward(self, metadata, mpnet, vggish, mert, panns, mel_stats):
        raw_inputs = [metadata, mpnet, vggish, mert, panns, mel_stats]
        encs = [self.encoders[k](v) for k, v in zip(MODALITY_DIMS.keys(), raw_inputs)]

        # Presence mask
        presence = torch.stack([
            (x.abs().sum(dim=-1) > 0).float() for x in raw_inputs
        ], dim=-1)  # (B, 6)

        stacked = torch.stack(encs, dim=1)   # (B, 6, 128)
        summary = stacked.mean(dim=1)         # (B, 128)

        preds = []
        for gate_net, fusion, head in zip(self.task_gates, self.task_fusions, self.task_heads):
            logits = gate_net(summary)                        # (B, 6)
            logits = logits + (1.0 - presence) * (-1e9)
            gates = torch.softmax(logits, dim=-1).unsqueeze(-1)  # (B, 6, 1)
            gated = (stacked * gates).view(-1, FUSED_DIM)        # (B, 768)
            preds.append(head(fusion(gated)))                     # (B, 1)

        return torch.cat(preds, dim=-1)  # (B, 4)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
