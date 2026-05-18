"""
Cross-Modal Attention Architectures — Experiment G

Adds a Transformer encoder layer across the modalities *after* they are compressed
by the modality encoders, but *before* the gating and MLP fusion. This allows 
modalities (especially the 4 audio modalities) to contextualize each other.

Usage:
    from dl.utils.fusion_attention import AttentionTaskGatedFusionMLP
"""

import torch
import torch.nn as nn

from .fusion import _build_encoders, ENCODED_DIM, NUM_MODALITIES, FUSED_DIM, ResidualBlock

class CrossModalAttention(nn.Module):
    """
    Applies Self-Attention across the encoded modalities before fusion.
    Expects input shape: (B, NUM_MODALITIES, ENCODED_DIM)
    """
    def __init__(self, embed_dim: int = ENCODED_DIM, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        # batch_first=True means (Batch, Seq, Feature)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        """
        x: (B, NUM_MODALITIES, ENCODED_DIM)
        presence: (B, NUM_MODALITIES) float tensor where 1 means present, 0 means missing.
        """
        # key_padding_mask expects True for elements that should be IGNORED
        key_padding_mask = (presence == 0.0)
        
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class AttentionTaskGatedFusionMLP(nn.Module):
    """
    Experiment G: Adds CrossModalAttention before Task-Gated Fusion.
    
    Architecture:
        Shared encoders → 6 × 128d
        ↓
        CrossModalAttention (lets audio/text/metadata attend to each other)
        ↓
        Per-target gate  → weighted 768d fused repr
        Per-target head  → scalar prediction
    """

    def __init__(self, num_targets: int = 4, dropout_enc: float = 0.3, 
                 dropout_fusion: float = 0.5, metadata_dim: int = 30):
        super().__init__()
        self.encoders = _build_encoders(dropout_enc, metadata_dim=metadata_dim)
        self.num_targets = num_targets
        
        # Cross-modal attention layer
        self.cross_attn = CrossModalAttention(embed_dim=ENCODED_DIM, num_heads=4, dropout=dropout_enc)

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
            for _ in range(num_targets)
        ])

        # One scalar output per target
        self.task_heads = nn.ModuleList([
            nn.Linear(256, 1) for _ in range(num_targets)
        ])

    def forward(self, metadata, mpnet, vggish, mert, panns, mel_stats):
        raw_inputs = [metadata, mpnet, vggish, mert, panns, mel_stats]
        keys = ["metadata", "mpnet", "vggish", "mert", "panns", "mel_stats"]
        encs = [self.encoders[k](v) for k, v in zip(keys, raw_inputs)]

        # Presence mask (B, 6)
        presence = torch.stack([
            (x.abs().sum(dim=-1) > 0).float() for x in raw_inputs
        ], dim=-1)

        stacked = torch.stack(encs, dim=1)  # (B, 6, 128)
        
        # Apply cross-modal attention
        stacked = self.cross_attn(stacked, presence) # (B, 6, 128)
        
        # Summary for gating
        summary = stacked.mean(dim=1)  # (B, 128)

        preds = []
        for gate_net, fusion, head in zip(self.task_gates, self.task_fusions, self.task_heads):
            # Compute gate weights (B, 6)
            logits = gate_net(summary)
            # Mask out missing modalities (set logit to -inf)
            logits = logits + (1.0 - presence) * (-1e9)
            gates = torch.softmax(logits, dim=-1)
            
            # Apply gates
            gates_exp = gates.unsqueeze(-1)  # (B, 6, 1)
            gated = (stacked * gates_exp).view(-1, FUSED_DIM)  # (B, 768)

            preds.append(head(fusion(gated)))

        return torch.cat(preds, dim=-1)  # (B, 4)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
