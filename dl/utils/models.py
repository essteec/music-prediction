"""
Neural network architectures for music prediction.
"""

import torch
import torch.nn as nn


class MusicMLP(nn.Module):
    """
    Multi-Layer Perceptron for music attribute prediction.
    Based on HitMusicNet's MusicPopNet architecture.
    
    Architecture:
        Input (414) → FC1 (207) → ReLU → Dropout
                   → FC2 (138) → ReLU → Dropout  
                   → FC3 (4)   → (No activation - raw regression)
    
    Note: No final activation to allow predicting different ranges:
        - valence, energy, danceability: [0, 1] (raw values)
        - popularity: [0, ~4.6] (log1p transformed)
    """
    
    def __init__(self, input_size=414, num_targets=4, dropout=0.5):
        super(MusicMLP, self).__init__()
        
        self.input_size = input_size
        self.num_targets = num_targets
        
        # Layer sizes from HitMusicNet paper
        hidden1 = input_size // 2      # 414 → 207
        hidden2 = input_size // 3      # 414 → 138
        
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_targets)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output layer (no activation - raw regression)
        x = self.fc3(x)
        
        return x
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleXORNetwork(nn.Module):
    """
    Simple network for XOR problem (learning exercise).
    
    Architecture:
        Input (2) → Hidden (8) → ReLU → Output (1) → Sigmoid
    """
    
    def __init__(self):
        super(SimpleXORNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 8)  # More hidden units for XOR
        self.fc2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    # Test model creation
    print("Testing MusicMLP...")
    model = MusicMLP(input_size=414, num_targets=4)
    print(f"  Input size: {model.input_size}")
    print(f"  Output size: {model.num_targets}")
    print(f"  Parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(32, 414)  # Batch of 32
    y = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Output range: [{y.min().item():.3f}, {y.max().item():.3f}]")
    
    print("\n✓ Model test successful!")
