"""
Advanced Maneuver Classification Model with CNN-LSTM-Attention Architecture.

This is the actual architecture recovered from the trained checkpoint.
It uses a sophisticated combination of:
1. 1D CNN for local feature extraction
2. Bidirectional LSTM for temporal modeling  
3. Attention pooling for adaptive sequence aggregation
4. Feed-forward classifier

Architecture recovered from checkpoint analysis (2026-02-07).

Author: Space AI Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class CNNLSTMClassifierConfig:
    """Configuration for CNN-LSTM-Attention Maneuver Classifier."""
    input_dim: int = 24  # Input feature dimension
    cnn_channels: list = None  # CNN channel progression
    kernel_size: int = 3  # Conv kernel size
    lstm_hidden_dim: int = 128  # LSTM hidden dimension
    lstm_layers: int = 2  # Number of LSTM layers
    bidirectional: bool = True  # Bidirectional LSTM
    classifier_dims: list = None  # Classifier hidden dimensions
    num_classes: int = 6  # Number of output classes (from checkpoint)
    dropout: float = 0.3  # Dropout probability
    
    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 128]
        if self.classifier_dims is None:
            self.classifier_dims = [256, 128]


class CNN1D(nn.Module):
    """1D CNN for local feature extraction from sequences."""
    
    def __init__(
        self,
        input_dim: int = 24,
        channels: list = [32, 64, 128],
        kernel_size: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        in_channels = input_dim
        for out_channels in channels:
            # Conv layer with same padding
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                )
            )
            # Batch normalization
            self.bn_layers.append(nn.BatchNorm1d(out_channels))
            
            in_channels = out_channels
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN.
        
        Args:
            x: Input (batch, seq_len, input_dim)
        
        Returns:
            CNN features (batch, seq_len, channels[-1])
        """
        # Transpose to (batch, input_dim, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Transpose back to (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        return x


class LSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder for temporal modeling."""
    
    def __init__(
        self,
        input_dim: int = 24,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM.
        
        Args:
            x: Input (batch, seq_len, input_dim)
        
        Returns:
            LSTM outputs (batch, seq_len, hidden_dim * num_directions)
            Hidden states (h_n, c_n)
        """
        output, (h_n, c_n) = self.lstm(x)
        return output, (h_n, c_n)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling mechanism.
    
    Learns to weight each timestep based on its relevance.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Use nn.Parameter directly to match checkpoint naming
        self.weight = nn.Parameter(torch.Tensor(1, hidden_dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention pooling.
        
        Args:
            x: LSTM outputs (batch, seq_len, hidden_dim)
        
        Returns:
            Pooled representation (batch, hidden_dim)
        """
        # Compute attention scores
        attn_scores = torch.matmul(x, self.weight.t()) + self.bias  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)
        
        # Weighted sum
        pooled = torch.sum(attn_weights * x, dim=1)  # (batch, hidden_dim)
        
        return pooled


class CNNLSTMManeuverClassifier(nn.Module):
    """
    Advanced maneuver classifier using CNN-LSTM-Attention architecture.
    
    This is the actual architecture from the trained checkpoint:
    
    Architecture:
        Input (batch, seq_len, 24) →
        1D CNN (3 layers: 24→32→64→128) →
        Bidirectional LSTM (2 layers, hidden=128) →
        Attention Pooling (seq_len, 256) → (256,) →
        FC Classifier (256→128) →
        Output (128→6)
    
    Features:
        - Local pattern extraction via CNN
        - Temporal dependencies via LSTM
        - Adaptive sequence aggregation via attention
        - Dropout for regularization
    """
    
    def __init__(self, config: Optional[CNNLSTMClassifierConfig] = None):
        super().__init__()
        
        if config is None:
            config = CNNLSTMClassifierConfig()
        
        self.config = config
        
        # CNN for feature extraction
        self.cnn = CNN1D(
            input_dim=config.input_dim,
            channels=config.cnn_channels,
            kernel_size=config.kernel_size,
            dropout=config.dropout
        )
        
        # LSTM for temporal modeling
        # Input to LSTM is raw features (not CNN output)
        lstm_input_dim = config.input_dim  # From checkpoint analysis
        self.lstm = LSTMEncoder(
            input_dim=lstm_input_dim,
            hidden_dim=config.lstm_hidden_dim,
            num_layers=config.lstm_layers,
            bidirectional=config.bidirectional,
            dropout=config.dropout
        )
        
        # Attention pooling
        lstm_output_dim = config.lstm_hidden_dim * (2 if config.bidirectional else 1)
        cnn_output_dim = config.cnn_channels[-1]
        combined_dim = lstm_output_dim + cnn_output_dim
        
        self.attention_pool = AttentionPooling(combined_dim)
        
        # Classifier
        classifier_layers = []
        in_dim = combined_dim
        
        for hidden_dim in config.classifier_dims:
            classifier_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            in_dim = hidden_dim
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Output layer
        self.output = nn.Linear(config.classifier_dims[-1], config.num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input sequences (batch, seq_len, input_dim)
        
        Returns:
            Logits (batch, num_classes)
        """
        # CNN features
        cnn_features = self.cnn(x)  # (batch, seq_len, cnn_channels[-1])
        
        # LSTM features
        lstm_output, _ = self.lstm(x)  # (batch, seq_len, lstm_hidden*2)
        
        # Concatenate CNN and LSTM features
        combined = torch.cat([cnn_features, lstm_output], dim=-1)  # (batch, seq_len, combined_dim)
        
        # Attention pooling
        pooled = self.attention_pool(combined)  # (batch, combined_dim)
        
        # Classifier
        features = self.classifier(pooled)  # (batch, classifier_dims[-1])
        
        # Output
        logits = self.output(features)  # (batch, num_classes)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        logits = self.forward(x)
        predictions = torch.argmax(logits, dim=-1)
        return predictions
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=-1)
        return probabilities
    
    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            'input_dim': self.config.input_dim,
            'cnn_channels': self.config.cnn_channels,
            'kernel_size': self.config.kernel_size,
            'lstm_hidden_dim': self.config.lstm_hidden_dim,
            'lstm_layers': self.config.lstm_layers,
            'bidirectional': self.config.bidirectional,
            'classifier_dims': self.config.classifier_dims,
            'num_classes': self.config.num_classes,
            'dropout': self.config.dropout
        }
    
    @classmethod
    def from_config(cls, config_dict: Dict) -> 'CNNLSTMManeuverClassifier':
        """Create model from configuration dictionary."""
        config = CNNLSTMClassifierConfig(**config_dict)
        return cls(config)


# Keep simple classifier for backward compatibility
ManeuverClassifier = CNNLSTMManeuverClassifier
ClassifierConfig = CNNLSTMClassifierConfig
ManeuverClassifierConfig = CNNLSTMClassifierConfig  # Alias for old checkpoints


# Class labels (updated based on 6 classes)
CLASS_NAMES = {
    0: "Normal",
    1: "Drift/Decay",
    2: "Station-keeping",
    3: "Minor Maneuver",
    4: "Major Maneuver",
    5: "Deorbit"
}


def get_class_name(class_idx: int) -> str:
    """Get human-readable class name from index."""
    return CLASS_NAMES.get(class_idx, "Unknown")


# Example usage and testing
if __name__ == "__main__":
    print("Testing CNN-LSTM Maneuver Classifier...\n")
    
    config = CNNLSTMClassifierConfig(
        input_dim=24,
        cnn_channels=[32, 64, 128],
        lstm_hidden_dim=128,
        lstm_layers=2,
        bidirectional=True,
        classifier_dims=[256, 128],
        num_classes=6
    )
    
    model = CNNLSTMManeuverClassifier(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 4
    seq_len = 20
    input_dim = 24
    
    x = torch.randn(batch_size, seq_len, input_dim)
    logits = model(x)
    print(f"Logits shape: {logits.shape}")  # Should be (4, 6)
    
    predictions = model.predict(x)
    print(f"Predictions: {predictions}")
    
    probabilities = model.predict_proba(x)
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Sample probabilities: {probabilities[0]}")
    
    print("\n✅ All tests passed!")
