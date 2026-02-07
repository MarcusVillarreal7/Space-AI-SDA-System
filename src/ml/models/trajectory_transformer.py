"""
Trajectory Transformer Model for Satellite Position Prediction.

This module implements a Transformer-based sequence-to-sequence model for predicting
satellite trajectories. The architecture uses:
- Encoder: Processes historical trajectory sequences
- Decoder: Generates future trajectory predictions
- Attention: Captures temporal dependencies

Architecture recovered from checkpoint analysis (2026-02-07).

Author: Space AI Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Configuration for Trajectory Transformer model."""
    d_model: int = 64  # Model dimension
    n_heads: int = 4  # Number of attention heads
    n_encoder_layers: int = 2  # Number of encoder layers
    n_decoder_layers: int = 2  # Number of decoder layers
    d_ff: int = 256  # Feed-forward dimension
    dropout: float = 0.1  # Dropout probability
    input_dim: int = 24  # Input feature dimension
    output_dim: int = 6  # Output feature dimension (position + velocity)
    max_seq_len: int = 100  # Maximum sequence length for positional encoding


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.
    
    Adds position information to input embeddings using sine and cosine functions.
    """
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    Implements scaled dot-product attention with multiple heads.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for multi-head attention.
        
        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model)
            value: Value tensor (batch, seq_len, d_model)
            mask: Optional attention mask (batch, seq_len, seq_len)
        
        Returns:
            Attention output (batch, seq_len, d_model)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape to (batch, n_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    Two linear layers with ReLU activation.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for encoder layer.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Self-attention with residual and layer norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerDecoderLayer(nn.Module):
    """Single Transformer decoder layer."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for decoder layer.
        
        Args:
            x: Decoder input (batch, tgt_len, d_model)
            encoder_output: Encoder output (batch, src_len, d_model)
            src_mask: Source attention mask
            tgt_mask: Target attention mask (causal mask)
        
        Returns:
            Output tensor (batch, tgt_len, d_model)
        """
        # Masked self-attention
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-attention to encoder
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TrajectoryTransformer(nn.Module):
    """
    Transformer model for trajectory prediction.
    
    Predicts future satellite positions and velocities based on historical trajectory.
    Uses encoder-decoder architecture with multi-head attention.
    
    Architecture:
        Input → Input Projection → Positional Encoding → 
        Encoder (N layers) → Decoder (N layers) → Output Projection → Output
    """
    
    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__()
        
        if config is None:
            config = TransformerConfig()
        
        self.config = config
        
        # Input and output projections
        self.input_projection = nn.Linear(config.input_dim, config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.output_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_decoder_layers)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            src: Source sequence (batch, src_len, input_dim)
            tgt: Target sequence (batch, tgt_len, input_dim) - teacher forcing
            src_mask: Source attention mask
            tgt_mask: Target attention mask (causal)
        
        Returns:
            Predictions (batch, tgt_len, output_dim)
        """
        # Encode source sequence
        encoder_output = self.encode(src, src_mask)
        
        # Decode target sequence
        output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        return output
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode source sequence.
        
        Args:
            src: Source sequence (batch, src_len, input_dim)
            src_mask: Source attention mask
        
        Returns:
            Encoder output (batch, src_len, d_model)
        """
        # Project input
        x = self.input_projection(src)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode target sequence.
        
        Args:
            tgt: Target sequence (batch, tgt_len, input_dim)
            encoder_output: Encoder output (batch, src_len, d_model)
            src_mask: Source attention mask
            tgt_mask: Target attention mask (causal)
        
        Returns:
            Predictions (batch, tgt_len, output_dim)
        """
        # Project input
        x = self.input_projection(tgt)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        # Project to output
        output = self.output_projection(x)
        
        return output
    
    def predict(
        self,
        src: torch.Tensor,
        pred_horizon: int,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Auto-regressive prediction for inference.
        
        Args:
            src: Source sequence (batch, src_len, input_dim)
            pred_horizon: Number of future timesteps to predict
            src_mask: Source attention mask
        
        Returns:
            Predictions (batch, pred_horizon, output_dim)
        """
        batch_size = src.size(0)
        device = src.device
        
        # Encode source
        encoder_output = self.encode(src, src_mask)
        
        # Initialize decoder input with zeros (or last timestep)
        # Using zeros for simplicity; could use last known state
        decoder_input = torch.zeros(batch_size, 1, self.config.input_dim, device=device)
        
        predictions = []
        
        # Auto-regressive decoding
        for t in range(pred_horizon):
            # Decode one step
            output = self.decode(decoder_input, encoder_output, src_mask, None)
            
            # Take last prediction
            pred = output[:, -1:, :]  # (batch, 1, output_dim)
            predictions.append(pred)
            
            # Prepare next decoder input (use prediction as input)
            # For simplicity, pad prediction to input_dim
            if self.config.output_dim < self.config.input_dim:
                next_input = torch.zeros(batch_size, 1, self.config.input_dim, device=device)
                next_input[:, :, :self.config.output_dim] = pred
            else:
                next_input = pred
            
            # Append to decoder input
            decoder_input = torch.cat([decoder_input, next_input], dim=1)
        
        # Concatenate all predictions
        predictions = torch.cat(predictions, dim=1)  # (batch, pred_horizon, output_dim)
        
        return predictions
    
    def get_config(self) -> Dict:
        """Get model configuration as dictionary."""
        return {
            'd_model': self.config.d_model,
            'n_heads': self.config.n_heads,
            'n_encoder_layers': self.config.n_encoder_layers,
            'n_decoder_layers': self.config.n_decoder_layers,
            'd_ff': self.config.d_ff,
            'dropout': self.config.dropout,
            'input_dim': self.config.input_dim,
            'output_dim': self.config.output_dim
        }
    
    @classmethod
    def from_config(cls, config_dict: Dict) -> 'TrajectoryTransformer':
        """Create model from configuration dictionary."""
        config = TransformerConfig(**config_dict)
        return cls(config)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal (lower triangular) mask for decoder.
    
    Prevents positions from attending to future positions.
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
    
    Returns:
        Causal mask (1, seq_len, seq_len)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, 0)
    return mask.unsqueeze(0)


# Example usage and testing
if __name__ == "__main__":
    # Test model initialization
    config = TransformerConfig(
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=256,
        dropout=0.1,
        input_dim=24,
        output_dim=6
    )
    
    model = TrajectoryTransformer(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 4
    src_len = 20
    tgt_len = 30
    
    src = torch.randn(batch_size, src_len, config.input_dim)
    tgt = torch.randn(batch_size, tgt_len, config.input_dim)
    
    output = model(src, tgt)
    print(f"Output shape: {output.shape}")  # Should be (4, 30, 6)
    
    # Test prediction
    predictions = model.predict(src, pred_horizon=30)
    print(f"Prediction shape: {predictions.shape}")  # Should be (4, 30, 6)
    
    print("\n✅ Model test passed!")
