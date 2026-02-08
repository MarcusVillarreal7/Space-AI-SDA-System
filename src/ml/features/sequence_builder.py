"""Build sequences from trajectory features for ML models."""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler


@dataclass
class SequenceConfig:
    """Configuration for sequence building."""
    history_length: int = 20  # Input sequence length (timesteps)
    prediction_horizon: int = 30  # How many steps to predict ahead
    stride: int = 5  # Sliding window stride
    padding: str = 'zero'  # 'zero' or 'edge' or 'none'
    normalize: bool = True  # Whether to normalize features
    normalization_method: str = 'standard'  # 'standard' or 'minmax'


class TrajectorySequenceBuilder:
    """Build sequences from trajectory features using sliding windows."""
    
    def __init__(self, config: Optional[SequenceConfig] = None):
        self.config = config or SequenceConfig()
        self.scaler = None
        
    def build_sequences(self, features: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Build input-output sequences from features using sliding window.
        
        Args:
            features: (T, D) array of features over time
        
        Returns:
            Dictionary with:
                - 'history': (N, seq_len, D) input sequences
                - 'target': (N, pred_horizon, D) target sequences  
                - 'mask': (N, seq_len) validity mask (1=valid, 0=padding)
        """
        T, D = features.shape
        seq_len = self.config.history_length
        pred_horizon = self.config.prediction_horizon
        stride = self.config.stride
        
        # Normalize if requested
        if self.config.normalize:
            if self.scaler is None:
                if self.config.normalization_method == 'standard':
                    self.scaler = StandardScaler()
                else:
                    self.scaler = MinMaxScaler()
                features = self.scaler.fit_transform(features)
            else:
                features = self.scaler.transform(features)
        
        # Generate sequences using sliding window
        history_list = []
        target_list = []
        mask_list = []
        
        for i in range(0, T - seq_len - pred_horizon + 1, stride):
            # Input sequence
            hist_seq = features[i:i + seq_len]
            
            # Target sequence (future values)
            tgt_seq = features[i + seq_len:i + seq_len + pred_horizon]
            
            # Mask (all valid for now)
            mask = np.ones(seq_len)
            
            history_list.append(hist_seq)
            target_list.append(tgt_seq)
            mask_list.append(mask)
        
        # Handle padding if needed
        if len(history_list) == 0 and self.config.padding != 'none':
            # Not enough data, use padding
            if T >= seq_len:
                hist_seq = features[:seq_len]
                mask = np.ones(seq_len)
            elif T > 0:
                # Pad to seq_len
                if self.config.padding == 'zero':
                    hist_seq = np.zeros((seq_len, D))
                    hist_seq[:T] = features
                else:  # edge
                    hist_seq = np.repeat(features[[-1]], seq_len, axis=0)
                    hist_seq[:T] = features
                mask = np.zeros(seq_len)
                mask[:T] = 1
            else:
                # No data at all
                hist_seq = np.zeros((seq_len, D))
                mask = np.zeros(seq_len)
            
            # For target, just pad with zeros
            tgt_seq = np.zeros((pred_horizon, D))
            
            history_list.append(hist_seq)
            target_list.append(tgt_seq)
            mask_list.append(mask)
        
        if len(history_list) == 0:
            # Return empty tensors
            return {
                'history': torch.zeros((0, seq_len, D), dtype=torch.float32),
                'target': torch.zeros((0, pred_horizon, D), dtype=torch.float32),
                'mask': torch.zeros((0, seq_len), dtype=torch.float32)
            }
        
        # Convert to tensors
        history = torch.from_numpy(np.array(history_list)).float()
        target = torch.from_numpy(np.array(target_list)).float()
        mask = torch.from_numpy(np.array(mask_list)).float()
        
        return {
            'history': history,
            'target': target,
            'mask': mask
        }
