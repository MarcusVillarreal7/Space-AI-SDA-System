"""
Behavior Autoencoder for Anomaly Detection.

A symmetric autoencoder that compresses 19D behavioral feature vectors
into a 6D latent space.  Objects whose reconstruction error exceeds a
learned threshold are flagged as anomalous.

Architecture:
    Encoder: 19 → 32 → 16 → 6  (latent)
    Decoder:  6 → 16 → 32 → 19

Activation: LeakyReLU  |  Normalization: LayerNorm  |  Regularization: Dropout

Author: Space AI Team
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.ml.anomaly.behavior_features import FEATURE_DIM


@dataclass
class AutoencoderConfig:
    """Hyperparameters for BehaviorAutoencoder."""
    input_dim: int = FEATURE_DIM        # 19
    hidden_dims: tuple = (32, 16)       # encoder hidden layers
    latent_dim: int = 6
    dropout: float = 0.1
    negative_slope: float = 0.01        # LeakyReLU

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "latent_dim": self.latent_dim,
            "dropout": self.dropout,
            "negative_slope": self.negative_slope,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AutoencoderConfig":
        d = dict(d)
        if "hidden_dims" in d:
            d["hidden_dims"] = tuple(d["hidden_dims"])
        return cls(**d)


class BehaviorAutoencoder(nn.Module):
    """
    Symmetric autoencoder for behavioral anomaly detection.

    Forward returns ``(reconstruction, latent)`` so callers can inspect
    both the reconstruction error and the latent representation.
    """

    def __init__(self, config: AutoencoderConfig | None = None):
        super().__init__()
        self.config = config or AutoencoderConfig()
        c = self.config

        # --- Encoder ---
        encoder_layers = []
        in_dim = c.input_dim
        for h in c.hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.LeakyReLU(c.negative_slope),
                nn.Dropout(c.dropout),
            ])
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, c.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Decoder (mirror of encoder) ---
        decoder_layers = []
        in_dim = c.latent_dim
        for h in reversed(c.hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.LeakyReLU(c.negative_slope),
                nn.Dropout(c.dropout),
            ])
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, c.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim) feature vectors.

        Returns:
            (reconstruction, latent) both as Tensors.
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent representation only."""
        return self.encoder(x)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error, shape (batch,)."""
        self.eval()
        with torch.no_grad():
            recon, _ = self.forward(x)
            err = ((x - recon) ** 2).mean(dim=1)
        return err

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_config(self) -> dict:
        return self.config.to_dict()

    @classmethod
    def from_config(cls, config_dict: dict) -> "BehaviorAutoencoder":
        return cls(AutoencoderConfig.from_dict(config_dict))
