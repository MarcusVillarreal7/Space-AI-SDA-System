"""
Anomaly Detector — High-Level API.

Wraps the BehaviorAutoencoder and BehaviorFeatureExtractor into a single
``AnomalyDetector`` that can fit on normal data, score new observations,
and persist/load models to disk.

Workflow:
    1. ``fit(profiles)``   — train autoencoder, compute threshold
    2. ``score(profile)``  — return anomaly score (reconstruction error)
    3. ``detect(profile)`` — return AnomalyResult with score, is_anomaly, explanation

Author: Space AI Team
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.ml.anomaly.autoencoder import AutoencoderConfig, BehaviorAutoencoder
from src.ml.anomaly.behavior_features import BehaviorProfile, FEATURE_DIM
from src.ml.anomaly.explainer import AnomalyExplainer


@dataclass
class AnomalyResult:
    """Output of the anomaly detector."""
    object_id: str
    anomaly_score: float          # Reconstruction error (MSE)
    threshold: float              # Learned threshold
    is_anomaly: bool
    percentile: float             # Where score falls in training distribution (0-100)
    top_features: List[str]       # Features with highest reconstruction error
    explanation: str              # Human-readable explanation


class AnomalyDetector:
    """
    End-to-end anomaly detector for satellite behavioral profiles.

    Args:
        config: Autoencoder hyperparameters.
        threshold_percentile: Percentile of training reconstruction errors
            used as the anomaly threshold (default 95th percentile).
        device: Torch device (auto-detected if None).
    """

    def __init__(
        self,
        config: AutoencoderConfig | None = None,
        threshold_percentile: float = 95.0,
        device: Optional[str] = None,
    ):
        self.config = config or AutoencoderConfig()
        self.model = BehaviorAutoencoder(self.config)
        self.threshold_percentile = threshold_percentile
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Learned during fit()
        self.threshold: float = 0.0
        self.training_errors: Optional[np.ndarray] = None
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self._fitted = False

        self.explainer = AnomalyExplainer()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        profiles: List[BehaviorProfile],
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        verbose: bool = False,
    ) -> dict:
        """
        Train the autoencoder on a collection of *normal* behavior profiles.

        Args:
            profiles: List of BehaviorProfile (assumed to be non-anomalous).
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            lr: Learning rate.
            verbose: Print per-epoch loss.

        Returns:
            Dictionary with training metrics (final_loss, threshold, param_count).
        """
        features = np.stack([p.features for p in profiles])

        # Compute normalization stats
        self.feature_means = features.mean(axis=0)
        self.feature_stds = features.std(axis=0) + 1e-8
        features_norm = (features - self.feature_means) / self.feature_stds

        tensor = torch.tensor(features_norm, dtype=torch.float32)
        dataset = TensorDataset(tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        final_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                recon, _ = self.model(batch)
                loss = criterion(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)

            epoch_loss /= len(dataset)
            final_loss = epoch_loss
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}  loss={epoch_loss:.6f}")

        # Compute threshold from training data
        self.model.eval()
        with torch.no_grad():
            all_tensor = tensor.to(self.device)
            recon, _ = self.model(all_tensor)
            errors = ((all_tensor - recon) ** 2).mean(dim=1).cpu().numpy()

        self.training_errors = errors
        self.threshold = float(np.percentile(errors, self.threshold_percentile))
        self._fitted = True

        return {
            "final_loss": final_loss,
            "threshold": self.threshold,
            "param_count": self.model.param_count(),
            "n_training": len(profiles),
            "threshold_percentile": self.threshold_percentile,
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, profile: BehaviorProfile) -> float:
        """Return the anomaly score (reconstruction MSE) for a single profile."""
        self._check_fitted()
        x = self._normalize(profile.features)
        tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
        return float(self.model.reconstruction_error(tensor).item())

    def score_batch(self, profiles: List[BehaviorProfile]) -> np.ndarray:
        """Return anomaly scores for a batch of profiles."""
        self._check_fitted()
        features = np.stack([self._normalize(p.features) for p in profiles])
        tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        return self.model.reconstruction_error(tensor).cpu().numpy()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, profile: BehaviorProfile) -> AnomalyResult:
        """
        Score a profile and return a full AnomalyResult with explanation.
        """
        self._check_fitted()

        anomaly_score = self.score(profile)
        is_anomaly = anomaly_score > self.threshold

        # Percentile in training distribution
        if self.training_errors is not None and len(self.training_errors) > 0:
            percentile = float(
                (self.training_errors < anomaly_score).mean() * 100.0
            )
        else:
            percentile = 0.0

        # Per-feature reconstruction error for explainability
        x = self._normalize(profile.features)
        tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            recon, _ = self.model(tensor)
            per_feature_err = ((tensor - recon) ** 2).squeeze(0).cpu().numpy()

        top_features = self.explainer.top_features(per_feature_err, n=3)
        explanation = self.explainer.explain(
            anomaly_score=anomaly_score,
            threshold=self.threshold,
            is_anomaly=is_anomaly,
            percentile=percentile,
            per_feature_error=per_feature_err,
            profile=profile,
        )

        return AnomalyResult(
            object_id=profile.object_id,
            anomaly_score=anomaly_score,
            threshold=self.threshold,
            is_anomaly=is_anomaly,
            percentile=percentile,
            top_features=top_features,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model weights, config, and normalization stats."""
        self._check_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), path / "autoencoder.pt")

        meta = {
            "config": self.config.to_dict(),
            "threshold": self.threshold,
            "threshold_percentile": self.threshold_percentile,
            "feature_means": self.feature_means.tolist(),
            "feature_stds": self.feature_stds.tolist(),
        }
        with open(path / "anomaly_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        if self.training_errors is not None:
            np.save(path / "training_errors.npy", self.training_errors)

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "AnomalyDetector":
        """Load a fitted AnomalyDetector from disk."""
        path = Path(path)

        with open(path / "anomaly_meta.json") as f:
            meta = json.load(f)

        config = AutoencoderConfig.from_dict(meta["config"])
        detector = cls(
            config=config,
            threshold_percentile=meta["threshold_percentile"],
            device=device,
        )
        detector.model.load_state_dict(
            torch.load(path / "autoencoder.pt", map_location=detector.device, weights_only=True)
        )
        detector.threshold = meta["threshold"]
        detector.feature_means = np.array(meta["feature_means"], dtype=np.float32)
        detector.feature_stds = np.array(meta["feature_stds"], dtype=np.float32)

        errors_path = path / "training_errors.npy"
        if errors_path.exists():
            detector.training_errors = np.load(errors_path)

        detector._fitted = True
        return detector

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Z-score normalize using training statistics."""
        return (features - self.feature_means) / self.feature_stds

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "AnomalyDetector has not been fitted. Call fit() first."
            )
