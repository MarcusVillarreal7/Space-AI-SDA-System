"""
ConjunctionService — Periodic pairwise collision analysis.

Runs every N simulation ticks, pre-filters object pairs by distance,
runs the trained CollisionPredictor on filtered pairs, and stores
the top-20 riskiest conjunction pairs.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

COLLISION_CHECKPOINT = Path("checkpoints/collision_predictor/best_model.pt")


class ConjunctionService:
    """
    Periodic conjunction analysis service.

    Pre-filters pairs within 1000 km in same orbital regime,
    then runs the trained CollisionPredictor neural network.
    """

    def __init__(self, run_interval: int = 10):
        """
        Args:
            run_interval: Run analysis every N simulation ticks.
        """
        self._model = None
        self._model_loaded = False
        self._feature_extractor = None
        self._run_interval = run_interval
        self._tick_counter = 0
        self._top_pairs: list[dict] = []
        self._analyzed_count = 0
        self._last_run_iso = ""

    def _load_model(self) -> None:
        if self._model_loaded:
            return
        self._model_loaded = True
        if not COLLISION_CHECKPOINT.exists():
            logger.warning("Collision predictor checkpoint not found: %s", COLLISION_CHECKPOINT)
            return
        try:
            from src.ml.models.collision_predictor import CollisionPredictor
            checkpoint = torch.load(COLLISION_CHECKPOINT, map_location="cpu", weights_only=False)
            config = checkpoint.get("model_config")
            if config:
                self._model = CollisionPredictor.from_config(config)
            else:
                self._model = CollisionPredictor()
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.eval()

            from src.ml.features.trajectory_features import TrajectoryFeatureExtractor, FeatureConfig
            self._feature_extractor = TrajectoryFeatureExtractor(FeatureConfig(
                include_position=True,
                include_velocity=True,
                include_orbital_elements=True,
                include_derived_features=True,
                include_temporal_features=True,
                include_uncertainty=False,
            ))
            logger.info("ConjunctionService: CollisionPredictor loaded")
        except Exception:
            logger.exception("Failed to load CollisionPredictor")

    async def on_tick(self, timestep: int) -> None:
        """Called on each simulation tick. Runs analysis every N ticks."""
        self._tick_counter += 1
        if self._tick_counter % self._run_interval != 0:
            return
        self._analyze(timestep)

    def _analyze(self, timestep: int) -> None:
        """Run conjunction analysis for the current timestep."""
        self._load_model()

        from src.api.main import app_state
        catalog = app_state.get("catalog")
        if catalog is None or not catalog.is_loaded:
            return

        t0 = time.perf_counter()
        ts = min(timestep, catalog.n_timesteps - 1)

        # Get all positions at current timestep
        positions = catalog.positions[:, ts, :]  # (N, 3)
        velocities = catalog.velocities[:, ts, :]  # (N, 3)
        n = len(positions)

        # Pre-filter: compute pairwise distances using broadcasting
        # Only compute upper triangle to avoid O(N^2) memory
        # Chunk into regime groups for efficiency
        close_pairs = []
        threshold_km = 1000.0

        # Distance-based pre-filter using numpy broadcasting (chunked)
        chunk_size = 200
        for i_start in range(0, n, chunk_size):
            i_end = min(i_start + chunk_size, n)
            pos_i = positions[i_start:i_end]  # (chunk, 3)

            for j_start in range(i_start, n, chunk_size):
                j_end = min(j_start + chunk_size, n)
                pos_j = positions[j_start:j_end]  # (chunk2, 3)

                # Pairwise distances: (chunk, chunk2)
                diff = pos_i[:, np.newaxis, :] - pos_j[np.newaxis, :, :]  # (c1, c2, 3)
                dists = np.linalg.norm(diff, axis=2)  # (c1, c2)

                # Find close pairs (upper triangle only to avoid duplicates)
                for li in range(dists.shape[0]):
                    for lj in range(dists.shape[1]):
                        gi = i_start + li
                        gj = j_start + lj
                        if gi >= gj:
                            continue
                        if dists[li, lj] < threshold_km:
                            close_pairs.append((gi, gj, dists[li, lj]))

        self._analyzed_count = len(close_pairs)

        if not close_pairs or self._model is None:
            # Even without model, report distance-based results
            close_pairs.sort(key=lambda x: x[2])
            self._top_pairs = []
            for gi, gj, dist in close_pairs[:20]:
                oid1 = int(catalog.object_ids[gi])
                oid2 = int(catalog.object_ids[gj])
                rel_vel = np.linalg.norm(velocities[gj] - velocities[gi])
                self._top_pairs.append({
                    "object1_id": oid1,
                    "object1_name": catalog.object_names[gi],
                    "object2_id": oid2,
                    "object2_name": catalog.object_names[gj],
                    "risk_score": round(max(0, 1.0 - dist / threshold_km), 3),
                    "miss_distance_km": round(dist, 2),
                    "time_to_closest_approach_s": round(dist / (rel_vel + 1e-10), 1),
                })
            self._last_run_iso = datetime.utcnow().isoformat()
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("Conjunction analysis (distance-only): %d pairs, %d close, %.1fms",
                        n * (n - 1) // 2, len(close_pairs), elapsed)
            return

        # Extract features for close pair objects using the neural model
        unique_indices = sorted(set(i for p in close_pairs for i in (p[0], p[1])))
        idx_to_feature = {}

        for idx in unique_indices:
            # Use last 20 timesteps for features
            start_ts = max(0, ts - 19)
            pos_seq = catalog.positions[idx, start_ts:ts + 1]  # (<=20, 3)
            vel_seq = catalog.velocities[idx, start_ts:ts + 1]
            timestamps = np.arange(len(pos_seq)) * 60.0
            try:
                features = self._feature_extractor.extract_features(pos_seq, vel_seq, timestamps)
                idx_to_feature[idx] = features.mean(axis=0).astype(np.float32)
            except Exception:
                idx_to_feature[idx] = np.zeros(24, dtype=np.float32)

        # Batch predict with CollisionPredictor
        f1_batch = np.stack([idx_to_feature[p[0]] for p in close_pairs])
        f2_batch = np.stack([idx_to_feature[p[1]] for p in close_pairs])
        f1_batch = np.nan_to_num(f1_batch, nan=0.0, posinf=1e6, neginf=-1e6)
        f2_batch = np.nan_to_num(f2_batch, nan=0.0, posinf=1e6, neginf=-1e6)

        with torch.no_grad():
            t1 = torch.from_numpy(f1_batch)
            t2 = torch.from_numpy(f2_batch)
            output = self._model(t1, t2)  # (N, 3)
            risk_scores = output[:, 0].numpy()
            ttca_values = output[:, 1].numpy()
            miss_values = output[:, 2].numpy()

        # Build results sorted by risk
        results = []
        for k, (gi, gj, dist) in enumerate(close_pairs):
            oid1 = int(catalog.object_ids[gi])
            oid2 = int(catalog.object_ids[gj])
            results.append({
                "object1_id": oid1,
                "object1_name": catalog.object_names[gi],
                "object2_id": oid2,
                "object2_name": catalog.object_names[gj],
                "risk_score": round(float(risk_scores[k]), 3),
                "miss_distance_km": round(float(miss_values[k]), 2),
                "time_to_closest_approach_s": round(float(ttca_values[k]), 1),
            })

        results.sort(key=lambda x: x["risk_score"], reverse=True)
        self._top_pairs = results[:20]
        self._last_run_iso = datetime.utcnow().isoformat()

        elapsed = (time.perf_counter() - t0) * 1000
        top_risk = results[0]["risk_score"] if results else 0
        logger.info("Conjunction analysis: %d close pairs, top risk=%.3f, %.1fms",
                    len(close_pairs), top_risk, elapsed)

    def get_results(self) -> dict:
        """Get latest conjunction analysis results."""
        return {
            "pairs": self._top_pairs,
            "analyzed_pairs": self._analyzed_count,
            "timestamp": self._last_run_iso,
        }
