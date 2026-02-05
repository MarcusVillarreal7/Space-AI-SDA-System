"""
Multi-Object Tracker for Space Domain Awareness.

This module provides the main tracking pipeline that orchestrates all components:
- Kalman filters (EKF/UKF)
- Data association (Hungarian/GNN)
- Track management
- Maneuver detection

Classes:
    TrackerConfig: Configuration dataclass
    MultiObjectTracker: Main tracking orchestrator

References:
    - Bar-Shalom, Y. "Multitarget-Multisensor Tracking"
    - Blackman, S. "Multiple-Target Tracking with Radar Applications"
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time

from src.tracking.kalman_filters import ExtendedKalmanFilter, UnscentedKalmanFilter
from src.tracking.data_association import (
    Measurement, HungarianAssociator, GNNAssociator, Association
)
from src.tracking.track_manager import TrackManager, Track, TrackState
from src.tracking.maneuver_detection import InnovationDetector, ManeuverEvent
from src.utils.logging_config import get_logger

logger = get_logger("tracking.mot")


@dataclass
class TrackerConfig:
    """
    Configuration for MultiObjectTracker.
    
    Attributes:
        filter_type: 'ekf' or 'ukf'
        association_method: 'hungarian' or 'gnn'
        gate_threshold: Chi-square gating threshold
        max_association_cost: Maximum cost for valid association
        confirmation_threshold: Hits needed to confirm track
        deletion_threshold: Misses before track deletion
        coast_threshold: Misses before entering coast state
        max_coast_time: Maximum time in coast state (seconds)
        maneuver_detection: Enable maneuver detection
        maneuver_threshold: Chi-square threshold for maneuver detection
        process_noise_scale: Scale factor for process noise
        measurement_noise_scale: Scale factor for measurement noise
    """
    filter_type: str = "ukf"
    association_method: str = "hungarian"
    gate_threshold: float = 9.0
    max_association_cost: float = 100.0
    confirmation_threshold: int = 3
    deletion_threshold: int = 5
    coast_threshold: int = 3
    max_coast_time: float = 300.0
    maneuver_detection: bool = True
    maneuver_threshold: float = 13.8
    process_noise_scale: float = 1.0
    measurement_noise_scale: float = 1.0


class MultiObjectTracker:
    """
    Multi-object tracker for space objects.
    
    Orchestrates the complete tracking pipeline:
    1. Predict all tracks forward in time
    2. Associate measurements to tracks
    3. Update tracks with measurements
    4. Detect maneuvers
    5. Initialize new tracks
    6. Delete lost tracks
    """
    
    def __init__(self, config: Optional[TrackerConfig] = None):
        """
        Initialize multi-object tracker.
        
        Args:
            config: Tracker configuration (uses defaults if None)
        """
        self.config = config or TrackerConfig()
        
        # Initialize components
        self.track_manager = TrackManager(
            filter_type=self.config.filter_type,
            confirmation_threshold=self.config.confirmation_threshold,
            deletion_threshold=self.config.deletion_threshold,
            coast_threshold=self.config.coast_threshold,
            max_coast_time=self.config.max_coast_time,
            process_noise_std=self.config.process_noise_scale * 0.01
        )
        
        # Data association
        if self.config.association_method == "hungarian":
            self.associator = HungarianAssociator(
                gate_threshold=self.config.gate_threshold,
                max_cost=self.config.max_association_cost
            )
        elif self.config.association_method == "gnn":
            self.associator = GNNAssociator(
                gate_threshold=self.config.gate_threshold,
                max_cost=self.config.max_association_cost
            )
        else:
            raise ValueError(f"Unknown association method: {self.config.association_method}")
        
        # Maneuver detection
        if self.config.maneuver_detection:
            self.maneuver_detector = InnovationDetector(
                threshold=self.config.maneuver_threshold
            )
        else:
            self.maneuver_detector = None
        
        # Statistics
        self.current_time = 0.0
        self.last_update_time: Optional[float] = None
        self.update_count = 0
        self.total_measurements = 0
        self.total_associations = 0
        self.maneuver_events: List[ManeuverEvent] = []
        
        logger.info(
            f"MultiObjectTracker initialized: "
            f"filter={self.config.filter_type}, "
            f"association={self.config.association_method}, "
            f"maneuver_detection={self.config.maneuver_detection}"
        )
    
    def update(
        self,
        measurements: List[Measurement],
        timestamp: float
    ) -> List[Track]:
        """
        Update tracker with new measurements.
        
        This is the main tracking pipeline that:
        1. Predicts all tracks forward
        2. Associates measurements to tracks
        3. Updates matched tracks
        4. Detects maneuvers
        5. Initializes new tracks
        6. Prunes old tracks
        
        Args:
            measurements: List of sensor measurements
            timestamp: Current time
            
        Returns:
            List of all active tracks
        """
        start_time = time.time()
        
        # Update statistics
        self.current_time = timestamp
        self.update_count += 1
        self.total_measurements += len(measurements)
        
        logger.debug(
            f"Update {self.update_count}: {len(measurements)} measurements, "
            f"{len(self.track_manager.tracks)} tracks"
        )
        
        # Step 1: Predict all tracks to current time
        dt = timestamp - self.last_update_time if self.last_update_time is not None else 0.0
        if dt > 0:
            for track_id in list(self.track_manager.tracks.keys()):
                self.track_manager.predict_track(track_id, dt)
        
        self.last_update_time = timestamp
        
        # Step 2: Data association
        associations, unassociated_tracks, unassociated_measurements = self._associate(
            measurements, timestamp
        )
        
        self.total_associations += len(associations)
        
        logger.debug(
            f"Association: {len(associations)} matched, "
            f"{len(unassociated_tracks)} unmatched tracks, "
            f"{len(unassociated_measurements)} unmatched measurements"
        )
        
        # Step 3: Update matched tracks
        for assoc in associations:
            track = self.track_manager.tracks.get(assoc.track_id)
            if track is None:
                continue
            
            # Get measurement
            meas = next((m for m in measurements if m.measurement_id == assoc.measurement_id), None)
            if meas is None:
                continue
            
            # Update track
            self.track_manager.update_track(assoc.track_id, meas)
            
            # Step 4: Maneuver detection
            if self.maneuver_detector is not None:
                # Compute innovation for maneuver detection
                predicted_pos = track.get_position()
                innovation = meas.position - predicted_pos
                
                # Get innovation covariance (S = H P H^T + R)
                P = track.filter.covariance
                H = np.eye(3, 6)  # Measurement matrix (position only)
                S = H @ P @ H.T + meas.covariance
                
                # Check for maneuver
                maneuver_event = self.maneuver_detector.detect(
                    innovation, S, timestamp, assoc.track_id
                )
                
                if maneuver_event is not None:
                    self.maneuver_events.append(maneuver_event)
                    track.is_maneuvering = True
                    
                    # Increase process noise for maneuvering track
                    if hasattr(track.filter, 'Q'):
                        track.filter.Q *= 10.0
                else:
                    # Reset process noise if no longer maneuvering
                    if track.is_maneuvering:
                        track.is_maneuvering = False
                        if hasattr(track.filter, 'Q'):
                            track.filter.Q /= 10.0
        
        # Step 5: Handle unassociated measurements (initialize new tracks)
        for meas_id in unassociated_measurements:
            meas = next((m for m in measurements if m.measurement_id == meas_id), None)
            if meas is not None:
                new_track = self.track_manager.create_track(meas)
                logger.debug(f"Initialized new track {new_track.track_id}")
        
        # Step 6: Prune old tracks
        deleted_tracks = self.track_manager.prune_tracks(timestamp)
        if deleted_tracks:
            logger.debug(f"Deleted {len(deleted_tracks)} tracks")
            
            # Clean up maneuver detector
            if self.maneuver_detector is not None:
                for track_id in deleted_tracks:
                    self.maneuver_detector.reset(track_id)
        
        # Update timing
        elapsed = time.time() - start_time
        logger.debug(f"Update completed in {elapsed*1000:.2f}ms")
        
        return list(self.track_manager.tracks.values())
    
    def _associate(
        self,
        measurements: List[Measurement],
        timestamp: float
    ) -> Tuple[List[Association], List[int], List[int]]:
        """
        Associate measurements to tracks.
        
        Args:
            measurements: List of measurements
            timestamp: Current time
            
        Returns:
            Tuple of (associations, unassociated_track_ids, unassociated_measurement_ids)
        """
        if not measurements:
            # No measurements - all tracks unassociated
            track_ids = list(self.track_manager.tracks.keys())
            return [], track_ids, []
        
        if not self.track_manager.tracks:
            # No tracks - all measurements unassociated
            meas_ids = [m.measurement_id for m in measurements]
            return [], [], meas_ids
        
        # Get track predictions
        track_predictions = []
        for track_id, track in self.track_manager.tracks.items():
            pred_pos = track.get_position()
            pred_cov = track.get_position_covariance()  # Position covariance only
            track_predictions.append((track_id, pred_pos, pred_cov))
        
        # Run association
        associations, unassoc_tracks, unassoc_meas = self.associator.associate(
            track_predictions, measurements
        )
        
        return associations, unassoc_tracks, unassoc_meas
    
    def get_confirmed_tracks(self) -> List[Track]:
        """Get all confirmed tracks."""
        return self.track_manager.get_confirmed_tracks()
    
    def get_track(self, track_id: int) -> Optional[Track]:
        """Get a specific track by ID."""
        return self.track_manager.tracks.get(track_id)
    
    def get_statistics(self) -> dict:
        """
        Get tracker statistics.
        
        Returns:
            Dictionary of performance metrics
        """
        track_stats = self.track_manager.get_statistics()
        
        stats = {
            'update_count': self.update_count,
            'current_time': self.current_time,
            'total_measurements': self.total_measurements,
            'total_associations': self.total_associations,
            'association_rate': self.total_associations / max(1, self.total_measurements),
            'maneuver_events': len(self.maneuver_events),
            **track_stats
        }
        
        # Add per-update averages
        if self.update_count > 0:
            stats['avg_measurements_per_update'] = self.total_measurements / self.update_count
            stats['avg_tracks_per_update'] = track_stats.get('total_tracks', 0)
        
        return stats
    
    def get_maneuver_events(self) -> List[ManeuverEvent]:
        """Get all detected maneuver events."""
        return self.maneuver_events.copy()
    
    def reset(self):
        """Reset tracker to initial state."""
        self.track_manager = TrackManager(
            filter_type=self.config.filter_type,
            confirmation_threshold=self.config.confirmation_threshold,
            deletion_threshold=self.config.deletion_threshold,
            coast_threshold=self.config.coast_threshold,
            max_coast_time=self.config.max_coast_time
        )
        
        if self.maneuver_detector is not None:
            self.maneuver_detector = InnovationDetector(
                threshold=self.config.maneuver_threshold
            )
        
        self.current_time = 0.0
        self.last_update_time = None
        self.update_count = 0
        self.total_measurements = 0
        self.total_associations = 0
        self.maneuver_events = []
        
        logger.info("Tracker reset")
    
    def __repr__(self) -> str:
        return (
            f"MultiObjectTracker("
            f"tracks={len(self.track_manager.tracks)}, "
            f"updates={self.update_count}, "
            f"filter={self.config.filter_type}, "
            f"association={self.config.association_method})"
        )
