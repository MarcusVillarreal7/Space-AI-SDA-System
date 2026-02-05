"""
Track Management for Multi-Object Tracking.

This module handles the lifecycle of tracks, including initialization,
confirmation, maintenance, and deletion.

Classes:
    TrackState: Enum for track states
    Track: Represents a single tracked object
    TrackManager: Manages multiple tracks

References:
    - Bar-Shalom, Y. "Tracking and Data Association"
    - Blackman, S. "Multiple-Target Tracking with Radar Applications"
"""

import numpy as np
from enum import Enum
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import time

from src.tracking.kalman_filters import StateVector, ExtendedKalmanFilter, UnscentedKalmanFilter
from src.tracking.data_association import Measurement
from src.utils.logging_config import get_logger

logger = get_logger("tracking.manager")


class TrackState(Enum):
    """Track lifecycle states."""
    TENTATIVE = "tentative"      # New track, not yet confirmed
    CONFIRMED = "confirmed"      # Confirmed track with sufficient updates
    COASTED = "coasted"          # Track with no recent measurements (coasting)
    DELETED = "deleted"          # Track marked for deletion


@dataclass
class Track:
    """
    Represents a single tracked space object.
    
    Attributes:
        track_id: Unique track identifier
        state: Current track state (TENTATIVE, CONFIRMED, etc.)
        filter: Kalman filter for state estimation
        last_update_time: Timestamp of last measurement update
        creation_time: Timestamp when track was created
        hit_count: Number of successful measurement associations
        miss_count: Number of consecutive missed detections
        covariance_trace: Trace of position covariance (uncertainty metric)
        norad_id: Optional NORAD catalog ID if identified
    """
    track_id: int
    state: TrackState
    filter: ExtendedKalmanFilter  # or UnscentedKalmanFilter
    last_update_time: float
    creation_time: float
    hit_count: int = 0
    miss_count: int = 0
    covariance_trace: float = 0.0
    is_maneuvering: bool = False
    norad_id: Optional[int] = None
    
    def get_state_vector(self) -> StateVector:
        """Get current state estimate."""
        return self.filter.get_state()
    
    def get_position(self) -> np.ndarray:
        """Get current position estimate."""
        return self.filter.state[:3]
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        return self.filter.state[3:]
    
    def get_position_covariance(self) -> np.ndarray:
        """Get position covariance (3x3)."""
        P = self.filter.covariance
        return P[:3, :3]
    
    def update_covariance_trace(self):
        """Update covariance trace metric."""
        pos_cov = self.get_position_covariance()
        self.covariance_trace = np.trace(pos_cov)
    
    def predict(self, dt: float):
        """
        Predict track state forward in time.
        
        Args:
            dt: Time step (seconds)
        """
        self.filter.predict(dt)
        self.update_covariance_trace()
    
    def update(self, measurement: Measurement):
        """
        Update track with measurement.
        
        Args:
            measurement: Measurement to incorporate
        """
        # The Kalman filter's update method only takes the measurement position
        # It internally uses H = [I_3x3, 0_3x3] for position-only measurements
        self.filter.update(measurement.position)
        self.last_update_time = measurement.timestamp
        self.hit_count += 1
        self.miss_count = 0
        self.update_covariance_trace()
    
    def mark_missed(self):
        """Mark track as having a missed detection."""
        self.miss_count += 1
    
    def age(self, current_time: float) -> float:
        """
        Get track age in seconds.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Age in seconds
        """
        return current_time - self.creation_time
    
    def time_since_update(self, current_time: float) -> float:
        """
        Get time since last update in seconds.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Time since last update in seconds
        """
        return current_time - self.last_update_time


class TrackManager:
    """
    Manages multiple tracks and their lifecycle.
    
    Handles track initialization, confirmation, coasting, and deletion
    based on configurable thresholds.
    """
    
    def __init__(
        self,
        confirmation_threshold: int = 3,
        deletion_threshold: int = 5,
        coast_threshold: int = 3,
        max_coast_time: float = 300.0,  # 5 minutes
        filter_type: str = "ekf",
        process_noise_std: float = 0.01,
        initial_position_std: float = 0.1,
        initial_velocity_std: float = 0.01
    ):
        """
        Initialize track manager.
        
        Args:
            confirmation_threshold: Hits needed to confirm track
            deletion_threshold: Consecutive misses before deletion
            coast_threshold: Misses before entering coast state
            max_coast_time: Maximum time to coast without measurements (seconds)
            filter_type: "ekf" or "ukf"
            process_noise_std: Process noise standard deviation (km/s²)
            initial_position_std: Initial position uncertainty (km)
            initial_velocity_std: Initial velocity uncertainty (km/s)
        """
        self.confirmation_threshold = confirmation_threshold
        self.deletion_threshold = deletion_threshold
        self.coast_threshold = coast_threshold
        self.max_coast_time = max_coast_time
        self.filter_type = filter_type
        self.process_noise_std = process_noise_std
        self.initial_position_std = initial_position_std
        self.initial_velocity_std = initial_velocity_std
        
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        
        logger.info(f"TrackManager initialized: confirm={confirmation_threshold}, "
                   f"delete={deletion_threshold}, coast={coast_threshold}, "
                   f"filter={filter_type}")
    
    def create_track(
        self,
        measurement: Measurement,
        initial_velocity: Optional[np.ndarray] = None
    ) -> Track:
        """
        Create a new tentative track from a measurement.
        
        Args:
            measurement: Initial measurement
            initial_velocity: Optional initial velocity estimate (default: zero)
            
        Returns:
            New Track object
        """
        track_id = self.next_track_id
        self.next_track_id += 1
        
        # Initialize state
        if initial_velocity is None:
            initial_velocity = np.zeros(3)
        
        # Create state vector as numpy array [x, y, z, vx, vy, vz]
        initial_state_array = np.concatenate([
            measurement.position.copy(),
            initial_velocity.copy()
        ])
        
        # Initialize covariance
        P = np.eye(6)
        P[:3, :3] *= self.initial_position_std ** 2
        P[3:, 3:] *= self.initial_velocity_std ** 2
        
        # Create filter
        if self.filter_type == "ekf":
            filter_obj = ExtendedKalmanFilter(
                initial_state=initial_state_array,
                initial_covariance=P,
                process_noise_std=self.process_noise_std
            )
        else:  # ukf
            filter_obj = UnscentedKalmanFilter(
                initial_state=initial_state_array,
                initial_covariance=P,
                process_noise_std=self.process_noise_std
            )
        
        # Create track
        track = Track(
            track_id=track_id,
            state=TrackState.TENTATIVE,
            filter=filter_obj,
            last_update_time=measurement.timestamp,
            creation_time=measurement.timestamp,
            hit_count=1,
            miss_count=0
        )
        
        track.update_covariance_trace()
        
        self.tracks[track_id] = track
        logger.debug(f"Created track {track_id} at position {measurement.position}")
        
        return track
    
    def update_track(self, track_id: int, measurement: Measurement):
        """
        Update an existing track with a measurement.
        
        Args:
            track_id: ID of track to update
            measurement: Measurement to incorporate
        """
        if track_id not in self.tracks:
            logger.warning(f"Track {track_id} not found for update")
            return
        
        track = self.tracks[track_id]
        
        # Predict to measurement time
        dt = measurement.timestamp - track.last_update_time
        if dt > 0:
            track.predict(dt)
        
        # Update with measurement
        track.update(measurement)
        
        # Update track state based on hit count
        if track.state == TrackState.TENTATIVE:
            if track.hit_count >= self.confirmation_threshold:
                track.state = TrackState.CONFIRMED
                logger.info(f"Track {track_id} CONFIRMED (hits: {track.hit_count})")
        elif track.state == TrackState.COASTED:
            track.state = TrackState.CONFIRMED
            logger.debug(f"Track {track_id} returned from COASTED to CONFIRMED")
    
    def predict_track(self, track_id: int, dt: float):
        """
        Predict a track forward in time without measurement.
        
        Args:
            track_id: ID of track to predict
            dt: Time step (seconds)
        """
        if track_id not in self.tracks:
            logger.warning(f"Track {track_id} not found for prediction")
            return
        
        track = self.tracks[track_id]
        track.predict(dt)
        track.mark_missed()
        
        # Update track state based on miss count
        if track.state == TrackState.CONFIRMED:
            if track.miss_count >= self.coast_threshold:
                track.state = TrackState.COASTED
                logger.debug(f"Track {track_id} entered COASTED state (misses: {track.miss_count})")
    
    def delete_track(self, track_id: int):
        """
        Mark a track for deletion.
        
        Args:
            track_id: ID of track to delete
        """
        if track_id in self.tracks:
            self.tracks[track_id].state = TrackState.DELETED
            logger.info(f"Track {track_id} marked for DELETION")
    
    def prune_tracks(self, current_time: float):
        """
        Remove deleted tracks and tracks that have coasted too long.
        
        Args:
            current_time: Current timestamp
        """
        to_delete = []
        
        for track_id, track in self.tracks.items():
            # Delete tracks marked for deletion
            if track.state == TrackState.DELETED:
                to_delete.append(track_id)
            
            # Delete tracks with too many consecutive misses
            elif track.miss_count >= self.deletion_threshold:
                logger.info(f"Track {track_id} deleted: too many misses ({track.miss_count})")
                to_delete.append(track_id)
            
            # Delete coasted tracks that have exceeded max coast time
            elif track.state == TrackState.COASTED:
                time_since_update = track.time_since_update(current_time)
                if time_since_update > self.max_coast_time:
                    logger.info(f"Track {track_id} deleted: coast timeout ({time_since_update:.1f}s)")
                    to_delete.append(track_id)
        
        # Remove tracks
        for track_id in to_delete:
            del self.tracks[track_id]
        
        if to_delete:
            logger.debug(f"Pruned {len(to_delete)} tracks")
    
    def get_confirmed_tracks(self) -> List[Track]:
        """Get all confirmed tracks."""
        return [t for t in self.tracks.values() if t.state == TrackState.CONFIRMED]
    
    def get_all_active_tracks(self) -> List[Track]:
        """Get all active tracks (confirmed + coasted)."""
        return [
            t for t in self.tracks.values() 
            if t.state in [TrackState.CONFIRMED, TrackState.COASTED]
        ]
    
    def get_track(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        return self.tracks.get(track_id)
    
    def get_track_count(self) -> Dict[str, int]:
        """Get count of tracks by state."""
        counts = {state.value: 0 for state in TrackState}
        for track in self.tracks.values():
            counts[track.state.value] += 1
        return counts
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get tracking statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self.tracks:
            return {
                'total_tracks': 0,
                'confirmed_tracks': 0,
                'mean_covariance_trace': 0.0,
                'mean_hit_count': 0.0,
            }
        
        confirmed = self.get_confirmed_tracks()
        
        return {
            'total_tracks': len(self.tracks),
            'confirmed_tracks': len(confirmed),
            'mean_covariance_trace': np.mean([t.covariance_trace for t in confirmed]) if confirmed else 0.0,
            'mean_hit_count': np.mean([t.hit_count for t in confirmed]) if confirmed else 0.0,
        }
