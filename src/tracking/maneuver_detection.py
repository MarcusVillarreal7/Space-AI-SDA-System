"""
Maneuver Detection for Space Object Tracking.

This module provides algorithms for detecting orbital maneuvers (thrust events)
by analyzing filter innovations and residuals.

Classes:
    ManeuverDetector: Base class for maneuver detection
    InnovationDetector: Chi-square test on innovation sequence
    MMAEDetector: Multiple Model Adaptive Estimation (optional)

References:
    - Bar-Shalom, Y. "Estimation with Applications to Tracking and Navigation"
    - Li, X.R. "Multiple-Model Estimation with Variable Structure"
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from scipy.stats import chi2
from collections import deque

from src.utils.logging_config import get_logger

logger = get_logger("tracking.maneuver")


@dataclass
class ManeuverEvent:
    """
    Represents a detected maneuver event.
    
    Attributes:
        timestamp: Time of detection
        track_id: ID of maneuvering track
        innovation_magnitude: Size of innovation (residual)
        chi_square_statistic: Chi-square test statistic
        confidence: Detection confidence (0-1)
        duration: How long maneuver has been detected (seconds)
    """
    timestamp: float
    track_id: int
    innovation_magnitude: float
    chi_square_statistic: float
    confidence: float
    duration: float = 0.0


class ManeuverDetector(ABC):
    """
    Base class for maneuver detection algorithms.
    """
    
    @abstractmethod
    def detect(
        self,
        innovation: np.ndarray,
        innovation_covariance: np.ndarray,
        timestamp: float,
        track_id: int
    ) -> Optional[ManeuverEvent]:
        """
        Detect if a maneuver has occurred.
        
        Args:
            innovation: Measurement residual (z - h(x))
            innovation_covariance: Innovation covariance matrix S
            timestamp: Current time
            track_id: Track identifier
            
        Returns:
            ManeuverEvent if detected, None otherwise
        """
        pass
    
    @abstractmethod
    def reset(self, track_id: int):
        """Reset detector state for a track."""
        pass


class InnovationDetector(ManeuverDetector):
    """
    Maneuver detection using chi-square test on innovation sequence.
    
    Detects maneuvers by monitoring the normalized innovation squared (NIS):
        d² = νᵀ S⁻¹ ν
    
    where ν is the innovation and S is the innovation covariance.
    
    A maneuver is detected when d² exceeds a threshold for multiple consecutive
    time steps.
    """
    
    def __init__(
        self,
        threshold: float = 13.8,  # 99.9% confidence for 3 DOF
        window_size: int = 3,
        min_detections: int = 2,
        cooldown_time: float = 300.0  # 5 minutes
    ):
        """
        Initialize innovation-based maneuver detector.
        
        Args:
            threshold: Chi-square threshold for detection
            window_size: Number of innovations to track
            min_detections: Minimum detections in window to confirm
            cooldown_time: Time to wait after detection before detecting again
        """
        self.threshold = threshold
        self.window_size = window_size
        self.min_detections = min_detections
        self.cooldown_time = cooldown_time
        
        # Track state per track ID
        self.innovation_history: dict = {}  # track_id -> deque of (time, d²)
        self.last_detection: dict = {}  # track_id -> timestamp
        self.maneuver_start: dict = {}  # track_id -> timestamp
        
        logger.info(
            f"InnovationDetector initialized: threshold={threshold:.2f}, "
            f"window={window_size}, min_detections={min_detections}"
        )
    
    def detect(
        self,
        innovation: np.ndarray,
        innovation_covariance: np.ndarray,
        timestamp: float,
        track_id: int
    ) -> Optional[ManeuverEvent]:
        """
        Detect maneuver using chi-square test.
        
        Args:
            innovation: Measurement residual (3D position)
            innovation_covariance: Innovation covariance (3x3)
            timestamp: Current time
            track_id: Track identifier
            
        Returns:
            ManeuverEvent if maneuver detected, None otherwise
        """
        # Initialize history for new tracks
        if track_id not in self.innovation_history:
            self.innovation_history[track_id] = deque(maxlen=self.window_size)
        
        # Compute normalized innovation squared (chi-square statistic)
        try:
            S_inv = np.linalg.inv(innovation_covariance)
            d_squared = float(innovation.T @ S_inv @ innovation)
        except np.linalg.LinAlgError:
            logger.warning(f"Track {track_id}: Singular innovation covariance")
            return None
        
        # Add to history
        self.innovation_history[track_id].append((timestamp, d_squared))
        
        # Check cooldown
        if track_id in self.last_detection:
            time_since_last = timestamp - self.last_detection[track_id]
            if time_since_last < self.cooldown_time:
                return None
        
        # Count recent exceedances
        history = self.innovation_history[track_id]
        exceedances = sum(1 for _, d2 in history if d2 > self.threshold)
        
        # Detect if enough exceedances in window
        if exceedances >= self.min_detections and len(history) >= self.min_detections:
            # Calculate confidence based on how much threshold is exceeded
            max_d_squared = max(d2 for _, d2 in history)
            confidence = min(1.0, max_d_squared / self.threshold)
            
            # Calculate duration if continuing maneuver
            if track_id in self.maneuver_start:
                duration = timestamp - self.maneuver_start[track_id]
            else:
                self.maneuver_start[track_id] = timestamp
                duration = 0.0
            
            self.last_detection[track_id] = timestamp
            
            event = ManeuverEvent(
                timestamp=timestamp,
                track_id=track_id,
                innovation_magnitude=float(np.linalg.norm(innovation)),
                chi_square_statistic=d_squared,
                confidence=confidence,
                duration=duration
            )
            
            logger.warning(
                f"Maneuver detected: Track {track_id}, "
                f"χ²={d_squared:.2f} (threshold={self.threshold:.2f}), "
                f"confidence={confidence:.2%}"
            )
            
            return event
        else:
            # Clear maneuver start if below threshold
            if track_id in self.maneuver_start and exceedances == 0:
                del self.maneuver_start[track_id]
        
        return None
    
    def reset(self, track_id: int):
        """Reset detector state for a track."""
        if track_id in self.innovation_history:
            del self.innovation_history[track_id]
        if track_id in self.last_detection:
            del self.last_detection[track_id]
        if track_id in self.maneuver_start:
            del self.maneuver_start[track_id]
    
    def get_statistics(self, track_id: int) -> dict:
        """Get detection statistics for a track."""
        if track_id not in self.innovation_history:
            return {}
        
        history = list(self.innovation_history[track_id])
        if not history:
            return {}
        
        d_squared_values = [d2 for _, d2 in history]
        
        return {
            'mean_chi_square': np.mean(d_squared_values),
            'max_chi_square': np.max(d_squared_values),
            'exceedance_rate': sum(1 for d2 in d_squared_values if d2 > self.threshold) / len(d_squared_values),
            'window_size': len(history),
            'is_maneuvering': track_id in self.maneuver_start
        }


class MMAEDetector(ManeuverDetector):
    """
    Multiple Model Adaptive Estimation (MMAE) for maneuver detection.
    
    Runs multiple filters in parallel:
    - Model 1: No maneuver (low process noise)
    - Model 2: Maneuver (high process noise)
    
    Computes model probabilities based on likelihood of measurements.
    Detects maneuver when maneuver model probability exceeds threshold.
    
    Note: This is a simplified MMAE. Full IMM would include model switching.
    """
    
    def __init__(
        self,
        threshold: float = 0.7,  # Probability threshold
        window_size: int = 5
    ):
        """
        Initialize MMAE detector.
        
        Args:
            threshold: Model probability threshold for detection
            window_size: Number of likelihoods to average
        """
        self.threshold = threshold
        self.window_size = window_size
        
        # Track state
        self.likelihood_history: dict = {}  # track_id -> deque of likelihoods
        self.model_probabilities: dict = {}  # track_id -> [p_no_maneuver, p_maneuver]
        
        logger.info(
            f"MMAEDetector initialized: threshold={threshold:.2f}, "
            f"window={window_size}"
        )
    
    def detect(
        self,
        innovation: np.ndarray,
        innovation_covariance: np.ndarray,
        timestamp: float,
        track_id: int
    ) -> Optional[ManeuverEvent]:
        """
        Detect maneuver using MMAE.
        
        Args:
            innovation: Measurement residual
            innovation_covariance: Innovation covariance
            timestamp: Current time
            track_id: Track identifier
            
        Returns:
            ManeuverEvent if maneuver detected, None otherwise
        """
        # Initialize for new tracks
        if track_id not in self.likelihood_history:
            self.likelihood_history[track_id] = deque(maxlen=self.window_size)
            self.model_probabilities[track_id] = [0.5, 0.5]  # Equal priors
        
        # Compute likelihood for no-maneuver model
        try:
            S_inv = np.linalg.inv(innovation_covariance)
            d_squared = float(innovation.T @ S_inv @ innovation)
            
            # Gaussian likelihood
            det_S = np.linalg.det(innovation_covariance)
            likelihood_no_maneuver = np.exp(-0.5 * d_squared) / np.sqrt((2*np.pi)**3 * det_S)
            
            # Maneuver model has higher covariance (less sensitive to large innovations)
            S_maneuver = innovation_covariance * 10.0  # 10x higher variance
            S_maneuver_inv = np.linalg.inv(S_maneuver)
            d_squared_maneuver = float(innovation.T @ S_maneuver_inv @ innovation)
            det_S_maneuver = np.linalg.det(S_maneuver)
            likelihood_maneuver = np.exp(-0.5 * d_squared_maneuver) / np.sqrt((2*np.pi)**3 * det_S_maneuver)
            
        except np.linalg.LinAlgError:
            logger.warning(f"Track {track_id}: Singular covariance in MMAE")
            return None
        
        # Update model probabilities (Bayes rule)
        p_prev = self.model_probabilities[track_id]
        p_no_maneuver = likelihood_no_maneuver * p_prev[0]
        p_maneuver = likelihood_maneuver * p_prev[1]
        
        # Normalize
        total = p_no_maneuver + p_maneuver
        if total > 0:
            p_no_maneuver /= total
            p_maneuver /= total
        else:
            p_no_maneuver, p_maneuver = 0.5, 0.5
        
        self.model_probabilities[track_id] = [p_no_maneuver, p_maneuver]
        
        # Detect if maneuver probability exceeds threshold
        if p_maneuver > self.threshold:
            event = ManeuverEvent(
                timestamp=timestamp,
                track_id=track_id,
                innovation_magnitude=float(np.linalg.norm(innovation)),
                chi_square_statistic=d_squared,
                confidence=p_maneuver,
                duration=0.0
            )
            
            logger.warning(
                f"Maneuver detected (MMAE): Track {track_id}, "
                f"P(maneuver)={p_maneuver:.2%}"
            )
            
            return event
        
        return None
    
    def reset(self, track_id: int):
        """Reset detector state for a track."""
        if track_id in self.likelihood_history:
            del self.likelihood_history[track_id]
        if track_id in self.model_probabilities:
            del self.model_probabilities[track_id]
    
    def get_model_probabilities(self, track_id: int) -> Optional[Tuple[float, float]]:
        """Get current model probabilities for a track."""
        if track_id in self.model_probabilities:
            return tuple(self.model_probabilities[track_id])
        return None


def compute_innovation_statistics(innovations: List[np.ndarray]) -> dict:
    """
    Compute statistics on a sequence of innovations.
    
    Args:
        innovations: List of innovation vectors
        
    Returns:
        Dictionary of statistics
    """
    if not innovations:
        return {}
    
    innovations_array = np.array(innovations)
    norms = np.linalg.norm(innovations_array, axis=1)
    
    return {
        'mean_norm': float(np.mean(norms)),
        'std_norm': float(np.std(norms)),
        'max_norm': float(np.max(norms)),
        'mean_innovation': innovations_array.mean(axis=0).tolist(),
        'innovation_covariance': np.cov(innovations_array.T).tolist()
    }
