"""
Tracking Engine - Multi-Object Tracking for Space Domain Awareness

This module provides state estimation and tracking capabilities for space objects
using Kalman filters, data association, and track management.

Components:
- kalman_filters: EKF and UKF implementations
- data_association: Hungarian algorithm and GNN
- track_manager: Track lifecycle management
- maneuver_detection: Anomaly detection
- multi_object_tracker: Main tracking orchestration

Example:
    >>> from src.tracking import MultiObjectTracker
    >>> tracker = MultiObjectTracker()
    >>> tracks = tracker.process_measurements(measurements, timestamp)
"""

__version__ = "0.1.0"
__author__ = "Space AI Project"

# Import main classes for easy access
# (Will be uncommented as we implement each module)
from .kalman_filters import ExtendedKalmanFilter, UnscentedKalmanFilter, StateVector
from .data_association import HungarianAssociator, GNNAssociator, Measurement, Association
from .track_manager import Track, TrackManager, TrackState
# from .maneuver_detection import InnovationDetector
# from .multi_object_tracker import MultiObjectTracker

__all__ = [
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "StateVector",
    "HungarianAssociator",
    "GNNAssociator",
    "Measurement",
    "Association",
    "Track",
    "TrackManager",
    "TrackState",
    # "InnovationDetector",
    # "MultiObjectTracker",
]
