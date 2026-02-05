"""
Data Association algorithms for multi-object tracking.

This module provides algorithms for associating measurements to tracks,
including optimal assignment (Hungarian algorithm) and greedy approaches (GNN).

Classes:
    CostCalculator: Compute association costs using Mahalanobis distance
    HungarianAssociator: Optimal assignment using Hungarian algorithm
    GNNAssociator: Greedy nearest neighbor association

References:
    - Kuhn, H. "The Hungarian Method for the Assignment Problem"
    - Bar-Shalom, Y. "Multitarget-Multisensor Tracking"
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from src.utils.logging_config import get_logger

logger = get_logger("tracking.association")


@dataclass
class Measurement:
    """
    Represents a sensor measurement.
    
    Attributes:
        position: Measured position [x, y, z] in km
        covariance: Measurement covariance (3x3)
        timestamp: Time of measurement
        sensor_id: ID of sensor that made measurement
        measurement_id: Unique measurement ID
    """
    position: np.ndarray
    covariance: np.ndarray
    timestamp: float
    sensor_id: str
    measurement_id: int


@dataclass
class Association:
    """
    Represents a track-to-measurement association.
    
    Attributes:
        track_id: ID of associated track
        measurement_id: ID of associated measurement
        cost: Association cost (Mahalanobis distance)
    """
    track_id: int
    measurement_id: int
    cost: float


class CostCalculator:
    """
    Calculate association costs between tracks and measurements.
    
    Uses Mahalanobis distance to account for uncertainty in both
    track prediction and measurement.
    """
    
    def __init__(self, gate_threshold: float = 9.0):
        """
        Initialize cost calculator.
        
        Args:
            gate_threshold: Chi-square gating threshold (default: 9.0 for 3D, 99% confidence)
        """
        self.gate_threshold = gate_threshold
        logger.debug(f"CostCalculator initialized with gate threshold: {gate_threshold}")
    
    def mahalanobis_distance(
        self,
        predicted_position: np.ndarray,
        predicted_covariance: np.ndarray,
        measurement_position: np.ndarray,
        measurement_covariance: np.ndarray
    ) -> float:
        """
        Calculate Mahalanobis distance between prediction and measurement.
        
        The Mahalanobis distance accounts for uncertainty:
        d² = (z - ẑ)ᵀ S⁻¹ (z - ẑ)
        
        where S = H P Hᵀ + R (innovation covariance)
        
        Args:
            predicted_position: Predicted position [x, y, z]
            predicted_covariance: Predicted position covariance (3x3)
            measurement_position: Measured position [x, y, z]
            measurement_covariance: Measurement covariance (3x3)
            
        Returns:
            Mahalanobis distance (scalar)
        """
        # Innovation (residual)
        innovation = measurement_position - predicted_position
        
        # Innovation covariance
        S = predicted_covariance + measurement_covariance
        
        # Mahalanobis distance: d² = νᵀ S⁻¹ ν
        try:
            S_inv = np.linalg.inv(S)
            distance_squared = innovation.T @ S_inv @ innovation
            return float(np.sqrt(distance_squared))
        except np.linalg.LinAlgError:
            # If S is singular, return large distance
            logger.warning("Singular innovation covariance, returning large distance")
            return 1e6
    
    def gate_measurement(
        self,
        predicted_position: np.ndarray,
        predicted_covariance: np.ndarray,
        measurement_position: np.ndarray,
        measurement_covariance: np.ndarray
    ) -> bool:
        """
        Check if measurement is within validation gate.
        
        Uses chi-square test: d² < χ²(α, df)
        
        Args:
            predicted_position: Predicted position
            predicted_covariance: Predicted covariance
            measurement_position: Measured position
            measurement_covariance: Measurement covariance
            
        Returns:
            True if measurement is within gate, False otherwise
        """
        distance = self.mahalanobis_distance(
            predicted_position, predicted_covariance,
            measurement_position, measurement_covariance
        )
        
        # Chi-square test
        distance_squared = distance ** 2
        return distance_squared < self.gate_threshold
    
    def build_cost_matrix(
        self,
        track_predictions: List[Tuple[np.ndarray, np.ndarray]],
        measurements: List[Measurement]
    ) -> np.ndarray:
        """
        Build cost matrix for track-to-measurement association.
        
        Args:
            track_predictions: List of (predicted_position, predicted_covariance) tuples
            measurements: List of measurements
            
        Returns:
            Cost matrix (n_tracks x n_measurements)
            Entries are Mahalanobis distances, or infinity if gated out
        """
        n_tracks = len(track_predictions)
        n_measurements = len(measurements)
        
        cost_matrix = np.full((n_tracks, n_measurements), np.inf)
        
        for i, (pred_pos, pred_cov) in enumerate(track_predictions):
            for j, meas in enumerate(measurements):
                # Check gate first
                if self.gate_measurement(pred_pos, pred_cov, meas.position, meas.covariance):
                    # Calculate cost
                    cost_matrix[i, j] = self.mahalanobis_distance(
                        pred_pos, pred_cov, meas.position, meas.covariance
                    )
        
        logger.debug(f"Built cost matrix: {n_tracks} tracks x {n_measurements} measurements")
        logger.debug(f"  Valid associations: {np.sum(np.isfinite(cost_matrix))}")
        
        return cost_matrix


class HungarianAssociator:
    """
    Optimal track-to-measurement association using Hungarian algorithm.
    
    Solves the assignment problem to minimize total association cost.
    Complexity: O(n³) where n = max(tracks, measurements)
    """
    
    def __init__(self, gate_threshold: float = 9.0, max_cost: float = 100.0):
        """
        Initialize Hungarian associator.
        
        Args:
            gate_threshold: Chi-square gating threshold
            max_cost: Maximum allowed association cost
        """
        self.cost_calculator = CostCalculator(gate_threshold)
        self.max_cost = max_cost
        logger.info(f"HungarianAssociator initialized (gate: {gate_threshold}, max_cost: {max_cost})")
    
    def associate(
        self,
        track_predictions: List[Tuple[int, np.ndarray, np.ndarray]],
        measurements: List[Measurement]
    ) -> Tuple[List[Association], List[int], List[int]]:
        """
        Associate tracks to measurements using Hungarian algorithm.
        
        Args:
            track_predictions: List of (track_id, predicted_position, predicted_covariance)
            measurements: List of measurements
            
        Returns:
            Tuple of:
            - associations: List of Association objects
            - unassociated_tracks: List of track IDs without measurements
            - unassociated_measurements: List of measurement IDs without tracks
        """
        if len(track_predictions) == 0 or len(measurements) == 0:
            # No associations possible
            unassoc_tracks = [tid for tid, _, _ in track_predictions]
            unassoc_meas = [m.measurement_id for m in measurements]
            logger.debug("No associations: empty tracks or measurements")
            return [], unassoc_tracks, unassoc_meas
        
        # Extract track IDs and predictions
        track_ids = [tid for tid, _, _ in track_predictions]
        predictions = [(pos, cov) for _, pos, cov in track_predictions]
        
        # Build cost matrix
        cost_matrix = self.cost_calculator.build_cost_matrix(predictions, measurements)
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Extract valid associations
        associations = []
        associated_tracks = set()
        associated_measurements = set()
        
        for i, j in zip(row_ind, col_ind):
            cost = cost_matrix[i, j]
            if np.isfinite(cost) and cost < self.max_cost:
                associations.append(Association(
                    track_id=track_ids[i],
                    measurement_id=measurements[j].measurement_id,
                    cost=cost
                ))
                associated_tracks.add(track_ids[i])
                associated_measurements.add(measurements[j].measurement_id)
        
        # Find unassociated tracks and measurements
        unassociated_tracks = [tid for tid in track_ids if tid not in associated_tracks]
        unassociated_measurements = [
            m.measurement_id for m in measurements 
            if m.measurement_id not in associated_measurements
        ]
        
        logger.info(f"Hungarian association: {len(associations)} associations, "
                   f"{len(unassociated_tracks)} unassoc tracks, "
                   f"{len(unassociated_measurements)} unassoc measurements")
        
        return associations, unassociated_tracks, unassociated_measurements


class GNNAssociator:
    """
    Greedy Nearest Neighbor (GNN) association.
    
    Greedily assigns measurements to tracks based on minimum cost.
    Faster than Hungarian (O(n²)) but suboptimal.
    """
    
    def __init__(self, gate_threshold: float = 9.0, max_cost: float = 100.0):
        """
        Initialize GNN associator.
        
        Args:
            gate_threshold: Chi-square gating threshold
            max_cost: Maximum allowed association cost
        """
        self.cost_calculator = CostCalculator(gate_threshold)
        self.max_cost = max_cost
        logger.info(f"GNNAssociator initialized (gate: {gate_threshold}, max_cost: {max_cost})")
    
    def associate(
        self,
        track_predictions: List[Tuple[int, np.ndarray, np.ndarray]],
        measurements: List[Measurement]
    ) -> Tuple[List[Association], List[int], List[int]]:
        """
        Associate tracks to measurements using greedy nearest neighbor.
        
        Args:
            track_predictions: List of (track_id, predicted_position, predicted_covariance)
            measurements: List of measurements
            
        Returns:
            Tuple of:
            - associations: List of Association objects
            - unassociated_tracks: List of track IDs without measurements
            - unassociated_measurements: List of measurement IDs without tracks
        """
        if len(track_predictions) == 0 or len(measurements) == 0:
            unassoc_tracks = [tid for tid, _, _ in track_predictions]
            unassoc_meas = [m.measurement_id for m in measurements]
            logger.debug("No associations: empty tracks or measurements")
            return [], unassoc_tracks, unassoc_meas
        
        # Extract track IDs and predictions
        track_ids = [tid for tid, _, _ in track_predictions]
        predictions = [(pos, cov) for _, pos, cov in track_predictions]
        
        # Build cost matrix
        cost_matrix = self.cost_calculator.build_cost_matrix(predictions, measurements)
        
        # Greedy assignment
        associations = []
        associated_tracks = set()
        associated_measurements = set()
        
        # Flatten cost matrix to get all (track, measurement, cost) tuples
        candidates = []
        for i in range(len(track_ids)):
            for j in range(len(measurements)):
                cost = cost_matrix[i, j]
                if np.isfinite(cost) and cost < self.max_cost:
                    candidates.append((i, j, cost))
        
        # Sort by cost (ascending)
        candidates.sort(key=lambda x: x[2])
        
        # Greedily assign
        for i, j, cost in candidates:
            track_id = track_ids[i]
            meas_id = measurements[j].measurement_id
            
            # Only assign if both track and measurement are unassociated
            if track_id not in associated_tracks and meas_id not in associated_measurements:
                associations.append(Association(
                    track_id=track_id,
                    measurement_id=meas_id,
                    cost=cost
                ))
                associated_tracks.add(track_id)
                associated_measurements.add(meas_id)
        
        # Find unassociated
        unassociated_tracks = [tid for tid in track_ids if tid not in associated_tracks]
        unassociated_measurements = [
            m.measurement_id for m in measurements 
            if m.measurement_id not in associated_measurements
        ]
        
        logger.info(f"GNN association: {len(associations)} associations, "
                   f"{len(unassociated_tracks)} unassoc tracks, "
                   f"{len(unassociated_measurements)} unassoc measurements")
        
        return associations, unassociated_tracks, unassociated_measurements


def compute_association_metrics(
    associations: List[Association],
    ground_truth: Optional[Dict[int, int]] = None
) -> Dict[str, float]:
    """
    Compute metrics for association performance.
    
    Args:
        associations: List of associations
        ground_truth: Optional dict mapping track_id to true measurement_id
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'num_associations': len(associations),
        'mean_cost': np.mean([a.cost for a in associations]) if associations else 0.0,
        'max_cost': np.max([a.cost for a in associations]) if associations else 0.0,
    }
    
    if ground_truth is not None:
        # Calculate accuracy
        correct = sum(
            1 for a in associations 
            if ground_truth.get(a.track_id) == a.measurement_id
        )
        metrics['accuracy'] = correct / len(associations) if associations else 0.0
    
    return metrics
