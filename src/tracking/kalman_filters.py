"""
Kalman Filter implementations for orbital state estimation.

This module provides Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF)
implementations optimized for tracking space objects with nonlinear orbital dynamics.

Classes:
    KalmanFilter: Abstract base class for all Kalman filters
    ExtendedKalmanFilter: EKF with linearized dynamics
    UnscentedKalmanFilter: UKF with sigma point transform

References:
    - Bar-Shalom, Y., et al. "Estimation with Applications to Tracking and Navigation"
    - Simon, D. "Optimal State Estimation"
    - Julier, S. "The Unscented Kalman Filter"
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
from datetime import datetime

from src.utils.logging_config import get_logger

logger = get_logger("tracking.kalman")


# Earth gravitational parameter (km³/s²)
MU_EARTH = 398600.4418

# Earth radius (km)
R_EARTH = 6378.137

# J2 perturbation coefficient
J2 = 1.08263e-3


@dataclass
class StateVector:
    """
    Represents a 6-dimensional state vector for orbital tracking.
    
    Attributes:
        position: Position in ECI frame [x, y, z] (km)
        velocity: Velocity in ECI frame [vx, vy, vz] (km/s)
        timestamp: Time of state
        covariance: 6x6 covariance matrix (optional)
    """
    position: np.ndarray  # [x, y, z] in km
    velocity: np.ndarray  # [vx, vy, vz] in km/s
    timestamp: datetime
    covariance: Optional[np.ndarray] = None  # 6x6 matrix
    
    def to_vector(self) -> np.ndarray:
        """Convert to 6D state vector [x, y, z, vx, vy, vz]."""
        return np.concatenate([self.position, self.velocity])
    
    @classmethod
    def from_vector(cls, state: np.ndarray, timestamp: datetime, 
                    covariance: Optional[np.ndarray] = None) -> 'StateVector':
        """Create StateVector from 6D array."""
        return cls(
            position=state[:3],
            velocity=state[3:],
            timestamp=timestamp,
            covariance=covariance
        )


class KalmanFilter(ABC):
    """
    Abstract base class for Kalman filters.
    
    Provides common interface for state estimation with different filter types.
    Subclasses must implement predict() and update() methods.
    
    Attributes:
        state: Current state estimate (6D: position + velocity)
        covariance: State covariance matrix (6x6)
        process_noise: Process noise covariance Q (6x6)
        measurement_noise: Measurement noise covariance R (3x3 or variable)
    """
    
    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise_std: float = 1.0,  # m/s²
        measurement_noise_std: float = 50.0  # m
    ):
        """
        Initialize Kalman filter.
        
        Args:
            initial_state: Initial state vector [x, y, z, vx, vy, vz] (km, km/s)
            initial_covariance: Initial covariance matrix (6x6)
            process_noise_std: Process noise standard deviation (m/s²)
            measurement_noise_std: Measurement noise standard deviation (m)
        """
        self.state = initial_state.copy()
        self.covariance = initial_covariance.copy()
        
        # Process noise (only on velocity components)
        # Convert m/s² to km/s²
        q_std = process_noise_std / 1000.0
        self.process_noise = np.diag([0, 0, 0, q_std**2, q_std**2, q_std**2])
        
        # Measurement noise (position measurements)
        # Convert m to km
        r_std = measurement_noise_std / 1000.0
        self.measurement_noise = np.eye(3) * r_std**2
        
        logger.debug(f"Initialized {self.__class__.__name__}")
        logger.debug(f"  Process noise std: {process_noise_std} m/s²")
        logger.debug(f"  Measurement noise std: {measurement_noise_std} m")
    
    @abstractmethod
    def predict(self, dt: float) -> None:
        """
        Predict state forward by dt seconds.
        
        Args:
            dt: Time step in seconds
        """
        pass
    
    @abstractmethod
    def update(self, measurement: np.ndarray) -> None:
        """
        Update state with measurement.
        
        Args:
            measurement: Measurement vector (typically position [x, y, z] in km)
        """
        pass
    
    def get_state(self) -> np.ndarray:
        """Get current state estimate."""
        return self.state.copy()
    
    def get_covariance(self) -> np.ndarray:
        """Get current covariance matrix."""
        return self.covariance.copy()
    
    def get_position(self) -> np.ndarray:
        """Get current position estimate [x, y, z] in km."""
        return self.state[:3].copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate [vx, vy, vz] in km/s."""
        return self.state[3:].copy()
    
    def get_position_uncertainty(self) -> float:
        """Get position uncertainty (trace of position covariance) in km."""
        return np.sqrt(np.trace(self.covariance[:3, :3]))
    
    def get_velocity_uncertainty(self) -> float:
        """Get velocity uncertainty (trace of velocity covariance) in km/s."""
        return np.sqrt(np.trace(self.covariance[3:, 3:]))


class ExtendedKalmanFilter(KalmanFilter):
    """
    Extended Kalman Filter for nonlinear orbital dynamics.
    
    Uses first-order linearization (Jacobian) to handle nonlinear orbital motion.
    Includes J2 perturbation for improved accuracy.
    
    State: [x, y, z, vx, vy, vz] in ECI frame
    Measurement: [x, y, z] position in ECI frame
    
    Example:
        >>> initial_state = np.array([7000, 0, 0, 0, 7.5, 0])  # km, km/s
        >>> initial_cov = np.eye(6) * 100  # 100 km² position, 0.1 (km/s)²
        >>> ekf = ExtendedKalmanFilter(initial_state, initial_cov)
        >>> ekf.predict(60.0)  # Predict 60 seconds forward
        >>> ekf.update(measurement)  # Update with measurement
    """
    
    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise_std: float = 1.0,
        measurement_noise_std: float = 50.0,
        include_j2: bool = True
    ):
        """
        Initialize Extended Kalman Filter.
        
        Args:
            initial_state: Initial state [x, y, z, vx, vy, vz]
            initial_covariance: Initial covariance (6x6)
            process_noise_std: Process noise std dev (m/s²)
            measurement_noise_std: Measurement noise std dev (m)
            include_j2: Whether to include J2 perturbation
        """
        super().__init__(initial_state, initial_covariance, 
                        process_noise_std, measurement_noise_std)
        self.include_j2 = include_j2
        logger.info(f"EKF initialized (J2: {include_j2})")
    
    def predict(self, dt: float) -> None:
        """
        Predict state forward using orbital dynamics.
        
        Uses RK4 integration for state propagation and linearized dynamics
        for covariance propagation.
        
        Args:
            dt: Time step in seconds
        """
        # Propagate state using RK4
        self.state = self._rk4_step(self.state, dt)
        
        # Propagate covariance using linearized dynamics
        F = self._compute_state_transition_matrix(self.state, dt)
        self.covariance = F @ self.covariance @ F.T + self.process_noise
        
        logger.debug(f"EKF predict: dt={dt:.1f}s, pos_unc={self.get_position_uncertainty():.3f}km")
    
    def update(self, measurement: np.ndarray) -> None:
        """
        Update state with position measurement.
        
        Args:
            measurement: Position measurement [x, y, z] in km
        """
        # Measurement model: H = [I_3x3, 0_3x3] (observe position only)
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        
        # Innovation
        predicted_measurement = H @ self.state
        innovation = measurement - predicted_measurement
        
        # Innovation covariance
        S = H @ self.covariance @ H.T + self.measurement_noise
        
        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ innovation
        
        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(6) - K @ H
        self.covariance = I_KH @ self.covariance @ I_KH.T + K @ self.measurement_noise @ K.T
        
        innovation_norm = np.linalg.norm(innovation)
        logger.debug(f"EKF update: innovation={innovation_norm:.3f}km")
    
    def _orbital_dynamics(self, state: np.ndarray) -> np.ndarray:
        """
        Compute state derivative (orbital dynamics).
        
        Includes two-body dynamics and optional J2 perturbation.
        
        Args:
            state: State vector [x, y, z, vx, vy, vz]
            
        Returns:
            State derivative [vx, vy, vz, ax, ay, az]
        """
        pos = state[:3]
        vel = state[3:]
        
        r = np.linalg.norm(pos)
        
        # Two-body acceleration
        acc = -MU_EARTH / r**3 * pos
        
        # J2 perturbation
        if self.include_j2:
            x, y, z = pos
            factor = 1.5 * J2 * MU_EARTH * R_EARTH**2 / r**5
            
            acc[0] += factor * x * (5 * z**2 / r**2 - 1)
            acc[1] += factor * y * (5 * z**2 / r**2 - 1)
            acc[2] += factor * z * (5 * z**2 / r**2 - 3)
        
        return np.concatenate([vel, acc])
    
    def _rk4_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        """
        Runge-Kutta 4th order integration step.
        
        Args:
            state: Current state
            dt: Time step in seconds
            
        Returns:
            Propagated state
        """
        k1 = self._orbital_dynamics(state)
        k2 = self._orbital_dynamics(state + 0.5 * dt * k1)
        k3 = self._orbital_dynamics(state + 0.5 * dt * k2)
        k4 = self._orbital_dynamics(state + dt * k3)
        
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def _compute_state_transition_matrix(self, state: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute state transition matrix (Jacobian of dynamics).
        
        Uses finite differences for numerical Jacobian.
        
        Args:
            state: State vector
            dt: Time step
            
        Returns:
            6x6 state transition matrix F
        """
        # Numerical Jacobian using finite differences
        epsilon = 1e-6
        n = len(state)
        F = np.zeros((n, n))
        
        state_nominal = self._rk4_step(state, dt)
        
        for i in range(n):
            state_perturbed = state.copy()
            state_perturbed[i] += epsilon
            state_perturbed_prop = self._rk4_step(state_perturbed, dt)
            F[:, i] = (state_perturbed_prop - state_nominal) / epsilon
        
        return F


class UnscentedKalmanFilter(KalmanFilter):
    """
    Unscented Kalman Filter for nonlinear orbital dynamics.
    
    Uses sigma point transform to handle nonlinearities without linearization.
    Generally more accurate than EKF for highly nonlinear systems.
    
    State: [x, y, z, vx, vy, vz] in ECI frame
    Measurement: [x, y, z] position in ECI frame
    
    Example:
        >>> initial_state = np.array([7000, 0, 0, 0, 7.5, 0])
        >>> initial_cov = np.eye(6) * 100
        >>> ukf = UnscentedKalmanFilter(initial_state, initial_cov)
        >>> ukf.predict(60.0)
        >>> ukf.update(measurement)
    """
    
    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise_std: float = 1.0,
        measurement_noise_std: float = 50.0,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        include_j2: bool = True
    ):
        """
        Initialize Unscented Kalman Filter.
        
        Args:
            initial_state: Initial state [x, y, z, vx, vy, vz]
            initial_covariance: Initial covariance (6x6)
            process_noise_std: Process noise std dev (m/s²)
            measurement_noise_std: Measurement noise std dev (m)
            alpha: Spread of sigma points (typically 1e-3)
            beta: Prior knowledge parameter (2 for Gaussian)
            kappa: Secondary scaling parameter (0 or 3-n)
            include_j2: Whether to include J2 perturbation
        """
        super().__init__(initial_state, initial_covariance,
                        process_noise_std, measurement_noise_std)
        
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.include_j2 = include_j2
        
        # Compute sigma point parameters
        n = len(initial_state)
        self.lambda_ = alpha**2 * (n + kappa) - n
        self.gamma = np.sqrt(n + self.lambda_)
        
        # Weights for mean
        self.Wm = np.zeros(2 * n + 1)
        self.Wm[0] = self.lambda_ / (n + self.lambda_)
        self.Wm[1:] = 1.0 / (2 * (n + self.lambda_))
        
        # Weights for covariance
        self.Wc = self.Wm.copy()
        self.Wc[0] += (1 - alpha**2 + beta)
        
        logger.info(f"UKF initialized (J2: {include_j2}, alpha: {alpha})")
    
    def predict(self, dt: float) -> None:
        """
        Predict state forward using unscented transform.
        
        Args:
            dt: Time step in seconds
        """
        # Generate sigma points
        sigma_points = self._generate_sigma_points(self.state, self.covariance)
        
        # Propagate sigma points through dynamics
        sigma_points_prop = np.array([
            self._propagate_state(sp, dt) for sp in sigma_points
        ])
        
        # Compute predicted mean and covariance
        self.state = np.sum(self.Wm[:, np.newaxis] * sigma_points_prop, axis=0)
        
        diff = sigma_points_prop - self.state
        self.covariance = np.sum(
            self.Wc[:, np.newaxis, np.newaxis] * 
            (diff[:, :, np.newaxis] @ diff[:, np.newaxis, :]),
            axis=0
        ) + self.process_noise
        
        logger.debug(f"UKF predict: dt={dt:.1f}s, pos_unc={self.get_position_uncertainty():.3f}km")
    
    def update(self, measurement: np.ndarray) -> None:
        """
        Update state with measurement using unscented transform.
        
        Args:
            measurement: Position measurement [x, y, z] in km
        """
        # Generate sigma points from predicted state
        sigma_points = self._generate_sigma_points(self.state, self.covariance)
        
        # Transform sigma points through measurement model (identity for position)
        sigma_measurements = sigma_points[:, :3]  # Extract position
        
        # Predicted measurement
        predicted_measurement = np.sum(self.Wm[:, np.newaxis] * sigma_measurements, axis=0)
        
        # Innovation
        innovation = measurement - predicted_measurement
        
        # Innovation covariance
        diff_z = sigma_measurements - predicted_measurement
        Pzz = np.sum(
            self.Wc[:, np.newaxis, np.newaxis] *
            (diff_z[:, :, np.newaxis] @ diff_z[:, np.newaxis, :]),
            axis=0
        ) + self.measurement_noise
        
        # Cross-covariance
        diff_x = sigma_points - self.state
        Pxz = np.sum(
            self.Wc[:, np.newaxis, np.newaxis] *
            (diff_x[:, :, np.newaxis] @ diff_z[:, np.newaxis, :]),
            axis=0
        )
        
        # Kalman gain
        K = Pxz @ np.linalg.inv(Pzz)
        
        # Update state and covariance
        self.state = self.state + K @ innovation
        self.covariance = self.covariance - K @ Pzz @ K.T
        
        innovation_norm = np.linalg.norm(innovation)
        logger.debug(f"UKF update: innovation={innovation_norm:.3f}km")
    
    def _generate_sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Generate sigma points for unscented transform.
        
        Args:
            mean: State mean
            cov: State covariance
            
        Returns:
            Array of sigma points (2n+1 x n)
        """
        n = len(mean)
        sigma_points = np.zeros((2 * n + 1, n))
        
        # Central point
        sigma_points[0] = mean
        
        # Compute matrix square root
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use eigendecomposition
            eigval, eigvec = np.linalg.eigh(cov)
            eigval = np.maximum(eigval, 0)  # Ensure positive
            L = eigvec @ np.diag(np.sqrt(eigval))
        
        # Positive sigma points
        for i in range(n):
            sigma_points[i + 1] = mean + self.gamma * L[:, i]
        
        # Negative sigma points
        for i in range(n):
            sigma_points[n + i + 1] = mean - self.gamma * L[:, i]
        
        return sigma_points
    
    def _propagate_state(self, state: np.ndarray, dt: float) -> np.ndarray:
        """
        Propagate single state through dynamics (same as EKF).
        
        Args:
            state: State vector
            dt: Time step
            
        Returns:
            Propagated state
        """
        # Use same RK4 integration as EKF
        return self._rk4_step(state, dt)
    
    def _orbital_dynamics(self, state: np.ndarray) -> np.ndarray:
        """Orbital dynamics (same as EKF)."""
        pos = state[:3]
        vel = state[3:]
        
        r = np.linalg.norm(pos)
        acc = -MU_EARTH / r**3 * pos
        
        if self.include_j2:
            x, y, z = pos
            factor = 1.5 * J2 * MU_EARTH * R_EARTH**2 / r**5
            acc[0] += factor * x * (5 * z**2 / r**2 - 1)
            acc[1] += factor * y * (5 * z**2 / r**2 - 1)
            acc[2] += factor * z * (5 * z**2 / r**2 - 3)
        
        return np.concatenate([vel, acc])
    
    def _rk4_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        """RK4 integration (same as EKF)."""
        k1 = self._orbital_dynamics(state)
        k2 = self._orbital_dynamics(state + 0.5 * dt * k1)
        k3 = self._orbital_dynamics(state + 0.5 * dt * k2)
        k4 = self._orbital_dynamics(state + dt * k3)
        
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
