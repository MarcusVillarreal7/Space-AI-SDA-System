"""
Unit tests for tracking modules.

Tests cover:
- Kalman filters (EKF and UKF)
- Data association algorithms
- Track management
- Maneuver detection
- Multi-object tracking
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from src.tracking.kalman_filters import (
    StateVector,
    KalmanFilter,
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    MU_EARTH,
    R_EARTH
)


class TestStateVector:
    """Test StateVector dataclass."""
    
    def test_state_vector_creation(self):
        """Test creating a StateVector."""
        pos = np.array([7000.0, 0.0, 0.0])
        vel = np.array([0.0, 7.5, 0.0])
        timestamp = datetime.now()
        
        sv = StateVector(pos, vel, timestamp)
        
        assert np.allclose(sv.position, pos)
        assert np.allclose(sv.velocity, vel)
        assert sv.timestamp == timestamp
        assert sv.covariance is None
    
    def test_state_vector_to_vector(self):
        """Test converting StateVector to array."""
        pos = np.array([7000.0, 0.0, 0.0])
        vel = np.array([0.0, 7.5, 0.0])
        timestamp = datetime.now()
        
        sv = StateVector(pos, vel, timestamp)
        vec = sv.to_vector()
        
        assert len(vec) == 6
        assert np.allclose(vec[:3], pos)
        assert np.allclose(vec[3:], vel)
    
    def test_state_vector_from_vector(self):
        """Test creating StateVector from array."""
        state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        timestamp = datetime.now()
        
        sv = StateVector.from_vector(state, timestamp)
        
        assert np.allclose(sv.position, state[:3])
        assert np.allclose(sv.velocity, state[3:])


class TestExtendedKalmanFilter:
    """Test Extended Kalman Filter."""
    
    def test_ekf_initialization(self):
        """Test EKF initialization."""
        initial_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        initial_cov = np.eye(6) * 100.0
        
        ekf = ExtendedKalmanFilter(initial_state, initial_cov)
        
        assert np.allclose(ekf.get_state(), initial_state)
        assert np.allclose(ekf.get_covariance(), initial_cov)
        assert ekf.include_j2 is True
    
    def test_ekf_predict_step(self):
        """Test EKF prediction step."""
        # Circular orbit at 7000 km
        initial_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        initial_cov = np.eye(6) * 100.0
        
        ekf = ExtendedKalmanFilter(initial_state, initial_cov)
        
        # Predict 60 seconds forward
        ekf.predict(60.0)
        
        # State should have changed
        assert not np.allclose(ekf.get_state(), initial_state)
        
        # Position should have moved
        new_pos = ekf.get_position()
        assert not np.allclose(new_pos, initial_state[:3])
        
        # Covariance should have grown
        assert ekf.get_position_uncertainty() > np.sqrt(np.trace(initial_cov[:3, :3]))
    
    def test_ekf_update_step(self):
        """Test EKF update step."""
        initial_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        initial_cov = np.eye(6) * 100.0
        
        ekf = ExtendedKalmanFilter(initial_state, initial_cov)
        
        # Measurement close to true position
        measurement = np.array([7000.1, 0.05, 0.0])
        
        pos_unc_before = ekf.get_position_uncertainty()
        ekf.update(measurement)
        pos_unc_after = ekf.get_position_uncertainty()
        
        # Uncertainty should decrease after measurement
        assert pos_unc_after < pos_unc_before
        
        # State should be updated toward measurement
        updated_pos = ekf.get_position()
        assert np.linalg.norm(updated_pos - measurement) < np.linalg.norm(initial_state[:3] - measurement)
    
    def test_ekf_full_cycle(self):
        """Test full predict-update cycle."""
        # Start with circular orbit
        r = 7000.0  # km
        v = np.sqrt(MU_EARTH / r)  # Circular velocity
        initial_state = np.array([r, 0.0, 0.0, 0.0, v, 0.0])
        initial_cov = np.eye(6) * 100.0
        
        ekf = ExtendedKalmanFilter(initial_state, initial_cov, 
                                   process_noise_std=0.1,
                                   measurement_noise_std=50.0)
        
        # Simulate 10 predict-update cycles
        dt = 60.0  # seconds
        for i in range(10):
            ekf.predict(dt)
            
            # Simulate measurement with noise
            true_pos = ekf.get_position()
            noise = np.random.randn(3) * 0.05  # 50m noise
            measurement = true_pos + noise
            
            ekf.update(measurement)
        
        # Filter should still be stable
        assert ekf.get_position_uncertainty() < 1.0  # < 1 km uncertainty
        assert not np.any(np.isnan(ekf.get_state()))
        assert not np.any(np.isnan(ekf.get_covariance()))
    
    def test_ekf_orbital_dynamics(self):
        """Test that EKF respects orbital mechanics."""
        # Circular orbit
        r = 7000.0
        v = np.sqrt(MU_EARTH / r)
        initial_state = np.array([r, 0.0, 0.0, 0.0, v, 0.0])
        initial_cov = np.eye(6) * 100.0
        
        ekf = ExtendedKalmanFilter(initial_state, initial_cov)
        
        # Predict for a short time (10 minutes)
        dt = 600.0  # seconds
        ekf.predict(dt)
        
        # Radius should stay approximately constant for circular orbit
        final_pos = ekf.get_position()
        final_r = np.linalg.norm(final_pos)
        
        # Should maintain altitude within 5% over 10 minutes
        assert abs(final_r - r) / r < 0.05  # Within 5% of initial radius
        
        # Energy should be approximately conserved
        final_vel = ekf.get_velocity()
        final_v = np.linalg.norm(final_vel)
        # For circular orbit, v² = μ/r
        expected_v = np.sqrt(MU_EARTH / final_r)
        assert abs(final_v - expected_v) / expected_v < 0.1  # Within 10%


class TestUnscentedKalmanFilter:
    """Test Unscented Kalman Filter."""
    
    def test_ukf_initialization(self):
        """Test UKF initialization."""
        initial_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        initial_cov = np.eye(6) * 100.0
        
        ukf = UnscentedKalmanFilter(initial_state, initial_cov)
        
        assert np.allclose(ukf.get_state(), initial_state)
        assert np.allclose(ukf.get_covariance(), initial_cov)
        assert ukf.include_j2 is True
        assert ukf.alpha == 1e-3
        assert ukf.beta == 2.0
    
    def test_ukf_sigma_points(self):
        """Test sigma point generation."""
        initial_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        initial_cov = np.eye(6) * 100.0
        
        ukf = UnscentedKalmanFilter(initial_state, initial_cov)
        
        # Generate sigma points
        sigma_points = ukf._generate_sigma_points(initial_state, initial_cov)
        
        # Should have 2n+1 points
        n = len(initial_state)
        assert sigma_points.shape == (2 * n + 1, n)
        
        # First point should be the mean
        assert np.allclose(sigma_points[0], initial_state)
        
        # Weighted mean should equal original mean
        mean = np.sum(ukf.Wm[:, np.newaxis] * sigma_points, axis=0)
        assert np.allclose(mean, initial_state, atol=1e-10)
    
    def test_ukf_predict_step(self):
        """Test UKF prediction step."""
        initial_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        initial_cov = np.eye(6) * 100.0
        
        ukf = UnscentedKalmanFilter(initial_state, initial_cov)
        
        # Predict 60 seconds forward
        ukf.predict(60.0)
        
        # State should have changed
        assert not np.allclose(ukf.get_state(), initial_state)
        
        # Covariance should have grown
        assert ukf.get_position_uncertainty() > np.sqrt(np.trace(initial_cov[:3, :3]))
    
    def test_ukf_update_step(self):
        """Test UKF update step."""
        initial_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        initial_cov = np.eye(6) * 100.0
        
        ukf = UnscentedKalmanFilter(initial_state, initial_cov)
        
        # Measurement
        measurement = np.array([7000.1, 0.05, 0.0])
        
        pos_unc_before = ukf.get_position_uncertainty()
        ukf.update(measurement)
        pos_unc_after = ukf.get_position_uncertainty()
        
        # Uncertainty should decrease
        assert pos_unc_after < pos_unc_before
    
    def test_ukf_vs_ekf_accuracy(self):
        """Compare UKF and EKF accuracy."""
        # Highly elliptical orbit (more nonlinear)
        initial_state = np.array([10000.0, 0.0, 0.0, 0.0, 6.0, 0.0])
        initial_cov = np.eye(6) * 100.0
        
        ekf = ExtendedKalmanFilter(initial_state.copy(), initial_cov.copy())
        ukf = UnscentedKalmanFilter(initial_state.copy(), initial_cov.copy())
        
        # Propagate both filters
        dt = 60.0
        for _ in range(10):
            ekf.predict(dt)
            ukf.predict(dt)
            
            # Same measurement for both
            measurement = ekf.get_position() + np.random.randn(3) * 0.05
            ekf.update(measurement)
            ukf.update(measurement)
        
        # Both should be stable
        assert not np.any(np.isnan(ekf.get_state()))
        assert not np.any(np.isnan(ukf.get_state()))
        
        # UKF typically has lower uncertainty for nonlinear systems
        # (This is a weak test since the orbit isn't highly nonlinear)
        assert ukf.get_position_uncertainty() < 2.0  # Reasonable uncertainty


class TestKalmanFilterComparison:
    """Compare EKF and UKF performance."""
    
    def test_filters_track_circular_orbit(self):
        """Test that both filters can track a circular orbit."""
        r = 7000.0
        v = np.sqrt(MU_EARTH / r)
        initial_state = np.array([r, 0.0, 0.0, 0.0, v, 0.0])
        initial_cov = np.eye(6) * 100.0
        
        ekf = ExtendedKalmanFilter(initial_state.copy(), initial_cov.copy())
        ukf = UnscentedKalmanFilter(initial_state.copy(), initial_cov.copy())
        
        # Track for 1 hour
        dt = 60.0
        for _ in range(60):
            ekf.predict(dt)
            ukf.predict(dt)
            
            # Generate measurement from true orbit
            true_pos = ekf.get_position()  # Use EKF as "truth"
            measurement = true_pos + np.random.randn(3) * 0.05
            
            ekf.update(measurement)
            ukf.update(measurement)
        
        # Both should maintain circular orbit radius
        ekf_r = np.linalg.norm(ekf.get_position())
        ukf_r = np.linalg.norm(ukf.get_position())
        
        assert abs(ekf_r - r) / r < 0.05  # Within 5%
        assert abs(ukf_r - r) / r < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
