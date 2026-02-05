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
from src.tracking.data_association import (
    Measurement,
    Association,
    CostCalculator,
    HungarianAssociator,
    GNNAssociator,
    compute_association_metrics
)
from src.tracking.track_manager import (
    Track,
    TrackState,
    TrackManager
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


class TestDataAssociation:
    """Test data association algorithms."""
    
    def test_measurement_creation(self):
        """Test creating a Measurement object."""
        pos = np.array([7000.0, 0.0, 0.0])
        cov = np.eye(3) * 0.05**2  # 50m std dev
        
        meas = Measurement(
            position=pos,
            covariance=cov,
            timestamp=0.0,
            sensor_id="sensor_1",
            measurement_id=1
        )
        
        assert np.allclose(meas.position, pos)
        assert np.allclose(meas.covariance, cov)
        assert meas.measurement_id == 1
    
    def test_mahalanobis_distance(self):
        """Test Mahalanobis distance calculation."""
        calc = CostCalculator()
        
        # Predicted position and covariance
        pred_pos = np.array([7000.0, 0.0, 0.0])
        pred_cov = np.eye(3) * 0.1**2  # 100m std dev
        
        # Measurement close to prediction
        meas_pos = np.array([7000.05, 0.0, 0.0])  # 50m away
        meas_cov = np.eye(3) * 0.05**2  # 50m std dev
        
        distance = calc.mahalanobis_distance(pred_pos, pred_cov, meas_pos, meas_cov)
        
        # Distance should be reasonable (not too large)
        assert distance > 0
        assert distance < 10.0  # Should be well within gate
    
    def test_gating(self):
        """Test measurement gating."""
        calc = CostCalculator(gate_threshold=9.0)
        
        pred_pos = np.array([7000.0, 0.0, 0.0])
        pred_cov = np.eye(3) * 0.1**2
        
        # Close measurement (should pass gate)
        close_meas = np.array([7000.05, 0.0, 0.0])
        meas_cov = np.eye(3) * 0.05**2
        
        assert calc.gate_measurement(pred_pos, pred_cov, close_meas, meas_cov) is True
        
        # Far measurement (should fail gate)
        far_meas = np.array([7010.0, 0.0, 0.0])  # 10 km away
        assert calc.gate_measurement(pred_pos, pred_cov, far_meas, meas_cov) is False
    
    def test_cost_matrix_construction(self):
        """Test building cost matrix."""
        calc = CostCalculator()
        
        # Two track predictions
        track_preds = [
            (np.array([7000.0, 0.0, 0.0]), np.eye(3) * 0.1**2),
            (np.array([7000.0, 100.0, 0.0]), np.eye(3) * 0.1**2),
        ]
        
        # Two measurements
        measurements = [
            Measurement(np.array([7000.05, 0.0, 0.0]), np.eye(3) * 0.05**2, 0.0, "s1", 1),
            Measurement(np.array([7000.0, 100.05, 0.0]), np.eye(3) * 0.05**2, 0.0, "s1", 2),
        ]
        
        cost_matrix = calc.build_cost_matrix(track_preds, measurements)
        
        # Should be 2x2
        assert cost_matrix.shape == (2, 2)
        
        # Diagonal should have low cost (correct associations)
        assert cost_matrix[0, 0] < 1.0
        assert cost_matrix[1, 1] < 1.0
        
        # Off-diagonal should have high cost or be gated out
        assert cost_matrix[0, 1] > cost_matrix[0, 0]
        assert cost_matrix[1, 0] > cost_matrix[1, 1]
    
    def test_hungarian_association(self):
        """Test Hungarian algorithm association."""
        associator = HungarianAssociator(gate_threshold=9.0, max_cost=100.0)
        
        # Three tracks
        track_preds = [
            (1, np.array([7000.0, 0.0, 0.0]), np.eye(3) * 0.1**2),
            (2, np.array([7000.0, 100.0, 0.0]), np.eye(3) * 0.1**2),
            (3, np.array([7000.0, 200.0, 0.0]), np.eye(3) * 0.1**2),
        ]
        
        # Two measurements (one track will be unassociated)
        measurements = [
            Measurement(np.array([7000.05, 0.0, 0.0]), np.eye(3) * 0.05**2, 0.0, "s1", 1),
            Measurement(np.array([7000.0, 100.05, 0.0]), np.eye(3) * 0.05**2, 0.0, "s1", 2),
        ]
        
        associations, unassoc_tracks, unassoc_meas = associator.associate(track_preds, measurements)
        
        # Should have 2 associations
        assert len(associations) == 2
        
        # Should have 1 unassociated track
        assert len(unassoc_tracks) == 1
        assert 3 in unassoc_tracks
        
        # No unassociated measurements
        assert len(unassoc_meas) == 0
        
        # Check associations are correct
        assoc_dict = {a.track_id: a.measurement_id for a in associations}
        assert assoc_dict[1] == 1
        assert assoc_dict[2] == 2
    
    def test_gnn_association(self):
        """Test GNN association."""
        associator = GNNAssociator(gate_threshold=9.0, max_cost=100.0)
        
        # Same setup as Hungarian test
        track_preds = [
            (1, np.array([7000.0, 0.0, 0.0]), np.eye(3) * 0.1**2),
            (2, np.array([7000.0, 100.0, 0.0]), np.eye(3) * 0.1**2),
        ]
        
        measurements = [
            Measurement(np.array([7000.05, 0.0, 0.0]), np.eye(3) * 0.05**2, 0.0, "s1", 1),
            Measurement(np.array([7000.0, 100.05, 0.0]), np.eye(3) * 0.05**2, 0.0, "s1", 2),
        ]
        
        associations, unassoc_tracks, unassoc_meas = associator.associate(track_preds, measurements)
        
        # Should have 2 associations
        assert len(associations) == 2
        
        # No unassociated
        assert len(unassoc_tracks) == 0
        assert len(unassoc_meas) == 0
    
    def test_association_with_no_tracks(self):
        """Test association when there are no tracks."""
        associator = HungarianAssociator()
        
        measurements = [
            Measurement(np.array([7000.0, 0.0, 0.0]), np.eye(3) * 0.05**2, 0.0, "s1", 1),
        ]
        
        associations, unassoc_tracks, unassoc_meas = associator.associate([], measurements)
        
        assert len(associations) == 0
        assert len(unassoc_tracks) == 0
        assert len(unassoc_meas) == 1
    
    def test_association_with_no_measurements(self):
        """Test association when there are no measurements."""
        associator = HungarianAssociator()
        
        track_preds = [
            (1, np.array([7000.0, 0.0, 0.0]), np.eye(3) * 0.1**2),
        ]
        
        associations, unassoc_tracks, unassoc_meas = associator.associate(track_preds, [])
        
        assert len(associations) == 0
        assert len(unassoc_tracks) == 1
        assert len(unassoc_meas) == 0
    
    def test_association_metrics(self):
        """Test computing association metrics."""
        associations = [
            Association(track_id=1, measurement_id=1, cost=0.5),
            Association(track_id=2, measurement_id=2, cost=1.0),
            Association(track_id=3, measurement_id=3, cost=0.8),
        ]
        
        metrics = compute_association_metrics(associations)
        
        assert metrics['num_associations'] == 3
        assert metrics['mean_cost'] == pytest.approx((0.5 + 1.0 + 0.8) / 3)
        assert metrics['max_cost'] == 1.0
        
        # Test with ground truth
        ground_truth = {1: 1, 2: 2, 3: 4}  # Track 3 is wrong
        metrics = compute_association_metrics(associations, ground_truth)
        
        assert metrics['accuracy'] == pytest.approx(2.0 / 3.0)


class TestTrackManager:
    """Test track management functionality."""
    
    def test_track_creation(self):
        """Test creating a new track."""
        manager = TrackManager()
        
        meas = Measurement(
            position=np.array([7000.0, 0.0, 0.0]),
            covariance=np.eye(3) * 0.05**2,
            timestamp=0.0,
            sensor_id="s1",
            measurement_id=1
        )
        
        track = manager.create_track(meas)
        
        assert track.track_id == 1
        assert track.state == TrackState.TENTATIVE
        assert track.hit_count == 1
        assert track.miss_count == 0
        assert np.allclose(track.get_position(), meas.position)
    
    def test_track_confirmation(self):
        """Test track confirmation after sufficient hits."""
        manager = TrackManager(confirmation_threshold=3)
        
        # Create track
        meas1 = Measurement(
            position=np.array([7000.0, 0.0, 0.0]),
            covariance=np.eye(3) * 0.05**2,
            timestamp=0.0,
            sensor_id="s1",
            measurement_id=1
        )
        track = manager.create_track(meas1)
        assert track.state == TrackState.TENTATIVE
        
        # Update with 2 more measurements
        for i in range(2):
            meas = Measurement(
                position=np.array([7000.0 + 0.01 * (i+1), 0.0, 0.0]),
                covariance=np.eye(3) * 0.05**2,
                timestamp=(i+1) * 10.0,
                sensor_id="s1",
                measurement_id=i+2
            )
            manager.update_track(track.track_id, meas)
        
        # Should be confirmed now
        assert track.state == TrackState.CONFIRMED
        assert track.hit_count == 3
    
    def test_track_coasting(self):
        """Test track entering coast state after misses."""
        manager = TrackManager(confirmation_threshold=2, coast_threshold=3)
        
        # Create and confirm track
        meas1 = Measurement(
            position=np.array([7000.0, 0.0, 0.0]),
            covariance=np.eye(3) * 0.05**2,
            timestamp=0.0,
            sensor_id="s1",
            measurement_id=1
        )
        track = manager.create_track(meas1)
        
        meas2 = Measurement(
            position=np.array([7000.01, 0.0, 0.0]),
            covariance=np.eye(3) * 0.05**2,
            timestamp=10.0,
            sensor_id="s1",
            measurement_id=2
        )
        manager.update_track(track.track_id, meas2)
        
        assert track.state == TrackState.CONFIRMED
        
        # Miss 3 times
        for i in range(3):
            manager.predict_track(track.track_id, 10.0)
        
        # Should be coasting
        assert track.state == TrackState.COASTED
        assert track.miss_count == 3
    
    def test_track_deletion(self):
        """Test track deletion after too many misses."""
        manager = TrackManager(deletion_threshold=5)
        
        # Create track
        meas = Measurement(
            position=np.array([7000.0, 0.0, 0.0]),
            covariance=np.eye(3) * 0.05**2,
            timestamp=0.0,
            sensor_id="s1",
            measurement_id=1
        )
        track = manager.create_track(meas)
        track_id = track.track_id
        
        # Miss 5 times
        for i in range(5):
            manager.predict_track(track_id, 10.0)
        
        # Prune tracks
        manager.prune_tracks(50.0)
        
        # Track should be deleted
        assert track_id not in manager.tracks
    
    def test_track_coast_timeout(self):
        """Test track deletion after coasting too long."""
        manager = TrackManager(
            confirmation_threshold=2,
            coast_threshold=2,
            max_coast_time=100.0
        )
        
        # Create and confirm track
        meas1 = Measurement(
            position=np.array([7000.0, 0.0, 0.0]),
            covariance=np.eye(3) * 0.05**2,
            timestamp=0.0,
            sensor_id="s1",
            measurement_id=1
        )
        track = manager.create_track(meas1)
        
        meas2 = Measurement(
            position=np.array([7000.01, 0.0, 0.0]),
            covariance=np.eye(3) * 0.05**2,
            timestamp=10.0,
            sensor_id="s1",
            measurement_id=2
        )
        manager.update_track(track.track_id, meas2)
        
        # Enter coast state
        manager.predict_track(track.track_id, 10.0)
        manager.predict_track(track.track_id, 10.0)
        assert track.state == TrackState.COASTED
        
        # Prune after coast timeout
        manager.prune_tracks(200.0)  # 200s > 100s timeout
        
        # Track should be deleted
        assert track.track_id not in manager.tracks
    
    def test_track_return_from_coast(self):
        """Test track returning from coast state with new measurement."""
        manager = TrackManager(confirmation_threshold=2, coast_threshold=2)
        
        # Create and confirm track
        meas1 = Measurement(
            position=np.array([7000.0, 0.0, 0.0]),
            covariance=np.eye(3) * 0.05**2,
            timestamp=0.0,
            sensor_id="s1",
            measurement_id=1
        )
        track = manager.create_track(meas1)
        
        meas2 = Measurement(
            position=np.array([7000.01, 0.0, 0.0]),
            covariance=np.eye(3) * 0.05**2,
            timestamp=10.0,
            sensor_id="s1",
            measurement_id=2
        )
        manager.update_track(track.track_id, meas2)
        
        # Enter coast
        manager.predict_track(track.track_id, 10.0)
        manager.predict_track(track.track_id, 10.0)
        assert track.state == TrackState.COASTED
        
        # New measurement
        meas3 = Measurement(
            position=np.array([7000.02, 0.0, 0.0]),
            covariance=np.eye(3) * 0.05**2,
            timestamp=40.0,
            sensor_id="s1",
            measurement_id=3
        )
        manager.update_track(track.track_id, meas3)
        
        # Should return to confirmed
        assert track.state == TrackState.CONFIRMED
        assert track.miss_count == 0
    
    def test_get_confirmed_tracks(self):
        """Test retrieving confirmed tracks."""
        manager = TrackManager(confirmation_threshold=2)
        
        # Create 3 tracks
        for i in range(3):
            meas = Measurement(
                position=np.array([7000.0 + i * 100.0, 0.0, 0.0]),
                covariance=np.eye(3) * 0.05**2,
                timestamp=0.0,
                sensor_id="s1",
                measurement_id=i+1
            )
            track = manager.create_track(meas)
            
            # Confirm 2 of them
            if i < 2:
                meas2 = Measurement(
                    position=np.array([7000.0 + i * 100.0 + 0.01, 0.0, 0.0]),
                    covariance=np.eye(3) * 0.05**2,
                    timestamp=10.0,
                    sensor_id="s1",
                    measurement_id=i+10
                )
                manager.update_track(track.track_id, meas2)
        
        confirmed = manager.get_confirmed_tracks()
        assert len(confirmed) == 2
    
    def test_track_statistics(self):
        """Test computing track statistics."""
        manager = TrackManager(confirmation_threshold=2)
        
        # Create and confirm 2 tracks
        for i in range(2):
            meas = Measurement(
                position=np.array([7000.0 + i * 100.0, 0.0, 0.0]),
                covariance=np.eye(3) * 0.05**2,
                timestamp=0.0,
                sensor_id="s1",
                measurement_id=i+1
            )
            track = manager.create_track(meas)
            
            meas2 = Measurement(
                position=np.array([7000.0 + i * 100.0 + 0.01, 0.0, 0.0]),
                covariance=np.eye(3) * 0.05**2,
                timestamp=10.0,
                sensor_id="s1",
                measurement_id=i+10
            )
            manager.update_track(track.track_id, meas2)
        
        stats = manager.get_statistics()
        
        assert stats['total_tracks'] == 2
        assert stats['confirmed_tracks'] == 2
        assert stats['mean_hit_count'] == 2.0
    
    def test_ukf_filter_type(self):
        """Test creating tracks with UKF filter."""
        manager = TrackManager(filter_type="ukf")
        
        meas = Measurement(
            position=np.array([7000.0, 0.0, 0.0]),
            covariance=np.eye(3) * 0.05**2,
            timestamp=0.0,
            sensor_id="s1",
            measurement_id=1
        )
        
        track = manager.create_track(meas)
        
        # Check that filter is UKF
        assert isinstance(track.filter, UnscentedKalmanFilter)


class TestManeuverDetection:
    """Test maneuver detection algorithms."""
    
    def test_innovation_detector_initialization(self):
        """Test creating an InnovationDetector."""
        from src.tracking.maneuver_detection import InnovationDetector
        
        detector = InnovationDetector(threshold=13.8, window_size=3)
        
        assert detector.threshold == 13.8
        assert detector.window_size == 3
        assert len(detector.innovation_history) == 0
    
    def test_innovation_detector_no_maneuver(self):
        """Test that small innovations don't trigger detection."""
        from src.tracking.maneuver_detection import InnovationDetector
        
        detector = InnovationDetector(threshold=13.8, window_size=3, min_detections=2)
        
        # Small innovation (normal tracking)
        innovation = np.array([0.01, 0.01, 0.01])  # 10m error
        innovation_cov = np.eye(3) * 0.1**2
        
        event = detector.detect(innovation, innovation_cov, 0.0, track_id=1)
        
        assert event is None
    
    def test_innovation_detector_maneuver(self):
        """Test that large innovations trigger detection."""
        from src.tracking.maneuver_detection import InnovationDetector
        
        detector = InnovationDetector(threshold=9.0, window_size=3, min_detections=2)
        
        # Large innovations (maneuver)
        innovation = np.array([1.0, 1.0, 1.0])  # 1km error
        innovation_cov = np.eye(3) * 0.05**2
        
        # Need multiple detections
        event1 = detector.detect(innovation, innovation_cov, 0.0, track_id=1)
        event2 = detector.detect(innovation, innovation_cov, 10.0, track_id=1)
        event3 = detector.detect(innovation, innovation_cov, 20.0, track_id=1)
        
        # Should detect after min_detections
        assert event2 is not None or event3 is not None
    
    def test_innovation_detector_reset(self):
        """Test resetting detector state."""
        from src.tracking.maneuver_detection import InnovationDetector
        
        detector = InnovationDetector()
        
        # Add some history
        innovation = np.array([0.1, 0.1, 0.1])
        innovation_cov = np.eye(3) * 0.1**2
        detector.detect(innovation, innovation_cov, 0.0, track_id=1)
        
        assert 1 in detector.innovation_history
        
        # Reset
        detector.reset(track_id=1)
        
        assert 1 not in detector.innovation_history
    
    def test_mmae_detector(self):
        """Test MMAE detector initialization."""
        from src.tracking.maneuver_detection import MMAEDetector
        
        detector = MMAEDetector(threshold=0.7)
        
        assert detector.threshold == 0.7
        assert len(detector.model_probabilities) == 0


class TestMultiObjectTracker:
    """Test multi-object tracker."""
    
    def test_tracker_initialization(self):
        """Test creating a MultiObjectTracker."""
        from src.tracking.multi_object_tracker import MultiObjectTracker, TrackerConfig
        
        config = TrackerConfig(filter_type="ekf", association_method="hungarian")
        tracker = MultiObjectTracker(config)
        
        assert tracker.config.filter_type == "ekf"
        assert tracker.config.association_method == "hungarian"
        assert tracker.update_count == 0
    
    def test_tracker_single_measurement(self):
        """Test tracker with single measurement."""
        from src.tracking.multi_object_tracker import MultiObjectTracker
        from src.tracking.data_association import Measurement
        
        tracker = MultiObjectTracker()
        
        # Single measurement
        meas = Measurement(
            position=np.array([7000.0, 0.0, 0.0]),
            covariance=np.eye(3) * 0.05**2,
            timestamp=0.0,
            sensor_id="s1",
            measurement_id=1
        )
        
        tracks = tracker.update([meas], 0.0)
        
        # Should create one track
        assert len(tracks) == 1
        assert tracks[0].track_id == 1
    
    def test_tracker_multiple_updates(self):
        """Test tracker with multiple updates."""
        from src.tracking.multi_object_tracker import MultiObjectTracker
        from src.tracking.data_association import Measurement
        
        tracker = MultiObjectTracker()
        
        # First measurement
        meas1 = Measurement(
            position=np.array([7000.0, 0.0, 0.0]),
            covariance=np.eye(3) * 0.05**2,
            timestamp=0.0,
            sensor_id="s1",
            measurement_id=1
        )
        
        tracks = tracker.update([meas1], 0.0)
        assert len(tracks) == 1
        
        # Second measurement (close to first)
        meas2 = Measurement(
            position=np.array([7000.01, 0.0, 0.0]),
            covariance=np.eye(3) * 0.05**2,
            timestamp=10.0,
            sensor_id="s1",
            measurement_id=2
        )
        
        tracks = tracker.update([meas2], 10.0)
        
        # Should still be one track (associated)
        assert len(tracks) == 1
        assert tracks[0].hit_count == 2
    
    def test_tracker_multiple_objects(self):
        """Test tracker with multiple objects."""
        from src.tracking.multi_object_tracker import MultiObjectTracker
        from src.tracking.data_association import Measurement
        
        tracker = MultiObjectTracker()
        
        # Two measurements far apart
        measurements = [
            Measurement(
                position=np.array([7000.0, 0.0, 0.0]),
                covariance=np.eye(3) * 0.05**2,
                timestamp=0.0,
                sensor_id="s1",
                measurement_id=1
            ),
            Measurement(
                position=np.array([7000.0, 1000.0, 0.0]),
                covariance=np.eye(3) * 0.05**2,
                timestamp=0.0,
                sensor_id="s1",
                measurement_id=2
            )
        ]
        
        tracks = tracker.update(measurements, 0.0)
        
        # Should create two tracks
        assert len(tracks) == 2
    
    def test_tracker_statistics(self):
        """Test tracker statistics."""
        from src.tracking.multi_object_tracker import MultiObjectTracker
        from src.tracking.data_association import Measurement
        
        tracker = MultiObjectTracker()
        
        meas = Measurement(
            position=np.array([7000.0, 0.0, 0.0]),
            covariance=np.eye(3) * 0.05**2,
            timestamp=0.0,
            sensor_id="s1",
            measurement_id=1
        )
        
        tracker.update([meas], 0.0)
        
        stats = tracker.get_statistics()
        
        assert stats['update_count'] == 1
        assert stats['total_measurements'] == 1
        assert stats['total_tracks'] == 1
    
    def test_tracker_reset(self):
        """Test resetting tracker."""
        from src.tracking.multi_object_tracker import MultiObjectTracker
        from src.tracking.data_association import Measurement
        
        tracker = MultiObjectTracker()
        
        meas = Measurement(
            position=np.array([7000.0, 0.0, 0.0]),
            covariance=np.eye(3) * 0.05**2,
            timestamp=0.0,
            sensor_id="s1",
            measurement_id=1
        )
        
        tracker.update([meas], 0.0)
        assert tracker.update_count == 1
        
        tracker.reset()
        
        assert tracker.update_count == 0
        assert len(tracker.track_manager.tracks) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
