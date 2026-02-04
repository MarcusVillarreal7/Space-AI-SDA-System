"""
Unit tests for utility modules.
"""

import pytest
import numpy as np
from pathlib import Path

from src.utils.config_loader import Config, SimulationConfig, TrackingConfig, MLConfig
from src.utils.metrics import PerformanceMetrics, rmse, mae, position_error
from src.utils.coordinates import (
    eci_to_ecef,
    ecef_to_eci,
    ecef_to_geodetic,
    geodetic_to_ecef,
    orbital_elements_to_state_vector,
)


class TestConfigLoader:
    """Test configuration loading and validation."""
    
    def test_simulation_config_defaults(self):
        """Test default simulation configuration."""
        config = SimulationConfig()
        assert config.num_objects == 100
        assert config.duration_hours == 24.0
        assert config.time_step_seconds == 60.0
        assert config.num_sensors == 3
    
    def test_simulation_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = SimulationConfig(num_objects=50)
        assert config.num_objects == 50
        
        # Invalid config (negative objects)
        with pytest.raises(ValueError):
            SimulationConfig(num_objects=-1)
    
    def test_tracking_config_defaults(self):
        """Test default tracking configuration."""
        config = TrackingConfig()
        assert config.process_noise_std == 1.0
        assert config.track_init_threshold == 3
        assert config.association_method == "hungarian"
    
    def test_ml_config_defaults(self):
        """Test default ML configuration."""
        config = MLConfig()
        assert config.predictor_hidden_dim == 256
        assert config.predictor_num_layers == 4
        assert config.num_classes == 4


class TestMetrics:
    """Test performance metrics."""
    
    def test_performance_metrics_recording(self):
        """Test metric recording."""
        metrics = PerformanceMetrics()
        metrics.record("latency", 0.1)
        metrics.record("latency", 0.2)
        metrics.record("latency", 0.15)
        
        stats = metrics.get_stats("latency")
        assert stats["count"] == 3
        assert stats["mean"] == pytest.approx(0.15, abs=0.01)
    
    def test_rmse(self):
        """Test RMSE calculation."""
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.1, 2.1, 2.9])
        
        error = rmse(predictions, targets)
        assert error == pytest.approx(0.1, abs=0.01)
    
    def test_mae(self):
        """Test MAE calculation."""
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.1, 2.1, 2.9])
        
        error = mae(predictions, targets)
        assert error == pytest.approx(0.1, abs=0.01)
    
    def test_position_error(self):
        """Test 3D position error."""
        pred = np.array([1.0, 0.0, 0.0])
        true = np.array([0.0, 0.0, 0.0])
        
        error = position_error(pred, true)
        assert error == pytest.approx(1.0)


class TestCoordinates:
    """Test coordinate transformations."""
    
    def test_eci_ecef_roundtrip(self):
        """Test ECI to ECEF and back."""
        pos_eci = np.array([7000.0, 0.0, 0.0])
        gmst = 0.0
        
        pos_ecef = eci_to_ecef(pos_eci, gmst)
        pos_eci_back = ecef_to_eci(pos_ecef, gmst)
        
        np.testing.assert_array_almost_equal(pos_eci, pos_eci_back, decimal=10)
    
    def test_geodetic_ecef_roundtrip(self):
        """Test geodetic to ECEF and back."""
        lat = np.radians(45.0)
        lon = np.radians(0.0)
        alt = 0.0
        
        pos_ecef = geodetic_to_ecef(lat, lon, alt)
        lat_back, lon_back, alt_back = ecef_to_geodetic(pos_ecef)
        
        assert lat == pytest.approx(lat_back, abs=1e-6)
        assert lon == pytest.approx(lon_back, abs=1e-6)
        assert alt == pytest.approx(alt_back, abs=0.01)
    
    def test_orbital_elements_circular(self):
        """Test orbital elements for circular orbit."""
        # Circular orbit at 7000 km
        a = 7000.0  # Semi-major axis
        e = 0.0     # Eccentricity (circular)
        i = 0.0     # Inclination
        omega = 0.0 # RAAN
        w = 0.0     # Argument of periapsis
        nu = 0.0    # True anomaly
        
        pos, vel = orbital_elements_to_state_vector(a, e, i, omega, w, nu)
        
        # For circular orbit at nu=0, position should be [a, 0, 0]
        assert pos[0] == pytest.approx(a, abs=0.1)
        assert pos[1] == pytest.approx(0.0, abs=0.1)
        assert pos[2] == pytest.approx(0.0, abs=0.1)
        
        # Velocity should be perpendicular to position
        assert np.dot(pos, vel) == pytest.approx(0.0, abs=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
