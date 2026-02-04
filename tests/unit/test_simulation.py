"""
Unit tests for simulation layer.
Tests TLE loading, orbital mechanics, sensors, noise, and data generation.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.simulation.tle_loader import TLE, TLELoader
from src.simulation.orbital_mechanics import SGP4Propagator, StateVector, OrbitalElements
from src.simulation.sensor_models import RadarSensor, OpticalSensor, Measurement
from src.simulation.noise_models import GaussianNoise, SystematicBias, CorrelatedNoise
from src.simulation.data_generator import DatasetGenerator, Dataset
from src.utils.config_loader import SimulationConfig


class TestTLE:
    """Test TLE data class."""
    
    def test_tle_creation(self):
        """Test TLE creation from lines."""
        name = "ISS (ZARYA)"
        line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
        line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
        
        tle = TLE.from_lines(name, line1, line2)
        
        assert tle.name == name
        assert tle.catalog_number == 25544
        assert tle.line1 == line1
        assert tle.line2 == line2
        assert tle.epoch > 0
    
    def test_tle_repr(self):
        """Test TLE string representation."""
        name = "TEST SAT"
        line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
        line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
        
        tle = TLE.from_lines(name, line1, line2)
        repr_str = repr(tle)
        
        assert "TEST SAT" in repr_str
        assert "25544" in repr_str


class TestTLELoader:
    """Test TLE loader."""
    
    def test_loader_initialization(self):
        """Test TLE loader initialization."""
        loader = TLELoader()
        assert loader.tles == []
    
    def test_filter_by_altitude(self):
        """Test altitude filtering."""
        # Create mock TLEs with different mean motions
        # Higher mean motion = lower altitude
        loader = TLELoader()
        
        # LEO satellite (mean motion ~15 rev/day, ~400km altitude)
        leo_line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
        leo_line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
        leo_tle = TLE.from_lines("LEO", leo_line1, leo_line2)
        
        loader.tles = [leo_tle]
        
        # Filter for LEO altitudes (200-2000 km)
        filtered = loader.filter_by_altitude(200, 2000)
        assert len(filtered) == 1


class TestStateVector:
    """Test StateVector data class."""
    
    def test_state_vector_creation(self):
        """Test StateVector creation."""
        time = datetime.now(timezone.utc)
        position = np.array([7000.0, 0.0, 0.0])
        velocity = np.array([0.0, 7.5, 0.0])
        
        state = StateVector(time=time, position=position, velocity=velocity)
        
        assert np.array_equal(state.position, position)
        assert np.array_equal(state.velocity, velocity)
        assert state.frame == "ECI"
    
    def test_state_vector_speed(self):
        """Test speed calculation."""
        time = datetime.now(timezone.utc)
        velocity = np.array([3.0, 4.0, 0.0])
        
        state = StateVector(
            time=time,
            position=np.array([7000.0, 0.0, 0.0]),
            velocity=velocity
        )
        
        assert state.speed == pytest.approx(5.0)  # 3-4-5 triangle
    
    def test_state_vector_altitude(self):
        """Test altitude calculation."""
        time = datetime.now(timezone.utc)
        # Position at ~400 km altitude
        position = np.array([6778.137, 0.0, 0.0])  # Earth radius + 400 km
        
        state = StateVector(
            time=time,
            position=position,
            velocity=np.array([0.0, 7.5, 0.0])
        )
        
        assert state.altitude == pytest.approx(400.0, abs=1.0)
    
    def test_state_vector_to_dict(self):
        """Test dictionary conversion."""
        time = datetime.now(timezone.utc)
        state = StateVector(
            time=time,
            position=np.array([7000.0, 0.0, 0.0]),
            velocity=np.array([0.0, 7.5, 0.0])
        )
        
        state_dict = state.to_dict()
        
        assert 'time' in state_dict
        assert 'position_x' in state_dict
        assert 'velocity_x' in state_dict
        assert 'altitude_km' in state_dict
        assert 'speed_km_s' in state_dict


class TestSGP4Propagator:
    """Test SGP4 propagator."""
    
    @pytest.fixture
    def iss_tle(self):
        """ISS TLE for testing."""
        name = "ISS (ZARYA)"
        line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
        line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
        return TLE.from_lines(name, line1, line2)
    
    def test_propagator_initialization(self, iss_tle):
        """Test propagator initialization."""
        propagator = SGP4Propagator(iss_tle)
        assert propagator.tle == iss_tle
        assert propagator.satellite is not None
    
    def test_propagate_single_time(self, iss_tle):
        """Test single time propagation."""
        propagator = SGP4Propagator(iss_tle)
        time = datetime(2008, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
        
        state = propagator.propagate(time)
        
        assert isinstance(state, StateVector)
        assert state.position.shape == (3,)
        assert state.velocity.shape == (3,)
        assert state.altitude > 300  # ISS altitude > 300 km
        assert state.altitude < 500  # ISS altitude < 500 km
    
    def test_propagate_batch(self, iss_tle):
        """Test batch propagation."""
        propagator = SGP4Propagator(iss_tle)
        start = datetime(2008, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
        times = [start + timedelta(minutes=i) for i in range(10)]
        
        states = propagator.propagate_batch(times)
        
        assert len(states) == 10
        assert all(isinstance(s, StateVector) for s in states)
        
        # Check positions change over time
        pos_0 = states[0].position
        pos_9 = states[9].position
        distance = np.linalg.norm(pos_9 - pos_0)
        assert distance > 0  # Position should change


class TestRadarSensor:
    """Test radar sensor model."""
    
    def test_radar_initialization(self):
        """Test radar sensor initialization."""
        radar = RadarSensor(
            name="Test-Radar",
            location_lat_lon_alt=(40.0, -105.0, 1.5),
            max_range_km=3000,
            accuracy_m=50
        )
        
        assert radar.name == "Test-Radar"
        assert radar.max_range == 3000
        assert radar.accuracy == pytest.approx(0.05)  # 50m in km
        assert radar.sensor_type == "radar"
    
    def test_radar_visibility_range(self):
        """Test range-based visibility."""
        radar = RadarSensor(
            name="Test-Radar",
            location_lat_lon_alt=(40.0, -105.0, 1.5),
            max_range_km=3000,
            accuracy_m=50
        )
        
        time = datetime.now(timezone.utc)
        
        # Close position (should be visible if above horizon)
        close_pos = np.array([6700.0, 0.0, 0.0])
        
        # Far position (beyond range)
        far_pos = np.array([20000.0, 0.0, 0.0])
        
        # Note: Actual visibility depends on elevation angle
        # Just test that the method runs without error
        radar.can_observe(close_pos, time)
        radar.can_observe(far_pos, time)
    
    def test_radar_measurement(self):
        """Test radar measurement generation."""
        radar = RadarSensor(
            name="Test-Radar",
            location_lat_lon_alt=(40.0, -105.0, 1.5),
            max_range_km=3000,
            accuracy_m=50
        )
        
        time = datetime.now(timezone.utc)
        true_position = np.array([7000.0, 0.0, 0.0])
        
        # Measurement without noise
        measurement = radar.measure(true_position, object_id=1, time=time, add_noise=False)
        
        assert isinstance(measurement, Measurement)
        assert measurement.sensor_id == "Test-Radar"
        assert measurement.object_id == 1
        np.testing.assert_array_almost_equal(measurement.position_measured, true_position)
        
        # Measurement with noise
        noisy_measurement = radar.measure(true_position, object_id=1, time=time, add_noise=True)
        error = np.linalg.norm(noisy_measurement.position_measured - true_position)
        
        # Error should be non-zero and reasonable (< 1km for 50m accuracy)
        assert error > 0
        assert error < 1.0  # Should be within 1 km


class TestOpticalSensor:
    """Test optical sensor model."""
    
    def test_optical_initialization(self):
        """Test optical sensor initialization."""
        optical = OpticalSensor(
            name="Test-Optical",
            location_lat_lon_alt=(19.8, -155.5, 4.2),
            max_range_km=40000,
            accuracy_m=500
        )
        
        assert optical.name == "Test-Optical"
        assert optical.max_range == 40000
        assert optical.accuracy == pytest.approx(0.5)  # 500m in km
        assert optical.sensor_type == "optical"
    
    def test_optical_measurement(self):
        """Test optical measurement generation."""
        optical = OpticalSensor(
            name="Test-Optical",
            location_lat_lon_alt=(19.8, -155.5, 4.2),
            max_range_km=40000,
            accuracy_m=500
        )
        
        time = datetime.now(timezone.utc)
        true_position = np.array([7000.0, 0.0, 0.0])
        
        measurement = optical.measure(true_position, object_id=1, time=time, add_noise=True)
        
        assert isinstance(measurement, Measurement)
        assert measurement.sensor_id == "Test-Optical"


class TestGaussianNoise:
    """Test Gaussian noise model."""
    
    def test_gaussian_noise_statistics(self):
        """Test Gaussian noise has correct statistics."""
        std_dev = 0.05  # 50m
        noise_model = GaussianNoise(std_dev=std_dev, seed=42)
        
        # Generate many samples
        samples = np.array([
            noise_model.add_noise(np.zeros(3))
            for _ in range(1000)
        ])
        
        # Check mean is close to zero
        mean = np.mean(samples)
        assert abs(mean) < 0.01  # Within 10m of zero
        
        # Check std dev is close to target
        measured_std = np.std(samples)
        assert measured_std == pytest.approx(std_dev, rel=0.1)  # Within 10%
    
    def test_gaussian_covariance(self):
        """Test covariance matrix generation."""
        std_dev = 0.05
        noise_model = GaussianNoise(std_dev=std_dev, seed=42)
        
        cov = noise_model.get_covariance_matrix(dim=3)
        
        assert cov.shape == (3, 3)
        # Diagonal should be variance
        np.testing.assert_array_almost_equal(np.diag(cov), [std_dev**2] * 3)
        # Off-diagonal should be zero
        assert cov[0, 1] == 0
        assert cov[0, 2] == 0
        assert cov[1, 2] == 0


class TestSystematicBias:
    """Test systematic bias model."""
    
    def test_systematic_bias(self):
        """Test systematic bias addition."""
        bias_vector = np.array([0.01, 0.0, 0.0])  # 10m in X
        bias_model = SystematicBias(bias_vector)
        
        position = np.array([7000.0, 0.0, 0.0])
        biased = bias_model.add_bias(position)
        
        expected = position + bias_vector
        np.testing.assert_array_almost_equal(biased, expected)


class TestCorrelatedNoise:
    """Test correlated noise model."""
    
    def test_correlated_noise_initialization(self):
        """Test correlated noise initialization."""
        noise_model = CorrelatedNoise(
            std_dev=0.05,
            correlation_time=60.0,
            time_step=1.0,
            seed=42
        )
        
        assert noise_model.std_dev == 0.05
        assert noise_model.correlation_time == 60.0
        assert noise_model.alpha > 0
        assert noise_model.alpha < 1
    
    def test_correlated_noise_persistence(self):
        """Test that noise is correlated over time."""
        noise_model = CorrelatedNoise(
            std_dev=0.05,
            correlation_time=60.0,
            time_step=1.0,
            seed=42
        )
        
        # Generate two consecutive samples
        sample1 = noise_model.add_noise(np.zeros(3))
        sample2 = noise_model.add_noise(np.zeros(3))
        
        # Samples should be correlated (not independent)
        # Correlation should be positive
        correlation = np.dot(sample1, sample2) / (np.linalg.norm(sample1) * np.linalg.norm(sample2))
        assert correlation > 0  # Should be positively correlated


class TestDatasetGenerator:
    """Test dataset generator."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        config = SimulationConfig(num_objects=10, duration_hours=1.0)
        generator = DatasetGenerator(config)
        
        assert generator.config.num_objects == 10
        assert generator.config.duration_hours == 1.0
    
    def test_sensor_network_creation(self):
        """Test sensor network creation."""
        generator = DatasetGenerator()
        sensors = generator.create_sensor_network()
        
        assert len(sensors) > 0
        assert all(hasattr(s, 'can_observe') for s in sensors)


class TestDataset:
    """Test Dataset class."""
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        import pandas as pd
        
        ground_truth = pd.DataFrame({
            'time': [datetime.now(timezone.utc)],
            'object_id': [0],
            'position_x': [7000.0],
            'position_y': [0.0],
            'position_z': [0.0]
        })
        
        measurements = pd.DataFrame({
            'time': [datetime.now(timezone.utc)],
            'sensor_id': ['Radar-1'],
            'object_id': [0],
            'measured_x': [7000.1],
            'measured_y': [0.0],
            'measured_z': [0.0]
        })
        
        metadata = {'num_objects': 1, 'num_sensors': 1}
        
        dataset = Dataset(
            ground_truth=ground_truth,
            measurements=measurements,
            metadata=metadata
        )
        
        assert len(dataset.ground_truth) == 1
        assert len(dataset.measurements) == 1
    
    def test_dataset_statistics(self):
        """Test dataset statistics calculation."""
        import pandas as pd
        
        ground_truth = pd.DataFrame({
            'time': [datetime.now(timezone.utc)] * 3,
            'object_id': [0, 1, 2],
            'position_x': [7000.0, 7100.0, 7200.0],
            'position_y': [0.0, 0.0, 0.0],
            'position_z': [0.0, 0.0, 0.0]
        })
        
        measurements = pd.DataFrame({
            'time': [datetime.now(timezone.utc)] * 5,
            'sensor_id': ['Radar-1'] * 3 + ['Radar-2'] * 2,
            'object_id': [0, 1, 2, 0, 1],
            'measured_x': [7000.0] * 5,
            'measured_y': [0.0] * 5,
            'measured_z': [0.0] * 5
        })
        
        metadata = {'num_objects': 3, 'num_sensors': 2}
        
        dataset = Dataset(
            ground_truth=ground_truth,
            measurements=measurements,
            metadata=metadata
        )
        
        stats = dataset.get_statistics()
        
        assert stats['num_objects'] == 3
        assert stats['num_sensors'] == 2
        assert stats['num_measurements'] == 5
        assert 'measurements_by_sensor' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
