"""
Sensor models for space object detection and tracking.
Includes radar and optical sensors with realistic characteristics.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional
from abc import ABC, abstractmethod
import numpy as np

from src.utils.coordinates import (
    geodetic_to_ecef,
    range_azimuth_elevation,
    EARTH_RADIUS_KM
)
from src.utils.logging_config import get_logger

logger = get_logger("simulation")


@dataclass
class Measurement:
    """Sensor measurement of a space object."""
    
    time: datetime
    sensor_id: str
    object_id: int
    position_measured: np.ndarray  # [x, y, z] in km (ECI frame)
    position_true: Optional[np.ndarray] = None  # Ground truth (if available)
    covariance: Optional[np.ndarray] = None  # 3x3 covariance matrix
    range_km: Optional[float] = None
    azimuth_rad: Optional[float] = None
    elevation_rad: Optional[float] = None
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        self.position_measured = np.asarray(self.position_measured)
        if self.position_true is not None:
            self.position_true = np.asarray(self.position_true)
        if self.covariance is not None:
            self.covariance = np.asarray(self.covariance)
    
    @property
    def measurement_error(self) -> Optional[float]:
        """Calculate measurement error magnitude (km)."""
        if self.position_true is not None:
            return float(np.linalg.norm(self.position_measured - self.position_true))
        return None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'time': self.time.isoformat(),
            'sensor_id': self.sensor_id,
            'object_id': self.object_id,
            'measured_x': float(self.position_measured[0]),
            'measured_y': float(self.position_measured[1]),
            'measured_z': float(self.position_measured[2]),
            'range_km': self.range_km,
            'azimuth_deg': np.degrees(self.azimuth_rad) if self.azimuth_rad else None,
            'elevation_deg': np.degrees(self.elevation_rad) if self.elevation_rad else None,
            'error_km': self.measurement_error,
        }


class BaseSensor(ABC):
    """Abstract base class for all sensors."""
    
    def __init__(
        self,
        name: str,
        location_lat_lon_alt: Tuple[float, float, float],
        max_range_km: float,
        accuracy_m: float,
        field_of_view_deg: float,
        min_elevation_deg: float = 10.0
    ):
        """
        Initialize sensor.
        
        Args:
            name: Sensor identifier
            location_lat_lon_alt: (latitude_deg, longitude_deg, altitude_km)
            max_range_km: Maximum detection range
            accuracy_m: Measurement accuracy (1-sigma) in meters
            field_of_view_deg: Field of view in degrees
            min_elevation_deg: Minimum elevation angle for detection
        """
        self.name = name
        self.location_geodetic = location_lat_lon_alt
        
        # Convert to ECEF for calculations
        lat_rad = np.radians(location_lat_lon_alt[0])
        lon_rad = np.radians(location_lat_lon_alt[1])
        alt_km = location_lat_lon_alt[2]
        self.location_ecef = geodetic_to_ecef(lat_rad, lon_rad, alt_km)
        
        self.max_range = max_range_km
        self.accuracy = accuracy_m / 1000.0  # Convert to km
        self.fov = np.radians(field_of_view_deg)
        self.min_elevation = np.radians(min_elevation_deg)
        
        logger.info(
            f"Initialized {self.__class__.__name__} '{name}' at "
            f"({location_lat_lon_alt[0]:.2f}°, {location_lat_lon_alt[1]:.2f}°, "
            f"{location_lat_lon_alt[2]:.1f} km)"
        )
    
    def can_observe(self, target_position_eci: np.ndarray, time: datetime) -> bool:
        """
        Check if target is visible to sensor.
        
        Args:
            target_position_eci: Target position in ECI frame (km)
            time: Observation time
        
        Returns:
            True if target is visible
        """
        # Calculate range, azimuth, elevation
        range_km, azimuth, elevation = range_azimuth_elevation(
            self.location_ecef,
            target_position_eci
        )
        
        # Check range limit
        if range_km > self.max_range:
            return False
        
        # Check minimum elevation (above horizon)
        if elevation < self.min_elevation:
            return False
        
        # Check FOV (simplified: elevation-based cone)
        # More sophisticated sensors might have azimuth-dependent FOV
        if elevation > np.pi/2:  # Above 90° (shouldn't happen)
            return False
        
        # Check Earth occultation (line of sight)
        if self._is_occluded(target_position_eci):
            return False
        
        return True
    
    def _is_occluded(self, target_position_eci: np.ndarray) -> bool:
        """
        Check if target is occluded by Earth.
        
        Args:
            target_position_eci: Target position in ECI frame (km)
        
        Returns:
            True if occluded
        """
        # Vector from sensor to target
        sensor_to_target = target_position_eci - self.location_ecef
        
        # Distance from Earth center to line of sight
        # Using point-to-line distance formula
        sensor_mag = np.linalg.norm(self.location_ecef)
        target_mag = np.linalg.norm(target_position_eci)
        
        # If both sensor and target are above surface, check line of sight
        if sensor_mag > EARTH_RADIUS_KM and target_mag > EARTH_RADIUS_KM:
            # Cross product gives perpendicular distance
            cross = np.cross(self.location_ecef, sensor_to_target)
            distance_to_center = np.linalg.norm(cross) / np.linalg.norm(sensor_to_target)
            
            # If line of sight passes through Earth, it's occluded
            if distance_to_center < EARTH_RADIUS_KM:
                return True
        
        return False
    
    @abstractmethod
    def measure(
        self,
        target_position_eci: np.ndarray,
        object_id: int,
        time: datetime,
        add_noise: bool = True
    ) -> Measurement:
        """
        Generate measurement of target.
        
        Args:
            target_position_eci: True target position in ECI frame (km)
            object_id: Object identifier
            time: Measurement time
            add_noise: Whether to add measurement noise
        
        Returns:
            Measurement object
        """
        pass
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"location={self.location_geodetic}, "
            f"range={self.max_range}km, "
            f"accuracy={self.accuracy*1000:.0f}m)"
        )


class RadarSensor(BaseSensor):
    """
    Ground-based radar sensor model.
    
    Characteristics:
    - Medium range (typically 1000-5000 km)
    - High accuracy (10-100m)
    - Wide field of view (60-120°)
    - All-weather operation
    """
    
    def __init__(
        self,
        name: str,
        location_lat_lon_alt: Tuple[float, float, float],
        max_range_km: float = 3000.0,
        accuracy_m: float = 50.0,
        field_of_view_deg: float = 120.0,
        min_elevation_deg: float = 10.0
    ):
        """
        Initialize radar sensor.
        
        Example:
            >>> radar = RadarSensor(
            ...     name="Radar-CONUS-1",
            ...     location_lat_lon_alt=(40.0, -105.0, 1.5),
            ...     max_range_km=3000,
            ...     accuracy_m=50
            ... )
        """
        super().__init__(
            name=name,
            location_lat_lon_alt=location_lat_lon_alt,
            max_range_km=max_range_km,
            accuracy_m=accuracy_m,
            field_of_view_deg=field_of_view_deg,
            min_elevation_deg=min_elevation_deg
        )
        self.sensor_type = "radar"
    
    def measure(
        self,
        target_position_eci: np.ndarray,
        object_id: int,
        time: datetime,
        add_noise: bool = True
    ) -> Measurement:
        """
        Generate radar measurement.
        
        Radar measurements have:
        - Isotropic Gaussian noise (equal in all directions)
        - High accuracy
        - Range, azimuth, elevation information
        """
        # Calculate range, azimuth, elevation
        range_km, azimuth, elevation = range_azimuth_elevation(
            self.location_ecef,
            target_position_eci
        )
        
        # Add measurement noise if requested
        if add_noise:
            # Gaussian noise in Cartesian coordinates
            noise = np.random.normal(0, self.accuracy, size=3)
            measured_position = target_position_eci + noise
        else:
            measured_position = target_position_eci.copy()
        
        # Covariance matrix (isotropic)
        covariance = np.eye(3) * (self.accuracy ** 2)
        
        return Measurement(
            time=time,
            sensor_id=self.name,
            object_id=object_id,
            position_measured=measured_position,
            position_true=target_position_eci,
            covariance=covariance,
            range_km=range_km,
            azimuth_rad=azimuth,
            elevation_rad=elevation
        )


class OpticalSensor(BaseSensor):
    """
    Ground-based optical sensor model (telescope).
    
    Characteristics:
    - Long range (up to 40000 km for GEO)
    - Lower accuracy (100-1000m)
    - Narrow field of view (10-30°)
    - Weather-dependent (clear skies only)
    """
    
    def __init__(
        self,
        name: str,
        location_lat_lon_alt: Tuple[float, float, float],
        max_range_km: float = 40000.0,
        accuracy_m: float = 500.0,
        field_of_view_deg: float = 30.0,
        min_elevation_deg: float = 20.0
    ):
        """
        Initialize optical sensor.
        
        Example:
            >>> optical = OpticalSensor(
            ...     name="Optical-Hawaii",
            ...     location_lat_lon_alt=(19.8, -155.5, 4.2),
            ...     max_range_km=40000,
            ...     accuracy_m=500
            ... )
        """
        super().__init__(
            name=name,
            location_lat_lon_alt=location_lat_lon_alt,
            max_range_km=max_range_km,
            accuracy_m=accuracy_m,
            field_of_view_deg=field_of_view_deg,
            min_elevation_deg=min_elevation_deg
        )
        self.sensor_type = "optical"
    
    def measure(
        self,
        target_position_eci: np.ndarray,
        object_id: int,
        time: datetime,
        add_noise: bool = True
    ) -> Measurement:
        """
        Generate optical measurement.
        
        Optical measurements have:
        - Anisotropic noise (better in angular than range)
        - Lower accuracy than radar
        - Angular (azimuth, elevation) information primary
        """
        # Calculate range, azimuth, elevation
        range_km, azimuth, elevation = range_azimuth_elevation(
            self.location_ecef,
            target_position_eci
        )
        
        # Add measurement noise if requested
        if add_noise:
            # Optical sensors have better angular accuracy than range accuracy
            # Simulate this with anisotropic noise
            # Higher noise in radial direction, lower in tangential
            noise = np.random.normal(0, self.accuracy, size=3)
            
            # Scale noise based on direction (simplified model)
            # In reality, would convert to range/azimuth/elevation and add noise there
            measured_position = target_position_eci + noise
        else:
            measured_position = target_position_eci.copy()
        
        # Covariance matrix (slightly anisotropic for optical)
        # Higher uncertainty in range direction
        covariance = np.eye(3) * (self.accuracy ** 2)
        covariance[2, 2] *= 1.5  # Higher uncertainty in Z (simplified)
        
        return Measurement(
            time=time,
            sensor_id=self.name,
            object_id=object_id,
            position_measured=measured_position,
            position_true=target_position_eci,
            covariance=covariance,
            range_km=range_km,
            azimuth_rad=azimuth,
            elevation_rad=elevation
        )


def create_sensor_network(sensor_configs: list) -> list:
    """
    Create a network of sensors from configuration.
    
    Args:
        sensor_configs: List of sensor configuration dictionaries
    
    Returns:
        List of sensor objects
    
    Example:
        >>> configs = [
        ...     {
        ...         'name': 'Radar-1',
        ...         'type': 'radar',
        ...         'location': [40.0, -105.0, 1.5],
        ...         'max_range_km': 3000,
        ...         'accuracy_m': 50
        ...     }
        ... ]
        >>> sensors = create_sensor_network(configs)
    """
    sensors = []
    
    for config in sensor_configs:
        sensor_type = config.get('type', 'radar').lower()
        
        if sensor_type == 'radar':
            sensor = RadarSensor(
                name=config['name'],
                location_lat_lon_alt=tuple(config['location']),
                max_range_km=config.get('max_range_km', 3000.0),
                accuracy_m=config.get('accuracy_m', 50.0),
                field_of_view_deg=config.get('fov_deg', 120.0),
                min_elevation_deg=config.get('min_elevation_deg', 10.0)
            )
        elif sensor_type == 'optical':
            sensor = OpticalSensor(
                name=config['name'],
                location_lat_lon_alt=tuple(config['location']),
                max_range_km=config.get('max_range_km', 40000.0),
                accuracy_m=config.get('accuracy_m', 500.0),
                field_of_view_deg=config.get('fov_deg', 30.0),
                min_elevation_deg=config.get('min_elevation_deg', 20.0)
            )
        else:
            logger.warning(f"Unknown sensor type: {sensor_type}")
            continue
        
        sensors.append(sensor)
    
    logger.info(f"Created sensor network with {len(sensors)} sensors")
    return sensors


# Example usage
if __name__ == "__main__":
    from datetime import datetime, timezone
    
    # Create example sensors
    radar = RadarSensor(
        name="Radar-CONUS-1",
        location_lat_lon_alt=(40.0, -105.0, 1.5),  # Colorado
        max_range_km=3000,
        accuracy_m=50
    )
    
    optical = OpticalSensor(
        name="Optical-Hawaii",
        location_lat_lon_alt=(19.8, -155.5, 4.2),  # Mauna Kea
        max_range_km=40000,
        accuracy_m=500
    )
    
    print(f"Radar: {radar}")
    print(f"Optical: {optical}")
    
    # Test with example satellite position (ISS-like orbit)
    # Position in ECI frame (km)
    iss_position = np.array([6700.0, 0.0, 0.0])
    
    # Check visibility
    now = datetime.now(timezone.utc)
    
    print(f"\nVisibility test for position {iss_position}:")
    print(f"  Radar can observe: {radar.can_observe(iss_position, now)}")
    print(f"  Optical can observe: {optical.can_observe(iss_position, now)}")
    
    # Generate measurements
    if radar.can_observe(iss_position, now):
        measurement = radar.measure(iss_position, object_id=1, time=now)
        print(f"\nRadar measurement:")
        print(f"  Range: {measurement.range_km:.1f} km")
        print(f"  Elevation: {np.degrees(measurement.elevation_rad):.1f}°")
        print(f"  Error: {measurement.measurement_error*1000:.1f} m")
