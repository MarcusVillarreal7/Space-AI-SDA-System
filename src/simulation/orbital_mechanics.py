"""
Orbital mechanics and satellite propagation using SGP4/SDP4.
Provides high-accuracy satellite position and velocity prediction.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple
import numpy as np

from skyfield.api import EarthSatellite, load, wgs84
from skyfield.timelib import Time

from src.simulation.tle_loader import TLE
from src.utils.logging_config import get_logger

logger = get_logger("simulation")


@dataclass
class StateVector:
    """Satellite state vector (position and velocity)."""
    
    time: datetime
    position: np.ndarray  # [x, y, z] in km (ECI frame)
    velocity: np.ndarray  # [vx, vy, vz] in km/s (ECI frame)
    frame: str = "ECI"
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        self.position = np.asarray(self.position)
        self.velocity = np.asarray(self.velocity)
    
    @property
    def speed(self) -> float:
        """Magnitude of velocity vector (km/s)."""
        return float(np.linalg.norm(self.velocity))
    
    @property
    def altitude(self) -> float:
        """Altitude above Earth's surface (km)."""
        r = np.linalg.norm(self.position)
        return r - 6378.137  # WGS84 Earth radius
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'time': self.time.isoformat(),
            'position_x': float(self.position[0]),
            'position_y': float(self.position[1]),
            'position_z': float(self.position[2]),
            'velocity_x': float(self.velocity[0]),
            'velocity_y': float(self.velocity[1]),
            'velocity_z': float(self.velocity[2]),
            'altitude_km': self.altitude,
            'speed_km_s': self.speed,
        }


@dataclass
class OrbitalElements:
    """Classical orbital elements."""
    
    semi_major_axis: float  # km
    eccentricity: float
    inclination: float  # radians
    raan: float  # Right Ascension of Ascending Node (radians)
    argument_of_perigee: float  # radians
    true_anomaly: float  # radians
    
    @property
    def period_minutes(self) -> float:
        """Orbital period in minutes."""
        mu = 398600.4418  # Earth's gravitational parameter (km³/s²)
        return 2 * np.pi * np.sqrt(self.semi_major_axis ** 3 / mu) / 60
    
    @property
    def apogee_km(self) -> float:
        """Apogee altitude (km)."""
        r_apogee = self.semi_major_axis * (1 + self.eccentricity)
        return r_apogee - 6378.137
    
    @property
    def perigee_km(self) -> float:
        """Perigee altitude (km)."""
        r_perigee = self.semi_major_axis * (1 - self.eccentricity)
        return r_perigee - 6378.137


class SGP4Propagator:
    """
    Satellite propagator using SGP4/SDP4 algorithm via Skyfield.
    
    Provides high-accuracy orbital propagation for Earth-orbiting satellites.
    """
    
    def __init__(self, tle: TLE):
        """
        Initialize propagator from TLE.
        
        Args:
            tle: Two-Line Element set
        
        Example:
            >>> from src.simulation.tle_loader import TLE
            >>> tle = TLE.from_lines("ISS", line1, line2)
            >>> propagator = SGP4Propagator(tle)
            >>> state = propagator.propagate(datetime.now(timezone.utc))
        """
        self.tle = tle
        self.ts = load.timescale()
        
        # Create Skyfield satellite object
        self.satellite = EarthSatellite(
            tle.line1,
            tle.line2,
            name=tle.name,
            ts=self.ts
        )
        
        logger.debug(f"Initialized propagator for {tle.name}")
    
    def propagate(self, time: datetime) -> StateVector:
        """
        Propagate satellite to specific time.
        
        Args:
            time: Target time (must be timezone-aware UTC)
        
        Returns:
            StateVector with position and velocity in ECI frame
        
        Example:
            >>> from datetime import datetime, timezone
            >>> time = datetime.now(timezone.utc)
            >>> state = propagator.propagate(time)
            >>> print(f"Position: {state.position} km")
            >>> print(f"Altitude: {state.altitude:.1f} km")
        """
        # Ensure time is timezone-aware UTC
        if time.tzinfo is None:
            time = time.replace(tzinfo=timezone.utc)
        
        # Convert to Skyfield time
        t = self.ts.from_datetime(time)
        
        # Get geocentric position and velocity
        geocentric = self.satellite.at(t)
        
        # Extract position and velocity in ECI frame (GCRS)
        position = geocentric.position.km
        velocity = geocentric.velocity.km_per_s
        
        return StateVector(
            time=time,
            position=position,
            velocity=velocity,
            frame='ECI'
        )
    
    def propagate_batch(self, times: List[datetime]) -> List[StateVector]:
        """
        Propagate satellite to multiple times efficiently.
        
        Args:
            times: List of target times
        
        Returns:
            List of StateVectors
        
        Example:
            >>> from datetime import datetime, timedelta, timezone
            >>> start = datetime.now(timezone.utc)
            >>> times = [start + timedelta(minutes=i) for i in range(100)]
            >>> states = propagator.propagate_batch(times)
        """
        # Ensure all times are timezone-aware UTC
        times = [t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t for t in times]
        
        # Convert to Skyfield times
        t_array = self.ts.from_datetimes(times)
        
        # Batch propagation
        geocentric = self.satellite.at(t_array)
        
        # Extract positions and velocities
        positions = geocentric.position.km.T  # Transpose to (N, 3)
        velocities = geocentric.velocity.km_per_s.T
        
        # Create StateVector objects
        states = [
            StateVector(time=time, position=pos, velocity=vel)
            for time, pos, vel in zip(times, positions, velocities)
        ]
        
        return states
    
    def get_orbital_elements(self, time: datetime) -> OrbitalElements:
        """
        Calculate classical orbital elements at specific time.
        
        Args:
            time: Target time
        
        Returns:
            OrbitalElements object
        
        Note:
            This is an approximation derived from the state vector.
            For precise orbital elements, use the TLE directly.
        """
        state = self.propagate(time)
        
        # Calculate orbital elements from state vector
        r = state.position
        v = state.velocity
        
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        
        # Specific angular momentum
        h = np.cross(r, v)
        h_mag = np.linalg.norm(h)
        
        # Node vector
        k = np.array([0, 0, 1])
        n = np.cross(k, h)
        n_mag = np.linalg.norm(n)
        
        # Eccentricity vector
        mu = 398600.4418  # km³/s²
        e_vec = ((v_mag**2 - mu/r_mag) * r - np.dot(r, v) * v) / mu
        ecc = np.linalg.norm(e_vec)
        
        # Semi-major axis
        energy = v_mag**2 / 2 - mu / r_mag
        a = -mu / (2 * energy)
        
        # Inclination
        inc = np.arccos(h[2] / h_mag)
        
        # RAAN
        if n_mag > 1e-10:
            raan = np.arccos(n[0] / n_mag)
            if n[1] < 0:
                raan = 2 * np.pi - raan
        else:
            raan = 0.0
        
        # Argument of perigee
        if n_mag > 1e-10 and ecc > 1e-10:
            arg_pe = np.arccos(np.dot(n, e_vec) / (n_mag * ecc))
            if e_vec[2] < 0:
                arg_pe = 2 * np.pi - arg_pe
        else:
            arg_pe = 0.0
        
        # True anomaly
        if ecc > 1e-10:
            true_anom = np.arccos(np.dot(e_vec, r) / (ecc * r_mag))
            if np.dot(r, v) < 0:
                true_anom = 2 * np.pi - true_anom
        else:
            true_anom = 0.0
        
        return OrbitalElements(
            semi_major_axis=a,
            eccentricity=ecc,
            inclination=inc,
            raan=raan,
            argument_of_perigee=arg_pe,
            true_anomaly=true_anom
        )
    
    def get_ground_track(self, time: datetime) -> Tuple[float, float, float]:
        """
        Get ground track position (latitude, longitude, altitude).
        
        Args:
            time: Target time
        
        Returns:
            Tuple of (latitude_deg, longitude_deg, altitude_km)
        
        Example:
            >>> lat, lon, alt = propagator.get_ground_track(datetime.now(timezone.utc))
            >>> print(f"Satellite over: {lat:.2f}°N, {lon:.2f}°E at {alt:.0f} km")
        """
        # Ensure time is timezone-aware UTC
        if time.tzinfo is None:
            time = time.replace(tzinfo=timezone.utc)
        
        # Convert to Skyfield time
        t = self.ts.from_datetime(time)
        
        # Get geographic position
        geocentric = self.satellite.at(t)
        subpoint = wgs84.subpoint(geocentric)
        
        return (
            subpoint.latitude.degrees,
            subpoint.longitude.degrees,
            subpoint.elevation.km
        )


# Example usage
if __name__ == "__main__":
    from src.simulation.tle_loader import TLELoader
    from pathlib import Path
    
    # Load TLE data
    tle_file = Path("data/raw/stations.tle")
    if tle_file.exists():
        loader = TLELoader()
        tles = loader.load_from_file(tle_file)
        
        if tles:
            # Use ISS (first TLE)
            tle = tles[0]
            print(f"Propagating: {tle.name}")
            
            # Create propagator
            propagator = SGP4Propagator(tle)
            
            # Propagate to current time
            now = datetime.now(timezone.utc)
            state = propagator.propagate(now)
            
            print(f"\nState Vector:")
            print(f"  Time: {state.time}")
            print(f"  Position: {state.position} km")
            print(f"  Velocity: {state.velocity} km/s")
            print(f"  Altitude: {state.altitude:.1f} km")
            print(f"  Speed: {state.speed:.3f} km/s")
            
            # Get orbital elements
            elements = propagator.get_orbital_elements(now)
            print(f"\nOrbital Elements:")
            print(f"  Semi-major axis: {elements.semi_major_axis:.1f} km")
            print(f"  Eccentricity: {elements.eccentricity:.6f}")
            print(f"  Inclination: {np.degrees(elements.inclination):.2f}°")
            print(f"  Period: {elements.period_minutes:.2f} minutes")
            print(f"  Apogee: {elements.apogee_km:.1f} km")
            print(f"  Perigee: {elements.perigee_km:.1f} km")
            
            # Get ground track
            lat, lon, alt = propagator.get_ground_track(now)
            print(f"\nGround Track:")
            print(f"  Latitude: {lat:.2f}°")
            print(f"  Longitude: {lon:.2f}°")
            print(f"  Altitude: {alt:.1f} km")
    else:
        print(f"TLE file not found: {tle_file}")
        print("Run: python scripts/download_tle_data.py --categories stations")
