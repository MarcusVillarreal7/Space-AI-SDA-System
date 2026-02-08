"""Trajectory feature extraction for ML models."""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import datetime


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    include_position: bool = True
    include_velocity: bool = True
    include_orbital_elements: bool = True
    include_derived_features: bool = True
    include_temporal_features: bool = True
    include_uncertainty: bool = True
    
    def get_feature_dim(self) -> int:
        """Calculate total feature dimensionality."""
        dim = 0
        if self.include_position:
            dim += 3  # x, y, z
        if self.include_velocity:
            dim += 3  # vx, vy, vz
        if self.include_orbital_elements:
            dim += 6  # a, e, i, RAAN, omega, nu
        if self.include_derived_features:
            dim += 8  # altitude, speed, period, apogee, perigee, etc.
        if self.include_temporal_features:
            dim += 4  # sin/cos of hour and day
        if self.include_uncertainty:
            dim += 4  # position/velocity uncertainty, covariance trace, track quality
        return dim


class TrajectoryFeatureExtractor:
    """Extract features from trajectory data."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.feature_dim = self.config.get_feature_dim()
    
    def extract_features(
        self, 
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: np.ndarray
    ) -> np.ndarray:
        """
        Extract features from trajectory data.
        
        Args:
            positions: (T, 3) array of positions [x, y, z] in km
            velocities: (T, 3) array of velocities [vx, vy, vz] in km/s
            timestamps: (T,) array of timestamps in seconds
        
        Returns:
            features: (T, D) array where T=timesteps, D=feature_dim
        """
        T = len(positions)
        features_list = []
        
        for t in range(T):
            feature_vec = []
            
            pos = positions[t]
            vel = velocities[t]
            timestamp = timestamps[t]
            
            # 1. Position features (3D)
            if self.config.include_position:
                feature_vec.extend(pos)
            
            # 2. Velocity features (3D)
            if self.config.include_velocity:
                feature_vec.extend(vel)
            
            # 3. Orbital elements (6D)
            if self.config.include_orbital_elements:
                orbital = self._compute_orbital_elements(pos, vel)
                feature_vec.extend(orbital)
            
            # 4. Derived features (8D)
            if self.config.include_derived_features:
                derived = self._compute_derived_features(pos, vel)
                feature_vec.extend(derived)
            
            # 5. Temporal features (4D)
            if self.config.include_temporal_features:
                temporal = self._compute_temporal_features(timestamp)
                feature_vec.extend(temporal)
            
            # 6. Uncertainty features (4D) - set to defaults
            if self.config.include_uncertainty:
                uncertainty = np.array([0.0, 0.0, 0.0, 1.0])  # defaults
                feature_vec.extend(uncertainty)
            
            features_list.append(feature_vec)
        
        return np.array(features_list)
    
    def _compute_orbital_elements(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Compute Keplerian orbital elements."""
        mu = 398600.4418  # km^3/s^2 (Earth gravitational parameter)
        
        r = np.linalg.norm(pos)
        v = np.linalg.norm(vel)
        
        # Specific orbital energy
        energy = v**2 / 2 - mu / r
        
        # Semi-major axis
        a = -mu / (2 * energy) if energy < 0 else r  # fallback for parabolic/hyperbolic
        
        # Angular momentum vector
        h_vec = np.cross(pos, vel)
        h = np.linalg.norm(h_vec)
        
        # Eccentricity vector
        e_vec = np.cross(vel, h_vec) / mu - pos / r
        e = np.linalg.norm(e_vec)
        
        # Inclination
        i = np.arccos(np.clip(h_vec[2] / h, -1, 1)) if h > 1e-10 else 0.0
        
        # RAAN (Right Ascension of Ascending Node)
        n_vec = np.cross([0, 0, 1], h_vec)
        n = np.linalg.norm(n_vec)
        if n > 1e-10:
            Omega = np.arccos(np.clip(n_vec[0] / n, -1, 1))
            if n_vec[1] < 0:
                Omega = 2 * np.pi - Omega
        else:
            Omega = 0.0
        
        # Argument of periapsis
        if n > 1e-10 and e > 1e-6:
            omega = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1, 1))
            if e_vec[2] < 0:
                omega = 2 * np.pi - omega
        else:
            omega = 0.0
        
        # True anomaly
        if e > 1e-6:
            nu = np.arccos(np.clip(np.dot(e_vec, pos) / (e * r), -1, 1))
            if np.dot(pos, vel) < 0:
                nu = 2 * np.pi - nu
        else:
            nu = 0.0
        
        return np.array([a, e, i, Omega, omega, nu])
    
    def _compute_derived_features(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Compute derived orbital features."""
        R_earth = 6371.0  # km
        mu = 398600.4418  # km^3/s^2
        
        r = np.linalg.norm(pos)
        v = np.linalg.norm(vel)
        
        # Altitude
        altitude = r - R_earth
        
        # Speed
        speed = v
        
        # Semi-major axis (for period calculation)
        energy = v**2 / 2 - mu / r
        a = -mu / (2 * energy) if energy < 0 else r
        
        # Orbital period
        period = 2 * np.pi * np.sqrt(a**3 / mu) if a > 0 else 0.0
        
        # Eccentricity (for apogee/perigee)
        h_vec = np.cross(pos, vel)
        e_vec = np.cross(vel, h_vec) / mu - pos / r
        e = np.linalg.norm(e_vec)
        
        # Apogee and perigee
        apogee = a * (1 + e) - R_earth if a > 0 else 0.0
        perigee = a * (1 - e) - R_earth if a > 0 else 0.0
        
        # Position/velocity uncertainty (defaults)
        pos_unc = 0.0
        vel_unc = 0.0
        
        return np.array([altitude, speed, period, apogee, perigee, pos_unc, vel_unc, 0.0])
    
    def _compute_temporal_features(self, timestamp: float) -> np.ndarray:
        """Compute temporal cyclic features."""
        # Convert timestamp (seconds) to datetime
        # Assume timestamp is seconds since epoch of trajectory start
        # For simplicity, use modulo to extract cyclic patterns
        
        # Hour of day (24-hour cycle) - use timestamp modulo
        hour = (timestamp / 3600.0) % 24.0
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        
        # Day of year (365-day cycle)
        day_of_year = (timestamp / 86400.0) % 365.0
        day_sin = np.sin(2 * np.pi * day_of_year / 365.0)
        day_cos = np.cos(2 * np.pi * day_of_year / 365.0)
        
        return np.array([hour_sin, hour_cos, day_sin, day_cos])
