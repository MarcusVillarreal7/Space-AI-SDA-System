"""
Coordinate transformation utilities for space tracking.
Handles conversions between different reference frames.
"""

import numpy as np
from typing import Tuple


def eci_to_ecef(position_eci: np.ndarray, gmst: float) -> np.ndarray:
    """
    Convert Earth-Centered Inertial (ECI) to Earth-Centered Earth-Fixed (ECEF).
    
    Args:
        position_eci: Position vector in ECI frame [x, y, z] (km)
        gmst: Greenwich Mean Sidereal Time (radians)
    
    Returns:
        Position vector in ECEF frame [x, y, z] (km)
    
    Example:
        >>> pos_eci = np.array([7000.0, 0.0, 0.0])
        >>> gmst = 0.0  # 0 degrees
        >>> pos_ecef = eci_to_ecef(pos_eci, gmst)
    """
    # Rotation matrix from ECI to ECEF
    cos_gmst = np.cos(gmst)
    sin_gmst = np.sin(gmst)
    
    rotation_matrix = np.array([
        [cos_gmst, sin_gmst, 0],
        [-sin_gmst, cos_gmst, 0],
        [0, 0, 1]
    ])
    
    return rotation_matrix @ position_eci


def ecef_to_eci(position_ecef: np.ndarray, gmst: float) -> np.ndarray:
    """
    Convert Earth-Centered Earth-Fixed (ECEF) to Earth-Centered Inertial (ECI).
    
    Args:
        position_ecef: Position vector in ECEF frame [x, y, z] (km)
        gmst: Greenwich Mean Sidereal Time (radians)
    
    Returns:
        Position vector in ECI frame [x, y, z] (km)
    """
    # Rotation matrix from ECEF to ECI (inverse of ECI to ECEF)
    cos_gmst = np.cos(gmst)
    sin_gmst = np.sin(gmst)
    
    rotation_matrix = np.array([
        [cos_gmst, -sin_gmst, 0],
        [sin_gmst, cos_gmst, 0],
        [0, 0, 1]
    ])
    
    return rotation_matrix @ position_ecef


def ecef_to_geodetic(position_ecef: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert ECEF to geodetic coordinates (latitude, longitude, altitude).
    Uses WGS84 ellipsoid model.
    
    Args:
        position_ecef: Position vector in ECEF frame [x, y, z] (km)
    
    Returns:
        Tuple of (latitude, longitude, altitude) in (radians, radians, km)
    
    Example:
        >>> pos_ecef = np.array([6378.137, 0.0, 0.0])  # On equator
        >>> lat, lon, alt = ecef_to_geodetic(pos_ecef)
        >>> print(f"Lat: {np.degrees(lat):.2f}°, Lon: {np.degrees(lon):.2f}°")
    """
    # WGS84 ellipsoid parameters
    a = 6378.137  # Semi-major axis (km)
    f = 1 / 298.257223563  # Flattening
    e2 = 2 * f - f**2  # Eccentricity squared
    
    x, y, z = position_ecef
    
    # Longitude
    lon = np.arctan2(y, x)
    
    # Iterative calculation for latitude
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2))
    
    # Iterate to converge on latitude
    for _ in range(5):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        lat = np.arctan2(z + e2 * N * np.sin(lat), p)
    
    # Altitude
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N
    
    return lat, lon, alt


def geodetic_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
    """
    Convert geodetic coordinates to ECEF.
    Uses WGS84 ellipsoid model.
    
    Args:
        lat: Latitude (radians)
        lon: Longitude (radians)
        alt: Altitude above ellipsoid (km)
    
    Returns:
        Position vector in ECEF frame [x, y, z] (km)
    """
    # WGS84 ellipsoid parameters
    a = 6378.137  # Semi-major axis (km)
    f = 1 / 298.257223563  # Flattening
    e2 = 2 * f - f**2  # Eccentricity squared
    
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + alt) * np.sin(lat)
    
    return np.array([x, y, z])


def range_azimuth_elevation(
    observer_ecef: np.ndarray,
    target_ecef: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate range, azimuth, and elevation from observer to target.
    
    Args:
        observer_ecef: Observer position in ECEF [x, y, z] (km)
        target_ecef: Target position in ECEF [x, y, z] (km)
    
    Returns:
        Tuple of (range, azimuth, elevation) in (km, radians, radians)
        Azimuth: 0 = North, π/2 = East
        Elevation: 0 = horizon, π/2 = zenith
    """
    # Observer geodetic coordinates
    lat, lon, _ = ecef_to_geodetic(observer_ecef)
    
    # Relative position vector
    relative = target_ecef - observer_ecef
    
    # Rotation matrix from ECEF to topocentric (East-North-Up)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    
    rotation = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
    ])
    
    # Transform to topocentric coordinates
    enu = rotation @ relative
    east, north, up = enu
    
    # Calculate range, azimuth, elevation
    range_km = np.linalg.norm(enu)
    azimuth = np.arctan2(east, north)
    elevation = np.arcsin(up / range_km)
    
    return range_km, azimuth, elevation


def orbital_elements_to_state_vector(
    a: float, e: float, i: float, omega: float, w: float, nu: float, mu: float = 398600.4418
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert classical orbital elements to state vector (position and velocity).
    
    Args:
        a: Semi-major axis (km)
        e: Eccentricity
        i: Inclination (radians)
        omega: Right ascension of ascending node (radians)
        w: Argument of periapsis (radians)
        nu: True anomaly (radians)
        mu: Gravitational parameter (km^3/s^2), default is Earth
    
    Returns:
        Tuple of (position, velocity) in ECI frame (km, km/s)
    """
    # Position and velocity in perifocal frame
    p = a * (1 - e**2)
    r_pqw = np.array([
        p * np.cos(nu) / (1 + e * np.cos(nu)),
        p * np.sin(nu) / (1 + e * np.cos(nu)),
        0
    ])
    
    v_pqw = np.sqrt(mu / p) * np.array([
        -np.sin(nu),
        e + np.cos(nu),
        0
    ])
    
    # Rotation matrix from perifocal to ECI
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    cos_w = np.cos(w)
    sin_w = np.sin(w)
    
    R = np.array([
        [cos_omega * cos_w - sin_omega * sin_w * cos_i,
         -cos_omega * sin_w - sin_omega * cos_w * cos_i,
         sin_omega * sin_i],
        [sin_omega * cos_w + cos_omega * sin_w * cos_i,
         -sin_omega * sin_w + cos_omega * cos_w * cos_i,
         -cos_omega * sin_i],
        [sin_w * sin_i,
         cos_w * sin_i,
         cos_i]
    ])
    
    # Transform to ECI
    r_eci = R @ r_pqw
    v_eci = R @ v_pqw
    
    return r_eci, v_eci


# Constants
EARTH_RADIUS_KM = 6378.137
EARTH_MU = 398600.4418  # km^3/s^2
