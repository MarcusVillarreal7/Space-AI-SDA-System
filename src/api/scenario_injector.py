"""
ScenarioInjector — Injects realistic threat scenarios into the simulation.

Replaces objects 990-996 with adversary trajectories generated via
Keplerian two-body propagation with impulsive maneuvers. These create
5-8 ELEVATED/CRITICAL objects that demonstrate the full assessment pipeline.

Scenarios target the 6 high-value assets defined in src/ml/intent/asset_catalog.py.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Earth gravitational parameter (km^3/s^2)
MU = 398600.4418
EARTH_RADIUS = 6371.0


@dataclass
class Scenario:
    """Definition of a threat scenario."""
    object_idx: int          # Index into catalog arrays (NOT object_id)
    name: str                # Adversary designation
    description: str
    target_asset: Optional[str]  # Asset ID from asset_catalog.py
    expected_tier: str       # ELEVATED or CRITICAL


def _rotation_matrix_z(angle_rad: float) -> np.ndarray:
    """Rotation matrix about Z axis."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rotation_matrix_x(angle_rad: float) -> np.ndarray:
    """Rotation matrix about X axis."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _circular_velocity(r_km: float) -> float:
    """Circular orbital velocity at radius r."""
    return math.sqrt(MU / r_km)


def _propagate_kepler(r0: np.ndarray, v0: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple Keplerian propagation for one timestep.
    Uses Euler integration with gravitational acceleration.
    Good enough for ~60s steps over 24h.
    """
    r_mag = np.linalg.norm(r0)
    if r_mag < 100:
        return r0.copy(), v0.copy()
    a_grav = -MU * r0 / (r_mag ** 3)
    v1 = v0 + a_grav * dt
    r1 = r0 + v1 * dt
    return r1, v1


def _generate_orbit(
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float,
    n_timesteps: int,
    dt: float = 60.0,
    initial_true_anomaly_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a full orbit trajectory."""
    r = EARTH_RADIUS + altitude_km
    v_circ = _circular_velocity(r)

    # Initial state in orbital plane
    ta = math.radians(initial_true_anomaly_deg)
    r0 = np.array([r * math.cos(ta), r * math.sin(ta), 0.0])
    v0 = np.array([-v_circ * math.sin(ta), v_circ * math.cos(ta), 0.0])

    # Rotate to 3D orientation
    R = _rotation_matrix_z(math.radians(raan_deg)) @ _rotation_matrix_x(math.radians(inclination_deg))
    r0 = R @ r0
    v0 = R @ v0

    positions = np.zeros((n_timesteps, 3))
    velocities = np.zeros((n_timesteps, 3))
    positions[0] = r0
    velocities[0] = v0

    for i in range(1, n_timesteps):
        positions[i], velocities[i] = _propagate_kepler(positions[i - 1], velocities[i - 1], dt)

    return positions, velocities


def _inject_rendezvous(
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    n_timesteps: int,
    dt: float = 60.0,
    closing_rate_km_s: float = 0.3,
    start_offset_km: float = 500.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a rendezvous trajectory that approaches a target.
    Starts offset from target and closes at the given rate.
    """
    # Direction from target to offset starting position
    r_target = target_pos[0]
    v_target = target_vel[0]

    # Approach direction: perpendicular to velocity in orbital plane
    v_hat = v_target / (np.linalg.norm(v_target) + 1e-10)
    r_hat = r_target / (np.linalg.norm(r_target) + 1e-10)
    approach_dir = np.cross(v_hat, r_hat)
    approach_dir = approach_dir / (np.linalg.norm(approach_dir) + 1e-10)

    positions = np.zeros((n_timesteps, 3))
    velocities = np.zeros((n_timesteps, 3))

    # Start offset from target, closing gradually
    for i in range(n_timesteps):
        t = i * dt
        # Distance decreases over time
        frac = min(1.0, t / (n_timesteps * dt))
        dist = start_offset_km * (1.0 - 0.95 * frac)  # Get to 5% of start distance

        # Follow the target orbit + offset
        if i < len(target_pos):
            base_pos = target_pos[i]
            base_vel = target_vel[i]
        else:
            base_pos = target_pos[-1]
            base_vel = target_vel[-1]

        positions[i] = base_pos + approach_dir * dist
        # Velocity matches target + closing component
        velocities[i] = base_vel - approach_dir * closing_rate_km_s * (1.0 - frac)

    return positions, velocities


def _inject_shadowing(
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    n_timesteps: int,
    standoff_km: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a GEO shadowing trajectory: maintains standoff distance.
    """
    positions = np.zeros((n_timesteps, 3))
    velocities = np.zeros((n_timesteps, 3))

    for i in range(n_timesteps):
        if i < len(target_pos):
            tp = target_pos[i]
            tv = target_vel[i]
        else:
            tp = target_pos[-1]
            tv = target_vel[-1]

        # Offset in velocity direction (along-track)
        v_hat = tv / (np.linalg.norm(tv) + 1e-10)
        positions[i] = tp + v_hat * standoff_km
        velocities[i] = tv.copy()

    return positions, velocities


def _inject_collision_course(
    target_pos: np.ndarray,
    n_timesteps: int,
    dt: float = 60.0,
    closing_speed_km_s: float = 7.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a retrograde near-circular LEO orbit for collision-course debris.

    Uses a nearly equatorial retrograde orbit at the same altitude as the target.
    This gives ~15 km/s relative velocity at crossing points — a realistic
    head-on collision scenario. The slight inclination offset (5 deg) provides
    out-of-plane separation so close approaches happen near the line of nodes
    rather than every pass.
    """
    r_mag = np.linalg.norm(target_pos[0])
    altitude_km = r_mag - EARTH_RADIUS

    # Retrograde orbit: inclination 178° = nearly equatorial, opposite direction.
    # 2° offset gives ~236 km out-of-plane distance at crossing points,
    # well within the 500 km warning radius for proximity detection.
    # Relative velocity at crossing: ~15.3 km/s.
    positions, velocities = _generate_orbit(
        altitude_km=altitude_km,
        inclination_deg=178.0,
        raan_deg=0.0,
        n_timesteps=n_timesteps,
        dt=dt,
        initial_true_anomaly_deg=180.0,
    )

    return positions, velocities


def _inject_phasing_burns(
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    n_timesteps: int,
    dt: float = 60.0,
    n_burns: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a trajectory with phasing burns approaching a target.
    """
    r_mag = np.linalg.norm(target_pos[0])
    v_circ = _circular_velocity(r_mag)

    # Start in a slightly different orbit
    offset_angle = math.radians(30)
    r0 = _rotation_matrix_z(offset_angle) @ target_pos[0]
    v0 = _rotation_matrix_z(offset_angle) @ target_vel[0]
    # Slightly lower orbit = faster
    r0 = r0 * 0.998
    v_scale = _circular_velocity(np.linalg.norm(r0)) / v_circ
    v0 = v0 * v_scale

    positions = np.zeros((n_timesteps, 3))
    velocities = np.zeros((n_timesteps, 3))
    positions[0] = r0
    velocities[0] = v0

    burn_times = [n_timesteps // (n_burns + 1) * (b + 1) for b in range(n_burns)]

    for i in range(1, n_timesteps):
        positions[i], velocities[i] = _propagate_kepler(positions[i - 1], velocities[i - 1], dt)

        # Apply burns
        if i in burn_times:
            # Burn toward target
            tidx = min(i, len(target_pos) - 1)
            to_target = target_pos[tidx] - positions[i]
            dist = np.linalg.norm(to_target)
            if dist > 1.0:
                burn_dir = to_target / dist
                dv = 0.05  # 50 m/s per burn
                velocities[i] += burn_dir * dv
                logger.debug("Phasing burn %d at timestep %d, dist=%.0f km", burn_times.index(i), i, dist)

    return positions, velocities


def _inject_sudden_maneuver(
    altitude_km: float,
    inclination_deg: float,
    n_timesteps: int,
    dt: float = 60.0,
    maneuver_timestep: int = 1150,
    dv_km_s: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a normal orbit with a sudden large maneuver.
    """
    positions, velocities = _generate_orbit(
        altitude_km, inclination_deg, raan_deg=45.0,
        n_timesteps=n_timesteps, dt=dt,
    )

    if maneuver_timestep < n_timesteps:
        # Apply sudden delta-V in cross-track direction
        r = positions[maneuver_timestep]
        v = velocities[maneuver_timestep]
        cross = np.cross(r, v)
        cross = cross / (np.linalg.norm(cross) + 1e-10)
        velocities[maneuver_timestep] += cross * dv_km_s

        # Re-propagate after maneuver
        for i in range(maneuver_timestep + 1, n_timesteps):
            positions[i], velocities[i] = _propagate_kepler(
                positions[i - 1], velocities[i - 1], dt
            )

    return positions, velocities


def _inject_approach_then_evade(
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    n_timesteps: int,
    dt: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Approach a target then perform evasive maneuver (inspect-then-leave).
    """
    # Phase 1 (0-60%): approach
    approach_end = int(n_timesteps * 0.6)
    positions = np.zeros((n_timesteps, 3))
    velocities = np.zeros((n_timesteps, 3))

    # Generate approach trajectory
    r_hat = target_pos[0] / (np.linalg.norm(target_pos[0]) + 1e-10)
    v_hat = target_vel[0] / (np.linalg.norm(target_vel[0]) + 1e-10)

    start_offset = 200.0  # km
    for i in range(approach_end):
        frac = i / approach_end
        tidx = min(i, len(target_pos) - 1)
        offset = start_offset * (1.0 - 0.9 * frac)
        positions[i] = target_pos[tidx] + v_hat * offset
        velocities[i] = target_vel[tidx] - v_hat * 0.1 * (1 - frac)

    # Phase 2 (60-100%): evasive maneuver — large delta-V away
    escape_dir = np.cross(r_hat, v_hat)
    escape_dir = escape_dir / (np.linalg.norm(escape_dir) + 1e-10)

    positions[approach_end] = positions[approach_end - 1]
    velocities[approach_end] = velocities[approach_end - 1] + escape_dir * 0.3

    for i in range(approach_end + 1, n_timesteps):
        positions[i], velocities[i] = _propagate_kepler(
            positions[i - 1], velocities[i - 1], dt
        )

    return positions, velocities


class ScenarioInjector:
    """
    Injects threat scenarios into the simulation catalog at runtime.

    Repurposes objects 990-996 (last 7 in a 1000-object catalog) by
    replacing their trajectories with adversary scenarios.
    """

    # Asset positions from asset_catalog.py (ECI, km)
    ASSETS = {
        "ISS": {"pos": np.array([6771.0, 0.0, 0.0]), "vel": np.array([0.0, 7.66, 0.0])},
        "DSP-23": {"pos": np.array([42164.0, 0.0, 0.0]), "vel": np.array([0.0, 3.07, 0.0])},
        "SBIRS-GEO-1": {"pos": np.array([0.0, 42164.0, 0.0]), "vel": np.array([-3.07, 0.0, 0.0])},
        "TDRS-13": {"pos": np.array([-42164.0, 0.0, 0.0]), "vel": np.array([0.0, -3.07, 0.0])},
        "WGS-10": {"pos": np.array([0.0, -42164.0, 0.0]), "vel": np.array([3.07, 0.0, 0.0])},
    }

    # Adversary names and scenarios
    SCENARIOS = [
        Scenario(990, "COSMOS-2558", "Rendezvous approach toward ISS", "ISS", "CRITICAL"),
        Scenario(991, "LUCH/OLYMP", "GEO shadowing of DSP-23, 15km standoff", "DSP-23", "ELEVATED"),
        Scenario(992, "COSMOS-2542", "Sudden 0.5 km/s maneuver at t=1150", None, "ELEVATED"),
        Scenario(993, "DEBRIS-KZ-1A", "Collision course debris toward ISS", "ISS", "CRITICAL"),
        Scenario(994, "SJ-17", "3 phasing burns toward SBIRS-GEO-1", "SBIRS-GEO-1", "ELEVATED"),
        Scenario(995, "SHIJIAN-21", "GEO drift + station-keep near WGS-10", "WGS-10", "ELEVATED"),
        Scenario(996, "OBJECT-2024-999A", "Approach then evasive maneuver near TDRS-13", "TDRS-13", "ELEVATED"),
    ]

    def inject(self, catalog) -> list[int]:
        """
        Inject all threat scenarios into the catalog.

        Modifies catalog.positions, catalog.velocities, and catalog.object_names
        for objects at indices corresponding to IDs 990-996.

        Returns list of modified object IDs.
        """
        modified_ids = []
        n_timesteps = catalog.n_timesteps
        dt = 60.0  # seconds per timestep

        # Generate full-length asset trajectories for targeting
        asset_trajectories = {}
        for asset_id, asset in self.ASSETS.items():
            r_mag = np.linalg.norm(asset["pos"])
            alt = r_mag - EARTH_RADIUS
            v_circ = _circular_velocity(r_mag)
            # Propagate asset position for all timesteps
            pos = np.zeros((n_timesteps, 3))
            vel = np.zeros((n_timesteps, 3))
            pos[0] = asset["pos"]
            vel[0] = asset["vel"]
            for i in range(1, n_timesteps):
                pos[i], vel[i] = _propagate_kepler(pos[i - 1], vel[i - 1], dt)
            asset_trajectories[asset_id] = (pos, vel)

        for scenario in self.SCENARIOS:
            idx = catalog.get_object_index(scenario.object_idx)
            if idx is None:
                logger.warning("Object %d not found in catalog, skipping %s",
                               scenario.object_idx, scenario.name)
                continue

            logger.info("Injecting scenario: %s (ID %d) — %s",
                        scenario.name, scenario.object_idx, scenario.description)

            # Generate adversary trajectory
            if scenario.object_idx == 990:
                # COSMOS-2558: Rendezvous toward ISS
                tgt_pos, tgt_vel = asset_trajectories["ISS"]
                pos, vel = _inject_rendezvous(
                    tgt_pos, tgt_vel, n_timesteps, dt,
                    closing_rate_km_s=0.3, start_offset_km=500.0,
                )

            elif scenario.object_idx == 991:
                # LUCH/OLYMP: GEO shadowing DSP-23
                tgt_pos, tgt_vel = asset_trajectories["DSP-23"]
                pos, vel = _inject_shadowing(tgt_pos, tgt_vel, n_timesteps, standoff_km=15.0)

            elif scenario.object_idx == 992:
                # COSMOS-2542: Sudden maneuver in LEO
                pos, vel = _inject_sudden_maneuver(
                    altitude_km=600.0, inclination_deg=65.0,
                    n_timesteps=n_timesteps, dt=dt,
                    maneuver_timestep=1150, dv_km_s=0.5,
                )

            elif scenario.object_idx == 993:
                # DEBRIS-KZ-1A: Collision course toward ISS
                tgt_pos, _ = asset_trajectories["ISS"]
                pos, vel = _inject_collision_course(
                    tgt_pos, n_timesteps, dt, closing_speed_km_s=7.5,
                )

            elif scenario.object_idx == 994:
                # SJ-17: Phasing burns toward SBIRS
                tgt_pos, tgt_vel = asset_trajectories["SBIRS-GEO-1"]
                pos, vel = _inject_phasing_burns(
                    tgt_pos, tgt_vel, n_timesteps, dt, n_burns=3,
                )

            elif scenario.object_idx == 995:
                # SHIJIAN-21: GEO drift near WGS-10
                tgt_pos, tgt_vel = asset_trajectories["WGS-10"]
                pos, vel = _inject_shadowing(tgt_pos, tgt_vel, n_timesteps, standoff_km=20.0)

            elif scenario.object_idx == 996:
                # OBJECT-2024-999A: Approach then evade near TDRS-13
                tgt_pos, tgt_vel = asset_trajectories["TDRS-13"]
                pos, vel = _inject_approach_then_evade(tgt_pos, tgt_vel, n_timesteps, dt)

            else:
                continue

            # Write into catalog arrays
            catalog.positions[idx] = pos
            catalog.velocities[idx] = vel
            catalog.object_names[idx] = scenario.name

            # Set object type: 993 (DEBRIS-KZ-1A) is debris, all others are payloads
            if hasattr(catalog, "object_types") and catalog.object_types:
                catalog.object_types[idx] = (
                    "DEBRIS" if scenario.object_idx == 993 else "PAYLOAD"
                )

            # Update reference altitude/speed
            mid = n_timesteps // 2
            r_mid = np.linalg.norm(pos[mid])
            catalog.ref_altitudes[idx] = r_mid - EARTH_RADIUS
            catalog.ref_speeds[idx] = np.linalg.norm(vel[mid])
            catalog.regimes[idx] = _classify_regime(catalog.ref_altitudes[idx])

            modified_ids.append(scenario.object_idx)

        # Recompute geodetic coordinates for modified objects
        if modified_ids:
            self._recompute_geodetic(catalog, modified_ids)

        logger.info("Injected %d threat scenarios: %s", len(modified_ids), modified_ids)
        return modified_ids

    @staticmethod
    def _recompute_geodetic(catalog, object_ids: list[int]) -> None:
        """Recompute lat/lon/alt for modified objects."""
        from src.api.data_manager import _vectorized_eci_to_geodetic, _compute_gmst

        # Recompute GMST values
        gmst_values = np.array([_compute_gmst(t) for t in catalog.timestamps])

        for oid in object_ids:
            idx = catalog.get_object_index(oid)
            if idx is None:
                continue
            # Single object: (1, T, 3)
            pos_3d = catalog.positions[idx:idx + 1]
            lat, lon, alt = _vectorized_eci_to_geodetic(pos_3d, gmst_values)
            catalog.latitudes[idx] = lat[0]
            catalog.longitudes[idx] = lon[0]
            catalog.altitudes[idx] = alt[0]


def _classify_regime(alt_km: float) -> str:
    if alt_km < 2000:
        return "LEO"
    elif alt_km < 35000:
        return "MEO"
    elif alt_km < 36500:
        return "GEO"
    else:
        return "HEO"
