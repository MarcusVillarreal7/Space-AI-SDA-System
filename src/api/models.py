"""
Pydantic response schemas for the Space Domain Awareness API.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class ThreatTierEnum(str, Enum):
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MODERATE = "MODERATE"
    ELEVATED = "ELEVATED"
    CRITICAL = "CRITICAL"


class TrajectoryPoint(BaseModel):
    timestep: int
    time_iso: str
    lat: float
    lon: float
    alt_km: float
    position_x: float
    position_y: float
    position_z: float
    velocity_x: float
    velocity_y: float
    velocity_z: float


class ObjectPosition(BaseModel):
    id: int
    name: str
    lat: float
    lon: float
    alt_km: float
    threat_tier: ThreatTierEnum = ThreatTierEnum.MINIMAL


class ObjectSummary(BaseModel):
    id: int
    name: str
    regime: str
    altitude_km: float
    speed_km_s: float
    threat_tier: ThreatTierEnum = ThreatTierEnum.MINIMAL
    threat_score: float = 0.0


class ObjectDetail(BaseModel):
    id: int
    name: str
    regime: str
    altitude_km: float
    speed_km_s: float
    threat_tier: ThreatTierEnum = ThreatTierEnum.MINIMAL
    threat_score: float = 0.0
    trajectory: List[TrajectoryPoint] = []


class SimulationStatus(BaseModel):
    is_playing: bool
    speed: float
    timestep: int
    max_timestep: int
    time_iso: str


class ThreatSummaryResponse(BaseModel):
    total: int
    by_tier: dict[str, int]


class ThreatAssessmentResponse(BaseModel):
    object_id: int
    object_name: str
    threat_score: float
    threat_tier: ThreatTierEnum
    intent_score: float
    anomaly_score: float
    proximity_score: float
    pattern_score: float
    maneuver_class: str
    maneuver_confidence: float
    contributing_factors: List[str]
    explanation: str
    latency_ms: float


class AlertResponse(BaseModel):
    id: int
    object_id: int
    object_name: str
    threat_tier: ThreatTierEnum
    threat_score: float
    message: str
    timestamp: str


class WebSocketMessage(BaseModel):
    type: str
    timestep: int
    time_iso: str
    objects: List[ObjectPosition]


class MetricsResponse(BaseModel):
    objects_tracked: int
    websocket_connections: int
    api_requests: int
    avg_api_latency_ms: float
    assessments_completed: int
    uptime_seconds: float
