"""
Pydantic response schemas for the Space Domain Awareness API.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class ObjectTypeEnum(str, Enum):
    PAYLOAD = "PAYLOAD"
    DEBRIS = "DEBRIS"
    ROCKET_BODY = "ROCKET_BODY"


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
    object_type: ObjectTypeEnum = ObjectTypeEnum.PAYLOAD
    lat: float
    lon: float
    alt_km: float
    threat_tier: ThreatTierEnum = ThreatTierEnum.MINIMAL


class ObjectSummary(BaseModel):
    id: int
    name: str
    object_type: ObjectTypeEnum = ObjectTypeEnum.PAYLOAD
    regime: str
    altitude_km: float
    speed_km_s: float
    threat_tier: ThreatTierEnum = ThreatTierEnum.MINIMAL
    threat_score: float = 0.0


class ObjectDetail(BaseModel):
    id: int
    name: str
    object_type: ObjectTypeEnum = ObjectTypeEnum.PAYLOAD
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
    object_type: ObjectTypeEnum = ObjectTypeEnum.PAYLOAD
    threat_score: float
    threat_tier: ThreatTierEnum
    intent_score: float
    anomaly_score: float
    proximity_score: float
    pattern_score: float
    maneuver_class: str
    maneuver_confidence: float
    maneuver_probabilities: Optional[List[float]] = None
    contributing_factors: List[str]
    explanation: str
    latency_ms: float


class PredictedPoint(BaseModel):
    step: int
    lat: float
    lon: float
    alt_km: float
    position_x: float
    position_y: float
    position_z: float


class TrajectoryPredictionResponse(BaseModel):
    object_id: int
    object_name: str
    points: List[PredictedPoint]
    model: str = "TrajectoryTransformer"
    latency_ms: float


class ConjunctionPair(BaseModel):
    object1_id: int
    object1_name: str
    object2_id: int
    object2_name: str
    risk_score: float
    miss_distance_km: float
    time_to_closest_approach_s: float


class ConjunctionResponse(BaseModel):
    pairs: List[ConjunctionPair]
    analyzed_pairs: int
    timestamp: str


class AssessAllStatus(BaseModel):
    running: bool
    completed: int
    total: int


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


# --- Ingestion models ---

class TLEIngestRequest(BaseModel):
    name: str
    tle_line1: str
    tle_line2: str


class TLEIngestResponse(BaseModel):
    object_id: int
    name: str
    object_type: str
    timesteps: int
    regime: str
    altitude_km: float


class IngestionStatusResponse(BaseModel):
    ingested_count: int
    catalog_size: int


# --- Monitoring models ---

class PredictionLogResponse(BaseModel):
    id: int
    object_id: int
    object_name: str
    object_type: str
    threat_tier: str
    threat_score: float
    maneuver_class: Optional[str] = None
    intent_category: Optional[str] = None
    anomaly_score: Optional[float] = None
    proximity_score: Optional[float] = None
    latency_ms: float
    model_version: Optional[str] = None
    timestamp: str


class PredictionStatsResponse(BaseModel):
    total_predictions: int
    avg_latency_ms: float
    tier_distribution: dict[str, int]
    type_distribution: dict[str, int]


class IngestionLogResponse(BaseModel):
    id: int
    object_id: int
    object_name: str
    object_type: str
    source: str
    tle_epoch: Optional[str] = None
    timestamp: str
