export type ThreatTier = 'MINIMAL' | 'LOW' | 'MODERATE' | 'ELEVATED' | 'CRITICAL';
export type ObjectType = 'PAYLOAD' | 'DEBRIS' | 'ROCKET_BODY';

export const OBJECT_TYPE_LABELS: Record<ObjectType, string> = {
  PAYLOAD: 'Satellite',
  DEBRIS: 'Debris',
  ROCKET_BODY: 'Rocket Body',
};

export const OBJECT_TYPE_COLORS: Record<ObjectType, string> = {
  PAYLOAD: '#3b82f6',
  DEBRIS: '#6b7280',
  ROCKET_BODY: '#d97706',
};

export interface SatellitePosition {
  id: number;
  name: string;
  object_type: ObjectType;
  lat: number;
  lon: number;
  alt_km: number;
  threat_tier: ThreatTier;
}

export interface ObjectSummary {
  id: number;
  name: string;
  object_type: ObjectType;
  regime: string;
  altitude_km: number;
  speed_km_s: number;
  threat_tier: ThreatTier;
  threat_score: number;
}

export interface TrajectoryPoint {
  timestep: number;
  time_iso: string;
  lat: number;
  lon: number;
  alt_km: number;
  position_x: number;
  position_y: number;
  position_z: number;
  velocity_x: number;
  velocity_y: number;
  velocity_z: number;
}

export interface ObjectDetail extends ObjectSummary {
  trajectory: TrajectoryPoint[];
}

export interface ThreatAssessment {
  object_id: number;
  object_name: string;
  object_type: ObjectType;
  threat_score: number;
  threat_tier: ThreatTier;
  intent_score: number;
  anomaly_score: number;
  proximity_score: number;
  pattern_score: number;
  maneuver_class: string;
  maneuver_confidence: number;
  maneuver_probabilities: number[] | null;
  contributing_factors: string[];
  explanation: string;
  latency_ms: number;
}

export interface ThreatSummary {
  total: number;
  by_tier: Record<string, number>;
}

export interface Alert {
  id: number;
  object_id: number;
  object_name: string;
  threat_tier: ThreatTier;
  threat_score: number;
  message: string;
  timestamp: string;
}

export interface SimulationStatus {
  is_playing: boolean;
  speed: number;
  timestep: number;
  max_timestep: number;
  time_iso: string;
}

export interface SystemMetrics {
  objects_tracked: number;
  websocket_connections: number;
  api_requests: number;
  avg_api_latency_ms: number;
  assessments_completed: number;
  uptime_seconds: number;
}

export interface PredictedPoint {
  step: number;
  lat: number;
  lon: number;
  alt_km: number;
  position_x: number;
  position_y: number;
  position_z: number;
}

export interface TrajectoryPrediction {
  object_id: number;
  object_name: string;
  points: PredictedPoint[];
  model: string;
  latency_ms: number;
}

export interface AssessAllStatus {
  running: boolean;
  completed: number;
  total: number;
}

export interface ConjunctionPair {
  object1_id: number;
  object1_name: string;
  object2_id: number;
  object2_name: string;
  risk_score: number;
  miss_distance_km: number;
  time_to_closest_approach_s: number;
}

export interface WSMessage {
  type: string;
  timestep: number;
  time_iso: string;
  objects: SatellitePosition[];
}

export const TIER_COLORS: Record<ThreatTier, string> = {
  MINIMAL: '#22c55e',
  LOW: '#84cc16',
  MODERATE: '#eab308',
  ELEVATED: '#f97316',
  CRITICAL: '#ef4444',
};

export const TIER_ORDER: ThreatTier[] = ['MINIMAL', 'LOW', 'MODERATE', 'ELEVATED', 'CRITICAL'];
