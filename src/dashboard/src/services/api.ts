const BASE = '/api';

async function fetchJSON<T>(url: string): Promise<T> {
  const res = await fetch(`${BASE}${url}`);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

async function postJSON<T>(url: string, params?: Record<string, string | number>): Promise<T> {
  const query = params ? '?' + new URLSearchParams(
    Object.entries(params).map(([k, v]) => [k, String(v)])
  ).toString() : '';
  const res = await fetch(`${BASE}${url}${query}`, { method: 'POST' });
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

import type {
  ObjectSummary,
  ObjectDetail,
  SimulationStatus,
  ThreatSummary,
  ThreatAssessment,
  TrajectoryPrediction,
  AssessAllStatus,
  ConjunctionPair,
  Alert,
  SystemMetrics,
} from '../types';

export const api = {
  health: () => fetchJSON<{ status: string; objects_loaded: number }>('/health'),

  getObjects: (params?: { regime?: string; limit?: number; offset?: number }) => {
    const query = params ? '?' + new URLSearchParams(
      Object.entries(params)
        .filter(([, v]) => v !== undefined)
        .map(([k, v]) => [k, String(v)])
    ).toString() : '';
    return fetchJSON<ObjectSummary[]>(`/objects${query}`);
  },

  getObject: (id: number) => fetchJSON<ObjectDetail>(`/objects/${id}`),

  getSimulationStatus: () => fetchJSON<SimulationStatus>('/simulation/status'),

  play: () => postJSON('/simulation/play'),
  pause: () => postJSON('/simulation/pause'),
  setSpeed: (speed: number) => postJSON('/simulation/speed', { speed }),
  seek: (timestep: number) => postJSON('/simulation/seek', { timestep }),

  getThreatSummary: () => fetchJSON<ThreatSummary>('/threat/summary'),
  assessObject: (id: number) => fetchJSON<ThreatAssessment>(`/threat/object/${id}`),
  getAlerts: (limit = 50) => fetchJSON<Alert[]>(`/threat/alerts?limit=${limit}`),
  clearAlerts: async () => {
    const res = await fetch(`${BASE}/threat/alerts`, { method: 'DELETE' });
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json() as Promise<{ deleted: number }>;
  },
  predictTrajectory: (id: number) => fetchJSON<TrajectoryPrediction>(`/threat/object/${id}/prediction`),
  assessAll: () => postJSON<AssessAllStatus>('/threat/assess-all'),
  assessAllStatus: () => fetchJSON<AssessAllStatus>('/threat/assess-all/status'),
  getConjunctions: () => fetchJSON<{ pairs: ConjunctionPair[]; analyzed_pairs: number; timestamp: string }>('/threat/conjunctions'),

  getMetrics: () => fetchJSON<SystemMetrics>('/metrics'),
};
