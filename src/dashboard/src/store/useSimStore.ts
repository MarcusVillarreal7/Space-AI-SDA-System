import { create } from 'zustand';
import type {
  SatellitePosition,
  ThreatAssessment,
  ThreatSummary,
  Alert,
  ThreatTier,
} from '../types';

interface SimState {
  // Simulation
  isPlaying: boolean;
  speed: number;
  timestep: number;
  maxTimestep: number;
  timeIso: string;

  // Objects
  objects: SatellitePosition[];
  selectedObjectId: number | null;

  // Threat
  threatSummary: ThreatSummary | null;
  selectedAssessment: ThreatAssessment | null;
  alerts: Alert[];

  // UI
  isConnected: boolean;
  isLoading: boolean;
  bottomTab: string;

  // Actions
  setSimulation: (isPlaying: boolean, speed: number, timestep: number, maxTimestep: number, timeIso: string) => void;
  updatePositions: (timestep: number, timeIso: string, objects: SatellitePosition[]) => void;
  selectObject: (id: number | null) => void;
  setThreatSummary: (summary: ThreatSummary) => void;
  setSelectedAssessment: (assessment: ThreatAssessment | null) => void;
  setAlerts: (alerts: Alert[]) => void;
  addAlert: (alert: Alert) => void;
  setConnected: (connected: boolean) => void;
  setLoading: (loading: boolean) => void;
  setBottomTab: (tab: string) => void;
  setSpeed: (speed: number) => void;
  setPlaying: (playing: boolean) => void;
}

export const useSimStore = create<SimState>((set) => ({
  isPlaying: false,
  speed: 60,
  timestep: 0,
  maxTimestep: 1439,
  timeIso: '',
  objects: [],
  selectedObjectId: null,
  threatSummary: null,
  selectedAssessment: null,
  alerts: [],
  isConnected: false,
  isLoading: true,
  bottomTab: 'tracking',

  setSimulation: (isPlaying, speed, timestep, maxTimestep, timeIso) =>
    set({ isPlaying, speed, timestep, maxTimestep, timeIso }),

  updatePositions: (timestep, timeIso, objects) =>
    set({ timestep, timeIso, objects, isLoading: false }),

  selectObject: (id) => set({ selectedObjectId: id, selectedAssessment: null }),

  setThreatSummary: (summary) => set({ threatSummary: summary }),

  setSelectedAssessment: (assessment) => set({ selectedAssessment: assessment }),

  setAlerts: (alerts) => set({ alerts }),

  addAlert: (alert) =>
    set((state) => ({ alerts: [alert, ...state.alerts].slice(0, 100) })),

  setConnected: (connected) => set({ isConnected: connected }),

  setLoading: (loading) => set({ isLoading: loading }),

  setBottomTab: (tab) => set({ bottomTab: tab }),

  setSpeed: (speed) => set({ speed }),

  setPlaying: (playing) => set({ isPlaying: playing }),
}));
