import { create } from 'zustand';
import type {
  SatellitePosition,
  ThreatAssessment,
  ThreatSummary,
  TrajectoryPrediction,
  AssessAllStatus,
  Alert,
  ThreatTier,
  ObjectType,
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
  selectedPrediction: TrajectoryPrediction | null;
  alerts: Alert[];

  // Assess-all progress
  assessAllStatus: AssessAllStatus | null;

  // Deployment mode
  readOnly: boolean;

  // UI
  isConnected: boolean;
  isLoading: boolean;
  bottomTab: string;
  textFilter: string;
  typeFilter: 'ALL' | ObjectType;

  // Actions
  setSimulation: (isPlaying: boolean, speed: number, timestep: number, maxTimestep: number, timeIso: string) => void;
  updatePositions: (timestep: number, timeIso: string, objects: SatellitePosition[]) => void;
  selectObject: (id: number | null) => void;
  setThreatSummary: (summary: ThreatSummary) => void;
  setSelectedAssessment: (assessment: ThreatAssessment | null) => void;
  setSelectedPrediction: (prediction: TrajectoryPrediction | null) => void;
  setAlerts: (alerts: Alert[]) => void;
  addAlert: (alert: Alert) => void;
  setAssessAllStatus: (status: AssessAllStatus | null) => void;
  setReadOnly: (readOnly: boolean) => void;
  setConnected: (connected: boolean) => void;
  setLoading: (loading: boolean) => void;
  setBottomTab: (tab: string) => void;
  setTextFilter: (filter: string) => void;
  setTypeFilter: (filter: 'ALL' | ObjectType) => void;
  setSpeed: (speed: number) => void;
  setPlaying: (playing: boolean) => void;
  resetVersion: number;
  triggerReset: () => void;
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
  selectedPrediction: null,
  alerts: [],
  assessAllStatus: null,
  readOnly: false,
  isConnected: false,
  isLoading: true,
  bottomTab: 'tracking',
  textFilter: '',
  typeFilter: 'ALL',
  resetVersion: 0,

  setSimulation: (isPlaying, speed, timestep, maxTimestep, timeIso) =>
    set({ isPlaying, speed, timestep, maxTimestep, timeIso }),

  updatePositions: (timestep, timeIso, objects) =>
    set({ timestep, timeIso, objects, isLoading: false }),

  selectObject: (id) => set({ selectedObjectId: id, selectedAssessment: null, selectedPrediction: null }),

  setThreatSummary: (summary) => set({ threatSummary: summary }),

  setSelectedAssessment: (assessment) => set({ selectedAssessment: assessment }),

  setSelectedPrediction: (prediction) => set({ selectedPrediction: prediction }),

  setAlerts: (alerts) => set({ alerts }),

  addAlert: (alert) =>
    set((state) => ({ alerts: [alert, ...state.alerts].slice(0, 100) })),

  setAssessAllStatus: (status) => set({ assessAllStatus: status }),

  setReadOnly: (readOnly) => set({ readOnly }),

  setConnected: (connected) => set({ isConnected: connected }),

  setLoading: (loading) => set({ isLoading: loading }),

  setBottomTab: (tab) => set({ bottomTab: tab }),

  setTextFilter: (filter) => set({ textFilter: filter }),

  setTypeFilter: (filter) => set({ typeFilter: filter }),

  setSpeed: (speed) => set({ speed }),

  setPlaying: (playing) => set({ isPlaying: playing }),

  triggerReset: () =>
    set((state) => ({
      objects: state.objects.map((o) => ({ ...o, threat_tier: 'MINIMAL' as ThreatTier })),
      alerts: [],
      assessAllStatus: null,
      selectedAssessment: null,
      resetVersion: state.resetVersion + 1,
    })),
}));
