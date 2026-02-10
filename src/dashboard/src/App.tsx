import { useEffect } from 'react';
import { useWebSocket } from './services/websocket';
import { useSimStore } from './store/useSimStore';
import { api } from './services/api';
import { Header } from './components/Header';
import { Globe } from './components/Globe';
import { PlaybackControls } from './components/PlaybackControls';
import { ThreatSummary } from './components/ThreatSummary';
import { AlertFeed } from './components/AlertFeed';
import { ObjectDetail } from './components/ObjectDetail';
import { BottomTabs } from './components/BottomTabs';

export default function App() {
  useWebSocket();

  const isLoading = useSimStore((s) => s.isLoading);
  const selectedObjectId = useSimStore((s) => s.selectedObjectId);
  const setThreatSummary = useSimStore((s) => s.setThreatSummary);
  const setAlerts = useSimStore((s) => s.setAlerts);
  const setSimulation = useSimStore((s) => s.setSimulation);

  const selectObject = useSimStore((s) => s.selectObject);
  const isPlaying = useSimStore((s) => s.isPlaying);
  const setPlaying = useSimStore((s) => s.setPlaying);
  const speed = useSimStore((s) => s.speed);
  const setSpeed = useSimStore((s) => s.setSpeed);

  // Keyboard shortcuts
  useEffect(() => {
    const SPEEDS = [1, 10, 60, 360];
    const handler = (e: KeyboardEvent) => {
      // Ignore when typing in input fields
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      switch (e.key) {
        case ' ':
          e.preventDefault();
          if (isPlaying) {
            api.pause().then(() => setPlaying(false));
          } else {
            api.play().then(() => setPlaying(true));
          }
          break;
        case 'Escape':
          selectObject(null);
          break;
        case '+':
        case '=': {
          const idx = SPEEDS.indexOf(speed);
          if (idx < SPEEDS.length - 1) {
            const newSpeed = SPEEDS[idx + 1];
            api.setSpeed(newSpeed).then(() => setSpeed(newSpeed));
          }
          break;
        }
        case '-': {
          const idx = SPEEDS.indexOf(speed);
          if (idx > 0) {
            const newSpeed = SPEEDS[idx - 1];
            api.setSpeed(newSpeed).then(() => setSpeed(newSpeed));
          }
          break;
        }
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isPlaying, speed, selectObject, setPlaying, setSpeed]);

  // Load initial data
  useEffect(() => {
    api.getSimulationStatus().then((s) =>
      setSimulation(s.is_playing, s.speed, s.timestep, s.max_timestep, s.time_iso)
    ).catch(() => {});

    api.getThreatSummary().then(setThreatSummary).catch(() => {});
    api.getAlerts().then(setAlerts).catch(() => {});

    // Poll threat summary every 30s
    const interval = setInterval(() => {
      api.getThreatSummary().then(setThreatSummary).catch(() => {});
      api.getAlerts(20).then(setAlerts).catch(() => {});
    }, 30000);

    return () => clearInterval(interval);
  }, [setThreatSummary, setAlerts, setSimulation]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-space-900">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <h2 className="text-xl text-slate-300">Initializing Space Domain Awareness</h2>
          <p className="text-sm text-slate-500 mt-2">Loading 1,000 satellite tracks...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-space-900">
      <Header />
      <div className="flex flex-1 min-h-0">
        {/* Globe area */}
        <div className="flex-1 relative flex flex-col">
          <Globe />
          <PlaybackControls />
        </div>

        {/* Right panel */}
        <div className="w-96 flex flex-col border-l border-space-700 bg-space-800">
          {selectedObjectId !== null ? (
            <ObjectDetail />
          ) : (
            <>
              <ThreatSummary />
              <AlertFeed />
            </>
          )}
        </div>
      </div>

      {/* Bottom tabs */}
      <BottomTabs />
    </div>
  );
}
