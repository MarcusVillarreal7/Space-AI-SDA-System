import { useCallback, useEffect, useRef } from 'react';
import { useSimStore } from '../store/useSimStore';
import { api } from '../services/api';
import { TIER_COLORS, TIER_ORDER } from '../types';

export function ThreatSummary() {
  const summary = useSimStore((s) => s.threatSummary);
  const setThreatSummary = useSimStore((s) => s.setThreatSummary);
  const assessAllStatus = useSimStore((s) => s.assessAllStatus);
  const setAssessAllStatus = useSimStore((s) => s.setAssessAllStatus);
  const setAlerts = useSimStore((s) => s.setAlerts);
  const triggerReset = useSimStore((s) => s.triggerReset);
  const setPlaying = useSimStore((s) => s.setPlaying);
  const readOnly = useSimStore((s) => s.readOnly);
  const pollRef = useRef<ReturnType<typeof setInterval>>();

  const handleAssessAll = useCallback(async () => {
    try {
      const status = await api.assessAll();
      setAssessAllStatus(status);
    } catch (e) {
      console.error('Failed to start assess-all:', e);
    }
  }, [setAssessAllStatus]);

  const handleReset = useCallback(async () => {
    try {
      await api.resetSimulation();
      // Clear all frontend state: objects' tiers, alerts, assessments
      triggerReset();
      // Backend auto-plays after reset — sync frontend state
      setPlaying(true);
      // Refresh threat summary to show cleared tiers
      const newSummary = await api.getThreatSummary();
      setThreatSummary(newSummary);
    } catch (e) {
      console.error('Failed to reset:', e);
    }
  }, [triggerReset, setPlaying, setThreatSummary]);

  // Poll for progress when running
  useEffect(() => {
    if (!assessAllStatus?.running) {
      if (pollRef.current) clearInterval(pollRef.current);
      return;
    }

    pollRef.current = setInterval(async () => {
      try {
        const status = await api.assessAllStatus();
        setAssessAllStatus(status);

        // Refresh threat summary and alerts while running
        const [newSummary, alerts] = await Promise.all([
          api.getThreatSummary(),
          api.getAlerts(50),
        ]);
        setThreatSummary(newSummary);
        setAlerts(alerts);

        if (!status.running) {
          clearInterval(pollRef.current);
        }
      } catch {
        // ignore polling errors
      }
    }, 2000);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [assessAllStatus?.running, setAssessAllStatus, setThreatSummary, setAlerts]);

  if (!summary) {
    return (
      <div className="p-4">
        <h3 className="text-sm font-semibold text-slate-300 mb-3">Threat Distribution</h3>
        <p className="text-xs text-slate-500">Loading...</p>
      </div>
    );
  }

  const total = summary.total || 1;
  const isRunning = assessAllStatus?.running ?? false;
  const completed = assessAllStatus?.completed ?? 0;
  const assessTotal = assessAllStatus?.total ?? 0;
  const progressPct = assessTotal > 0 ? (completed / assessTotal) * 100 : 0;

  return (
    <div className="p-4 border-b border-space-700">
      <h3 className="text-sm font-semibold text-slate-300 mb-3">
        Threat Distribution
        <span className="text-slate-500 font-normal ml-2">{summary.total} objects</span>
      </h3>
      <div className="space-y-2">
        {TIER_ORDER.map((tier) => {
          const count = summary.by_tier[tier] || 0;
          const pct = (count / total) * 100;
          const color = TIER_COLORS[tier];
          return (
            <div key={tier} className="flex items-center gap-2">
              <span className="text-xs text-slate-400 w-20">{tier}</span>
              <div className="flex-1 h-3 bg-space-700 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{ width: `${pct}%`, backgroundColor: color }}
                />
              </div>
              <span className="text-xs text-slate-400 w-8 text-right">{count}</span>
            </div>
          );
        })}
      </div>

      {/* Assess-all button + progress */}
      <div className="mt-4 space-y-2">
        {isRunning ? (
          <div>
            <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
              <span>Assessing all objects...</span>
              <span>{completed}/{assessTotal}</span>
            </div>
            <div className="h-2 bg-space-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 rounded-full transition-all duration-300"
                style={{ width: `${progressPct}%` }}
              />
            </div>
          </div>
        ) : readOnly ? (
          <div className="py-2 px-3 bg-space-700/50 rounded border border-space-600 text-center">
            <p className="text-xs text-slate-400">
              Read-only dashboard — all 1,000 objects pre-assessed
            </p>
            <a
              href="https://github.com/MarcusVillarreal7/Space-AI-SDA-System"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[10px] text-blue-400 hover:text-blue-300 transition-colors"
            >
              See GitHub for full ML pipeline breakdown &rarr;
            </a>
          </div>
        ) : (
          <div className="flex gap-2">
            <button
              onClick={handleAssessAll}
              className="flex-1 py-2 px-3 bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium rounded transition-colors"
            >
              Run Full Assessment
            </button>
            <button
              onClick={handleReset}
              className="py-2 px-3 bg-space-700 hover:bg-space-600 text-slate-400 hover:text-slate-200 text-xs font-medium rounded transition-colors"
            >
              Reset
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
