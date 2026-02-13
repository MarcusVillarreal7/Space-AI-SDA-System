import { useState, useEffect } from 'react';

const STORAGE_KEY = 'sda-welcome-dismissed';

export function WelcomeModal() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    // Show on first visit (or if user hasn't dismissed yet this session)
    if (!sessionStorage.getItem(STORAGE_KEY)) {
      setVisible(true);
    }
  }, []);

  const dismiss = () => {
    sessionStorage.setItem(STORAGE_KEY, '1');
    setVisible(false);
  };

  if (!visible) return null;

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="bg-space-800 border border-space-600 rounded-lg shadow-2xl max-w-2xl w-full mx-4 max-h-[85vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-space-700">
          <h2 className="text-lg font-semibold text-slate-200">
            Space Domain Awareness AI System
          </h2>
          <button
            onClick={dismiss}
            className="text-slate-500 hover:text-slate-300 text-xl leading-none"
            title="Close"
          >
            &times;
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-4 overflow-y-auto text-sm text-slate-400 space-y-4 leading-relaxed">
          <p className="text-slate-300">
            This dashboard provides real-time situational awareness of <strong>1,000 space objects</strong> in
            Earth orbit, assessed through a multi-model ML pipeline for threat detection.
          </p>

          {/* Catalog */}
          <div>
            <h3 className="text-xs font-semibold text-slate-300 uppercase mb-1">Mixed Object Catalog</h3>
            <p>
              The catalog mirrors real-world SDA composition using CelesTrak TLE data:
              <strong> 600 active satellites</strong>, <strong>350 debris fragments</strong> (from Cosmos-2251, Fengyun-1C,
              and other breakup events), and <strong>50 spent rocket bodies</strong>.
              Each object type follows a different assessment pathway.
            </p>
          </div>

          {/* Pipeline */}
          <div>
            <h3 className="text-xs font-semibold text-slate-300 uppercase mb-1">Threat Assessment Pipeline</h3>
            <div className="bg-space-900/60 rounded p-3 font-mono text-xs text-slate-400 space-y-1">
              <p className="text-slate-500">PAYLOAD (active satellites) &mdash; full 6-step pipeline:</p>
              <p className="pl-3">1. Derive maneuver history (gravity-subtracted delta-V)</p>
              <p className="pl-3">2. CNN-LSTM maneuver classification (6 classes, 719K params)</p>
              <p className="pl-3">3. Closest approach scan against 6 protected assets</p>
              <p className="pl-3">4. Intent classification (7 escalation rules + co-orbital detection)</p>
              <p className="pl-3">5. Behavioral anomaly detection (autoencoder)</p>
              <p className="pl-3">6. Weighted threat scoring &rarr; 0-100 score &rarr; 5 threat tiers</p>
              <p className="text-slate-500 mt-2">DEBRIS / ROCKET BODY &mdash; collision-only path:</p>
              <p className="pl-3">Proximity scoring only (no maneuver or intent analysis)</p>
            </div>
          </div>

          {/* Threat Tiers */}
          <div>
            <h3 className="text-xs font-semibold text-slate-300 uppercase mb-1">Threat Tiers</h3>
            <div className="grid grid-cols-5 gap-1 text-center text-xs">
              <div className="bg-space-900/60 rounded p-2">
                <div className="w-3 h-3 rounded-full mx-auto mb-1" style={{ background: '#22c55e' }} />
                <span className="text-slate-300">MINIMAL</span>
                <p className="text-[10px] text-slate-500 mt-0.5">0-19</p>
              </div>
              <div className="bg-space-900/60 rounded p-2">
                <div className="w-3 h-3 rounded-full mx-auto mb-1" style={{ background: '#84cc16' }} />
                <span className="text-slate-300">LOW</span>
                <p className="text-[10px] text-slate-500 mt-0.5">20-39</p>
              </div>
              <div className="bg-space-900/60 rounded p-2">
                <div className="w-3 h-3 rounded-full mx-auto mb-1" style={{ background: '#eab308' }} />
                <span className="text-slate-300">MODERATE</span>
                <p className="text-[10px] text-slate-500 mt-0.5">40-59</p>
              </div>
              <div className="bg-space-900/60 rounded p-2">
                <div className="w-3 h-3 rounded-full mx-auto mb-1" style={{ background: '#f97316' }} />
                <span className="text-slate-300">ELEVATED</span>
                <p className="text-[10px] text-slate-500 mt-0.5">60-79</p>
              </div>
              <div className="bg-space-900/60 rounded p-2">
                <div className="w-3 h-3 rounded-full mx-auto mb-1" style={{ background: '#ef4444' }} />
                <span className="text-slate-300">CRITICAL</span>
                <p className="text-[10px] text-slate-500 mt-0.5">80-100</p>
              </div>
            </div>
          </div>

          {/* Scenarios */}
          <div>
            <h3 className="text-xs font-semibold text-slate-300 uppercase mb-1">Injected Threat Scenarios</h3>
            <p>
              7 adversary scenarios are injected on objects 990-996, modeled after real-world events:
              rendezvous approaches, GEO shadowing, evasive maneuvers, phasing burns, and collision-course debris.
              Run <strong>Full Assessment</strong> to trigger detection across all 1,000 objects.
            </p>
          </div>

          {/* How to use */}
          <div>
            <h3 className="text-xs font-semibold text-slate-300 uppercase mb-1">Getting Started</h3>
            <ul className="list-disc list-inside space-y-0.5">
              <li>Click any object on the globe to view its threat assessment</li>
              <li>Use the <strong>Tracking</strong> tab to filter by object type and sort by threat tier</li>
              <li>Press <strong>Full Assessment</strong> (in About tab) to assess all 1,000 objects</li>
              <li>The <strong>Alert Feed</strong> shows only ELEVATED and CRITICAL detections</li>
              <li>Keyboard: <strong>Space</strong> = play/pause, <strong>+/-</strong> = speed, <strong>Esc</strong> = deselect</li>
            </ul>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-space-700 flex justify-end">
          <button
            onClick={dismiss}
            className="px-5 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded transition-colors"
          >
            Continue to Dashboard
          </button>
        </div>
      </div>
    </div>
  );
}
