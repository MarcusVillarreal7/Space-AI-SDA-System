const PHASES = [
  { phase: 0, name: 'Simulation', status: 'complete', desc: 'Orbital mechanics, sensor models, 1000-satellite scenario generation' },
  { phase: 1, name: 'Tracking', status: 'complete', desc: 'Extended & Unscented Kalman Filters, multi-sensor data fusion' },
  { phase: 2, name: 'ML Foundation', status: 'complete', desc: 'Feature engineering, trajectory prediction, uncertainty quantification' },
  { phase: 3, name: 'ML Prediction', status: 'complete', desc: '7 ML models: classification, anomaly detection, threat scoring' },
  { phase: 4, name: 'Dashboard', status: 'active', desc: 'Operational visualization, real-time monitoring, 3D globe' },
];

const TECH = [
  { category: 'ML / Backend', items: ['PyTorch', 'NumPy/SciPy', 'scikit-learn', 'FastAPI', 'SQLite'] },
  { category: 'Frontend', items: ['React 18', 'TypeScript', 'CesiumJS/Resium', 'Recharts', 'TailwindCSS', 'Zustand'] },
  { category: 'Infrastructure', items: ['Python 3.12', 'CUDA 12.1', 'RTX 4080', 'WebSocket', 'Vite'] },
];

export function AboutTab() {
  return (
    <div className="grid grid-cols-2 gap-6">
      {/* Left: Project info */}
      <div>
        <h4 className="text-xs font-semibold text-slate-300 uppercase mb-2">Space Domain Awareness System</h4>
        <p className="text-xs text-slate-400 leading-relaxed mb-4">
          End-to-end AI system for monitoring and assessing threats in the space domain.
          Tracks 1,000 satellites with real-time threat scoring, maneuver classification,
          anomaly detection, and intent analysis. Built with production-grade ML pipelines
          and operational dashboard visualization.
        </p>

        <h4 className="text-xs font-semibold text-slate-300 uppercase mb-2">Project Phases</h4>
        <div className="space-y-1.5">
          {PHASES.map((p) => (
            <div key={p.phase} className="flex items-center gap-2">
              <span className={`w-4 h-4 flex items-center justify-center rounded-full text-[9px] font-bold ${
                p.status === 'complete'
                  ? 'bg-green-900/50 text-green-400 border border-green-700'
                  : 'bg-blue-900/50 text-blue-400 border border-blue-700'
              }`}>
                {p.status === 'complete' ? '\u2713' : p.phase}
              </span>
              <div>
                <span className="text-xs text-slate-300 font-medium">Phase {p.phase}: {p.name}</span>
                <span className="text-[10px] text-slate-500 ml-2">{p.desc}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Right: Tech stack */}
      <div>
        <h4 className="text-xs font-semibold text-slate-300 uppercase mb-2">Technology Stack</h4>
        <div className="space-y-3">
          {TECH.map((t) => (
            <div key={t.category}>
              <span className="text-[10px] text-slate-500 uppercase">{t.category}</span>
              <div className="flex flex-wrap gap-1 mt-1">
                {t.items.map((item) => (
                  <span
                    key={item}
                    className="text-[10px] px-1.5 py-0.5 bg-space-700 text-slate-300 rounded border border-space-600"
                  >
                    {item}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>

        <h4 className="text-xs font-semibold text-slate-300 uppercase mt-4 mb-2">Key Metrics</h4>
        <div className="grid grid-cols-2 gap-2">
          {[
            { label: 'Objects Tracked', value: '1,000' },
            { label: 'ML Models', value: '7' },
            { label: 'Assessment Throughput', value: '226 obj/sec' },
            { label: 'Test Coverage', value: '72%' },
            { label: 'Total Tests', value: '333+' },
            { label: 'Parameters', value: '1.18M' },
          ].map((m) => (
            <div key={m.label} className="text-xs">
              <span className="text-slate-500">{m.label}: </span>
              <span className="text-slate-300 font-mono">{m.value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
