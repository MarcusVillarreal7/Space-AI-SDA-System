const MODELS = [
  {
    name: 'Trajectory Transformer',
    type: 'Neural Network',
    architecture: 'Transformer + ParallelHead (pos/vel)',
    params: '371K',
    metrics: [
      { label: 'Position RMSE', value: '7.57 km' },
      { label: 'Velocity RMSE', value: '1.00 km/s' },
      { label: 'Training Epochs', value: '10' },
    ],
    description: 'Predicts future satellite positions and velocities from 20-step history sequences using multi-head self-attention.',
  },
  {
    name: 'Maneuver Classifier',
    type: 'Neural Network',
    architecture: 'CNN-LSTM-Attention',
    params: '719K',
    metrics: [
      { label: 'Accuracy', value: '84.5%' },
      { label: 'Classes', value: '6' },
      { label: 'Training Epochs', value: '50' },
    ],
    description: 'Classifies delta-V patterns into 6 maneuver types: Normal, Drift/Decay, Station-keeping, Minor/Major Maneuver, Deorbit.',
  },
  {
    name: 'Behavior Autoencoder',
    type: 'Neural Network',
    architecture: 'Dense Autoencoder (Encoder→Decoder)',
    params: '~2.5K',
    metrics: [
      { label: 'True Positive Rate', value: '100%' },
      { label: 'False Positive Rate', value: '5%' },
      { label: 'Reconstruction Error', value: 'Threshold-based' },
    ],
    description: 'Detects anomalous behavior by learning normal satellite behavior profiles and flagging high reconstruction errors.',
  },
  {
    name: 'Collision Risk Predictor',
    type: 'Neural Network',
    architecture: 'MLP with physics-informed features',
    params: '~90K',
    metrics: [
      { label: 'Method', value: 'Statistical risk + TTCA' },
      { label: 'Features', value: 'Relative state vectors' },
    ],
    description: 'Estimates collision probability using statistical models and time-to-closest-approach calculations.',
  },
  {
    name: 'Intent Classifier',
    type: 'Rule-based',
    architecture: 'Decision tree with proximity context',
    params: 'N/A',
    metrics: [
      { label: 'Categories', value: '10' },
      { label: 'Threat Levels', value: '5' },
    ],
    description: 'Classifies satellite intent (nominal, station-keeping, rendezvous, surveillance, evasive, etc.) from maneuver and proximity context.',
  },
  {
    name: 'Threat Scorer',
    type: 'Rule-based',
    architecture: 'Weighted fusion (intent + anomaly + proximity + pattern)',
    params: 'N/A',
    metrics: [
      { label: 'Score Range', value: '0-100' },
      { label: 'Tiers', value: '5 (MINIMAL → CRITICAL)' },
      { label: 'Sub-scores', value: '4 components' },
    ],
    description: 'Fuses all ML module outputs into a unified 0-100 threat score with tier classification and natural language explanation.',
  },
  {
    name: 'E2E Threat Pipeline',
    type: 'Pipeline',
    architecture: 'Sequential: Classification → Intent → Anomaly → Scoring',
    params: 'All models combined',
    metrics: [
      { label: 'Throughput', value: '226 obj/sec' },
      { label: 'Avg Latency', value: '3.35 ms' },
      { label: 'Batch Size', value: '1000 objects' },
    ],
    description: 'End-to-end pipeline that processes raw track data through all ML modules to produce a complete threat assessment.',
  },
];

export function MLPerformanceTab() {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
      {MODELS.map((model) => (
        <div
          key={model.name}
          className="bg-space-700/50 rounded-lg p-3 border border-space-600"
        >
          <div className="flex items-start justify-between mb-2">
            <h4 className="text-xs font-semibold text-slate-200">{model.name}</h4>
            <span className={`text-[10px] px-1.5 py-0.5 rounded ${
              model.type === 'Neural Network'
                ? 'bg-blue-900/50 text-blue-400'
                : model.type === 'Rule-based'
                ? 'bg-purple-900/50 text-purple-400'
                : 'bg-emerald-900/50 text-emerald-400'
            }`}>
              {model.type}
            </span>
          </div>
          <p className="text-[10px] text-slate-500 mb-2">{model.architecture}</p>
          {model.params !== 'N/A' && model.params !== 'All models combined' && (
            <p className="text-[10px] text-slate-400 mb-2">Parameters: <span className="text-slate-300">{model.params}</span></p>
          )}
          <div className="space-y-1">
            {model.metrics.map((m) => (
              <div key={m.label} className="flex justify-between text-[10px]">
                <span className="text-slate-500">{m.label}</span>
                <span className="text-slate-300 font-mono">{m.value}</span>
              </div>
            ))}
          </div>
          <p className="text-[10px] text-slate-500 mt-2 leading-relaxed">{model.description}</p>
        </div>
      ))}
    </div>
  );
}
