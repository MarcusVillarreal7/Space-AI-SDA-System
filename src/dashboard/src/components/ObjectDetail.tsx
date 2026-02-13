import { useEffect, useState } from 'react';
import { useSimStore } from '../store/useSimStore';
import { api } from '../services/api';
import { TierBadge } from './TierBadge';
import { ScoreGauge } from './ScoreGauge';
import type { ThreatAssessment, ObjectDetail as ObjectDetailType, ObjectType } from '../types';
import { TIER_COLORS, OBJECT_TYPE_LABELS, OBJECT_TYPE_COLORS } from '../types';
import {
  LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip,
  BarChart, Bar, Cell,
} from 'recharts';

const MANEUVER_CLASSES = ['Normal', 'Drift/Decay', 'Station-keeping', 'Minor Maneuver', 'Major Maneuver', 'Deorbit'];
const MANEUVER_COLORS = ['#22c55e', '#84cc16', '#3b82f6', '#eab308', '#f97316', '#ef4444'];

export function ObjectDetail() {
  const selectedObjectId = useSimStore((s) => s.selectedObjectId);
  const selectObject = useSimStore((s) => s.selectObject);
  const assessment = useSimStore((s) => s.selectedAssessment);
  const setAssessment = useSimStore((s) => s.setSelectedAssessment);
  const setPrediction = useSimStore((s) => s.setSelectedPrediction);
  const [detail, setDetail] = useState<ObjectDetailType | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (selectedObjectId === null) return;
    setLoading(true);

    // Fetch object detail, threat assessment, and trajectory prediction in parallel
    Promise.all([
      api.getObject(selectedObjectId).catch(() => null),
      api.assessObject(selectedObjectId).catch(() => null),
      api.predictTrajectory(selectedObjectId).catch(() => null),
    ]).then(([obj, threat, pred]) => {
      setDetail(obj);
      setAssessment(threat);
      setPrediction(pred);
      setLoading(false);
    });
  }, [selectedObjectId, setAssessment, setPrediction]);

  if (selectedObjectId === null) return null;

  if (loading) {
    return (
      <div className="p-4 flex items-center justify-center h-full">
        <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  // Altitude sparkline data (subsample trajectory)
  const altitudeData = detail?.trajectory
    ? detail.trajectory
        .filter((_, i) => i % 10 === 0) // Every 10th point
        .map((p) => ({ t: p.timestep, alt: Math.round(p.alt_km) }))
    : [];

  // Maneuver probability data
  const maneuverData = assessment?.maneuver_probabilities
    ? MANEUVER_CLASSES.map((name, i) => ({
        name: name.length > 12 ? name.slice(0, 10) + '..' : name,
        fullName: name,
        prob: Math.round((assessment.maneuver_probabilities![i] ?? 0) * 100),
        color: MANEUVER_COLORS[i],
      }))
    : null;

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Header */}
      <div className="p-4 border-b border-space-700 flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-slate-200">
              {detail?.name || `Object ${selectedObjectId}`}
            </h3>
            {detail?.object_type && (
              <span
                className="px-1.5 py-0.5 rounded text-[10px] font-medium border"
                style={{
                  color: OBJECT_TYPE_COLORS[detail.object_type as ObjectType] ?? '#6b7280',
                  borderColor: (OBJECT_TYPE_COLORS[detail.object_type as ObjectType] ?? '#6b7280') + '40',
                  backgroundColor: (OBJECT_TYPE_COLORS[detail.object_type as ObjectType] ?? '#6b7280') + '15',
                }}
              >
                {OBJECT_TYPE_LABELS[detail.object_type as ObjectType] ?? detail.object_type}
              </span>
            )}
          </div>
          <p className="text-xs text-slate-500">
            ID: {selectedObjectId} | {detail?.regime || '--'} | {detail?.altitude_km?.toFixed(0)} km
          </p>
        </div>
        <button
          onClick={() => selectObject(null)}
          className="text-slate-500 hover:text-slate-300 text-lg leading-none"
          title="Close (Esc)"
        >
          &times;
        </button>
      </div>

      {/* Threat Assessment */}
      {assessment && (
        <div className="p-4 border-b border-space-700">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-semibold text-slate-400 uppercase">Threat Assessment</span>
            <TierBadge tier={assessment.threat_tier} size="md" />
          </div>
          <div className="flex justify-around mb-4">
            <ScoreGauge score={assessment.threat_score} tier={assessment.threat_tier} label="Overall" />
            <ScoreGauge score={assessment.intent_score} tier={assessment.intent_score > 60 ? 'ELEVATED' : 'MINIMAL'} label="Intent" />
            <ScoreGauge score={assessment.anomaly_score} tier={assessment.anomaly_score > 60 ? 'ELEVATED' : 'MINIMAL'} label="Anomaly" />
          </div>
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-slate-400">Maneuver</span>
              <span className="text-slate-300">{assessment.maneuver_class} ({(assessment.maneuver_confidence * 100).toFixed(0)}%)</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-slate-400">Proximity</span>
              <span className="text-slate-300">{assessment.proximity_score.toFixed(1)}</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-slate-400">Pattern</span>
              <span className="text-slate-300">{assessment.pattern_score.toFixed(1)}</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-slate-400">Latency</span>
              <span className="text-slate-300">{assessment.latency_ms.toFixed(1)} ms</span>
            </div>
          </div>
          {assessment.contributing_factors.length > 0 && (
            <div className="mt-3">
              <span className="text-xs text-slate-500 font-medium">Contributing Factors:</span>
              <ul className="mt-1 space-y-0.5">
                {assessment.contributing_factors.map((f, i) => (
                  <li key={i} className="text-xs text-slate-400">&#8226; {f}</li>
                ))}
              </ul>
            </div>
          )}
          {assessment.explanation && (
            <div className="mt-3 p-2 bg-space-700/40 rounded border border-space-600">
              <span className="text-xs text-slate-500 font-medium block mb-1">Analysis:</span>
              {assessment.explanation.split('\n').map((line, i) => (
                <p key={i} className="text-xs text-slate-400 leading-relaxed">
                  {line}
                </p>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Collision-only note for non-PAYLOAD objects */}
      {detail?.object_type && detail.object_type !== 'PAYLOAD' && (
        <div className="px-4 py-2 border-b border-space-700 bg-space-700/30">
          <p className="text-[10px] text-slate-400 italic">
            Collision-only assessment (passive object) — maneuver classification and intent analysis skipped.
          </p>
        </div>
      )}

      {/* Maneuver Classification Probabilities (PAYLOAD only) */}
      {maneuverData && (!detail?.object_type || detail.object_type === 'PAYLOAD') && (
        <div className="p-4 border-b border-space-700">
          <span className="text-xs font-semibold text-slate-400 uppercase mb-2 block">
            CNN-LSTM Classification
          </span>
          <ResponsiveContainer width="100%" height={120}>
            <BarChart data={maneuverData} layout="vertical" margin={{ left: 0, right: 10, top: 0, bottom: 0 }}>
              <XAxis type="number" domain={[0, 100]} hide />
              <YAxis
                type="category"
                dataKey="name"
                width={75}
                tick={{ fontSize: 9, fill: '#94a3b8' }}
              />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 4, fontSize: 11 }}
                formatter={(v: number, _n: string, entry: any) => [`${v}%`, entry.payload.fullName]}
              />
              <Bar dataKey="prob" radius={[0, 3, 3, 0]}>
                {maneuverData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Altitude Chart */}
      {altitudeData.length > 0 && (
        <div className="p-4 border-b border-space-700">
          <span className="text-xs font-semibold text-slate-400 uppercase mb-2 block">Altitude Profile</span>
          <ResponsiveContainer width="100%" height={100}>
            <LineChart data={altitudeData}>
              <XAxis dataKey="t" hide />
              <YAxis
                domain={['dataMin - 10', 'dataMax + 10']}
                width={40}
                tick={{ fontSize: 10, fill: '#94a3b8' }}
              />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 4, fontSize: 11 }}
                labelFormatter={(v) => `Timestep ${v}`}
                formatter={(v: number) => [`${v} km`, 'Altitude']}
              />
              <Line type="monotone" dataKey="alt" stroke="#3b82f6" dot={false} strokeWidth={1.5} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
