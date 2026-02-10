import { useSimStore } from '../store/useSimStore';
import { TIER_COLORS, TIER_ORDER } from '../types';

export function ThreatSummary() {
  const summary = useSimStore((s) => s.threatSummary);

  if (!summary) {
    return (
      <div className="p-4">
        <h3 className="text-sm font-semibold text-slate-300 mb-3">Threat Distribution</h3>
        <p className="text-xs text-slate-500">Loading...</p>
      </div>
    );
  }

  const total = summary.total || 1;

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
    </div>
  );
}
