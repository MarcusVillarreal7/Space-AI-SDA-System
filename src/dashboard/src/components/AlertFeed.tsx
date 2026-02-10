import { useSimStore } from '../store/useSimStore';
import { TierBadge } from './TierBadge';

export function AlertFeed() {
  const alerts = useSimStore((s) => s.alerts);
  const selectObject = useSimStore((s) => s.selectObject);

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <h3 className="text-sm font-semibold text-slate-300 px-4 py-3 shrink-0">
        Alert Feed
        <span className="text-slate-500 font-normal ml-2">{alerts.length}</span>
      </h3>
      <div className="flex-1 overflow-y-auto px-4 pb-4 space-y-2">
        {alerts.length === 0 ? (
          <p className="text-xs text-slate-500">No alerts yet. Threat assessments generate alerts for ELEVATED and CRITICAL objects.</p>
        ) : (
          alerts.map((alert) => (
            <button
              key={alert.id}
              onClick={() => selectObject(alert.object_id)}
              className="w-full text-left p-2 rounded bg-space-700/50 hover:bg-space-700 transition-colors"
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-slate-300 font-medium">{alert.object_name}</span>
                <TierBadge tier={alert.threat_tier} />
              </div>
              <p className="text-xs text-slate-500 truncate">{alert.message}</p>
              <span className="text-[10px] text-slate-600">{alert.timestamp}</span>
            </button>
          ))
        )}
      </div>
    </div>
  );
}
