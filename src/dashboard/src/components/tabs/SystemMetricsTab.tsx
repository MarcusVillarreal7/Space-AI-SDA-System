import { useEffect, useState } from 'react';
import { api } from '../../services/api';
import type { SystemMetrics } from '../../types';

export function SystemMetricsTab() {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);

  useEffect(() => {
    const fetch = () => api.getMetrics().then(setMetrics).catch(() => {});
    fetch();
    const interval = setInterval(fetch, 5000);
    return () => clearInterval(interval);
  }, []);

  if (!metrics) return <p className="text-xs text-slate-500">Loading metrics...</p>;

  const cards = [
    { label: 'Objects Tracked', value: metrics.objects_tracked.toLocaleString(), icon: '🛰' },
    { label: 'WebSocket Clients', value: metrics.websocket_connections, icon: '🔗' },
    { label: 'API Requests', value: metrics.api_requests.toLocaleString(), icon: '📡' },
    { label: 'Avg API Latency', value: `${metrics.avg_api_latency_ms.toFixed(1)} ms`, icon: '⏱' },
    { label: 'Assessments', value: metrics.assessments_completed.toLocaleString(), icon: '🎯' },
    { label: 'Uptime', value: formatUptime(metrics.uptime_seconds), icon: '⏰' },
  ];

  return (
    <div className="grid grid-cols-3 lg:grid-cols-6 gap-3">
      {cards.map((card) => (
        <div
          key={card.label}
          className="bg-space-700/50 rounded-lg p-3 border border-space-600 text-center"
        >
          <div className="text-lg mb-1">{card.icon}</div>
          <div className="text-sm font-bold text-slate-200">{card.value}</div>
          <div className="text-[10px] text-slate-500">{card.label}</div>
        </div>
      ))}
    </div>
  );
}

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}
