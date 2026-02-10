import { useState } from 'react';
import { useSimStore } from '../store/useSimStore';
import { TrackingTab } from './tabs/TrackingTab';
import { MLPerformanceTab } from './tabs/MLPerformanceTab';
import { SystemMetricsTab } from './tabs/SystemMetricsTab';
import { AboutTab } from './tabs/AboutTab';

const TABS = [
  { id: 'tracking', label: 'Tracking' },
  { id: 'ml', label: 'ML Performance' },
  { id: 'system', label: 'System Metrics' },
  { id: 'about', label: 'About' },
];

export function BottomTabs() {
  const [isExpanded, setIsExpanded] = useState(false);
  const bottomTab = useSimStore((s) => s.bottomTab);
  const setBottomTab = useSimStore((s) => s.setBottomTab);

  return (
    <div
      className={`flex flex-col border-t border-space-700 bg-space-800 transition-all duration-300 ${
        isExpanded ? 'h-80' : 'h-10'
      }`}
    >
      {/* Tab bar */}
      <div className="flex items-center h-10 shrink-0 px-2 gap-1">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-slate-500 hover:text-slate-300 px-1 mr-1 text-xs"
          title={isExpanded ? 'Collapse' : 'Expand'}
        >
          {isExpanded ? '▼' : '▲'}
        </button>
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => {
              setBottomTab(tab.id);
              if (!isExpanded) setIsExpanded(true);
            }}
            className={`px-3 py-1.5 text-xs rounded-t transition-colors ${
              bottomTab === tab.id && isExpanded
                ? 'bg-space-700 text-slate-200'
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {isExpanded && (
        <div className="flex-1 overflow-auto px-4 py-2">
          {bottomTab === 'tracking' && <TrackingTab />}
          {bottomTab === 'ml' && <MLPerformanceTab />}
          {bottomTab === 'system' && <SystemMetricsTab />}
          {bottomTab === 'about' && <AboutTab />}
        </div>
      )}
    </div>
  );
}
