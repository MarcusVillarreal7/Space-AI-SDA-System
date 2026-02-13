import { useEffect, useState, useMemo } from 'react';
import { useSimStore } from '../../store/useSimStore';
import { api } from '../../services/api';
import { TierBadge } from '../TierBadge';
import type { ObjectSummary, ObjectType } from '../../types';
import { OBJECT_TYPE_LABELS, OBJECT_TYPE_COLORS } from '../../types';

type SortKey = 'id' | 'name' | 'object_type' | 'regime' | 'altitude_km' | 'threat_tier';
type SortDir = 'asc' | 'desc';
type TypeFilter = 'ALL' | ObjectType;

export function TrackingTab() {
  const [allObjects, setAllObjects] = useState<ObjectSummary[]>([]);
  const [filter, setFilter] = useState('');
  const [typeFilter, setTypeFilter] = useState<TypeFilter>('ALL');
  const [sortKey, setSortKey] = useState<SortKey>('id');
  const [sortDir, setSortDir] = useState<SortDir>('asc');
  const selectObject = useSimStore((s) => s.selectObject);
  const liveObjects = useSimStore((s) => s.objects);
  const resetVersion = useSimStore((s) => s.resetVersion);

  useEffect(() => {
    api.getObjects({ limit: 1000 }).then(setAllObjects).catch(() => {});
  }, [resetVersion]);

  // Merge live threat_tiers from WebSocket into the REST-fetched objects
  const mergedObjects = useMemo(() => {
    if (!liveObjects.length) return allObjects;
    const liveTiers = new Map(liveObjects.map((o) => [o.id, o.threat_tier]));
    return allObjects.map((obj) => {
      const liveTier = liveTiers.get(obj.id);
      return liveTier && liveTier !== obj.threat_tier
        ? { ...obj, threat_tier: liveTier }
        : obj;
    });
  }, [allObjects, liveObjects]);

  const filtered = useMemo(() => {
    let items = mergedObjects;
    if (typeFilter !== 'ALL') {
      items = items.filter((o) => o.object_type === typeFilter);
    }
    if (filter) {
      const lc = filter.toLowerCase();
      items = items.filter(
        (o) =>
          o.name.toLowerCase().includes(lc) ||
          o.regime.toLowerCase().includes(lc) ||
          String(o.id).includes(lc)
      );
    }
    items.sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (typeof av === 'number' && typeof bv === 'number') {
        return sortDir === 'asc' ? av - bv : bv - av;
      }
      return sortDir === 'asc'
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av));
    });
    return items;
  }, [mergedObjects, filter, typeFilter, sortKey, sortDir]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  const headerClass = "text-left text-xs font-medium text-slate-500 uppercase cursor-pointer hover:text-slate-300 px-2 py-1";

  const typeFilterOptions: { key: TypeFilter; label: string }[] = [
    { key: 'ALL', label: 'All' },
    { key: 'PAYLOAD', label: 'Satellite' },
    { key: 'DEBRIS', label: 'Debris' },
    { key: 'ROCKET_BODY', label: 'Rocket Body' },
  ];

  return (
    <div>
      <div className="flex items-center gap-3 mb-2">
        <input
          type="text"
          placeholder="Filter by name, ID, or regime..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="w-64 px-2 py-1 text-xs bg-space-700 border border-space-600 rounded text-slate-300 placeholder-slate-500 outline-none focus:border-blue-500"
        />
        <div className="flex gap-1">
          {typeFilterOptions.map((opt) => (
            <button
              key={opt.key}
              onClick={() => setTypeFilter(opt.key)}
              className={`px-2 py-0.5 text-[10px] rounded border transition-colors ${
                typeFilter === opt.key
                  ? 'bg-blue-600/30 border-blue-500 text-blue-300'
                  : 'bg-space-700 border-space-600 text-slate-400 hover:text-slate-300'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>
      <div className="overflow-x-auto max-h-56">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-space-800">
            <tr>
              <th className={headerClass} onClick={() => toggleSort('id')}>ID {sortKey === 'id' ? (sortDir === 'asc' ? '↑' : '↓') : ''}</th>
              <th className={headerClass} onClick={() => toggleSort('name')}>Name {sortKey === 'name' ? (sortDir === 'asc' ? '↑' : '↓') : ''}</th>
              <th className={headerClass} onClick={() => toggleSort('object_type')}>Type {sortKey === 'object_type' ? (sortDir === 'asc' ? '↑' : '↓') : ''}</th>
              <th className={headerClass} onClick={() => toggleSort('regime')}>Regime {sortKey === 'regime' ? (sortDir === 'asc' ? '↑' : '↓') : ''}</th>
              <th className={headerClass} onClick={() => toggleSort('altitude_km')}>Alt (km) {sortKey === 'altitude_km' ? (sortDir === 'asc' ? '↑' : '↓') : ''}</th>
              <th className={headerClass} onClick={() => toggleSort('threat_tier')}>Threat {sortKey === 'threat_tier' ? (sortDir === 'asc' ? '↑' : '↓') : ''}</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((obj) => (
              <tr
                key={obj.id}
                onClick={() => selectObject(obj.id)}
                className="cursor-pointer hover:bg-space-700 transition-colors"
              >
                <td className="px-2 py-1 text-slate-400">{obj.id}</td>
                <td className="px-2 py-1 text-slate-300">{obj.name}</td>
                <td className="px-2 py-1">
                  <span
                    className="px-1.5 py-0.5 rounded text-[10px] font-medium border"
                    style={{
                      color: OBJECT_TYPE_COLORS[obj.object_type] ?? '#6b7280',
                      borderColor: (OBJECT_TYPE_COLORS[obj.object_type] ?? '#6b7280') + '40',
                      backgroundColor: (OBJECT_TYPE_COLORS[obj.object_type] ?? '#6b7280') + '15',
                    }}
                  >
                    {OBJECT_TYPE_LABELS[obj.object_type] ?? obj.object_type}
                  </span>
                </td>
                <td className="px-2 py-1 text-slate-400">{obj.regime}</td>
                <td className="px-2 py-1 text-slate-400 font-mono">{obj.altitude_km.toFixed(0)}</td>
                <td className="px-2 py-1"><TierBadge tier={obj.threat_tier} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[10px] text-slate-600 mt-1">{filtered.length} of {allObjects.length} objects</p>
    </div>
  );
}
