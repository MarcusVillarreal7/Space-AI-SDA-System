import { useSimStore } from '../store/useSimStore';

export function Header() {
  const isConnected = useSimStore((s) => s.isConnected);
  const objects = useSimStore((s) => s.objects);
  const timeIso = useSimStore((s) => s.timeIso);

  const simTime = timeIso ? new Date(timeIso).toUTCString().replace('GMT', 'UTC') : '--';

  return (
    <header className="h-12 flex items-center justify-between px-4 bg-space-800 border-b border-space-700 shrink-0">
      <div className="flex items-center gap-3">
        <h1 className="text-base font-semibold text-slate-100 tracking-tight">
          Space Domain Awareness
        </h1>
        <span className={`flex items-center gap-1.5 text-xs font-medium px-2 py-0.5 rounded-full ${
          isConnected ? 'bg-green-900/50 text-green-400' : 'bg-red-900/50 text-red-400'
        }`}>
          <span className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`} />
          {isConnected ? 'LIVE' : 'OFFLINE'}
        </span>
      </div>
      <div className="flex items-center gap-6 text-sm text-slate-400">
        <span>{objects.length} objects tracked</span>
        <span className="font-mono text-xs text-slate-300">{simTime}</span>
      </div>
    </header>
  );
}
