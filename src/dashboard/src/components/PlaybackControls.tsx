import { useSimStore } from '../store/useSimStore';
import { api } from '../services/api';

const SPEEDS = [1, 10, 60, 360];

export function PlaybackControls() {
  const isPlaying = useSimStore((s) => s.isPlaying);
  const speed = useSimStore((s) => s.speed);
  const timestep = useSimStore((s) => s.timestep);
  const maxTimestep = useSimStore((s) => s.maxTimestep);
  const setPlaying = useSimStore((s) => s.setPlaying);
  const setSpeed = useSimStore((s) => s.setSpeed);

  const handlePlayPause = async () => {
    if (isPlaying) {
      await api.pause();
      setPlaying(false);
    } else {
      await api.play();
      setPlaying(true);
    }
  };

  const handleSpeedChange = async (newSpeed: number) => {
    await api.setSpeed(newSpeed);
    setSpeed(newSpeed);
  };

  const handleSeek = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const ts = parseInt(e.target.value, 10);
    await api.seek(ts);
  };

  const progress = maxTimestep > 0 ? (timestep / maxTimestep) * 100 : 0;

  return (
    <div className="h-14 flex items-center gap-4 px-4 bg-space-800 border-t border-space-700 shrink-0">
      {/* Play/Pause */}
      <button
        onClick={handlePlayPause}
        className="w-8 h-8 flex items-center justify-center rounded bg-blue-600 hover:bg-blue-500 transition-colors"
        title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
      >
        {isPlaying ? (
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
            <rect x="6" y="4" width="4" height="16" />
            <rect x="14" y="4" width="4" height="16" />
          </svg>
        ) : (
          <svg className="w-4 h-4 ml-0.5" fill="currentColor" viewBox="0 0 24 24">
            <polygon points="5,3 19,12 5,21" />
          </svg>
        )}
      </button>

      {/* Speed selector */}
      <div className="flex gap-1">
        {SPEEDS.map((s) => (
          <button
            key={s}
            onClick={() => handleSpeedChange(s)}
            className={`px-2 py-1 text-xs rounded transition-colors ${
              speed === s
                ? 'bg-blue-600 text-white'
                : 'bg-space-700 text-slate-400 hover:text-slate-200'
            }`}
          >
            {s}x
          </button>
        ))}
      </div>

      {/* Timeline slider */}
      <div className="flex-1 flex items-center gap-2">
        <input
          type="range"
          min={0}
          max={maxTimestep}
          value={timestep}
          onChange={handleSeek}
          className="flex-1 h-1.5 appearance-none bg-space-600 rounded-full cursor-pointer
            [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3
            [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full
            [&::-webkit-slider-thumb]:bg-blue-500"
        />
      </div>

      {/* Timestep display */}
      <span className="text-xs text-slate-400 font-mono w-24 text-right">
        {timestep} / {maxTimestep}
      </span>
    </div>
  );
}
