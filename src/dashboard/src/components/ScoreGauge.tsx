import { TIER_COLORS } from '../types';
import type { ThreatTier } from '../types';

interface Props {
  score: number;
  tier: ThreatTier;
  label: string;
  size?: number;
}

export function ScoreGauge({ score, tier, label, size = 80 }: Props) {
  const color = TIER_COLORS[tier];
  const radius = (size - 8) / 2;
  const circumference = Math.PI * radius; // Semi-circle
  const filled = (score / 100) * circumference;

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size / 2 + 8} viewBox={`0 0 ${size} ${size / 2 + 8}`}>
        {/* Background arc */}
        <path
          d={`M 4 ${size / 2} A ${radius} ${radius} 0 0 1 ${size - 4} ${size / 2}`}
          fill="none"
          stroke="#334155"
          strokeWidth="6"
          strokeLinecap="round"
        />
        {/* Filled arc */}
        <path
          d={`M 4 ${size / 2} A ${radius} ${radius} 0 0 1 ${size - 4} ${size / 2}`}
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={`${filled} ${circumference}`}
        />
        {/* Score text */}
        <text
          x={size / 2}
          y={size / 2 - 2}
          textAnchor="middle"
          fill={color}
          fontSize="16"
          fontWeight="bold"
        >
          {Math.round(score)}
        </text>
      </svg>
      <span className="text-xs text-slate-400 -mt-1">{label}</span>
    </div>
  );
}
