import { TIER_COLORS } from '../types';
import type { ThreatTier } from '../types';

interface Props {
  tier: ThreatTier;
  size?: 'sm' | 'md';
}

export function TierBadge({ tier, size = 'sm' }: Props) {
  const color = TIER_COLORS[tier];
  const isCritical = tier === 'CRITICAL';

  return (
    <span
      className={`inline-flex items-center gap-1 font-semibold rounded-full ${
        size === 'sm' ? 'text-xs px-2 py-0.5' : 'text-sm px-3 py-1'
      } ${isCritical ? 'pulse-critical' : ''}`}
      style={{
        color,
        backgroundColor: `${color}20`,
        border: `1px solid ${color}40`,
      }}
    >
      <span
        className="w-1.5 h-1.5 rounded-full"
        style={{ backgroundColor: color }}
      />
      {tier}
    </span>
  );
}
