"""
Export assessment snapshot for Railway read-only deployment.

Run this locally after a full assess-all completes to create a snapshot
of all ML pipeline results. The snapshot is committed to the repo so
Railway can warm-load correct threat tiers on startup without a GPU.

Usage:
    # 1. Start the backend locally
    python scripts/run_dashboard.py

    # 2. In a separate terminal, run this script
    python scripts/export_snapshot.py

    # 3. Commit the snapshot
    git add data/assessments_snapshot.json
    git commit -m "chore: update assessment snapshot for Railway deployment"
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    import os
    os.environ.setdefault("DATABASE_URL", "")  # Force SQLite for local export

    from src.api.database import init_db, get_all_cached_assessments

    print("Connecting to local database...")
    init_db()

    assessments = get_all_cached_assessments()

    if not assessments:
        print(
            "No cached assessments found.\n"
            "Run the backend, click 'Run Full Assessment' in the dashboard,\n"
            "wait for it to complete, then re-run this script."
        )
        sys.exit(1)

    snapshot_path = Path("data/assessments_snapshot.json")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_objects": len(assessments),
        "assessments": {str(k): v for k, v in assessments.items()},
    }

    snapshot_path.write_text(json.dumps(snapshot, indent=2))

    # Tier breakdown for confirmation
    tiers: dict[str, int] = {}
    for result in assessments.values():
        t = result.get("threat_tier", "UNKNOWN")
        tiers[t] = tiers.get(t, 0) + 1

    print(f"\nSnapshot exported to {snapshot_path}")
    print(f"  Total objects: {len(assessments)}")
    for tier, count in sorted(tiers.items()):
        print(f"  {tier}: {count}")
    print(
        "\nNext steps:\n"
        "  git add data/assessments_snapshot.json\n"
        "  git commit -m 'chore: update assessment snapshot for Railway deployment'"
    )


if __name__ == "__main__":
    main()
