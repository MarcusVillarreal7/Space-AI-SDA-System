#!/usr/bin/env python3
"""
Launch script for the Space Domain Awareness Dashboard.

Usage:
    python scripts/run_dashboard.py              # Production (serves built frontend)
    python scripts/run_dashboard.py --dev         # Development (hot reload, proxy to Vite)
    python scripts/run_dashboard.py --port 8080   # Custom port
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Space Domain Awareness Dashboard")
    parser.add_argument("--dev", action="store_true", help="Run in dev mode with hot reload")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()

    if args.dev:
        print("Starting in DEVELOPMENT mode...")
        print(f"  Backend:  http://{args.host}:{args.port}")
        print(f"  Frontend: http://localhost:5173 (Vite dev server)")
        print()

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        subprocess.run(
            [
                sys.executable, "-m", "uvicorn",
                "src.api.main:app",
                "--host", args.host,
                "--port", str(args.port),
                "--reload",
                "--reload-dir", "src/api",
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
        )
    else:
        dist_path = PROJECT_ROOT / "src" / "dashboard" / "dist"
        if not dist_path.exists():
            print("Frontend not built. Building...")
            dashboard_dir = PROJECT_ROOT / "src" / "dashboard"
            if not (dashboard_dir / "node_modules").exists():
                subprocess.run(["npm", "install"], cwd=str(dashboard_dir), check=True)
            subprocess.run(["npm", "run", "build"], cwd=str(dashboard_dir), check=True)

        print(f"Starting Space Domain Awareness Dashboard...")
        print(f"  URL: http://{args.host}:{args.port}")
        print()

        import uvicorn
        uvicorn.run(
            "src.api.main:app",
            host=args.host,
            port=args.port,
            log_level="info",
        )


if __name__ == "__main__":
    main()
