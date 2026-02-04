#!/usr/bin/env python3
"""
Verification script to ensure proper setup of the Space AI environment.
Run this after installation to check all dependencies and configurations.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_python_version():
    """Check Python version is 3.10+."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} (requires 3.10+)")
        return False


def check_imports():
    """Check all critical imports."""
    print("\nChecking critical imports...")
    
    imports = {
        "numpy": "NumPy",
        "scipy": "SciPy",
        "pandas": "Pandas",
        "torch": "PyTorch",
        "sklearn": "scikit-learn",
        "skyfield": "Skyfield",
        "fastapi": "FastAPI",
        "yaml": "PyYAML",
        "loguru": "Loguru",
        "pydantic": "Pydantic",
        "filterpy": "FilterPy",
    }
    
    all_good = True
    for module, name in imports.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - not installed")
            all_good = False
    
    return all_good


def check_directory_structure():
    """Check project directory structure."""
    print("\nChecking directory structure...")
    
    required_dirs = [
        "src",
        "src/simulation",
        "src/tracking",
        "src/ml",
        "src/ml/models",
        "src/ml/features",
        "src/ml/training",
        "src/api",
        "src/api/routes",
        "src/utils",
        "tests",
        "tests/unit",
        "tests/integration",
        "tests/scenarios",
        "tests/benchmarks",
        "config",
        "data",
        "data/raw",
        "data/processed",
        "data/ground_truth",
        "data/logs",
        "docs",
        "docs/design",
        "docs/validation",
        "docs/operations",
        "scripts",
        "notebooks",
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - missing")
            all_good = False
    
    return all_good


def check_files():
    """Check essential files exist."""
    print("\nChecking essential files...")
    
    required_files = [
        "README.md",
        "ARCHITECTURE.md",
        "DEVLOG.md",
        "CONTRIBUTING.md",
        "LICENSE",
        ".gitignore",
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "src/__init__.py",
        "src/utils/logging_config.py",
        "src/utils/config_loader.py",
        "src/utils/coordinates.py",
        "src/utils/metrics.py",
    ]
    
    all_good = True
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - missing")
            all_good = False
    
    return all_good


def check_configs():
    """Check configuration files."""
    print("\nChecking configuration...")
    
    config_dir = project_root / "config"
    
    if not config_dir.exists():
        print("⚠️  Config directory not found (will use defaults)")
        return True
    
    print("✅ Config directory exists")
    
    # Try to create default configs
    try:
        from src.utils.config_loader import Config
        config = Config()
        config.create_default_configs()
        print("✅ Default configurations created")
    except Exception as e:
        print(f"⚠️  Could not create default configs: {e}")
    
    return True


def check_git():
    """Check Git repository status."""
    print("\nChecking Git repository...")
    
    git_dir = project_root / ".git"
    if git_dir.exists():
        print("✅ Git repository initialized")
        return True
    else:
        print("❌ Git repository not initialized")
        return False


def test_utilities():
    """Test utility modules."""
    print("\nTesting utility modules...")
    
    try:
        from src.utils.config_loader import Config, SimulationConfig
        config = Config()
        sim_config = SimulationConfig()
        print(f"✅ Config loader works (default: {sim_config.num_objects} objects)")
    except Exception as e:
        print(f"❌ Config loader failed: {e}")
        return False
    
    try:
        from src.utils.logging_config import get_logger
        logger = get_logger("test")
        print("✅ Logging system works")
    except Exception as e:
        print(f"❌ Logging system failed: {e}")
        return False
    
    try:
        import numpy as np
        from src.utils.coordinates import eci_to_ecef
        pos = np.array([7000.0, 0.0, 0.0])
        result = eci_to_ecef(pos, 0.0)
        print(f"✅ Coordinate utilities work")
    except Exception as e:
        print(f"❌ Coordinate utilities failed: {e}")
        return False
    
    try:
        from src.utils.metrics import PerformanceMetrics
        metrics = PerformanceMetrics()
        metrics.record("test", 1.0)
        print("✅ Metrics system works")
    except Exception as e:
        print(f"❌ Metrics system failed: {e}")
        return False
    
    return True


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Space AI Environment Verification")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version()),
        ("Dependencies", check_imports()),
        ("Directory Structure", check_directory_structure()),
        ("Essential Files", check_files()),
        ("Git Repository", check_git()),
        ("Configuration", check_configs()),
        ("Utility Modules", test_utilities()),
    ]
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:.<40} {status}")
    
    all_passed = all(result for _, result in checks)
    
    print("=" * 60)
    if all_passed:
        print("✅ All checks passed! Environment is ready.")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Review DEVLOG.md for development progress")
        print("2. Check ARCHITECTURE.md for system design")
        print("3. Begin Phase 1: Simulation Layer")
        print("   - Implement orbital mechanics (SGP4)")
        print("   - Create sensor models")
        print("   - Generate synthetic datasets")
        print("\nTo start development:")
        print("  source venv/bin/activate  # Activate virtual environment")
        print("  python scripts/download_tle_data.py  # Download TLE data")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("=" * 60)
        print("\nCommon fixes:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run from project root directory")
        print("3. Ensure Python 3.10+ is installed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
