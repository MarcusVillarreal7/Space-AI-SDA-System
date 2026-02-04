#!/bin/bash
# Phase 0 Testing Script
# Run all tests to verify Phase 0 completion

set -e  # Exit on error

echo "=========================================="
echo "Phase 0 Testing Suite"
echo "=========================================="

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Virtual environment not activated!"
    echo "Please run: source venv/bin/activate"
    exit 1
fi

echo "✅ Virtual environment: $VIRTUAL_ENV"
echo ""

# Test 1: Run verification script
echo "=========================================="
echo "Test 1: Environment Verification"
echo "=========================================="
python scripts/verify_setup.py
echo ""

# Test 2: Unit tests
echo "=========================================="
echo "Test 2: Unit Tests"
echo "=========================================="
pytest tests/unit/test_utils.py -v
echo ""

# Test 3: Unit tests with coverage
echo "=========================================="
echo "Test 3: Code Coverage Report"
echo "=========================================="
pytest tests/unit/test_utils.py -v --cov=src/utils --cov-report=term --cov-report=html
echo ""

# Test 4: Individual utility tests
echo "=========================================="
echo "Test 4: Individual Utility Tests"
echo "=========================================="

echo -n "Testing logging system... "
python -c "from src.utils.logging_config import get_logger; logger = get_logger('test'); logger.info('Test message'); print('✅')" 2>/dev/null

echo -n "Testing config loader... "
python -c "from src.utils.config_loader import Config; c = Config(); c.create_default_configs(); print('✅')" 2>/dev/null

echo -n "Testing coordinates... "
python -c "import numpy as np; from src.utils.coordinates import eci_to_ecef; pos = np.array([7000.0, 0.0, 0.0]); result = eci_to_ecef(pos, 0.0); print('✅')" 2>/dev/null

echo -n "Testing metrics... "
python -c "from src.utils.metrics import PerformanceMetrics; m = PerformanceMetrics(); m.record('test', 1.0); print('✅')" 2>/dev/null

echo ""

# Summary
echo "=========================================="
echo "✅ Phase 0 Testing Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Environment verification: PASSED"
echo "  - Unit tests: PASSED"
echo "  - Code coverage: See htmlcov/index.html"
echo "  - Individual utilities: PASSED"
echo ""
echo "Phase 0 is fully validated and ready for Phase 1!"
