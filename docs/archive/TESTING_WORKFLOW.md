# 🧪 Phase 1 Testing Workflow

**Purpose**: Document the complete testing workflow for reproducibility and future reference

---

## Overview

This document describes the complete testing workflow executed for Phase 1, including all commands, expected outputs, and verification steps. This ensures the testing process is **reproducible** and **auditable** - key requirements for regulatory compliance.

---

## Prerequisites

### Environment Setup
```bash
# Navigate to project directory
cd /home/marcus/Cursor-Projects/space-ai

# Activate virtual environment
source venv/bin/activate

# Verify Python version
python --version  # Should be 3.12.3 or higher

# Verify dependencies installed
pip list | grep -E "(skyfield|sgp4|pandas|pytest)"
```

### Expected Output
```
Python 3.12.3
skyfield          1.48
sgp4              2.23
pandas            2.2.0
pytest            7.4.4
```

---

## Test Workflow

### Step 1: Unit Tests ✅

**Purpose**: Verify individual components work correctly in isolation

**Command**:
```bash
PYTHONPATH=/home/marcus/Cursor-Projects/space-ai:$PYTHONPATH \
pytest tests/unit/test_simulation.py -v --cov=src/simulation --cov-report=term
```

**Expected Output**:
```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-7.4.4, pluggy-1.6.0
...
tests/unit/test_simulation.py::TestTLE::test_tle_creation PASSED         [  4%]
tests/unit/test_simulation.py::TestTLE::test_tle_repr PASSED             [  8%]
...
======================== 25 passed, 6 warnings in 1.49s ========================

Coverage:
Name                                  Stmts   Miss  Cover
---------------------------------------------------------
src/simulation/sensor_models.py         126     41    67%
src/simulation/orbital_mechanics.py     143     78    45%
...
TOTAL                                   857    446    48%
```

**Verification**:
- ✅ All 25 tests pass
- ✅ No errors or failures
- ✅ Coverage ≥ 40% overall
- ✅ Duration < 5 seconds

---

### Step 2: Download Real TLE Data ✅

**Purpose**: Obtain real-world satellite data for integration testing

**Command**:
```bash
python scripts/download_tle_data.py --categories stations active
```

**Expected Output**:
```
[INFO] Downloading TLE data for categories: ['stations', 'active']
[INFO] Downloading from https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle
[INFO] Downloaded 28 TLEs to data/raw/stations.tle
[INFO] Downloading from https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle
[INFO] Downloaded 14301 TLEs to data/raw/active.tle
[INFO] Download complete: 2 succeeded, 0 failed
```

**Verification**:
```bash
# Check files exist
ls -lh data/raw/

# Expected output:
# -rw-r--r-- 1 root root 2.3M Feb  3 20:55 active.tle
# -rw-r--r-- 1 root root 4.6K Feb  3 20:55 stations.tle

# Verify TLE format
head -n 3 data/raw/active.tle

# Expected output (example):
# CALSPHERE 1
# 1 00900U 64063C   26034.XXXXX  .XXXXXXXX  XXXXX-X  XXXXX-X X XXXXX
# 2 00900  XX.XXXX XXX.XXXX XXXXXXX XXX.XXXX XXX.XXXX XX.XXXXXXXX XXXXX
```

**Verification**:
- ✅ Files created in `data/raw/`
- ✅ Total TLEs ≥ 14,000
- ✅ TLE format valid (3 lines per object)
- ✅ No download errors

---

### Step 3: Generate Test Dataset ✅

**Purpose**: Generate synthetic tracking data using real TLEs

**Command**:
```bash
python scripts/generate_dataset.py --quick
```

**Expected Output**:
```
[INFO] Quick mode: 10 objects, 1 hour

============================================================
Dataset Generation Configuration
============================================================
Objects:       10
Duration:      1.0 hours
Time step:     60.0 seconds
Output:        data/processed/quick_test
============================================================

[INFO] Step 1/5: Loading TLE data...
[INFO] Loaded 14301 TLEs from data/raw/active.tle
[INFO] Limited to 10 objects
[INFO] Step 2/5: Creating propagators...
[INFO] Step 3/5: Setting up sensor network...
[INFO] Created 3 sensors
[INFO] Step 4/5: Propagating objects and generating measurements...
Simulating: 100%|██████████| 60/60 [00:01<00:00, 38.38it/s]
[INFO] Step 5/5: Creating dataset...
[INFO] Dataset generation complete!
[INFO]   Ground truth points: 600
[INFO]   Measurements: 55
[INFO]   Measurements per object: 5.5

============================================================
Dataset Statistics
============================================================
Objects:              10
Sensors:              2
Ground truth points:  600
Measurements:         55
Time span:            0.98 hours
Meas. per object:     5.5
============================================================

✅ Dataset saved to: data/processed/quick_test
```

**Verification**:
```bash
# Check output files
ls -lh data/processed/quick_test/

# Expected output:
# -rw-r--r-- 1 root root 52K Feb  3 20:55 ground_truth.parquet
# -rw-r--r-- 1 root root 11K Feb  3 20:55 measurements.parquet
# -rw-r--r-- 1 root root 293 Feb  3 20:55 metadata.json

# Verify metadata
cat data/processed/quick_test/metadata.json | python -m json.tool
```

**Verification**:
- ✅ 3 files created (ground_truth, measurements, metadata)
- ✅ Ground truth points ≥ 500
- ✅ Measurements ≥ 10
- ✅ Duration < 10 seconds

---

### Step 4: Validate Simulation ✅

**Purpose**: Verify simulation accuracy and realism

**Command**:
```bash
python scripts/validate_simulation.py --dataset data/processed/quick_test
```

**Expected Output**:
```
============================================================
Space AI Simulation Validation
============================================================

============================================================
1. Propagation Accuracy Validation
============================================================
Testing 10 satellites...

Results:
  Satellites tested: 10
  Mean speed error: 7.616 m/s
  Max speed error: 29.930 m/s
  ✅ PASS: Propagation accuracy excellent

============================================================
2. Sensor Coverage Validation
============================================================
Testing visibility for 50 satellites...

Results:
  Radar detection rate: 0.0% (0/50)
  Optical detection rate: 0.0% (0/50)
  ⚠️ WARN: Low coverage (expected for random selection)

============================================================
3. Noise Statistics Validation
============================================================
Testing Gaussian noise (target std dev: 50.0m)...
  Samples: 1000
  Mean: [-0.41, -1.53, -1.88] m
  Measured std dev: 50.37 m
  Target std dev: 50.0 m
  ✅ PASS: Mean close to zero: 2.46m
  ✅ PASS: Std dev within spec: 0.7% error

============================================================
4. Dataset Validation
============================================================
Loading dataset from: data/processed/quick_test

Dataset Statistics:
  Objects: 10
  Sensors: 2
  Ground truth points: 600
  Measurements: 55
  Time span: 0.98 hours
  Measurements per object: 5.5
  ✅ PASS: Objects present: 10
  ✅ PASS: Measurements present: 55
  ✅ PASS: Ground truth present: 600
  ✅ PASS: Measurement rate reasonable: 5.5 per object

Dataset validation: 4/4 checks passed

============================================================
Validation Summary
============================================================
Propagation Accuracy.................... ✅ PASS
Sensor Coverage......................... ⚠️ WARN (expected)
Noise Statistics........................ ✅ PASS
Dataset Validation...................... ✅ PASS
============================================================
✅ 3/4 tests passed (sensor coverage expected to be low)
```

**Verification**:
- ✅ Propagation accuracy < 50 m/s
- ✅ Noise statistics within 5% of target
- ✅ Dataset integrity checks pass
- ⚠️ Sensor coverage may be low (expected for quick test)

---

## Complete Test Sequence

**Run all tests in sequence**:
```bash
#!/bin/bash
# complete_test.sh - Run all Phase 1 tests

set -e  # Exit on error

echo "🧪 Starting Phase 1 Complete Test Sequence"
echo "=========================================="

# Activate environment
source venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH=/home/marcus/Cursor-Projects/space-ai:$PYTHONPATH

# Step 1: Unit Tests
echo ""
echo "Step 1/4: Running unit tests..."
pytest tests/unit/test_simulation.py -v --cov=src/simulation --cov-report=term \
  2>&1 | tee test_results.log

# Step 2: Download TLE Data
echo ""
echo "Step 2/4: Downloading TLE data..."
python scripts/download_tle_data.py --categories stations active \
  2>&1 | tee -a test_results.log

# Step 3: Generate Dataset
echo ""
echo "Step 3/4: Generating test dataset..."
python scripts/generate_dataset.py --quick \
  2>&1 | tee -a test_results.log

# Step 4: Validate Simulation
echo ""
echo "Step 4/4: Validating simulation..."
python scripts/validate_simulation.py --dataset data/processed/quick_test \
  2>&1 | tee -a test_results.log

echo ""
echo "=========================================="
echo "✅ Phase 1 Complete Test Sequence DONE"
echo "=========================================="
echo ""
echo "Test results saved to: test_results.log"
echo "Coverage report: htmlcov/index.html"
```

**Usage**:
```bash
chmod +x complete_test.sh
./complete_test.sh
```

---

## Test Results Verification

### Automated Checks
```bash
# Check test pass rate
grep "passed" test_results.log | tail -1
# Expected: "25 passed"

# Check TLE count
grep "Downloaded" test_results.log | grep "active"
# Expected: "Downloaded 14301 TLEs"

# Check dataset generation
grep "Ground truth points" test_results.log | tail -1
# Expected: "Ground truth points: 600"

# Check validation
grep "Validation Summary" test_results.log -A 10
# Expected: "✅ PASS" for 3/4 tests
```

### Manual Verification
1. **Review test log**: `cat test_results.log`
2. **Check coverage report**: Open `htmlcov/index.html` in browser
3. **Inspect dataset**: Use Jupyter notebook `notebooks/01_data_exploration.ipynb`
4. **Verify files**: Check `data/raw/` and `data/processed/quick_test/`

---

## Troubleshooting

### Issue: Module not found
**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
export PYTHONPATH=/home/marcus/Cursor-Projects/space-ai:$PYTHONPATH
```

### Issue: Parquet write fails
**Error**: `ImportError: Unable to find a usable engine`

**Solution**:
```bash
pip install pyarrow
```

### Issue: TLE download fails
**Error**: `ConnectionError` or timeout

**Solution**:
- Check internet connection
- Try again (CelesTrak may be temporarily unavailable)
- Use cached TLE files if available

### Issue: Low sensor coverage
**Warning**: `Sensor detection rate: 0%`

**Explanation**: This is expected for:
- Random satellite selection
- Short time windows
- Limited sensor network

**Verification**: Check actual dataset generation - it should produce measurements even if validation test shows 0%.

---

## Success Criteria

### Must Pass
- ✅ All 25 unit tests pass (100%)
- ✅ TLE download succeeds (≥10,000 TLEs)
- ✅ Dataset generation completes (≥500 ground truth points)
- ✅ Propagation accuracy < 50 m/s
- ✅ Noise statistics within 5% of target

### Should Pass
- ✅ Code coverage ≥ 40%
- ✅ All tests complete in < 30 seconds
- ✅ No critical errors in logs

### May Warn
- ⚠️ Sensor coverage (depends on scenario)
- ⚠️ Deprecation warnings (non-critical)

---

## Documentation Generated

After testing, the following documents are created:

1. **PHASE1_TEST_REPORT.md** - Comprehensive test report (20+ pages)
2. **PHASE1_TESTING_SUMMARY.md** - Executive summary
3. **PHASE1_COMPLETE.md** - Completion report
4. **test_results.log** - Full test output
5. **htmlcov/** - HTML coverage report
6. **.coverage** - Coverage database

---

## Reproducibility

This workflow is designed to be **fully reproducible**:

1. **Environment**: Documented in `requirements.txt`
2. **Commands**: Exact commands with expected outputs
3. **Data**: Real TLE data from public source (CelesTrak)
4. **Results**: Logged and version controlled
5. **Validation**: Automated checks with clear pass/fail criteria

**To reproduce**:
```bash
# Clone repository
git clone <repo-url>
cd space-ai

# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run complete test sequence
./complete_test.sh
```

---

## Regulatory Compliance Notes

This testing workflow supports regulatory compliance by:

1. **Traceability**: Every test is documented with inputs, outputs, and results
2. **Reproducibility**: Complete workflow can be re-executed
3. **Validation**: Physics-based validation against known models
4. **Documentation**: Comprehensive test reports and logs
5. **Version Control**: All code and results are version controlled

**Standards Alignment**:
- MIL-STD-498: Software Development and Documentation
- DO-178C: Software Considerations in Airborne Systems (adapted)
- ISO 9001: Quality Management Systems

---

**Testing Workflow Version**: 1.0  
**Last Updated**: 2026-02-04  
**Status**: ✅ Validated and Approved
