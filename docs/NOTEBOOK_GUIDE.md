# 📓 Jupyter Notebook Guide

## Overview

The `01_data_exploration.ipynb` notebook provides a comprehensive demonstration of the complete tracking pipeline from simulation through validation.

---

## Notebook Structure

### Part 1: Phase 1 - Simulation Data

**Purpose**: Load and explore synthetic tracking dataset

**Visualizations**:
- Measurement coverage over time
- Measurements per sensor
- Ground truth trajectories (2D projection)
- Altitude distribution

**Key Metrics**:
- Number of objects
- Number of measurements
- Sensor coverage
- Time span

---

### Part 2: Phase 2 - Multi-Object Tracking

**Purpose**: Run tracking pipeline on simulation data

**Components**:
- Multi-object tracker configuration (UKF, Hungarian)
- Measurement processing
- Track state recording
- Maneuver detection

**Visualizations**:
- Track states over time
- Track completeness (hits per track)
- Tracked trajectories (2D projection)
- Position uncertainty over time

**Key Metrics**:
- Total tracks created
- Confirmed tracks
- Association rate
- Maneuver events detected

---

### Part 3: Performance Evaluation

**Purpose**: Validate tracking accuracy against ground truth

**Metrics**:
- Position RMSE/MAE (target: <100m)
- Velocity RMSE/MAE (target: <10 m/s)
- Track completeness
- Error distribution

**Visualizations**:
- Position error over time
- Error distribution histogram

**Assessment**:
- ✅ EXCELLENT: Position RMSE < 100m
- ⚠️ GOOD: Position RMSE < 200m
- ❌ NEEDS IMPROVEMENT: Position RMSE > 200m

---

## How to Use

### 1. Setup Environment

Ensure you're using the correct kernel:

```bash
# Activate virtual environment
source venv/bin/activate

# Verify kernel is registered
jupyter kernelspec list | grep space-ai

# If not registered, run:
python -m ipykernel install --user --name=space-ai --display-name="Python (space-ai)"
```

### 2. Launch Jupyter

```bash
# From project root
jupyter notebook notebooks/01_data_exploration.ipynb

# Or use Jupyter Lab
jupyter lab notebooks/01_data_exploration.ipynb
```

### 3. Run the Notebook

**Option A: Run All Cells**
- Click `Kernel > Restart & Run All`
- Wait for all cells to complete (~30 seconds)

**Option B: Run Step-by-Step**
- Click `Cell > Run Cells` or press `Shift+Enter`
- Review output after each cell

### 4. Interpret Results

**Phase 1 Output**:
- Should show 10 objects, ~50-60 measurements
- Trajectories should be smooth orbits
- Altitudes should be 400-800 km

**Phase 2 Output**:
- Should create ~10 tracks
- Confirm most tracks (8-10 confirmed)
- Association rate should be >80%

**Phase 3 Output**:
- Position RMSE should be <100m (EXCELLENT)
- Velocity RMSE should be <10 m/s (EXCELLENT)
- Error distribution should be Gaussian

---

## Customization

### Using Different Datasets

Change the dataset path in Cell 3:

```python
# Use a different dataset
dataset_path = Path('../data/processed/your_dataset_name')
dataset = Dataset.load(dataset_path)
```

### Tracker Configuration

Modify tracker settings in Cell 6:

```python
# Example: Use EKF instead of UKF
tracker = MultiObjectTracker(
    filter_type="ekf",  # Changed from "ukf"
    association_method="gnn",  # Changed from "hungarian"
    confirmation_threshold=5,  # Changed from 3
    deletion_threshold=10,  # Changed from 5
    maneuver_detection_enabled=False  # Disabled
)
```

### Visualization Options

Adjust plot parameters:

```python
# Show more objects in trajectory plot
for obj_id in gt_df['object_id'].unique()[:10]:  # Changed from [:5]
    obj_data = gt_df[gt_df['object_id'] == obj_id]
    axes[1, 0].plot(obj_data['x'], obj_data['y'], alpha=0.7, label=f'Object {obj_id}')
```

---

## Troubleshooting

### Issue: ModuleNotFoundError

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**: 
1. Ensure you're using the `Python (space-ai)` kernel
2. Check that `sys.path.append('..')` is in Cell 1
3. Verify you're running from the `notebooks/` directory

### Issue: No matching tracks found

**Problem**: "⚠️ No matching tracks found for evaluation"

**Solution**:
- This happens when track IDs don't match object IDs
- For test datasets, track_id should equal object_id
- Check that tracker confirmed at least some tracks

### Issue: Poor tracking performance

**Problem**: Position RMSE > 200m

**Solution**:
1. Check measurement noise levels (should be ~50m)
2. Verify tracker configuration (use UKF for better accuracy)
3. Ensure sufficient measurements per object (>3)
4. Check for maneuvers (increase process noise if detected)

---

## Regenerating the Notebook

If the notebook gets corrupted or you want to rebuild it:

```bash
# From project root
python scripts/build_notebook.py
```

This will regenerate `notebooks/01_data_exploration.ipynb` from scratch.

---

## Interview Demo Tips

### What to Highlight

1. **Complete Pipeline**
   - "This notebook demonstrates the entire tracking pipeline from raw measurements to validated tracks"

2. **Realistic Simulation**
   - "Notice the measurement noise and gaps - this simulates real sensor limitations"

3. **Performance Metrics**
   - "We achieve sub-100m position accuracy, which meets defense standards"

4. **Uncertainty Quantification**
   - "The uncertainty plot shows how confidence improves with more measurements"

5. **Maneuver Detection**
   - "The system automatically detects orbital maneuvers and adapts"

### Live Demo Flow

1. **Start**: "Let me show you our space tracking system end-to-end"
2. **Phase 1**: "First, we simulate realistic sensor measurements with noise"
3. **Phase 2**: "Then our tracker processes these measurements in real-time"
4. **Phase 3**: "Finally, we validate against ground truth to prove accuracy"
5. **Results**: "As you can see, we achieve excellent performance metrics"

### Questions to Anticipate

**Q**: "How does this handle real-world data?"
- **A**: "The simulation includes realistic noise models. We can swap in real TLE data from CelesTrak with minimal changes."

**Q**: "What about scalability?"
- **A**: "This demo uses 10 objects, but the architecture scales to 1000+ objects. The Hungarian algorithm is O(n³) but we can switch to GNN for larger catalogs."

**Q**: "How do you ensure reliability?"
- **A**: "Notice the comprehensive testing - we validate every component with unit tests and this notebook provides end-to-end validation."

---

## Next Steps

After running this notebook:

1. **Test with larger datasets**:
   ```bash
   python scripts/generate_dataset.py --objects 100 --duration 24 --output large_test
   ```

2. **Test with real TLE data**:
   ```bash
   python scripts/download_tle_data.py --source celestrak --category active
   ```

3. **Explore Phase 3 (ML Prediction)**:
   - Train trajectory prediction models
   - Implement object classification
   - Add threat scoring

4. **Build operational dashboard (Phase 4)**:
   - Real-time tracking visualization
   - Alert system
   - Operator interface

---

## Additional Resources

- **Setup Guide**: `docs/JUPYTER_SETUP.md`
- **Phase 1 Documentation**: `docs/PHASE1_COMPLETE.md`
- **Phase 2 Documentation**: `docs/PHASE2_COMPLETE.md`
- **Testing Guide**: `docs/TESTING_WORKFLOW.md`

---

**Last Updated**: 2026-02-04  
**Notebook Version**: 1.0  
**Total Cells**: 14
