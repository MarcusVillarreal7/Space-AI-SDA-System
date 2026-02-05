#!/usr/bin/env python3
"""
Build comprehensive Phase 1 + Phase 2 notebook
"""

import json
from pathlib import Path

# Initialize notebook structure
notebook = {
    'cells': [],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python (space-ai)',
            'language': 'python',
            'name': 'space-ai'
        },
        'language_info': {
            'codemirror_mode': {'name': 'ipython', 'version': 3},
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.12.3'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

def add_markdown_cell(content):
    """Add a markdown cell"""
    notebook['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': content.split('\n')
    })

def add_code_cell(content):
    """Add a code cell"""
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': content.split('\n')
    })

# Cell 0: Title
add_markdown_cell("""# 🛰️ Phase 1 + Phase 2: Complete Tracking Pipeline

**Purpose**: Demonstrate end-to-end tracking from simulation to validated tracks

**Author**: Space AI Project  
**Date**: 2026-02-04

---

## Overview

This notebook demonstrates the complete tracking pipeline:

**Phase 1 (Simulation)**:
- Ground truth orbital trajectories
- Sensor measurements with realistic noise
- Data quality and coverage analysis

**Phase 2 (Tracking)**:
- Multi-object tracking with Kalman filters
- Data association (Hungarian algorithm)
- Track lifecycle management
- Maneuver detection

**Phase 3 (Validation)**:
- Performance metrics (RMSE, completeness)
- Error analysis
- Track quality assessment""")

# Cell 1: Setup
add_code_cell("""# Setup
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Phase 1 imports
from src.simulation.data_generator import Dataset
from src.utils.coordinates import eci_to_geodetic

# Phase 2 imports
from src.tracking import MultiObjectTracker, Measurement

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
%matplotlib inline

print("✅ All imports successful")""")

# Cell 2: Phase 1 Header
add_markdown_cell("""## Part 1: Phase 1 - Simulation Data

Load and explore the synthetic tracking dataset generated in Phase 1.""")

# Cell 3: Load Dataset
add_code_cell("""# Load dataset
print("="*60)
print("PHASE 1: SIMULATION DATA")
print("="*60)

dataset_path = Path('../data/processed/quick_test')
dataset = Dataset.load(dataset_path)

print(f"\\n📦 Dataset: {dataset_path.name}")
print(f"  • Objects: {len(dataset.ground_truth['object_id'].unique())}")
print(f"  • Ground truth points: {len(dataset.ground_truth)}")
print(f"  • Measurements: {len(dataset.measurements)}")
print(f"  • Sensors: {len(dataset.measurements['sensor_id'].unique())}")
print(f"  • Time span: {(dataset.ground_truth['time'].max() - dataset.ground_truth['time'].min()).total_seconds() / 3600:.2f} hours")""")

# Cell 4: Visualize Phase 1
add_code_cell("""# Visualize Phase 1 data
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Measurement coverage over time
measurements_df = dataset.measurements.copy()
measurements_df['time_hours'] = (measurements_df['time'] - measurements_df['time'].min()).dt.total_seconds() / 3600
axes[0, 0].scatter(measurements_df['time_hours'], measurements_df['object_id'], alpha=0.6, s=20, c=measurements_df['sensor_id'].astype('category').cat.codes)
axes[0, 0].set_xlabel('Time (hours)')
axes[0, 0].set_ylabel('Object ID')
axes[0, 0].set_title('Measurement Coverage (Phase 1)')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Measurements per sensor
sensor_counts = measurements_df.groupby('sensor_id').size()
axes[0, 1].bar(range(len(sensor_counts)), sensor_counts.values)
axes[0, 1].set_xlabel('Sensor ID')
axes[0, 1].set_ylabel('Number of Measurements')
axes[0, 1].set_title('Measurements per Sensor')
axes[0, 1].set_xticks(range(len(sensor_counts)))
axes[0, 1].set_xticklabels(sensor_counts.index)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Ground truth trajectories (2D projection)
gt_df = dataset.ground_truth.copy()
for obj_id in gt_df['object_id'].unique()[:5]:  # First 5 objects
    obj_data = gt_df[gt_df['object_id'] == obj_id]
    axes[1, 0].plot(obj_data['x'], obj_data['y'], alpha=0.7, label=f'Object {obj_id}')
axes[1, 0].set_xlabel('X (km)')
axes[1, 0].set_ylabel('Y (km)')
axes[1, 0].set_title('Ground Truth Trajectories (XY Plane)')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axis('equal')

# Plot 4: Altitude distribution
altitudes = np.sqrt(gt_df['x']**2 + gt_df['y']**2 + gt_df['z']**2) - 6378.137
axes[1, 1].hist(altitudes, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Altitude (km)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Altitude Distribution')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\\n✅ Phase 1 data visualized")""")

# Cell 5: Phase 2 Header
add_markdown_cell("""## Part 2: Phase 2 - Multi-Object Tracking

Run the tracking pipeline on the simulation data to generate tracks.""")

# Cell 6: Run Tracker
add_code_cell("""# Configure and run tracker
print("\\n" + "="*60)
print("PHASE 2: MULTI-OBJECT TRACKING")
print("="*60)

# Configure tracker
tracker = MultiObjectTracker(
    filter_type="ukf",
    association_method="hungarian",
    confirmation_threshold=3,
    deletion_threshold=5,
    maneuver_detection_enabled=True
)

print(f"\\n⚙️  Tracker Configuration:")
print(f"  • Filter: UKF")
print(f"  • Association: Hungarian")
print(f"  • Maneuver detection: Enabled")

# Process measurements
print(f"\\n🔄 Processing {len(measurements_df)} measurements...")

# Group by timestamp
grouped = measurements_df.groupby('timestamp')
timestamps = sorted(grouped.groups.keys())

all_tracks = []
update_count = 0

for timestamp in timestamps:
    # Get measurements at this time
    meas_group = grouped.get_group(timestamp)
    
    # Convert to Measurement objects
    measurements = []
    for idx, row in meas_group.iterrows():
        meas = Measurement(
            position=np.array([row['x'], row['y'], row['z']]),
            covariance=np.eye(3) * 0.05**2,  # 50m std dev
            timestamp=timestamp,
            sensor_id=row['sensor_id'],
            measurement_id=int(idx)
        )
        measurements.append(meas)
    
    # Update tracker
    tracks = tracker.update(measurements, timestamp)
    
    # Record track states
    for track in tracks:
        all_tracks.append({
            'timestamp': timestamp,
            'track_id': track.track_id,
            'state': track.state.value,
            'x': track.get_position()[0],
            'y': track.get_position()[1],
            'z': track.get_position()[2],
            'vx': track.get_velocity()[0],
            'vy': track.get_velocity()[1],
            'vz': track.get_velocity()[2],
            'hit_count': track.hit_count,
            'miss_count': track.miss_count,
            'is_maneuvering': track.is_maneuvering,
            'uncertainty': track.get_position_uncertainty()
        })
    
    update_count += 1

tracks_df = pd.DataFrame(all_tracks)

print(f"✅ Tracking complete!")
print(f"  • Updates processed: {update_count}")
print(f"  • Track states recorded: {len(tracks_df)}")""")

# Cell 7: Tracker Statistics
add_code_cell("""# Display tracker statistics
stats = tracker.get_statistics()
maneuver_events = tracker.get_maneuver_events()

print("\\n📊 Tracking Statistics:")
print(f"  • Total tracks: {stats['total_tracks']}")
print(f"  • Confirmed tracks: {stats['confirmed_tracks']}")
print(f"  • Tentative tracks: {stats['tentative_tracks']}")
print(f"  • Association rate: {stats['association_rate']:.1%}")
print(f"  • Maneuver events: {len(maneuver_events)}")

if len(maneuver_events) > 0:
    print(f"\\n⚠️  Maneuver Events Detected:")
    for event in maneuver_events[:5]:  # Show first 5
        print(f"  • Track {event.track_id} at t={event.timestamp:.1f}s (confidence: {event.confidence:.1%})")""")

# Cell 8: Visualize Phase 2
add_code_cell("""# Visualize Phase 2 results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Track states over time
for state in tracks_df['state'].unique():
    state_data = tracks_df[tracks_df['state'] == state]
    axes[0, 0].scatter(state_data['timestamp'], state_data['track_id'], 
                      label=state.upper(), alpha=0.6, s=20)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Track ID')
axes[0, 0].set_title('Track States Over Time')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Track completeness (hits per track)
confirmed_tracks = tracks_df[tracks_df['state'] == 'confirmed']
if len(confirmed_tracks) > 0:
    track_hits = confirmed_tracks.groupby('track_id')['hit_count'].max()
    axes[0, 1].bar(range(len(track_hits)), track_hits.values)
    axes[0, 1].set_xlabel('Track ID')
    axes[0, 1].set_ylabel('Hit Count')
    axes[0, 1].set_title('Track Completeness (Confirmed Tracks)')
    axes[0, 1].set_xticks(range(len(track_hits)))
    axes[0, 1].set_xticklabels(track_hits.index)
    axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Tracked trajectories (2D projection)
for track_id in tracks_df['track_id'].unique()[:5]:  # First 5 tracks
    track_data = tracks_df[tracks_df['track_id'] == track_id]
    axes[1, 0].plot(track_data['x'], track_data['y'], alpha=0.7, label=f'Track {track_id}')
axes[1, 0].set_xlabel('X (km)')
axes[1, 0].set_ylabel('Y (km)')
axes[1, 0].set_title('Tracked Trajectories (XY Plane)')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axis('equal')

# Plot 4: Position uncertainty over time
confirmed_tracks = tracks_df[tracks_df['state'] == 'confirmed']
if len(confirmed_tracks) > 0:
    for track_id in confirmed_tracks['track_id'].unique()[:5]:
        track_data = confirmed_tracks[confirmed_tracks['track_id'] == track_id]
        axes[1, 1].plot(track_data['timestamp'], track_data['uncertainty'], 
                       alpha=0.7, label=f'Track {track_id}')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Position Uncertainty (km)')
    axes[1, 1].set_title('Track Uncertainty Over Time')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\\n✅ Phase 2 results visualized")""")

# Cell 9: Phase 3 Header
add_markdown_cell("""## Part 3: Performance Evaluation

Compare tracked positions against ground truth to measure accuracy.""")

# Cell 10: Evaluate Performance
add_code_cell("""# Evaluate tracking performance
print("\\n" + "="*60)
print("PHASE 3: PERFORMANCE EVALUATION")
print("="*60)

# Prepare ground truth with timestamps
gt_df['timestamp'] = (gt_df['time'] - gt_df['time'].min()).dt.total_seconds()

# Merge tracks with ground truth (assuming track_id == object_id)
merged = pd.merge(
    tracks_df[tracks_df['state'] == 'confirmed'],
    gt_df,
    left_on=['timestamp', 'track_id'],
    right_on=['timestamp', 'object_id'],
    how='inner',
    suffixes=('_track', '_truth')
)

if len(merged) > 0:
    # Compute position errors
    merged['position_error'] = np.sqrt(
        (merged['x_track'] - merged['x_truth'])**2 +
        (merged['y_track'] - merged['y_truth'])**2 +
        (merged['z_track'] - merged['z_truth'])**2
    )
    
    # Compute velocity errors
    merged['velocity_error'] = np.sqrt(
        (merged['vx_track'] - merged['vx_truth'])**2 +
        (merged['vy_track'] - merged['vy_truth'])**2 +
        (merged['vz_track'] - merged['vz_truth'])**2
    )
    
    # Metrics
    pos_rmse = np.sqrt(np.mean(merged['position_error']**2))
    pos_mae = np.mean(merged['position_error'])
    vel_rmse = np.sqrt(np.mean(merged['velocity_error']**2))
    vel_mae = np.mean(merged['velocity_error'])
    
    print(f"\\n📊 Performance Metrics:")
    print(f"  • Position RMSE: {pos_rmse:.3f} km ({pos_rmse*1000:.1f} m)")
    print(f"  • Position MAE:  {pos_mae:.3f} km ({pos_mae*1000:.1f} m)")
    print(f"  • Velocity RMSE: {vel_rmse:.4f} km/s ({vel_rmse*1000:.2f} m/s)")
    print(f"  • Velocity MAE:  {vel_mae:.4f} km/s ({vel_mae*1000:.2f} m/s)")
    
    # Assessment
    print(f"\\n✅ Assessment:")
    if pos_rmse * 1000 < 100:
        print(f"  • Position accuracy: EXCELLENT ({pos_rmse*1000:.1f}m < 100m target) ✅")
    elif pos_rmse * 1000 < 200:
        print(f"  • Position accuracy: GOOD ({pos_rmse*1000:.1f}m < 200m) ⚠️")
    else:
        print(f"  • Position accuracy: NEEDS IMPROVEMENT ({pos_rmse*1000:.1f}m) ❌")
    
    if vel_rmse * 1000 < 10:
        print(f"  • Velocity accuracy: EXCELLENT ({vel_rmse*1000:.2f} m/s < 10 m/s target) ✅")
    else:
        print(f"  • Velocity accuracy: GOOD ({vel_rmse*1000:.2f} m/s) ⚠️")
else:
    print("\\n⚠️  No matching tracks found for evaluation")
    print("This may happen if track IDs don't match object IDs or if no tracks were confirmed.")""")

# Cell 11: Visualize Errors
add_code_cell("""# Visualize errors
if len(merged) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Position error over time
    axes[0].scatter(merged['timestamp'], merged['position_error'], alpha=0.5, s=10)
    axes[0].axhline(pos_rmse, color='r', linestyle='--', label=f'RMSE: {pos_rmse:.3f} km')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Position Error (km)')
    axes[0].set_title('Position Error Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error histogram
    axes[1].hist(merged['position_error'], bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(pos_rmse, color='r', linestyle='--', label=f'RMSE: {pos_rmse:.3f} km')
    axes[1].set_xlabel('Position Error (km)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Position Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\\n✅ Error analysis visualized")""")

# Cell 12: Summary Header
add_markdown_cell("""## Summary

This notebook demonstrates the complete tracking pipeline from simulation (Phase 1) through tracking (Phase 2) to validation (Phase 3).""")

# Cell 13: Final Summary
add_code_cell("""# Final summary
print("\\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f\"\"\"
✅ **Phase 1 (Simulation)**: 
   • Generated {len(measurements_df)} measurements from {len(gt_df['object_id'].unique())} objects
   • Time span: {(gt_df['time'].max() - gt_df['time'].min()).total_seconds() / 3600:.2f} hours
   • Sensors: {len(measurements_df['sensor_id'].unique())}

✅ **Phase 2 (Tracking)**: 
   • Processed {update_count} time steps
   • Created {stats['total_tracks']} tracks
   • Confirmed {stats['confirmed_tracks']} tracks
   • Association rate: {stats['association_rate']:.1%}
   • Detected {len(maneuver_events)} maneuvers
\"\"\")

if len(merged) > 0:
    print(f\"\"\"✅ **Phase 3 (Evaluation)**:
   • Position RMSE: {pos_rmse*1000:.1f} m (target: <100m)
   • Velocity RMSE: {vel_rmse*1000:.2f} m/s (target: <10 m/s)
   • Matched track states: {len(merged)}
\"\"\")

print(f\"\"\"
🎯 **Next Steps**:
   1. Test with larger datasets (100+ objects)
   2. Test with real TLE data from CelesTrak
   3. Implement Phase 3 (ML Prediction)
   4. Build operational dashboard (Phase 4)
\"\"\")

print("✨ Analysis complete!")""")

# Write notebook
output_path = Path(__file__).parent.parent / 'notebooks' / '01_data_exploration.ipynb'
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f'✅ Comprehensive notebook created: {output_path}')
print(f'   Total cells: {len(notebook["cells"])}')
