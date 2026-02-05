#!/usr/bin/env python3
"""
CLI script for running the multi-object tracker.

Processes sensor measurements and generates tracks using the complete
tracking pipeline (Kalman filters, data association, track management).
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import click
import pandas as pd
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tracking import MultiObjectTracker, TrackerConfig, Measurement
from src.simulation.data_generator import Dataset
from src.utils.logging_config import get_logger

logger = get_logger("tracker")


@click.command()
@click.option(
    '--input',
    '-i',
    'input_path',
    type=click.Path(exists=True),
    help='Path to measurements file (parquet or dataset directory)'
)
@click.option(
    '--output',
    '-o',
    default='results/tracking_run',
    type=click.Path(),
    help='Output directory for results'
)
@click.option(
    '--filter',
    'filter_type',
    type=click.Choice(['ekf', 'ukf']),
    default='ukf',
    help='Kalman filter type'
)
@click.option(
    '--association',
    type=click.Choice(['hungarian', 'gnn']),
    default='hungarian',
    help='Data association method'
)
@click.option(
    '--duration',
    '-d',
    type=float,
    help='Duration to process (hours). If not specified, processes all data'
)
@click.option(
    '--quick',
    is_flag=True,
    help='Quick test mode (10 minutes of data)'
)
@click.option(
    '--no-maneuver-detection',
    is_flag=True,
    help='Disable maneuver detection'
)
@click.option(
    '--verbose',
    '-v',
    is_flag=True,
    help='Verbose output'
)
def main(input_path, output, filter_type, association, duration, quick, no_maneuver_detection, verbose):
    """
    Run multi-object tracker on measurement data.
    
    Examples:
        # Track objects from Phase 1 dataset
        python scripts/run_tracker.py -i data/processed/scenario_001
        
        # Quick test (10 minutes)
        python scripts/run_tracker.py -i data/processed/scenario_001 --quick
        
        # Use EKF with GNN association
        python scripts/run_tracker.py -i data/processed/scenario_001 --filter ekf --association gnn
        
        # Process specific duration
        python scripts/run_tracker.py -i data/processed/scenario_001 -d 2.0
    """
    click.echo("╔" + "═"*78 + "╗")
    click.echo("║" + " "*78 + "║")
    click.echo("║" + "  🛰️  MULTI-OBJECT TRACKER - Space Domain Awareness".center(78) + "║")
    click.echo("║" + " "*78 + "║")
    click.echo("╚" + "═"*78 + "╝")
    click.echo()
    
    # Load data
    click.echo("📂 Loading data...")
    
    if input_path:
        input_path = Path(input_path)
        
        # Check if it's a dataset directory
        if input_path.is_dir():
            dataset = Dataset.load(input_path)
            measurements_df = dataset.measurements
            ground_truth_df = dataset.ground_truth if hasattr(dataset, 'ground_truth') else None
            click.echo(f"   ✅ Loaded dataset from {input_path}")
        else:
            # Load parquet file directly
            measurements_df = pd.read_parquet(input_path)
            ground_truth_df = None
            click.echo(f"   ✅ Loaded measurements from {input_path}")
    else:
        # Look for default dataset
        default_paths = list(Path("data/processed").glob("scenario_*"))
        if not default_paths:
            click.echo("   ❌ No dataset specified and no default found")
            click.echo("   Run: python scripts/generate_dataset.py")
            return
        
        dataset = Dataset.load(default_paths[0])
        measurements_df = dataset.measurements
        ground_truth_df = dataset.ground_truth if hasattr(dataset, 'ground_truth') else None
        click.echo(f"   ✅ Loaded default dataset from {default_paths[0]}")
    
    # Filter by duration
    if quick:
        duration = 10.0 / 60.0  # 10 minutes
    
    if duration is not None:
        max_time = measurements_df['timestamp'].min() + duration * 3600.0
        measurements_df = measurements_df[measurements_df['timestamp'] <= max_time]
        if ground_truth_df is not None:
            ground_truth_df = ground_truth_df[ground_truth_df['timestamp'] <= max_time]
        click.echo(f"   📊 Processing {duration:.2f} hours of data")
    
    click.echo(f"   📊 {len(measurements_df)} measurements")
    click.echo(f"   📊 {measurements_df['object_id'].nunique()} unique objects")
    click.echo()
    
    # Configure tracker
    click.echo("⚙️  Configuring tracker...")
    config = TrackerConfig(
        filter_type=filter_type,
        association_method=association,
        maneuver_detection=not no_maneuver_detection
    )
    
    tracker = MultiObjectTracker(config)
    
    click.echo(f"   ✅ Filter: {filter_type.upper()}")
    click.echo(f"   ✅ Association: {association}")
    click.echo(f"   ✅ Maneuver detection: {'enabled' if not no_maneuver_detection else 'disabled'}")
    click.echo()
    
    # Process measurements
    click.echo("🔄 Processing measurements...")
    
    # Group by timestamp
    grouped = measurements_df.groupby('timestamp')
    timestamps = sorted(grouped.groups.keys())
    
    all_tracks = []
    all_associations = []
    
    with click.progressbar(timestamps, label='Tracking') as bar:
        for timestamp in bar:
            # Get measurements at this time
            meas_group = grouped.get_group(timestamp)
            
            # Convert to Measurement objects
            measurements = []
            for idx, row in meas_group.iterrows():
                meas = Measurement(
                    position=np.array([row['x'], row['y'], row['z']]),
                    covariance=np.eye(3) * 0.05**2,  # 50m std dev
                    timestamp=timestamp,
                    sensor_id=row.get('sensor_id', 'unknown'),
                    measurement_id=int(row.get('measurement_id', idx))
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
                    'position_uncertainty': track.get_position_uncertainty()
                })
    
    click.echo()
    
    # Get statistics
    stats = tracker.get_statistics()
    maneuver_events = tracker.get_maneuver_events()
    
    click.echo("📊 Tracking Results:")
    click.echo(f"   • Total updates: {stats['update_count']}")
    click.echo(f"   • Total tracks: {stats['total_tracks']}")
    click.echo(f"   • Confirmed tracks: {stats['confirmed_tracks']}")
    click.echo(f"   • Tentative tracks: {stats['tentative_tracks']}")
    click.echo(f"   • Total measurements: {stats['total_measurements']}")
    click.echo(f"   • Association rate: {stats['association_rate']:.1%}")
    click.echo(f"   • Maneuver events: {len(maneuver_events)}")
    click.echo()
    
    # Save results
    click.echo("💾 Saving results...")
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tracks
    tracks_df = pd.DataFrame(all_tracks)
    tracks_path = output_dir / "tracks.parquet"
    tracks_df.to_parquet(tracks_path)
    click.echo(f"   ✅ Tracks saved to {tracks_path}")
    
    # Save statistics
    stats_path = output_dir / "statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    click.echo(f"   ✅ Statistics saved to {stats_path}")
    
    # Save maneuver events
    if maneuver_events:
        maneuvers_data = [
            {
                'timestamp': e.timestamp,
                'track_id': e.track_id,
                'innovation_magnitude': e.innovation_magnitude,
                'chi_square_statistic': e.chi_square_statistic,
                'confidence': e.confidence,
                'duration': e.duration
            }
            for e in maneuver_events
        ]
        maneuvers_path = output_dir / "maneuvers.json"
        with open(maneuvers_path, 'w') as f:
            json.dump(maneuvers_data, f, indent=2)
        click.echo(f"   ✅ Maneuver events saved to {maneuvers_path}")
    
    # Save configuration
    config_path = output_dir / "config_used.yaml"
    with open(config_path, 'w') as f:
        f.write(f"# Tracker Configuration\n")
        f.write(f"filter_type: {config.filter_type}\n")
        f.write(f"association_method: {config.association_method}\n")
        f.write(f"gate_threshold: {config.gate_threshold}\n")
        f.write(f"confirmation_threshold: {config.confirmation_threshold}\n")
        f.write(f"deletion_threshold: {config.deletion_threshold}\n")
        f.write(f"maneuver_detection: {config.maneuver_detection}\n")
    click.echo(f"   ✅ Configuration saved to {config_path}")
    
    click.echo()
    click.echo("✨ Tracking complete!")
    click.echo()
    click.echo(f"📁 Results saved to: {output_dir}")
    click.echo()
    click.echo("Next steps:")
    click.echo(f"  • Evaluate: python scripts/evaluate_tracking.py --tracks {tracks_path}")
    if ground_truth_df is not None:
        gt_path = output_dir.parent.parent / "processed" / input_path.name / "ground_truth.parquet"
        if gt_path.exists():
            click.echo(f"  • With ground truth: python scripts/evaluate_tracking.py --tracks {tracks_path} --ground-truth {gt_path}")


if __name__ == "__main__":
    main()
