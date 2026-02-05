#!/usr/bin/env python3
"""
CLI script for evaluating tracking performance.

Compares tracker output against ground truth to compute performance metrics
like position RMSE, track completeness, and false track rate.
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import click
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_config import get_logger

logger = get_logger("evaluation")


def compute_position_rmse(tracks_df, ground_truth_df):
    """Compute position RMSE between tracks and ground truth."""
    # Merge on timestamp and object_id
    merged = pd.merge(
        tracks_df,
        ground_truth_df,
        left_on=['timestamp', 'track_id'],
        right_on=['timestamp', 'object_id'],
        how='inner',
        suffixes=('_track', '_truth')
    )
    
    if len(merged) == 0:
        return None, None
    
    # Compute position errors
    errors = np.sqrt(
        (merged['x_track'] - merged['x_truth'])**2 +
        (merged['y_track'] - merged['y_truth'])**2 +
        (merged['z_track'] - merged['z_truth'])**2
    )
    
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(errors)
    
    return rmse, mae


def compute_velocity_rmse(tracks_df, ground_truth_df):
    """Compute velocity RMSE between tracks and ground truth."""
    merged = pd.merge(
        tracks_df,
        ground_truth_df,
        left_on=['timestamp', 'track_id'],
        right_on=['timestamp', 'object_id'],
        how='inner',
        suffixes=('_track', '_truth')
    )
    
    if len(merged) == 0:
        return None, None
    
    # Compute velocity errors
    errors = np.sqrt(
        (merged['vx_track'] - merged['vx_truth'])**2 +
        (merged['vy_track'] - merged['vy_truth'])**2 +
        (merged['vz_track'] - merged['vz_truth'])**2
    )
    
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(errors)
    
    return rmse, mae


def compute_track_completeness(tracks_df, ground_truth_df):
    """Compute track completeness (% of time each object is tracked)."""
    # Count timestamps per object in ground truth
    gt_counts = ground_truth_df.groupby('object_id')['timestamp'].count()
    
    # Count timestamps per track (only confirmed)
    confirmed_tracks = tracks_df[tracks_df['state'] == 'confirmed']
    track_counts = confirmed_tracks.groupby('track_id')['timestamp'].count()
    
    # Compute completeness
    completeness_values = []
    for obj_id in gt_counts.index:
        if obj_id in track_counts.index:
            completeness = track_counts[obj_id] / gt_counts[obj_id]
            completeness_values.append(min(1.0, completeness))
        else:
            completeness_values.append(0.0)
    
    mean_completeness = np.mean(completeness_values)
    
    return mean_completeness, completeness_values


def compute_false_track_rate(tracks_df, ground_truth_df):
    """Compute false track rate (% of tracks that don't match real objects)."""
    # Get unique track IDs (confirmed only)
    confirmed_tracks = tracks_df[tracks_df['state'] == 'confirmed']
    unique_tracks = confirmed_tracks['track_id'].unique()
    
    # Get unique object IDs in ground truth
    unique_objects = ground_truth_df['object_id'].unique()
    
    # Count tracks that match objects
    matched_tracks = set(unique_tracks) & set(unique_objects)
    
    # False track rate
    false_tracks = len(unique_tracks) - len(matched_tracks)
    false_track_rate = false_tracks / max(1, len(unique_tracks))
    
    return false_track_rate, false_tracks, len(unique_tracks)


def plot_position_errors(tracks_df, ground_truth_df, output_path):
    """Plot position errors over time."""
    merged = pd.merge(
        tracks_df,
        ground_truth_df,
        left_on=['timestamp', 'track_id'],
        right_on=['timestamp', 'object_id'],
        how='inner',
        suffixes=('_track', '_truth')
    )
    
    if len(merged) == 0:
        return
    
    # Compute errors
    merged['position_error'] = np.sqrt(
        (merged['x_track'] - merged['x_truth'])**2 +
        (merged['y_track'] - merged['y_truth'])**2 +
        (merged['z_track'] - merged['z_truth'])**2
    )
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Error over time
    axes[0].scatter(merged['timestamp'], merged['position_error'], alpha=0.3, s=1)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Position Error (km)')
    axes[0].set_title('Position Error Over Time')
    axes[0].grid(True, alpha=0.3)
    
    # Error histogram
    axes[1].hist(merged['position_error'], bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Position Error (km)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Position Error Distribution')
    axes[1].axvline(merged['position_error'].mean(), color='r', linestyle='--', label=f'Mean: {merged["position_error"].mean():.3f} km')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_track_completeness(completeness_values, output_path):
    """Plot track completeness distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(completeness_values, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Track Completeness')
    ax.set_ylabel('Number of Objects')
    ax.set_title('Track Completeness Distribution')
    ax.axvline(np.mean(completeness_values), color='r', linestyle='--', label=f'Mean: {np.mean(completeness_values):.1%}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


@click.command()
@click.option(
    '--tracks',
    '-t',
    'tracks_path',
    required=True,
    type=click.Path(exists=True),
    help='Path to tracks parquet file'
)
@click.option(
    '--ground-truth',
    '-g',
    'ground_truth_path',
    type=click.Path(exists=True),
    help='Path to ground truth parquet file'
)
@click.option(
    '--output',
    '-o',
    default='reports/evaluation',
    type=click.Path(),
    help='Output directory for evaluation results'
)
@click.option(
    '--metric',
    type=click.Choice(['all', 'position_rmse', 'velocity_rmse', 'completeness', 'false_tracks']),
    default='all',
    help='Specific metric to compute'
)
@click.option(
    '--threshold',
    type=float,
    help='Alert threshold for metrics (e.g., RMSE > threshold)'
)
def main(tracks_path, ground_truth_path, output, metric, threshold):
    """
    Evaluate tracking performance against ground truth.
    
    Examples:
        # Full evaluation
        python scripts/evaluate_tracking.py -t results/tracking_run/tracks.parquet -g data/processed/scenario_001/ground_truth.parquet
        
        # Check specific metric
        python scripts/evaluate_tracking.py -t results/tracking_run/tracks.parquet -g data/processed/scenario_001/ground_truth.parquet --metric position_rmse
        
        # With alert threshold
        python scripts/evaluate_tracking.py -t results/tracking_run/tracks.parquet -g data/processed/scenario_001/ground_truth.parquet --threshold 0.1
    """
    click.echo("╔" + "═"*78 + "╗")
    click.echo("║" + " "*78 + "║")
    click.echo("║" + "  📊 TRACKING EVALUATION - Performance Analysis".center(78) + "║")
    click.echo("║" + " "*78 + "║")
    click.echo("╚" + "═"*78 + "╝")
    click.echo()
    
    # Load tracks
    click.echo("📂 Loading data...")
    tracks_df = pd.read_parquet(tracks_path)
    click.echo(f"   ✅ Loaded {len(tracks_df)} track states")
    click.echo(f"   📊 {tracks_df['track_id'].nunique()} unique tracks")
    click.echo(f"   📊 {tracks_df['timestamp'].nunique()} time steps")
    
    # Basic statistics (no ground truth needed)
    click.echo()
    click.echo("📊 Track Statistics:")
    state_counts = tracks_df.groupby('state').size()
    for state, count in state_counts.items():
        click.echo(f"   • {state.capitalize()}: {count}")
    
    confirmed = tracks_df[tracks_df['state'] == 'confirmed']
    if len(confirmed) > 0:
        click.echo(f"   • Mean position uncertainty: {confirmed['position_uncertainty'].mean():.3f} km")
        click.echo(f"   • Maneuvering tracks: {confirmed['is_maneuvering'].sum()}")
    
    # If no ground truth, stop here
    if not ground_truth_path:
        click.echo()
        click.echo("ℹ️  No ground truth provided. Only basic statistics available.")
        click.echo("   For full evaluation, provide ground truth with --ground-truth")
        return
    
    # Load ground truth
    ground_truth_df = pd.read_parquet(ground_truth_path)
    click.echo(f"   ✅ Loaded {len(ground_truth_df)} ground truth states")
    click.echo(f"   📊 {ground_truth_df['object_id'].nunique()} unique objects")
    click.echo()
    
    # Compute metrics
    click.echo("🔍 Computing metrics...")
    metrics = {}
    
    if metric in ['all', 'position_rmse']:
        pos_rmse, pos_mae = compute_position_rmse(tracks_df, ground_truth_df)
        if pos_rmse is not None:
            metrics['position_rmse_km'] = pos_rmse
            metrics['position_mae_km'] = pos_mae
            click.echo(f"   ✅ Position RMSE: {pos_rmse:.3f} km ({pos_rmse*1000:.1f} m)")
            click.echo(f"   ✅ Position MAE: {pos_mae:.3f} km ({pos_mae*1000:.1f} m)")
            
            if threshold and pos_rmse > threshold:
                click.echo(f"   ⚠️  Position RMSE exceeds threshold ({threshold} km)!")
    
    if metric in ['all', 'velocity_rmse']:
        vel_rmse, vel_mae = compute_velocity_rmse(tracks_df, ground_truth_df)
        if vel_rmse is not None:
            metrics['velocity_rmse_km_s'] = vel_rmse
            metrics['velocity_mae_km_s'] = vel_mae
            click.echo(f"   ✅ Velocity RMSE: {vel_rmse:.4f} km/s ({vel_rmse*1000:.2f} m/s)")
            click.echo(f"   ✅ Velocity MAE: {vel_mae:.4f} km/s ({vel_mae*1000:.2f} m/s)")
    
    if metric in ['all', 'completeness']:
        completeness, completeness_values = compute_track_completeness(tracks_df, ground_truth_df)
        metrics['track_completeness'] = completeness
        click.echo(f"   ✅ Track Completeness: {completeness:.1%}")
        
        if threshold and completeness < threshold:
            click.echo(f"   ⚠️  Track completeness below threshold ({threshold:.1%})!")
    
    if metric in ['all', 'false_tracks']:
        false_rate, false_count, total_tracks = compute_false_track_rate(tracks_df, ground_truth_df)
        metrics['false_track_rate'] = false_rate
        metrics['false_track_count'] = false_count
        metrics['total_tracks'] = total_tracks
        click.echo(f"   ✅ False Track Rate: {false_rate:.1%} ({false_count}/{total_tracks})")
    
    click.echo()
    
    # Save results
    click.echo("💾 Saving results...")
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    click.echo(f"   ✅ Metrics saved to {metrics_path}")
    
    # Generate plots
    if metric in ['all', 'position_rmse']:
        plot_path = output_dir / "position_errors.png"
        plot_position_errors(tracks_df, ground_truth_df, plot_path)
        click.echo(f"   ✅ Position error plot saved to {plot_path}")
    
    if metric in ['all', 'completeness']:
        plot_path = output_dir / "track_completeness.png"
        plot_track_completeness(completeness_values, plot_path)
        click.echo(f"   ✅ Completeness plot saved to {plot_path}")
    
    # Generate report
    report_path = output_dir / "evaluation_report.md"
    with open(report_path, 'w') as f:
        f.write("# Tracking Evaluation Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Tracks File**: `{tracks_path}`\n\n")
        f.write(f"**Ground Truth**: `{ground_truth_path}`\n\n")
        f.write("## Performance Metrics\n\n")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'rate' in key or 'completeness' in key:
                    f.write(f"- **{key}**: {value:.2%}\n")
                else:
                    f.write(f"- **{key}**: {value:.4f}\n")
            else:
                f.write(f"- **{key}**: {value}\n")
        
        f.write("\n## Assessment\n\n")
        
        if 'position_rmse_km' in metrics:
            rmse_m = metrics['position_rmse_km'] * 1000
            if rmse_m < 100:
                f.write(f"✅ **Position Accuracy**: Excellent ({rmse_m:.1f}m RMSE < 100m target)\n\n")
            elif rmse_m < 200:
                f.write(f"⚠️ **Position Accuracy**: Good ({rmse_m:.1f}m RMSE, target: <100m)\n\n")
            else:
                f.write(f"❌ **Position Accuracy**: Needs improvement ({rmse_m:.1f}m RMSE >> 100m target)\n\n")
        
        if 'track_completeness' in metrics:
            comp = metrics['track_completeness']
            if comp > 0.95:
                f.write(f"✅ **Track Completeness**: Excellent ({comp:.1%} > 95% target)\n\n")
            elif comp > 0.85:
                f.write(f"⚠️ **Track Completeness**: Good ({comp:.1%}, target: >95%)\n\n")
            else:
                f.write(f"❌ **Track Completeness**: Needs improvement ({comp:.1%} << 95% target)\n\n")
        
        if 'false_track_rate' in metrics:
            ftr = metrics['false_track_rate']
            if ftr < 0.05:
                f.write(f"✅ **False Track Rate**: Excellent ({ftr:.1%} < 5% target)\n\n")
            elif ftr < 0.10:
                f.write(f"⚠️ **False Track Rate**: Acceptable ({ftr:.1%}, target: <5%)\n\n")
            else:
                f.write(f"❌ **False Track Rate**: Needs improvement ({ftr:.1%} >> 5% target)\n\n")
    
    click.echo(f"   ✅ Report saved to {report_path}")
    click.echo()
    click.echo("✨ Evaluation complete!")
    click.echo()
    click.echo(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
