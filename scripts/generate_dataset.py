#!/usr/bin/env python3
"""
CLI script for generating simulation datasets.
Provides easy interface to the DatasetGenerator.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
import click

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation.data_generator import DatasetGenerator
from src.utils.config_loader import SimulationConfig
from src.utils.logging_config import get_logger

logger = get_logger("simulation")


@click.command()
@click.option(
    '--objects',
    '-n',
    default=100,
    type=int,
    help='Number of objects to simulate'
)
@click.option(
    '--duration',
    '-d',
    default=24.0,
    type=float,
    help='Simulation duration in hours'
)
@click.option(
    '--timestep',
    '-t',
    default=60.0,
    type=float,
    help='Time step in seconds'
)
@click.option(
    '--output',
    '-o',
    default='data/processed/scenario_001',
    type=click.Path(),
    help='Output directory for dataset'
)
@click.option(
    '--tle-file',
    type=click.Path(exists=True),
    help='Path to TLE file (auto-detects if not specified)'
)
@click.option(
    '--seed',
    type=int,
    help='Random seed for reproducibility'
)
@click.option(
    '--quick',
    is_flag=True,
    help='Quick test (10 objects, 1 hour)'
)
def main(objects, duration, timestep, output, tle_file, seed, quick):
    """
    Generate synthetic space tracking dataset.
    
    Examples:
        # Generate default dataset (100 objects, 24 hours)
        python scripts/generate_dataset.py
        
        # Quick test
        python scripts/generate_dataset.py --quick
        
        # Custom scenario
        python scripts/generate_dataset.py -n 50 -d 12 -o data/processed/test
        
        # Reproducible dataset
        python scripts/generate_dataset.py --seed 42
    """
    # Quick mode overrides
    if quick:
        objects = 10
        duration = 1.0
        output = 'data/processed/quick_test'
        logger.info("Quick mode: 10 objects, 1 hour")
    
    # Create configuration
    config = SimulationConfig(
        num_objects=objects,
        duration_hours=duration,
        time_step_seconds=timestep
    )
    
    # Print configuration
    click.echo("\n" + "="*60)
    click.echo("Dataset Generation Configuration")
    click.echo("="*60)
    click.echo(f"Objects:       {config.num_objects}")
    click.echo(f"Duration:      {config.duration_hours} hours")
    click.echo(f"Time step:     {config.time_step_seconds} seconds")
    click.echo(f"Output:        {output}")
    if seed:
        click.echo(f"Random seed:   {seed}")
    click.echo("="*60 + "\n")
    
    # Confirm for large datasets
    num_steps = int(duration * 3600 / timestep)
    total_points = objects * num_steps
    
    if total_points > 100000 and not quick:
        click.confirm(
            f"This will generate ~{total_points:,} ground truth points. Continue?",
            abort=True
        )
    
    # Create generator
    generator = DatasetGenerator(config)
    
    # Generate dataset
    try:
        click.echo("Generating dataset...")
        dataset = generator.generate(
            tle_file=Path(tle_file) if tle_file else None,
            seed=seed
        )
        
        # Save dataset
        output_path = Path(output)
        dataset.save(output_path)
        
        # Print statistics
        stats = dataset.get_statistics()
        
        click.echo("\n" + "="*60)
        click.echo("Dataset Statistics")
        click.echo("="*60)
        click.echo(f"Objects:              {stats['num_objects']}")
        click.echo(f"Sensors:              {stats['num_sensors']}")
        click.echo(f"Ground truth points:  {stats['num_ground_truth_points']:,}")
        click.echo(f"Measurements:         {stats['num_measurements']:,}")
        click.echo(f"Time span:            {stats['time_span_hours']:.2f} hours")
        click.echo(f"Meas. per object:     {stats['measurements_per_object']:.1f}")
        
        click.echo("\nMeasurements by sensor:")
        for sensor_id, count in stats['measurements_by_sensor'].items():
            click.echo(f"  {sensor_id:20s}: {count:,}")
        
        click.echo("="*60)
        click.echo(f"\n✅ Dataset saved to: {output_path}")
        click.echo("\nNext steps:")
        click.echo(f"  1. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb")
        click.echo(f"  2. Validate: python scripts/validate_simulation.py --dataset {output}")
        click.echo(f"  3. Use for Phase 2 tracking")
        
        return 0
        
    except FileNotFoundError as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        click.echo("\nPlease download TLE data first:", err=True)
        click.echo("  python scripts/download_tle_data.py --categories stations active", err=True)
        return 1
    
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        logger.exception("Dataset generation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
