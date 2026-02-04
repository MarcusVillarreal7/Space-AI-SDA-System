"""
Data generation pipeline for space tracking simulation.
Orchestrates TLE loading, propagation, sensor measurements, and dataset creation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.simulation.tle_loader import TLELoader, TLE
from src.simulation.orbital_mechanics import SGP4Propagator, StateVector
from src.simulation.sensor_models import BaseSensor, create_sensor_network, Measurement
from src.utils.config_loader import SimulationConfig
from src.utils.logging_config import get_logger

logger = get_logger("simulation")


@dataclass
class Dataset:
    """
    Complete simulation dataset with ground truth and measurements.
    """
    
    ground_truth: pd.DataFrame  # True object states
    measurements: pd.DataFrame  # Sensor observations
    metadata: Dict  # Scenario information
    objects: List[TLE] = field(default_factory=list)  # TLE data
    sensors: List[BaseSensor] = field(default_factory=list)  # Sensor network
    
    def save(self, output_dir: Path):
        """
        Save dataset to disk.
        
        Args:
            output_dir: Directory to save dataset
        
        Creates:
            - ground_truth.parquet: True object states
            - measurements.parquet: Sensor observations
            - metadata.json: Scenario information
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save DataFrames as Parquet (efficient for large datasets)
        self.ground_truth.to_parquet(output_dir / "ground_truth.parquet", index=False)
        self.measurements.to_parquet(output_dir / "measurements.parquet", index=False)
        
        # Save metadata as JSON
        import json
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        logger.info(f"Dataset saved to {output_dir}")
        logger.info(f"  Ground truth points: {len(self.ground_truth)}")
        logger.info(f"  Measurements: {len(self.measurements)}")
    
    @classmethod
    def load(cls, input_dir: Path) -> "Dataset":
        """
        Load dataset from disk.
        
        Args:
            input_dir: Directory containing dataset
        
        Returns:
            Dataset object
        """
        input_dir = Path(input_dir)
        
        # Load DataFrames
        ground_truth = pd.read_parquet(input_dir / "ground_truth.parquet")
        measurements = pd.read_parquet(input_dir / "measurements.parquet")
        
        # Load metadata
        import json
        with open(input_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Dataset loaded from {input_dir}")
        
        return cls(
            ground_truth=ground_truth,
            measurements=measurements,
            metadata=metadata
        )
    
    def get_statistics(self) -> Dict:
        """
        Calculate dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'num_objects': self.ground_truth['object_id'].nunique(),
            'num_sensors': self.measurements['sensor_id'].nunique(),
            'num_ground_truth_points': len(self.ground_truth),
            'num_measurements': len(self.measurements),
            'time_span_hours': (
                pd.to_datetime(self.ground_truth['time'].max()) -
                pd.to_datetime(self.ground_truth['time'].min())
            ).total_seconds() / 3600,
            'measurements_per_object': len(self.measurements) / self.ground_truth['object_id'].nunique(),
        }
        
        # Measurement statistics by sensor
        sensor_stats = self.measurements.groupby('sensor_id').size().to_dict()
        stats['measurements_by_sensor'] = sensor_stats
        
        return stats


class DatasetGenerator:
    """
    Generate synthetic space tracking datasets.
    
    Orchestrates the complete simulation pipeline:
    1. Load TLE data
    2. Create sensor network
    3. Propagate satellites
    4. Generate measurements
    5. Create dataset
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize dataset generator.
        
        Args:
            config: Simulation configuration (uses defaults if None)
        
        Example:
            >>> from src.utils.config_loader import SimulationConfig
            >>> config = SimulationConfig(num_objects=50, duration_hours=12)
            >>> generator = DatasetGenerator(config)
            >>> dataset = generator.generate()
        """
        self.config = config or SimulationConfig()
        logger.info("Initialized DatasetGenerator")
        logger.info(f"  Objects: {self.config.num_objects}")
        logger.info(f"  Duration: {self.config.duration_hours} hours")
        logger.info(f"  Time step: {self.config.time_step_seconds} seconds")
    
    def load_objects(self, tle_file: Optional[Path] = None) -> List[TLE]:
        """
        Load TLE data for objects to track.
        
        Args:
            tle_file: Path to TLE file (uses config if None)
        
        Returns:
            List of TLE objects
        """
        loader = TLELoader()
        
        if tle_file is None:
            # Try to find TLE file in data/raw
            tle_files = list(Path("data/raw").glob("*.tle"))
            if not tle_files:
                logger.error("No TLE files found in data/raw/")
                raise FileNotFoundError(
                    "No TLE files found. Run: python scripts/download_tle_data.py"
                )
            tle_file = tle_files[0]
            logger.info(f"Using TLE file: {tle_file}")
        
        # Load TLEs
        tles = loader.load_from_file(tle_file)
        
        # Limit to configured number of objects
        if len(tles) > self.config.num_objects:
            tles = tles[:self.config.num_objects]
            logger.info(f"Limited to {self.config.num_objects} objects")
        
        return tles
    
    def create_sensor_network(self) -> List[BaseSensor]:
        """
        Create sensor network from configuration.
        
        Returns:
            List of sensors
        """
        # Default sensor network if not in config
        default_sensors = [
            {
                'name': 'Radar-CONUS-1',
                'type': 'radar',
                'location': [40.0, -105.0, 1.5],  # Colorado
                'max_range_km': 3000,
                'accuracy_m': 50,
                'fov_deg': 120
            },
            {
                'name': 'Radar-CONUS-2',
                'type': 'radar',
                'location': [35.0, -120.0, 0.5],  # California
                'max_range_km': 3000,
                'accuracy_m': 50,
                'fov_deg': 120
            },
            {
                'name': 'Optical-Hawaii',
                'type': 'optical',
                'location': [19.8, -155.5, 4.2],  # Mauna Kea
                'max_range_km': 40000,
                'accuracy_m': 500,
                'fov_deg': 30
            }
        ]
        
        sensors = create_sensor_network(default_sensors)
        return sensors
    
    def generate(
        self,
        tle_file: Optional[Path] = None,
        start_time: Optional[datetime] = None,
        seed: Optional[int] = None
    ) -> Dataset:
        """
        Generate complete simulation dataset.
        
        Args:
            tle_file: Path to TLE file (optional)
            start_time: Simulation start time (uses current time if None)
            seed: Random seed for reproducibility
        
        Returns:
            Dataset object with ground truth and measurements
        
        Example:
            >>> generator = DatasetGenerator()
            >>> dataset = generator.generate(seed=42)
            >>> print(f"Generated {len(dataset.measurements)} measurements")
        """
        if seed is not None:
            np.random.seed(seed)
            logger.info(f"Using random seed: {seed}")
        
        # 1. Load objects
        logger.info("Step 1/5: Loading TLE data...")
        tles = self.load_objects(tle_file)
        logger.info(f"Loaded {len(tles)} objects")
        
        # 2. Create propagators
        logger.info("Step 2/5: Creating propagators...")
        propagators = [SGP4Propagator(tle) for tle in tles]
        
        # 3. Create sensor network
        logger.info("Step 3/5: Setting up sensor network...")
        sensors = self.create_sensor_network()
        logger.info(f"Created {len(sensors)} sensors")
        
        # 4. Generate time steps
        if start_time is None:
            start_time = datetime.now(timezone.utc)
        
        num_steps = int(self.config.duration_hours * 3600 / self.config.time_step_seconds)
        time_steps = [
            start_time + timedelta(seconds=i * self.config.time_step_seconds)
            for i in range(num_steps)
        ]
        
        logger.info(f"Generating {num_steps} time steps...")
        
        # 5. Propagate and measure
        logger.info("Step 4/5: Propagating objects and generating measurements...")
        ground_truth_data = []
        measurement_data = []
        
        for time in tqdm(time_steps, desc="Simulating"):
            # Propagate all objects
            for obj_id, prop in enumerate(propagators):
                try:
                    state = prop.propagate(time)
                    
                    # Store ground truth
                    gt_dict = state.to_dict()
                    gt_dict['object_id'] = obj_id
                    gt_dict['object_name'] = tles[obj_id].name
                    ground_truth_data.append(gt_dict)
                    
                    # Generate measurements from visible sensors
                    for sensor in sensors:
                        if sensor.can_observe(state.position, time):
                            measurement = sensor.measure(
                                state.position,
                                object_id=obj_id,
                                time=time,
                                add_noise=True
                            )
                            
                            meas_dict = measurement.to_dict()
                            meas_dict['object_name'] = tles[obj_id].name
                            measurement_data.append(meas_dict)
                
                except Exception as e:
                    logger.warning(f"Failed to propagate object {obj_id} at {time}: {e}")
                    continue
        
        # 6. Create dataset
        logger.info("Step 5/5: Creating dataset...")
        
        ground_truth_df = pd.DataFrame(ground_truth_data)
        measurements_df = pd.DataFrame(measurement_data)
        
        # Convert time strings to datetime
        ground_truth_df['time'] = pd.to_datetime(ground_truth_df['time'])
        measurements_df['time'] = pd.to_datetime(measurements_df['time'])
        
        metadata = {
            'num_objects': len(tles),
            'num_sensors': len(sensors),
            'duration_hours': self.config.duration_hours,
            'time_step_seconds': self.config.time_step_seconds,
            'start_time': start_time.isoformat(),
            'end_time': time_steps[-1].isoformat(),
            'num_time_steps': num_steps,
            'seed': seed,
            'generated_at': datetime.now(timezone.utc).isoformat(),
        }
        
        dataset = Dataset(
            ground_truth=ground_truth_df,
            measurements=measurements_df,
            metadata=metadata,
            objects=tles,
            sensors=sensors
        )
        
        # Log statistics
        stats = dataset.get_statistics()
        logger.info("Dataset generation complete!")
        logger.info(f"  Ground truth points: {stats['num_ground_truth_points']}")
        logger.info(f"  Measurements: {stats['num_measurements']}")
        logger.info(f"  Measurements per object: {stats['measurements_per_object']:.1f}")
        logger.info(f"  Measurements by sensor:")
        for sensor_id, count in stats['measurements_by_sensor'].items():
            logger.info(f"    {sensor_id}: {count}")
        
        return dataset


# Example usage
if __name__ == "__main__":
    from src.utils.config_loader import SimulationConfig
    
    # Create configuration
    config = SimulationConfig(
        num_objects=10,  # Small test
        duration_hours=1.0,  # 1 hour
        time_step_seconds=60.0  # 1 minute steps
    )
    
    # Generate dataset
    generator = DatasetGenerator(config)
    
    try:
        dataset = generator.generate(seed=42)
        
        # Print statistics
        stats = dataset.get_statistics()
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Save dataset
        output_dir = Path("data/processed/test_scenario")
        dataset.save(output_dir)
        print(f"\nDataset saved to: {output_dir}")
        
        # Test loading
        loaded_dataset = Dataset.load(output_dir)
        print(f"Dataset loaded successfully!")
        print(f"  Ground truth shape: {loaded_dataset.ground_truth.shape}")
        print(f"  Measurements shape: {loaded_dataset.measurements.shape}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo generate a dataset, first download TLE data:")
        print("  python scripts/download_tle_data.py --categories stations")
