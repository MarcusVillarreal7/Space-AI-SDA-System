"""
Configuration management for the Space AI system.
Loads YAML configs with validation and environment variable support.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, validator


class SimulationConfig(BaseModel):
    """Configuration for simulation parameters."""
    
    num_objects: int = Field(100, ge=1, le=10000, description="Number of objects to simulate")
    duration_hours: float = Field(24.0, gt=0, description="Simulation duration in hours")
    time_step_seconds: float = Field(60.0, gt=0, description="Time step for propagation")
    
    # Sensor configuration
    num_sensors: int = Field(3, ge=1, description="Number of sensors in network")
    sensor_types: List[str] = Field(["radar", "optical"], description="Types of sensors")
    measurement_noise_std_m: float = Field(100.0, gt=0, description="Measurement noise std dev (meters)")
    measurement_cadence_seconds: float = Field(60.0, gt=0, description="Time between measurements")
    
    # Data output
    output_dir: Path = Field(Path("data/processed"), description="Output directory for datasets")
    save_ground_truth: bool = Field(True, description="Save ground truth trajectories")
    
    class Config:
        """Pydantic config."""
        validate_assignment = True


class TrackingConfig(BaseModel):
    """Configuration for tracking engine."""
    
    # Kalman filter parameters
    process_noise_std: float = Field(1.0, gt=0, description="Process noise standard deviation")
    measurement_noise_std: float = Field(100.0, gt=0, description="Measurement noise std dev")
    
    # Track management
    track_init_threshold: int = Field(3, ge=2, description="M in M/N track initiation logic")
    track_init_window: int = Field(5, ge=2, description="N in M/N track initiation logic")
    track_delete_threshold: int = Field(5, ge=1, description="Missed detections before deletion")
    
    # Data association
    association_gate_threshold: float = Field(9.21, gt=0, description="Chi-squared gate threshold (95% for 3D)")
    association_method: str = Field("hungarian", pattern="^(gnn|hungarian)$", description="Association algorithm")
    
    class Config:
        """Pydantic config."""
        validate_assignment = True


class MLConfig(BaseModel):
    """Configuration for ML models."""
    
    # Trajectory predictor
    predictor_hidden_dim: int = Field(256, ge=32, description="Hidden dimension for Transformer")
    predictor_num_layers: int = Field(4, ge=1, description="Number of Transformer layers")
    predictor_num_heads: int = Field(8, ge=1, description="Number of attention heads")
    predictor_dropout: float = Field(0.1, ge=0, le=0.5, description="Dropout rate")
    predictor_sequence_length: int = Field(10, ge=2, description="Input sequence length")
    predictor_prediction_horizon: int = Field(24, ge=1, description="Prediction horizon (hours)")
    
    # Classifier
    classifier_type: str = Field("ensemble", pattern="^(xgboost|neural|ensemble)$", description="Classifier type")
    num_classes: int = Field(4, ge=2, description="Number of object classes")
    class_names: List[str] = Field(
        ["operational_satellite", "debris", "maneuvering_object", "potential_threat"],
        description="Class labels"
    )
    
    # Training
    batch_size: int = Field(32, ge=1, description="Training batch size")
    learning_rate: float = Field(1e-4, gt=0, description="Learning rate")
    num_epochs: int = Field(100, ge=1, description="Number of training epochs")
    early_stopping_patience: int = Field(10, ge=1, description="Early stopping patience")
    
    # Uncertainty
    mc_dropout_samples: int = Field(50, ge=10, description="Monte Carlo dropout samples")
    
    class Config:
        """Pydantic config."""
        validate_assignment = True


class APIConfig(BaseModel):
    """Configuration for API server."""
    
    host: str = Field("0.0.0.0", description="API host")
    port: int = Field(8000, ge=1024, le=65535, description="API port")
    reload: bool = Field(False, description="Auto-reload on code changes")
    workers: int = Field(4, ge=1, description="Number of worker processes")
    
    # Database
    database_url: str = Field("postgresql://localhost/space_ai", description="Database connection URL")
    redis_url: str = Field("redis://localhost:6379", description="Redis connection URL")
    
    # Security
    enable_cors: bool = Field(True, description="Enable CORS")
    api_key_required: bool = Field(False, description="Require API key authentication")
    
    class Config:
        """Pydantic config."""
        validate_assignment = True


class Config:
    """Main configuration manager."""
    
    def __init__(self, config_dir: Path = Path("config")):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir
        self.simulation: Optional[SimulationConfig] = None
        self.tracking: Optional[TrackingConfig] = None
        self.ml: Optional[MLConfig] = None
        self.api: Optional[APIConfig] = None
    
    def load_all(self):
        """Load all configuration files."""
        self.simulation = self.load_config("simulation.yaml", SimulationConfig)
        self.tracking = self.load_config("tracking.yaml", TrackingConfig)
        self.ml = self.load_config("ml_models.yaml", MLConfig)
        self.api = self.load_config("api.yaml", APIConfig)
    
    def load_config(self, filename: str, config_class: type[BaseModel]) -> BaseModel:
        """
        Load and validate a configuration file.
        
        Args:
            filename: Config file name
            config_class: Pydantic model class for validation
        
        Returns:
            Validated configuration object
        
        Example:
            >>> config = Config()
            >>> sim_config = config.load_config("simulation.yaml", SimulationConfig)
            >>> print(f"Simulating {sim_config.num_objects} objects")
        """
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            # Return default configuration
            return config_class()
        
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            return config_class()
        
        return config_class(**config_dict)
    
    def save_config(self, config: BaseModel, filename: str):
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration object to save
            filename: Output filename
        """
        filepath = self.config_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(config.dict(), f, default_flow_style=False, sort_keys=False)
    
    def create_default_configs(self):
        """Create default configuration files if they don't exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        configs = [
            ("simulation.yaml", SimulationConfig()),
            ("tracking.yaml", TrackingConfig()),
            ("ml_models.yaml", MLConfig()),
            ("api.yaml", APIConfig()),
        ]
        
        for filename, config in configs:
            filepath = self.config_dir / filename
            if not filepath.exists():
                self.save_config(config, filename)


# Example usage:
if __name__ == "__main__":
    # Create default configs
    config_manager = Config()
    config_manager.create_default_configs()
    
    # Load all configs
    config_manager.load_all()
    
    print(f"Simulation: {config_manager.simulation.num_objects} objects")
    print(f"Tracking: {config_manager.tracking.association_method} association")
    print(f"ML: {config_manager.ml.predictor_num_layers} Transformer layers")
