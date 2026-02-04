"""
Noise models for sensor measurements.
Provides realistic measurement uncertainty and systematic biases.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger("simulation")


@dataclass
class NoiseParameters:
    """Parameters for noise generation."""
    
    std_dev: float  # Standard deviation (km)
    systematic_bias: Optional[np.ndarray] = None  # Constant bias vector (km)
    correlation_time: Optional[float] = None  # Temporal correlation (seconds)
    seed: Optional[int] = None  # Random seed for reproducibility
    
    def __post_init__(self):
        """Initialize random state if seed provided."""
        if self.seed is not None:
            np.random.seed(self.seed)


class GaussianNoise:
    """
    Gaussian (normal) noise model.
    
    Most common noise model for sensor measurements.
    Assumes independent, identically distributed noise.
    """
    
    def __init__(self, std_dev: float, seed: Optional[int] = None):
        """
        Initialize Gaussian noise model.
        
        Args:
            std_dev: Standard deviation in km
            seed: Random seed for reproducibility
        
        Example:
            >>> noise_model = GaussianNoise(std_dev=0.05)  # 50m std dev
            >>> position = np.array([7000.0, 0.0, 0.0])
            >>> noisy_position = noise_model.add_noise(position)
        """
        self.std_dev = std_dev
        self.rng = np.random.default_rng(seed)
        logger.debug(f"Initialized GaussianNoise with std_dev={std_dev*1000:.1f}m")
    
    def add_noise(self, measurement: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to measurement.
        
        Args:
            measurement: Clean measurement vector
        
        Returns:
            Noisy measurement
        """
        noise = self.rng.normal(0, self.std_dev, size=measurement.shape)
        return measurement + noise
    
    def get_covariance_matrix(self, dim: int = 3) -> np.ndarray:
        """
        Get covariance matrix for this noise model.
        
        Args:
            dim: Dimension of measurement space
        
        Returns:
            Covariance matrix (dim x dim)
        """
        return np.eye(dim) * (self.std_dev ** 2)


class SystematicBias:
    """
    Systematic (constant) bias in measurements.
    
    Represents sensor calibration errors or systematic offsets.
    """
    
    def __init__(self, bias_vector: np.ndarray):
        """
        Initialize systematic bias model.
        
        Args:
            bias_vector: Constant bias to add (km)
        
        Example:
            >>> # 10m bias in X direction
            >>> bias_model = SystematicBias(np.array([0.01, 0.0, 0.0]))
            >>> position = np.array([7000.0, 0.0, 0.0])
            >>> biased_position = bias_model.add_bias(position)
        """
        self.bias = np.asarray(bias_vector)
        logger.debug(f"Initialized SystematicBias: {self.bias*1000} m")
    
    def add_bias(self, measurement: np.ndarray) -> np.ndarray:
        """
        Add systematic bias to measurement.
        
        Args:
            measurement: Clean measurement
        
        Returns:
            Biased measurement
        """
        return measurement + self.bias


class CorrelatedNoise:
    """
    Temporally correlated noise model.
    
    Noise that persists over time (e.g., atmospheric effects).
    Uses first-order Markov process.
    """
    
    def __init__(
        self,
        std_dev: float,
        correlation_time: float,
        time_step: float,
        seed: Optional[int] = None
    ):
        """
        Initialize correlated noise model.
        
        Args:
            std_dev: Standard deviation (km)
            correlation_time: Correlation time constant (seconds)
            time_step: Time between measurements (seconds)
            seed: Random seed
        
        Example:
            >>> # Noise with 60-second correlation time
            >>> noise_model = CorrelatedNoise(
            ...     std_dev=0.05,
            ...     correlation_time=60.0,
            ...     time_step=1.0
            ... )
        """
        self.std_dev = std_dev
        self.correlation_time = correlation_time
        self.time_step = time_step
        self.rng = np.random.default_rng(seed)
        
        # Calculate correlation coefficient
        self.alpha = np.exp(-time_step / correlation_time)
        
        # Current noise state
        self.current_noise = np.zeros(3)
        
        logger.debug(
            f"Initialized CorrelatedNoise: std_dev={std_dev*1000:.1f}m, "
            f"correlation_time={correlation_time:.1f}s, alpha={self.alpha:.3f}"
        )
    
    def add_noise(self, measurement: np.ndarray) -> np.ndarray:
        """
        Add correlated noise to measurement.
        
        Uses first-order Markov process:
        noise(t) = alpha * noise(t-1) + sqrt(1 - alpha^2) * white_noise
        
        Args:
            measurement: Clean measurement
        
        Returns:
            Noisy measurement
        """
        # Generate white noise
        white_noise = self.rng.normal(0, self.std_dev, size=3)
        
        # Update correlated noise
        self.current_noise = (
            self.alpha * self.current_noise +
            np.sqrt(1 - self.alpha**2) * white_noise
        )
        
        return measurement + self.current_noise
    
    def reset(self):
        """Reset noise state (e.g., for new simulation run)."""
        self.current_noise = np.zeros(3)


class CompositeNoiseModel:
    """
    Composite noise model combining multiple noise sources.
    
    Allows combining Gaussian noise, systematic bias, and correlated noise.
    """
    
    def __init__(
        self,
        gaussian_std: float = 0.0,
        systematic_bias: Optional[np.ndarray] = None,
        correlated_std: float = 0.0,
        correlation_time: float = 60.0,
        time_step: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize composite noise model.
        
        Args:
            gaussian_std: Gaussian noise std dev (km)
            systematic_bias: Constant bias vector (km)
            correlated_std: Correlated noise std dev (km)
            correlation_time: Correlation time (seconds)
            time_step: Time between measurements (seconds)
            seed: Random seed
        
        Example:
            >>> noise_model = CompositeNoiseModel(
            ...     gaussian_std=0.03,  # 30m white noise
            ...     systematic_bias=np.array([0.01, 0.0, 0.0]),  # 10m X bias
            ...     correlated_std=0.02,  # 20m correlated noise
            ...     correlation_time=60.0
            ... )
        """
        self.components = []
        
        # Add Gaussian noise if specified
        if gaussian_std > 0:
            self.components.append(
                ('gaussian', GaussianNoise(gaussian_std, seed))
            )
        
        # Add systematic bias if specified
        if systematic_bias is not None:
            self.components.append(
                ('bias', SystematicBias(systematic_bias))
            )
        
        # Add correlated noise if specified
        if correlated_std > 0:
            self.components.append(
                ('correlated', CorrelatedNoise(
                    correlated_std, correlation_time, time_step, seed
                ))
            )
        
        logger.info(
            f"Initialized CompositeNoiseModel with {len(self.components)} components"
        )
    
    def add_noise(self, measurement: np.ndarray) -> np.ndarray:
        """
        Add all noise components to measurement.
        
        Args:
            measurement: Clean measurement
        
        Returns:
            Noisy measurement with all components applied
        """
        noisy_measurement = measurement.copy()
        
        for name, component in self.components:
            if name == 'gaussian':
                noisy_measurement = component.add_noise(noisy_measurement)
            elif name == 'bias':
                noisy_measurement = component.add_bias(noisy_measurement)
            elif name == 'correlated':
                noisy_measurement = component.add_noise(noisy_measurement)
        
        return noisy_measurement
    
    def get_total_covariance(self, dim: int = 3) -> np.ndarray:
        """
        Get total covariance matrix (Gaussian + correlated components).
        
        Note: Systematic bias doesn't contribute to covariance.
        
        Args:
            dim: Dimension of measurement space
        
        Returns:
            Total covariance matrix
        """
        total_variance = 0.0
        
        for name, component in self.components:
            if name == 'gaussian':
                total_variance += component.std_dev ** 2
            elif name == 'correlated':
                total_variance += component.std_dev ** 2
        
        return np.eye(dim) * total_variance
    
    def reset(self):
        """Reset all stateful components (e.g., correlated noise)."""
        for name, component in self.components:
            if hasattr(component, 'reset'):
                component.reset()


# Example usage and validation
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Testing noise models...")
    
    # Test 1: Gaussian noise statistics
    print("\n1. Gaussian Noise Test")
    gaussian = GaussianNoise(std_dev=0.05, seed=42)  # 50m std dev
    
    # Generate 1000 samples
    samples = np.array([gaussian.add_noise(np.zeros(3)) for _ in range(1000)])
    
    print(f"  Target std dev: 50.0 m")
    print(f"  Measured std dev: {np.std(samples) * 1000:.1f} m")
    print(f"  Mean: {np.mean(samples) * 1000:.1f} m (should be ~0)")
    
    # Test 2: Systematic bias
    print("\n2. Systematic Bias Test")
    bias = SystematicBias(np.array([0.01, 0.0, 0.0]))  # 10m in X
    
    position = np.array([7000.0, 0.0, 0.0])
    biased = bias.add_bias(position)
    
    print(f"  Original: {position}")
    print(f"  Biased: {biased}")
    print(f"  Difference: {(biased - position) * 1000} m")
    
    # Test 3: Correlated noise
    print("\n3. Correlated Noise Test")
    correlated = CorrelatedNoise(
        std_dev=0.05,
        correlation_time=60.0,
        time_step=1.0,
        seed=42
    )
    
    # Generate time series
    n_steps = 200
    noise_series = np.zeros((n_steps, 3))
    
    for i in range(n_steps):
        noise_series[i] = correlated.add_noise(np.zeros(3))
    
    # Calculate autocorrelation
    autocorr = np.correlate(noise_series[:, 0], noise_series[:, 0], mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr /= autocorr[0]
    
    print(f"  Autocorrelation at lag 0: {autocorr[0]:.3f}")
    print(f"  Autocorrelation at lag 60: {autocorr[60]:.3f}")
    print(f"  Expected at lag 60: {np.exp(-1):.3f}")
    
    # Test 4: Composite model
    print("\n4. Composite Noise Model Test")
    composite = CompositeNoiseModel(
        gaussian_std=0.03,  # 30m
        systematic_bias=np.array([0.01, 0.0, 0.0]),  # 10m
        correlated_std=0.02,  # 20m
        correlation_time=60.0,
        time_step=1.0,
        seed=42
    )
    
    # Generate samples
    samples = np.array([composite.add_noise(np.zeros(3)) for _ in range(1000)])
    
    print(f"  Mean (should be ~10m in X): {np.mean(samples, axis=0) * 1000} m")
    print(f"  Std dev (should be ~36m): {np.std(samples) * 1000:.1f} m")
    print(f"  Expected std: {np.sqrt(0.03**2 + 0.02**2) * 1000:.1f} m")
    
    print("\n✅ All noise model tests complete!")
