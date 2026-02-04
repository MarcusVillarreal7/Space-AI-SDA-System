"""
Performance metrics and evaluation utilities.
"""

import time
from contextlib import contextmanager
from typing import Dict, Any, Optional
import numpy as np


class PerformanceMetrics:
    """Track and report performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {}
    
    def record(self, metric_name: str, value: float):
        """
        Record a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """
        Get statistics for a metric.
        
        Args:
            metric_name: Name of the metric
        
        Returns:
            Dictionary with mean, std, min, max, median
        """
        if metric_name not in self.metrics:
            return {}
        
        values = np.array(self.metrics[metric_name])
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "count": len(values)
        }
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all metrics."""
        return {name: self.get_stats(name) for name in self.metrics.keys()}
    
    def reset(self):
        """Clear all metrics."""
        self.metrics.clear()


@contextmanager
def timer(metric_name: str, metrics: Optional[PerformanceMetrics] = None):
    """
    Context manager for timing code blocks.
    
    Args:
        metric_name: Name for the timing metric
        metrics: Optional PerformanceMetrics instance to record to
    
    Example:
        >>> metrics = PerformanceMetrics()
        >>> with timer("data_processing", metrics):
        ...     # Your code here
        ...     pass
        >>> print(metrics.get_stats("data_processing"))
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if metrics:
            metrics.record(metric_name, elapsed)


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
    
    Returns:
        RMSE value
    """
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
    
    Returns:
        MAE value
    """
    return float(np.mean(np.abs(predictions - targets)))


def position_error(pred_pos: np.ndarray, true_pos: np.ndarray) -> float:
    """
    Calculate 3D position error magnitude.
    
    Args:
        pred_pos: Predicted position [x, y, z]
        true_pos: True position [x, y, z]
    
    Returns:
        Position error in same units as input
    """
    return float(np.linalg.norm(pred_pos - true_pos))
