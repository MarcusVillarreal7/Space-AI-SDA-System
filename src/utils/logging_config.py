"""
Structured logging configuration for the Space AI system.
Provides JSON-formatted logs with context for debugging and auditing.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger


class LogConfig:
    """Centralized logging configuration."""
    
    LOG_DIR = Path("data/logs")
    LOG_FORMAT = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    @classmethod
    def setup(cls, log_level: str = "INFO", enable_json: bool = True):
        """
        Set up logging for the entire application.
        
        Args:
            log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_json: Whether to enable JSON logging to files
        """
        # Create log directory
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Remove default logger
        logger.remove()
        
        # Add console handler with colors
        logger.add(
            sys.stderr,
            format=cls.LOG_FORMAT,
            level=log_level,
            colorize=True,
        )
        
        if enable_json:
            # Add JSON file handlers for each component
            components = [
                "simulation",
                "tracking",
                "ml_predictions",
                "alerts",
                "performance",
                "validation",
            ]
            
            for component in components:
                logger.add(
                    cls.LOG_DIR / f"{component}.jsonl",
                    format="{message}",
                    level="INFO",
                    rotation="1 day",
                    retention="30 days",
                    compression="zip",
                    serialize=True,  # JSON format
                    filter=lambda record, comp=component: record["extra"].get("component") == comp,
                )
        
        # Add general application log
        logger.add(
            cls.LOG_DIR / "application.log",
            format=cls.LOG_FORMAT,
            level=log_level,
            rotation="500 MB",
            retention="7 days",
            compression="zip",
        )
        
        logger.info(f"Logging initialized at level {log_level}")


def get_logger(component: str):
    """
    Get a logger for a specific component.
    
    Args:
        component: Component name (e.g., 'simulation', 'tracking')
    
    Returns:
        Configured logger instance
    
    Example:
        >>> from src.utils.logging_config import get_logger
        >>> logger = get_logger("simulation")
        >>> logger.info("Starting simulation", object_count=100, duration_hours=24)
    """
    return logger.bind(component=component)


# Initialize logging on module import with default settings
# Can be reconfigured by calling LogConfig.setup() explicitly
try:
    LogConfig.setup(log_level="INFO", enable_json=True)
except Exception as e:
    # Fallback to basic logging if setup fails
    logging.basicConfig(level=logging.INFO)
    logging.warning(f"Failed to initialize advanced logging: {e}")
