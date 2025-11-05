"""Logging utilities using loguru."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    log_dir: str = "logs",
    log_file: str = "app.log",
    level: str = "INFO",
    rotation: str = "100 MB",
    retention: str = "10 days",
    compression: str = "zip",
) -> None:
    """Setup logger with file and console output.

    Args:
        log_dir: Directory for log files
        log_file: Name of log file
        level: Logging level
        rotation: When to rotate log file
        retention: How long to keep old logs
        compression: Compression format for old logs
    """
    # Remove default handler
    logger.remove()

    # Add console handler with custom format
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Add file handler
    log_path = Path(log_dir) / log_file
    logger.add(
        str(log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation=rotation,
        retention=retention,
        compression=compression,
        enqueue=True,  # Thread-safe
    )

    logger.info(f"Logger initialized. Logs will be saved to {log_path}")


def get_logger(name: Optional[str] = None):
    """Get logger instance.

    Args:
        name: Name for the logger (typically __name__)

    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger
