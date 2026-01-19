"""
Create logger named agentsociety as singleton for ray.
"""

import logging
import sys

__all__ = ["get_logger", "set_logger_level"]


def get_logger(name: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 清掉所有旧 handler
    logger.handlers.clear()

    # 加新的 handler
    handler = logging.StreamHandler(sys.stdout)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def set_logger_level(logger:logging.Logger, level: str):
    """Set the logger level"""
    logger.setLevel(level)
