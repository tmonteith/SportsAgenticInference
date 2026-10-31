"""Centralized logging configuration for SportsAgenticInference."""

from __future__ import annotations

import logging
import sys
from typing import Optional

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: Optional[str] = "INFO") -> logging.Logger:
    """Configure sportsagentic.inference loggers with consistent formatting."""
    resolved_level_name = (level or "INFO").upper()
    numeric_level = getattr(logging, resolved_level_name, logging.INFO)

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    app_logger = logging.getLogger("sportsagentic.inference")
    app_logger.setLevel(numeric_level)

    if not app_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        app_logger.addHandler(handler)
    else:
        for handler in app_logger.handlers:
            handler.setFormatter(formatter)

    app_logger.propagate = True
    return app_logger


def get_logger(component: str) -> logging.Logger:
    """Return a child logger under the 'sportsagentic.inference' namespace."""
    if component.startswith("sportsagentic.inference"):
        return logging.getLogger(component)
    if component.startswith("sportsagentic."):
        return logging.getLogger(component)
    return logging.getLogger(f"sportsagentic.inference.{component}")
