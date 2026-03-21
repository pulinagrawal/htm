"""Logging configuration for the HTM Reinforcement Learning project.

This module sets up a centralized logger with formatted output to stdout.
The logging level can be controlled via the DEBUG environment variable.

Attributes:
    logger: The configured logger instance for the entire project. Use this
        logger throughout the codebase by importing it: `from psu_capstone.log import logger`.
"""

import logging
import os
import sys
from typing import Any

logger = logging.getLogger("htmrl")
_handler = logging.StreamHandler(stream=sys.stdout)
_formatter = logging.Formatter(fmt="%(levelname)s [%(name)s]: %(message)s")
_handler.setFormatter(_formatter)
logger.addHandler(_handler)

if os.environ.get("DEBUG"):
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


def get_logger(source: Any | None = None) -> logging.Logger:
    """Return the shared logger or a child logger for a specific source.

    Creates child loggers with hierarchical names based on the source, which
    helps identify the origin of log messages in the output.

    Args:
        source: Source identifier - can be None (returns root logger), a string name,
            a class type, or a class instance. Class types and instances generate
            child loggers named after the class.

    Returns:
        The root logger or a child logger with an appropriate name suffix.
    """
    if source is None:
        return logger

    if isinstance(source, str):
        suffix = source
    elif isinstance(source, type):
        suffix = source.__name__
    else:
        suffix = source.__class__.__name__

    return logger.getChild(suffix)