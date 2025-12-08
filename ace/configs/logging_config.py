"""
Shared logging configuration for ACE project.

Usage:
    from configs.logging_config import setup_logging
    logger = setup_logging(__name__)
"""
import logging
import os
import sys


class StdoutFilter(logging.Filter):
    """Filter to only allow INFO and DEBUG levels."""
    def filter(self, record):
        return record.levelno <= logging.INFO


def setup_logging(name):
    """
    Configure logging with:
    - No prefix (no 'DEBUG:__main__' etc.)
    - INFO and DEBUG go to stdout
    - WARNING and above go to stderr

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    default_level = 'INFO' if sys.stdout.isatty() else 'WARNING'
    level = os.environ.get('LOGLEVEL', default_level).upper()
    logger.setLevel(level)

    # Formatter with no prefix - just the message
    formatter = logging.Formatter('%(message)s')

    # Handler for INFO and DEBUG -> stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(StdoutFilter())
    stdout_handler.setFormatter(formatter)

    # Handler for WARNING and above -> stderr
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger
