"""
Logging setup for the retrieval engine.

We configure a single root logger with a consistent format so every stage
(ingest -> chunk -> embed -> index -> search) emits traceable, timestamped
lines. Call ``configure_logging()`` once at application startup.
"""

import logging

from .config import get_settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def configure_logging() -> None:
    """Idempotently configure the root logger from settings.log_level."""
    global _configured
    if _configured:
        return

    level = getattr(logging, get_settings().log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format=_LOG_FORMAT)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, ensuring logging is configured first."""
    configure_logging()
    return logging.getLogger(name)
