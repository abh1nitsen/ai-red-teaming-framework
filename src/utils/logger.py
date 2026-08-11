"""
src/utils/logger.py
===================
Centralised logging configuration for the LLM Safety Framework.

Features:
    - Structured log format with timestamps, module name, and log level
    - Logs to both console (INFO+) and a rotating file (DEBUG+)
    - Single call: get_logger(__name__) in every module
    - Log file written to: logs/safety_framework.log (auto-created)

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Pipeline started")
    logger.warning("Groq API slow response")
    logger.error("Evaluator failed: %s", str(e))
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "safety_framework.log")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_LOG_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB per log file
BACKUP_COUNT = 3                       # Keep 3 rotated files


# ─────────────────────────────────────────────────────────────────────────────
# Logger Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Retrieve (or create) a named logger with console and file handlers.

    Call this once at the top of each module:
        logger = get_logger(__name__)

    Args:
        name:  Module name, typically __name__.
        level: Optional override for log level (e.g. logging.DEBUG).
               Defaults to INFO for console, DEBUG for file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # Capture everything; handlers filter

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── Console Handler (INFO and above) ──────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level or logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File Handler (DEBUG and above, rotating) ──────────────────────────────
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=MAX_LOG_SIZE_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (OSError, PermissionError) as e:
        # Log directory not writable (e.g. read-only filesystem) - console only
        logger.warning("Could not create log file at %s: %s. Console logging only.", LOG_FILE, e)

    # Prevent log records from bubbling up to the root logger
    logger.propagate = False

    return logger
