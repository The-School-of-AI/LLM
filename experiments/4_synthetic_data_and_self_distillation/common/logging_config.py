"""
logging_config.py — Centralized Logging for the Synthetic Data Pipeline

Features:
  • Dual output: console (INFO) + file (DEBUG)
  • Log files named by command + datetime: logs/generate-bank_20260208_150000.log
  • Configurable via --log-level CLI arg or LOG_LEVEL env var
  • Rich format with timestamps, level, module name
  • All pipeline modules use the same logger hierarchy: synth_pipeline.*

Usage:
    from common.logging_config import setup_logging, get_logger

    # In run_pipeline.py (once, at startup):
    setup_logging(command="generate-bank", log_level="DEBUG")

    # In any module:
    logger = get_logger(__name__)
    logger.info("Starting generation for skill %s", skill_id)
    logger.debug("Prompt template: %s", prompt[:100])
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Root logger name for the entire pipeline
ROOT_LOGGER_NAME = "synth_pipeline"

# Default log directory (relative to the pipeline root)
_PIPELINE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = _PIPELINE_ROOT / "logs"


class ColorFormatter(logging.Formatter):
    """Colored console formatter for readability."""

    COLORS = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        # Shorten the module name for console readability
        short_name = record.name.replace("synth_pipeline.", "")
        record.short_name = short_name
        msg = super().format(record)
        return f"{color}{msg}{self.RESET}"


def setup_logging(
    command: str = "pipeline",
    log_level: str | None = None,
    log_dir: str | Path | None = None,
    console_level: str | None = None,
) -> Path:
    """Initialize logging for a pipeline run.

    Args:
        command: The pipeline command being run (used in log filename).
        log_level: File log level (default: DEBUG). Also settable via LOG_LEVEL env var.
        log_dir: Directory for log files (default: <pipeline_root>/logs/).
        console_level: Console log level (default: INFO). Overrides log_level for console only.

    Returns:
        Path to the log file created.
    """
    # Resolve log level
    if log_level is None:
        log_level = os.environ.get("LOG_LEVEL", "DEBUG").upper()
    else:
        log_level = log_level.upper()

    if console_level is None:
        console_level = os.environ.get("CONSOLE_LOG_LEVEL", "INFO").upper()
    else:
        console_level = console_level.upper()

    # Resolve log directory
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build log filename: <command>_<YYYYMMDD_HHMMSS>.log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{command}_{timestamp}.log"
    log_path = log_dir / log_filename

    # Get the root pipeline logger
    root_logger = logging.getLogger(ROOT_LOGGER_NAME)
    root_logger.setLevel(logging.DEBUG)  # Capture everything; handlers filter

    # Remove any existing handlers (in case setup_logging is called multiple times)
    root_logger.handlers.clear()

    # ── File Handler (DEBUG — captures everything) ──────────────
    file_fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(getattr(logging, log_level, logging.DEBUG))
    file_handler.setFormatter(file_fmt)
    root_logger.addHandler(file_handler)

    # ── Console Handler (INFO — user-facing) ────────────────────
    console_fmt = ColorFormatter(
        fmt="[%(asctime)s] [%(levelname)-8s] [%(short_name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level, logging.INFO))
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # Log the session start
    root_logger.info("=" * 60)
    root_logger.info("  PIPELINE LOG STARTED")
    root_logger.info("  Command:    %s", command)
    root_logger.info("  Log file:   %s", log_path)
    root_logger.info("  File level: %s | Console level: %s", log_level, console_level)
    root_logger.info("  Timestamp:  %s", datetime.now().isoformat())
    root_logger.info("=" * 60)

    return log_path


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a pipeline module.

    Args:
        name: Module name (typically __name__). Will be prefixed with ROOT_LOGGER_NAME
              if not already.

    Returns:
        logging.Logger instance.

    Example:
        logger = get_logger(__name__)
        # If __name__ == "generation.dual_view_generator"
        # → logger name = "synth_pipeline.generation.dual_view_generator"
    """
    if name.startswith(ROOT_LOGGER_NAME):
        return logging.getLogger(name)

    # Strip common path prefixes for cleaner names
    for prefix in ("experiments.4_synthetic_data_and_self_distillation.", ""):
        if name.startswith(prefix) and prefix:
            name = name[len(prefix):]
            break

    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")

