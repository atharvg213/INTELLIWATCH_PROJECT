import logging
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configures application-wide logging with both console and optional file handlers.
    """
    resolved_level = log_level or os.getenv("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, resolved_level, logging.INFO)

    log_format = (
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout)
    ]

    target_file = log_file or os.getenv("LOG_FILE")
    if target_file:
        log_path = Path(target_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_path), encoding="utf-8"))

    # Reset any existing handlers to avoid duplicate logs in reload mode
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()

    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    app_logger = logging.getLogger("intelliwatch")
    app_logger.info(f"Logging initialized at level {resolved_level}")
    return app_logger
