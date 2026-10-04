"""Central logging setup for the training pipeline.

Design: attach handlers ONCE to the parent logger ("mlops"). Child loggers such
as "mlops.data" or "mlops.train" have no handlers of their own; their records
propagate up to the parent, so every module shares the same destinations and
format while still being individually named in the output.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT_LOGGER_NAME = "mlops"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-13s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    log_file: str = "logs/train.log",
) -> logging.Logger:
    """Configure the 'mlops' logger with a console and a rotating file handler."""
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    # The logger passes everything; each handler decides what it actually emits.
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # don't double-print via the root logger

    # Make setup idempotent: calling it twice must not duplicate output lines.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, console_level.upper()))
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, file_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger, e.g. get_logger('data') -> 'mlops.data'."""
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
