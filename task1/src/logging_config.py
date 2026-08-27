"""One JSON object per request, written to a log file (and stdout if LOG_STDOUT=1).

Only pipeline decisions are logged. No API keys, no request bodies sent to the
provider, no document content beyond a short preview.
"""
import json
import logging
import os

from . import config

_logger = None


def get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        config.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("rag")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            handlers = [logging.FileHandler(config.LOG_PATH)]
            if os.getenv("LOG_STDOUT") == "1":
                handlers.append(logging.StreamHandler())
            for handler in handlers:
                handler.setFormatter(logging.Formatter("%(message)s"))
                logger.addHandler(handler)
        _logger = logger
    return _logger


def log_request(record: dict) -> None:
    get_logger().info(json.dumps(record, ensure_ascii=False))
