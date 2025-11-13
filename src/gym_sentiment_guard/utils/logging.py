"""Structured logging helpers."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any


def json_log(message: str, **extra: Any) -> str:
    """Return a JSON-formatted log string."""
    payload = {'ts': time.time(), 'msg': message, **extra}
    return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger following project conventions."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)

    level = logging.DEBUG if os.getenv('GSG_DEBUG') else logging.INFO
    logger.setLevel(level)
    return logger
