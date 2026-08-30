"""One logger factory so every script prints in the same format."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "aigcdet", level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("aigcdet")
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("aigcdet") else f"aigcdet.{name}")
