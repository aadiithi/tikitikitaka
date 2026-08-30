"""Seeding, so that every number in the report is reproducible."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 1337, deterministic: bool = True) -> int:
    """Seed python, numpy and torch.

    Returns the seed so callers can log it. We seed PYTHONHASHSEED too because
    dataset shuffling that depends on set/dict iteration order is a classic
    source of "why did my numbers move by 0.4%" confusion.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # cudnn.benchmark picks different kernels run-to-run, which changes
            # float summation order and therefore the last decimal of our AUC.
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:  # torch is optional for pure-metric usage
        pass
    return seed
