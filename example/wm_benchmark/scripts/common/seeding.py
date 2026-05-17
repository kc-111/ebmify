"""Seed torch / numpy / Python's random for reproducible runs.

`set_seed` is the single entrypoint; everything else is convenience for
callers that want a deterministic generator handed back to them.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = False) -> torch.Generator:
    """Seed the global RNGs and return a fresh torch.Generator at ``seed``.

    ``deterministic=True`` flips cuDNN into deterministic + non-benchmark
    mode (slower, used only when bit-for-bit reproducibility matters).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
