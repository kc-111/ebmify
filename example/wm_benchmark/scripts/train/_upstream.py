"""Import the SHA-pinned upstream trainer modules as plain Python.

The ``upstream_scripts/`` tree is a verbatim snapshot, not a Python
package (no ``__init__.py``), so we load each ``upstream_scripts/train/
<method>.py`` via importlib. The local trainers then reuse upstream's
``get_data``, ``get_*_policy``, ``*_forward`` functions verbatim — which
keeps trained checkpoints binary-compatible with upstream eval.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_DIR = _BENCHMARK_ROOT / "upstream_scripts" / "train"


def load_upstream(method: str) -> ModuleType:
    """Load ``upstream_scripts/train/<method>.py`` as a module.

    The module is cached in ``sys.modules`` under ``swm_upstream_train.<method>``
    so re-imports are cheap.
    """
    key = f"swm_upstream_train.{method}"
    if key in sys.modules:
        return sys.modules[key]

    path = _UPSTREAM_DIR / f"{method}.py"
    if not path.is_file():
        raise FileNotFoundError(f"upstream trainer not found: {path}")

    spec = importlib.util.spec_from_file_location(key, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def to_omegaconf(cfg) -> "OmegaConf":  # noqa: F821 — lazy import
    """Convert a (possibly nested) dataclass cfg to an OmegaConf DictConfig.

    Upstream's builder functions assume Hydra-shaped configs with
    ``cfg.get(...)``, ``OmegaConf.to_container(...)``, etc., so a plain
    dataclass won't do; we go through ``OmegaConf.create`` once.
    """
    from omegaconf import OmegaConf

    from scripts.common.config import to_dict

    return OmegaConf.create(to_dict(cfg))
