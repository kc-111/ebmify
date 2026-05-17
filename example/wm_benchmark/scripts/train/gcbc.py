"""Local GCBC trainer — fork of ``upstream_scripts/train/gcbc.py``.

Reuses upstream's ``get_data`` + ``get_gcbc_policy`` verbatim; local
scaffolding handles config/logger/checkpoint.

Usage:

    python scripts/train/gcbc.py
    python scripts/train/gcbc.py dataset_name=tworoom_expert trainer.max_epochs=10
"""
from __future__ import annotations

import sys
from pathlib import Path

_BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
if str(_BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_ROOT))

from scripts.common import config as cfgmod  # noqa: E402
from scripts.train._runner import run_trainer  # noqa: E402
from scripts.train._upstream import load_upstream, to_omegaconf  # noqa: E402
from scripts.train.configs.gcbc import GCBCConfig  # noqa: E402

METHOD = "gcbc"


def _build(cfg: GCBCConfig):
    upstream = load_upstream("gcbc")
    oc = to_omegaconf(cfg)
    data = upstream.get_data(oc)
    module = upstream.get_gcbc_policy(oc)
    return data, module


def main(argv: list[str] | None = None) -> int:
    cfg = cfgmod.from_argv(GCBCConfig, argv, description=__doc__)
    run_trainer(cfg, _build, METHOD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
