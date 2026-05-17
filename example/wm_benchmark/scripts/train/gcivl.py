"""Local GC-IVL trainer — fork of ``upstream_scripts/train/gcivl.py``.

Two-phase: value (expectile loss, no Q), then actor (AWR). Reuses
upstream's ``get_data`` / ``get_ivl_value_model`` / ``get_ivl_actor_model``
verbatim.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
if str(_BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_ROOT))

from scripts.common import config as cfgmod  # noqa: E402
from scripts.train._runner import run_two_phase_trainer  # noqa: E402
from scripts.train._upstream import load_upstream, to_omegaconf  # noqa: E402
from scripts.train.configs.gcivl import GCIVLConfig  # noqa: E402

METHOD = "gcivl"


def _build_value(cfg: GCIVLConfig):
    upstream = load_upstream("gcivl")
    oc = to_omegaconf(cfg)
    data = upstream.get_data(oc, goal_probabilities=cfg.goal_probabilities.as_tuple())
    module = upstream.get_ivl_value_model(oc)
    return data, module


def _build_actor(cfg: GCIVLConfig, trained_value_model):
    upstream = load_upstream("gcivl")
    oc = to_omegaconf(cfg)
    data = upstream.get_data(oc, goal_probabilities=cfg.actor_goal_probabilities.as_tuple())
    module = upstream.get_ivl_actor_model(oc, trained_value_model)
    return data, module


def main(argv: list[str] | None = None) -> int:
    cfg = cfgmod.from_argv(GCIVLConfig, argv, description=__doc__)
    run_two_phase_trainer(cfg, _build_value, _build_actor, METHOD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
