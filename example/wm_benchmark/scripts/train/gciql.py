"""Local GC-IQL trainer — fork of ``upstream_scripts/train/gciql.py``.

Two-phase: value+critic (expectile loss), then actor (AWR). Reuses
upstream's ``get_data`` / ``get_gciql_critics_model`` / ``get_gciql_actor_model``
verbatim; local scaffolding handles config/logger/checkpoint.
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
from scripts.train.configs.gciql import GCIQLConfig  # noqa: E402

METHOD = "gciql"


def _build_value(cfg: GCIQLConfig):
    upstream = load_upstream("gciql")
    oc = to_omegaconf(cfg)
    data = upstream.get_data(oc, goal_probabilities=cfg.goal_probabilities.as_tuple())
    module = upstream.get_gciql_critics_model(oc)
    return data, module


def _build_actor(cfg: GCIQLConfig, trained_critics_model):
    upstream = load_upstream("gciql")
    oc = to_omegaconf(cfg)
    data = upstream.get_data(oc, goal_probabilities=cfg.actor_goal_probabilities.as_tuple())
    module = upstream.get_gciql_actor_model(oc, trained_critics_model)
    return data, module


def main(argv: list[str] | None = None) -> int:
    cfg = cfgmod.from_argv(GCIQLConfig, argv, description=__doc__)
    run_two_phase_trainer(cfg, _build_value, _build_actor, METHOD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
