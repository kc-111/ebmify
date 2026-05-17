"""Local HILP trainer — fork of ``upstream_scripts/train/hilp.py``.

Two-phase: Hilbert metric-value (φ(s)→ℝ^k with -||φ(s)−φ(g)||) then
actor (AWR). Note: HILP's ``get_data`` reads ``goal_probabilities`` off
``cfg`` directly (single dict, no positional arg), so we pass the
relevant block via the OmegaConf cfg.
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
from scripts.train.configs.hilp import HILPConfig  # noqa: E402

METHOD = "hilp"


def _build_value(cfg: HILPConfig):
    upstream = load_upstream("hilp")
    oc = to_omegaconf(cfg)
    data = upstream.get_data(oc)
    module = upstream.get_hilp_value_model(oc)
    return data, module


def _build_actor(cfg: HILPConfig, trained_value_model):
    upstream = load_upstream("hilp")
    # Swap goal_probabilities → actor_goal_probabilities on the cfg before
    # rebuilding the dataloader for the actor phase.
    from copy import deepcopy

    actor_cfg = deepcopy(cfg)
    actor_cfg.goal_probabilities = cfg.actor_goal_probabilities
    oc = to_omegaconf(actor_cfg)
    data = upstream.get_data(oc)
    module = upstream.get_hilp_actor_model(oc, trained_value_model)
    return data, module


def main(argv: list[str] | None = None) -> int:
    cfg = cfgmod.from_argv(HILPConfig, argv, description=__doc__)
    run_two_phase_trainer(cfg, _build_value, _build_actor, METHOD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
