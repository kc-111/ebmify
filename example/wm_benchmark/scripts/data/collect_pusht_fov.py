"""Forked from ``upstream_scripts/data/collect_pusht_fov.py`` —
WeakPolicy trajectories on ``swm/PushT-v1`` across every non-default
variation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import stable_worldmodel as swm
from stable_worldmodel.envs.pusht import WeakPolicy

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from _common import base_argparser, datasets_root  # noqa: E402

VARIATION_DEFAULT = {
    "agent.start_position",
    "block.start_position",
    "block.angle",
}


def main(argv: list[str] | None = None) -> int:
    p = base_argparser(__doc__, default_num_traj=100)
    p.add_argument("--dist-constraint", type=int, default=100)
    ns = p.parse_args(argv)

    rng = np.random.default_rng(ns.seed)
    base_kwargs = dict(num_envs=ns.num_envs,
                        max_episode_steps=ns.max_episode_steps,
                        render_mode="rgb_array")
    seed_world = swm.World("swm/PushT-v1", **base_kwargs)
    variation_list = set(seed_world.envs.single_variation_space.names())

    out_root = datasets_root(ns.cache_dir) / (ns.out_name or "pusht_fov")
    out_root.mkdir(parents=True, exist_ok=True)

    for var in variation_list:
        var = var.replace("variation.", "")
        if var in VARIATION_DEFAULT:
            continue
        world = swm.World("swm/PushT-v1", **base_kwargs)
        world.set_policy(WeakPolicy(dist_constraint=ns.dist_constraint))
        var_name = var.replace(".", "/")
        out = out_root / f"{var_name}.lance"
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"[collect_pusht_fov] {var} -> {out}")
        world.collect(
            out, episodes=ns.num_traj,
            seed=rng.integers(0, 1_000_000).item(),
            options={"variation": tuple([var] + list(VARIATION_DEFAULT))},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
