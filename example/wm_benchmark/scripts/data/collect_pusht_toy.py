"""Forked from ``upstream_scripts/data/collect_pusht_toy.py`` — 10
sharded ``WeakPolicy`` runs on ``swm/PushT-v1``.
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


def main(argv: list[str] | None = None) -> int:
    p = base_argparser(__doc__, default_num_traj=500)
    p.add_argument("--num-shards", type=int, default=10)
    p.add_argument("--dist-constraint", type=int, default=100)
    ns = p.parse_args(argv)

    world = swm.World(
        "swm/PushT-v1",
        num_envs=ns.num_envs, max_episode_steps=ns.max_episode_steps,
        render_mode="rgb_array",
    )
    world.set_policy(WeakPolicy(dist_constraint=ns.dist_constraint))
    rng = np.random.default_rng(ns.seed)

    out_root = datasets_root(ns.cache_dir) / (ns.out_name or "pusht_toy")
    out_root.mkdir(parents=True, exist_ok=True)
    for i in range(ns.num_shards):
        out = out_root / f"shard_{i}.lance"
        world.collect(out, episodes=ns.num_traj,
                      seed=rng.integers(0, 1_000_000).item())
        print(f"[collect_pusht_toy] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
