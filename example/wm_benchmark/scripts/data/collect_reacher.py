"""Forked from ``upstream_scripts/data/collect_reacher.py`` — RandomPolicy
trajectories on a DMC Reacher world.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np

import stable_worldmodel as swm
from stable_worldmodel.policy import RandomPolicy

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from _common import base_argparser, datasets_root  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = base_argparser(__doc__, default_num_traj=100)
    p.add_argument("--env-name", default="swm/ReacherDMControl-v0")
    ns = p.parse_args(argv)

    rng = np.random.default_rng(ns.seed)
    world = swm.World(ns.env_name,
                      num_envs=ns.num_envs,
                      max_episode_steps=ns.max_episode_steps)
    world.set_policy(RandomPolicy(seed=rng.integers(0, 1_000_000).item()))

    out = datasets_root(ns.cache_dir) / "dmc" / f"{ns.out_name or 'reacher_random'}.lance"
    out.parent.mkdir(parents=True, exist_ok=True)
    world.collect(out, episodes=ns.num_traj,
                  seed=rng.integers(0, 1_000_000).item())
    print(f"[collect_reacher] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
