"""Forked from ``upstream_scripts/data/collect_tworooms.py`` — ExpertPolicy
trajectories on ``swm/TwoRoom-v1``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import stable_worldmodel as swm
from stable_worldmodel.envs.two_room import ExpertPolicy

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from _common import base_argparser, datasets_root  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = base_argparser(__doc__, default_num_traj=100)
    p.add_argument("--action-noise", type=float, default=2.0)
    p.add_argument("--action-repeat-prob", type=float, default=0.05)
    ns = p.parse_args(argv)

    world = swm.World(
        "swm/TwoRoom-v1",
        num_envs=ns.num_envs,
        max_episode_steps=ns.max_episode_steps,
        render_mode="rgb_array",
    )
    world.set_policy(ExpertPolicy(action_noise=ns.action_noise,
                                    action_repeat_prob=ns.action_repeat_prob))

    rng = np.random.default_rng(ns.seed)
    out = datasets_root(ns.cache_dir) / f"{ns.out_name or 'tworoom_expert'}.lance"
    world.collect(out, episodes=ns.num_traj,
                  seed=rng.integers(0, 1_000_000).item())
    print(f"[collect_tworooms] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
