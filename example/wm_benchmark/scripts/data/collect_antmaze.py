"""Forked from ``upstream_scripts/data/collect_antmaze.py`` — random
trajectories on ``swm/OGBMaze-v0`` with ``loco_env_type='ant'``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import stable_worldmodel as swm
from stable_worldmodel.policy import RandomPolicy

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from _common import base_argparser, datasets_root  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = base_argparser(__doc__, default_num_traj=2)
    p.add_argument("--maze-type", default="teleport")
    ns = p.parse_args(argv)

    world = swm.World(
        "swm/OGBMaze-v0",
        num_envs=ns.num_envs,
        image_shape=(224, 224),
        loco_env_type="ant",
        maze_env_type="maze",
        maze_type=ns.maze_type,
        ob_type="pixels",
        max_episode_steps=ns.max_episode_steps,
    )
    world.set_policy(RandomPolicy())

    out = datasets_root(ns.cache_dir) / f"{ns.out_name or 'antmaze-teleport-navigate-v0'}.lance"
    world.collect(path=out, episodes=ns.num_traj, seed=ns.seed)
    print(f"[collect_antmaze] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
