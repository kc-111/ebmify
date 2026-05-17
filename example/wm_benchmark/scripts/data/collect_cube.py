"""Forked from ``upstream_scripts/data/collect_cube.py`` — ExpertPolicy on
``swm/OGBCube-v0`` (single-cube, multiview, data_collection mode).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np

import stable_worldmodel as swm
from stable_worldmodel.envs.ogbench import ExpertPolicy

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from _common import base_argparser, datasets_root  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = base_argparser(__doc__, default_num_traj=100)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--height", type=int, default=224)
    ns = p.parse_args(argv)

    rng = np.random.default_rng(ns.seed)
    world = swm.World(
        "swm/OGBCube-v0",
        num_envs=ns.num_envs, max_episode_steps=ns.max_episode_steps,
        env_type="single", multiview=True,
        width=ns.width, height=ns.height,
        visualize_info=False, terminate_at_goal=False,
        mode="data_collection",
    )
    world.set_policy(ExpertPolicy())

    out = datasets_root(ns.cache_dir) / "ogbench" / (
        f"{ns.out_name}.lance" if ns.out_name else "cube_single_multiview_expert.lance"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    world.collect(out, episodes=ns.num_traj,
                  seed=rng.integers(0, 1_000_000).item())
    print(f"[collect_cube] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
