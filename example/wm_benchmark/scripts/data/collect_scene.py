"""Forked from ``upstream_scripts/data/collect_scene.py`` — ExpertPolicy on
``swm/OGBScene-v0`` (single-view, data_collection mode, EGL renderer).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

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
        "swm/OGBScene-v0",
        num_envs=ns.num_envs, max_episode_steps=ns.max_episode_steps,
        multiview=False,
        width=ns.width, height=ns.height,
        visualize_info=False, terminate_at_goal=False,
        mode="data_collection",
    )
    world.set_policy(ExpertPolicy())

    out = datasets_root(ns.cache_dir) / "ogbench" / (
        f"{ns.out_name}.lance" if ns.out_name else "scene_single_expert.lance"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    world.collect(out, episodes=ns.num_traj,
                  seed=rng.integers(0, 1_000_000).item())
    print(f"[collect_scene] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
