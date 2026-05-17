"""Forked from ``upstream_scripts/data/collect_dmc.py`` — ExpertPolicy
trajectories on a chosen DMC env. Requires ``--expert-ckpt-path``
pointing at the upstream-supplied expert policy bundle.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np

import stable_worldmodel as swm
from stable_worldmodel.envs.dmcontrol import ExpertPolicy

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from _common import base_argparser, datasets_root  # noqa: E402

ENVS = {
    "swm/CartpoleDMControl-v0": "cartpole",
    "swm/WalkerDMControl-v0": "walker",
    "swm/QuadrupedDMControl-v0": "quadruped",
    "swm/BallInCupDMControl-v0": "ballincup",
    "swm/FingerDMControl-v0": "finger",
    "swm/HopperDMControl-v0": "hopper",
    "swm/CheetahDMControl-v0": "cheetah",
    "swm/ReacherDMControl-v0": "reacher",
    "swm/PendulumDMControl-v0": "pendulum",
}


def main(argv: list[str] | None = None) -> int:
    p = base_argparser(__doc__, default_num_traj=100)
    p.add_argument("--env-name", required=True, choices=sorted(ENVS))
    p.add_argument("--expert-ckpt-path", type=Path, required=True,
                   help="root dir containing <env>/expert_policy.zip and vec_normalize.pkl")
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--device", default="cpu")
    ns = p.parse_args(argv)

    short = ENVS[ns.env_name]
    rng = np.random.default_rng(ns.seed)
    world = swm.World(ns.env_name,
                      num_envs=ns.num_envs,
                      max_episode_steps=ns.max_episode_steps)
    world.set_policy(ExpertPolicy(
        ckpt_path=ns.expert_ckpt_path / f"{short}/expert_policy.zip",
        vec_normalize_path=ns.expert_ckpt_path / f"{short}/vec_normalize.pkl",
        noise_std=ns.noise_std,
        device=ns.device,
    ))

    out = datasets_root(ns.cache_dir) / "dmc" / f"{ns.out_name or short}_expert.lance"
    out.parent.mkdir(parents=True, exist_ok=True)
    world.collect(out, episodes=ns.num_traj,
                  seed=rng.integers(0, 1_000_000).item())
    print(f"[collect_dmc] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
