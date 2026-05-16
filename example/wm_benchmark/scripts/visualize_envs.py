"""Per-env policy rollouts → GIF per benchmark env.

For each spec we attach the same policy upstream's ``collect_<env>.py`` uses
(``RandomPolicy`` for pointmaze, ``WeakPolicy`` for pusht_fov, ``ExpertPolicy``
for tworooms and cube), step the underlying ``EnvPool`` directly, and capture
one ``(H, W, 3)`` frame per step from ``world.infos['pixels']`` (populated by
``AddPixelsWrapper`` on every step).

Random uniform action sampling barely moves Push-T/cube; the per-env policies
produce visibly purposeful trajectories instead.

Usage::

    source example/wm_benchmark/scripts/env.sh
    python example/wm_benchmark/scripts/visualize_envs.py
    python example/wm_benchmark/scripts/visualize_envs.py --env pointmaze --steps 120
    python example/wm_benchmark/scripts/visualize_envs.py --envs pointmaze tworooms --fps 20
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

# Force headless EGL before any MuJoCo import (default glfw silently produces
# all-black frames for swm/OGBMaze-v0). Set early or pointmaze renders black.
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import DATA_DIR  # noqa: E402

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.policy import RandomPolicy  # noqa: E402
from stable_worldmodel.envs.pusht import WeakPolicy  # noqa: E402
from stable_worldmodel.envs.two_room import ExpertPolicy as TwoRoomPolicy  # noqa: E402
from stable_worldmodel.envs.ogbench import ExpertPolicy as OGBenchPolicy  # noqa: E402

# Each entry: (tag, World kwargs, policy factory). Policy factory is a thunk
# so we build a fresh one per rollout. Policy choices mirror
# upstream_scripts/data/collect_<env>.py at SHA 463ab63. ``image_shape`` per
# spec is the (H, W) the env renders at: pointmaze's "teleport" layout is
# wider than tall, so a square 224x224 crops left/right walls — render at
# (224, 336) for the visualization (training-side benchmarks use 224x224).
SPECS = [
    ("pointmaze", dict(
        env_id="swm/OGBMaze-v0",
        loco_env_type="point", maze_env_type="maze", maze_type="teleport",
        ob_type="pixels", width=336, height=224, image_shape=(224, 336),
    ), lambda: RandomPolicy(seed=0)),
    ("pusht_fov", dict(
        env_id="swm/PushT-v1", render_mode="rgb_array",
    ), lambda: WeakPolicy(dist_constraint=100, seed=0)),
    ("tworooms", dict(
        env_id="swm/TwoRoom-v1", render_mode="rgb_array",
    ), lambda: TwoRoomPolicy(action_noise=2.0, action_repeat_prob=0.05, seed=0)),
    ("cube", dict(
        env_id="swm/OGBCube-v0", env_type="single", multiview=True,
        width=224, height=224, visualize_info=False, terminate_at_goal=False,
        mode="data_collection",
    ), lambda: OGBenchPolicy(seed=0)),
]
COMMON = dict(num_envs=1, image_shape=(224, 224), max_episode_steps=200)


PIXEL_KEYS = ("pixels", "pixels.front", "pixels.front_pixels", "pixels.side")


def _extract_frame(infos: dict) -> np.ndarray:
    """Pull one (H, W, 3) uint8 frame out of the stacked EnvPool infos dict.

    Pixels in stacked infos are shaped ``(num_envs, 1, H, W, C)`` (the leading
    ``1`` is the wrapper's time slot). We take env 0 and frame 0.
    """
    for key in PIXEL_KEYS:
        if key not in infos:
            continue
        v = np.asarray(infos[key])
        if v.ndim == 5:        # (num_envs, T, H, W, C)
            v = v[0, 0]
        elif v.ndim == 4:      # (num_envs, H, W, C)
            v = v[0]
        if v.dtype != np.uint8:
            v = (v * 255 if v.max() <= 1.0 else v).clip(0, 255).astype(np.uint8)
        # Copy: wrapper reuses the same backing buffer across steps, so views
        # appended to a list would all collapse to the final step's content.
        return np.array(v, copy=True)
    raise KeyError(f"no pixel key in infos; saw {sorted(infos.keys())[:8]}...")


def rollout(tag: str, kw: dict, make_policy, steps: int, seed: int) -> list[np.ndarray]:
    env_id = kw.pop("env_id")
    kw = {**COMMON, **kw}
    world = swm.World(env_id, **kw)
    try:
        policy = make_policy()
        world.set_policy(policy)
        world.reset(seed=seed)
        frames: list[np.ndarray] = [_extract_frame(world.infos)]
        rng = np.random.default_rng(seed)
        for _ in range(steps - 1):
            actions = policy.get_action(world.infos)
            _, _, terminateds, truncateds, infos = world.envs.step(actions)
            world.infos = infos
            frames.append(_extract_frame(infos))
            if bool(terminateds[0]) or bool(truncateds[0]):
                _, world.infos = world.envs.reset(seed=int(rng.integers(0, 1_000_000)))
        return frames
    finally:
        world.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--envs", nargs="+", choices=[t for t, _, _ in SPECS],
                        help="subset of envs to render (default: all)")
    parser.add_argument("--env", choices=[t for t, _, _ in SPECS],
                        help="alias for --envs <one>")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "gifs",
                        help="output dir (default: data/gifs)")
    ns = parser.parse_args()

    selected = ns.envs if ns.envs else ([ns.env] if ns.env else [t for t, _, _ in SPECS])
    ns.out.mkdir(parents=True, exist_ok=True)

    failures = 0
    for tag in selected:
        kw, make_policy = next((dict(kw), mk) for t, kw, mk in SPECS if t == tag)
        try:
            frames = rollout(tag, kw, make_policy, steps=ns.steps, seed=ns.seed)
            out_path = ns.out / f"{tag}.gif"
            imageio.mimsave(out_path, frames, fps=ns.fps, loop=0)
            h, w = frames[0].shape[:2]
            print(f"  {tag:<10s} → {out_path}  ({len(frames)} frames, {w}x{h}, {ns.fps} fps)")
        except Exception:
            failures += 1
            print(f"  {tag:<10s} FAIL")
            traceback.print_exc(limit=4)

    print(f"\n{len(selected) - failures}/{len(selected)} envs rendered")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
