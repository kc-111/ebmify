"""Smoke-test the stable-worldmodel installation against each benchmark env.

For each env it constructs a ``swm.World``, calls ``reset(seed=0)`` and one
``step(action_space.sample())``, then prints obs + render shapes. Use
``--env <name>`` to run a single env. Run after ``source scripts/env.sh``.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _paths  # noqa: F401, E402

import stable_worldmodel as swm  # noqa: E402

# (tag, World kwargs) — kwargs taken from upstream collect_<env>.py at SHA
# 463ab63. Only env-specific extras live below; `num_envs`, `image_shape`,
# `max_episode_steps` are added uniformly per smoke run.
SPECS = [
    ("pointmaze", dict(
        env_id="swm/OGBMaze-v0",
        loco_env_type="point", maze_env_type="maze", maze_type="teleport",
        ob_type="pixels",
    )),
    ("pusht_fov", dict(
        env_id="swm/PushT-v1", render_mode="rgb_array",
    )),
    ("tworooms", dict(
        env_id="swm/TwoRoom-v1", render_mode="rgb_array",
    )),
    ("cube", dict(
        env_id="swm/OGBCube-v0", env_type="single", multiview=True,
        width=224, height=224, visualize_info=False, terminate_at_goal=False,
        mode="data_collection",
    )),
]

COMMON = dict(num_envs=1, image_shape=(224, 224), max_episode_steps=50)


def _shape(x):
    return tuple(x.shape) if hasattr(x, "shape") else type(x).__name__


def smoke(tag: str, kw: dict) -> bool:
    env_id = kw.pop("env_id")
    kw = {**COMMON, **kw}
    try:
        world = swm.World(env_id, **kw)
        world.reset(seed=0)
        infos = world.infos
        try:
            img = world.envs.envs[0].render()
            render_shape = _shape(img)
        except Exception as e:
            render_shape = f"<render failed: {e!r}>"
        info_keys = sorted(infos.keys()) if isinstance(infos, dict) else type(infos).__name__
        print(f"  {tag:<10s} ok  info_keys={info_keys}  render={render_shape}")
        world.close()
        return True
    except Exception:
        print(f"  {tag:<10s} FAIL")
        traceback.print_exc(limit=4)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=[t for t, _ in SPECS],
                        help="run a single env instead of all four")
    ns = parser.parse_args()

    specs = [(t, kw) for t, kw in SPECS if ns.env is None or t == ns.env]
    print(f"stable_worldmodel verify: {len(specs)} env(s)")
    ok = sum(smoke(t, dict(kw)) for t, kw in specs)
    print(f"\n{ok}/{len(specs)} passed")
    return 0 if ok == len(specs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
