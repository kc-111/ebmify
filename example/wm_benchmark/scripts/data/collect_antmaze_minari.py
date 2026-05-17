"""Collect image-based AntMaze datasets from minari.

Minari ships D4RL AntMaze trajectories as state-action tuples. swm-side
training expects pixel observations, so this collector replays each
minari episode through a ``gymnasium-robotics`` ``AntMaze_*-v5`` env in
``render_mode='rgb_array'`` and stitches the frames back into a Lance
table with columns roughly matching what ``swm/OGBMaze-v0`` (ant) writes
via ``world.collect``.

Caveats:
- Replay rendering is slow (~5–20 ms per frame). The default
  ``--max-episodes 100`` keeps a single run minute-scale; pass a larger
  value explicitly for production collections.
- When a minari dataset stores qpos/qvel in ``infos``, we ``set_state``
  per step (exact pixel match for the recorded trajectory). Otherwise we
  fall back to replaying actions, which may diverge from the recorded
  state due to physics nondeterminism.
- Schema mirrors swm's OGBMaze antmaze: ``pixels``, ``action``,
  ``proprio``, ``goal_pixels``, ``reward``, ``terminated``,
  ``truncated``, ``step_idx``, ``id``, ``env_name``. ``episode_idx`` and
  ``step_idx`` are added by :func:`scripts.common.lance_io.write_lance`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterator, Mapping

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_BENCHMARK_ROOT = _HERE.parent.parent
if str(_BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_ROOT))

from _common import datasets_root  # noqa: E402
from scripts.common.lance_io import write_lance  # noqa: E402


VARIANT_TO_DATASET = {
    "umaze": "D4RL/antmaze/umaze-v1",
    "umaze-diverse": "D4RL/antmaze/umaze-diverse-v1",
    "medium-play": "D4RL/antmaze/medium-play-v1",
    "medium-diverse": "D4RL/antmaze/medium-diverse-v1",
    "large-play": "D4RL/antmaze/large-play-v1",
    "large-diverse": "D4RL/antmaze/large-diverse-v1",
}

VARIANT_TO_ENV = {
    "umaze": "AntMaze_UMaze-v5",
    "umaze-diverse": "AntMaze_UMaze-v5",
    "medium-play": "AntMaze_Medium-v5",
    "medium-diverse": "AntMaze_Medium_Diverse_GR-v5",
    "large-play": "AntMaze_Large-v5",
    "large-diverse": "AntMaze_Large_Diverse_GR-v5",
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, choices=sorted(VARIANT_TO_DATASET),
                   help="which D4RL antmaze split to pull")
    p.add_argument("--max-episodes", type=int, default=100,
                   help="cap on episodes (rendering is slow; default keeps runs minute-scale)")
    p.add_argument("--image-size", type=int, default=64,
                   help="square render resolution")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", type=str, default=None,
                   help="override $STABLEWM_HOME; defaults to swm's cache root")
    p.add_argument("--out-name", type=str, default=None,
                   help="lance output stem; default is antmaze-<variant>-minari-v0")
    p.add_argument("--overwrite", action="store_true",
                   help="clobber any existing lance file at the target path")
    p.add_argument("--prefer", choices=("state", "action"), default="state",
                   help="replay strategy: re-set state per step (exact) "
                        "or step actions (drifts but cheaper). 'state' silently "
                        "falls back to 'action' if qpos/qvel aren't stored.")
    return p.parse_args(argv)


def _open_env(variant: str, image_size: int):
    import gymnasium as gym
    try:
        import gymnasium_robotics  # noqa: F401  (registers AntMaze envs)
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "gymnasium-robotics is required for AntMaze rendering. "
            "Install with `pip install gymnasium-robotics`."
        ) from exc

    env_id = VARIANT_TO_ENV[variant]
    env = gym.make(
        env_id, render_mode="rgb_array",
        continuing_task=False,
        width=image_size, height=image_size,
    )
    return env


def _try_qstate(info: Mapping | None) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(info, Mapping):
        return None
    qpos = info.get("qpos")
    qvel = info.get("qvel")
    if qpos is None or qvel is None:
        return None
    return np.asarray(qpos), np.asarray(qvel)


def _proprio_from_obs(obs) -> np.ndarray:
    """AntMaze gym obs is a dict; we flatten the proprio portion."""
    if isinstance(obs, Mapping):
        # gymnasium-robotics AntMaze returns {'observation', 'achieved_goal', 'desired_goal'}
        return np.concatenate([
            np.asarray(obs.get("observation", []), dtype=np.float32).ravel(),
            np.asarray(obs.get("achieved_goal", []), dtype=np.float32).ravel(),
        ])
    return np.asarray(obs, dtype=np.float32).ravel()


def _render_goal(env, desired_goal: np.ndarray | None) -> np.ndarray | None:
    if desired_goal is None:
        return None
    try:
        # AntMaze stores the goal position; replace agent's xy with the goal
        # and snapshot to get a coarse "goal frame".
        u = env.unwrapped
        qpos = u.data.qpos.copy()
        qvel = np.zeros_like(u.data.qvel)
        qpos[0] = float(desired_goal[0])
        qpos[1] = float(desired_goal[1])
        u.set_state(qpos, qvel)
        frame = env.render()
        return np.asarray(frame, dtype=np.uint8)
    except Exception:
        return None


def _replay_episode(env, ep, prefer: str) -> dict[str, np.ndarray] | None:
    """Render a single minari episode → dict of (T,...) arrays."""
    actions = np.asarray(ep.actions)
    rewards = np.asarray(ep.rewards, dtype=np.float32)
    terms = np.asarray(getattr(ep, "terminations", ep.rewards.astype(bool)))
    truncs = np.asarray(getattr(ep, "truncations", np.zeros_like(rewards, dtype=bool)))
    T = int(actions.shape[0])
    if T == 0:
        return None

    obs0 = ep.observations
    obs_init = obs0[0] if isinstance(obs0, (list, np.ndarray)) and len(obs0) else obs0

    # Pull desired goal once (constant across episode for non-diverse splits).
    desired_goal = None
    if isinstance(obs_init, Mapping) and "desired_goal" in obs_init:
        desired_goal = np.asarray(obs_init["desired_goal"], dtype=np.float32)

    # Reset to the recorded initial state if available.
    env.reset(seed=0)
    qstate0 = None
    infos0 = getattr(ep, "infos", None)
    if isinstance(infos0, Mapping):
        first_info = {k: (v[0] if hasattr(v, "__getitem__") and len(np.shape(v)) >= 1 else v)
                      for k, v in infos0.items()}
        qstate0 = _try_qstate(first_info)

    use_state = (prefer == "state") and (qstate0 is not None)
    if use_state:
        env.unwrapped.set_state(qstate0[0], qstate0[1])

    pix_list: list[np.ndarray] = []
    proprio_list: list[np.ndarray] = []

    for t in range(T):
        # Step or set state
        if use_state and infos0 is not None:
            qpos_t = np.asarray(infos0["qpos"][t])
            qvel_t = np.asarray(infos0["qvel"][t])
            env.unwrapped.set_state(qpos_t, qvel_t)
            obs_t = env.unwrapped._get_obs() if hasattr(env.unwrapped, "_get_obs") else None
        else:
            obs_t, _, term, trunc, _ = env.step(actions[t])
            if term or trunc:
                # finish current step but stop early if env-terminated
                pass
        frame = env.render()
        pix_list.append(np.asarray(frame, dtype=np.uint8))
        proprio_list.append(_proprio_from_obs(obs_t) if obs_t is not None else np.zeros(1, dtype=np.float32))

    pixels = np.stack(pix_list, axis=0)
    # Pad proprio to a uniform width across timesteps.
    max_p = max(p.size for p in proprio_list)
    proprio = np.zeros((T, max_p), dtype=np.float32)
    for t, p in enumerate(proprio_list):
        proprio[t, :p.size] = p

    goal_frame = _render_goal(env, desired_goal)
    if goal_frame is None:
        goal_frame = pixels[-1]
    goal_pixels = np.broadcast_to(goal_frame[None], pixels.shape).copy()

    out: dict[str, np.ndarray] = {
        "pixels": pixels,
        "goal_pixels": goal_pixels,
        "action": actions.astype(np.float32),
        "proprio": proprio,
        "reward": rewards.astype(np.float32),
        "terminated": terms.astype(np.bool_),
        "truncated": truncs.astype(np.bool_),
    }
    out["env_name"] = np.array(["AntMaze"] * T)
    return out


def main(argv: list[str] | None = None) -> int:
    try:
        import minari  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "minari is required for this collector. "
            "Install with `pip install 'minari[hdf5]'`."
        ) from exc

    ns = _parse_args(argv)
    dataset_id = VARIANT_TO_DATASET[ns.variant]
    ds = minari.load_dataset(dataset_id, download=True)
    print(f"[collect_antmaze_minari] loaded {dataset_id} "
          f"({ds.total_episodes} episodes, capping at {ns.max_episodes})")

    env = _open_env(ns.variant, ns.image_size)

    out_path = datasets_root(ns.cache_dir) / (
        f"{ns.out_name}.lance" if ns.out_name else f"antmaze-{ns.variant}-minari-v0.lance"
    )

    def _episode_iter() -> Iterator[Mapping[str, np.ndarray]]:
        n = 0
        for ep in ds.iterate_episodes():
            if n >= ns.max_episodes:
                break
            try:
                rec = _replay_episode(env, ep, prefer=ns.prefer)
            except Exception as exc:  # keep collection going on per-episode hiccups
                print(f"[collect_antmaze_minari] episode {n} replay failed: {exc!r}")
                continue
            if rec is None:
                continue
            n += 1
            if n % 10 == 0:
                print(f"[collect_antmaze_minari] {n}/{ns.max_episodes} episodes rendered")
            yield rec

    write_lance(out_path, _episode_iter(), overwrite=ns.overwrite)
    print(f"[collect_antmaze_minari] wrote {out_path}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
