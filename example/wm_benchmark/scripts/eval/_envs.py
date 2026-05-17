"""Env registry mirroring ``scripts/verify.py``.

Each entry returns the kwargs passed to ``swm.World(...)`` for that
benchmark env. Centralizing this keeps eval_wm / eval_policy / verify in
sync — if upstream changes the canonical kwargs for an env, we change
them here and every consumer picks them up.
"""
from __future__ import annotations

from typing import Any


def get_world_kwargs(tag: str, *, max_episode_steps: int = 100) -> dict[str, Any]:
    """Return the ``swm.World`` kwargs for benchmark env ``tag``."""
    if tag == "pointmaze":
        return dict(
            env_name="swm/OGBMaze-v0",
            loco_env_type="point",
            maze_env_type="maze",
            maze_type="teleport",
            max_episode_steps=max_episode_steps,
            image_shape=(224, 224),
            render_mode="rgb_array",
        )
    if tag == "antmaze":
        return dict(
            env_name="swm/OGBMaze-v0",
            loco_env_type="ant",
            maze_env_type="maze",
            maze_type="teleport",
            max_episode_steps=max_episode_steps,
            image_shape=(224, 224),
            render_mode="rgb_array",
        )
    if tag == "pusht":
        return dict(
            env_name="swm/PushT-v1",
            max_episode_steps=max_episode_steps,
            image_shape=(224, 224),
            render_mode="rgb_array",
        )
    if tag == "tworooms":
        return dict(
            env_name="swm/TwoRoom-v1",
            max_episode_steps=max_episode_steps,
            image_shape=(224, 224),
            render_mode="rgb_array",
        )
    if tag == "cube":
        return dict(
            env_name="swm/OGBCube-v0",
            env_type="single",
            multiview=True,
            mode="data_collection",
            max_episode_steps=max_episode_steps,
            image_shape=(224, 224),
            render_mode="rgb_array",
        )
    raise ValueError(f"unknown env tag {tag!r}")
