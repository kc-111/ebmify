"""Per-rollout artifact recorder shared by ``eval_wm`` and ``eval_policy``.

For each evaluated episode, the recorder writes:

  - ``frames_ep<NN>.mp4``    via imageio[ffmpeg] (falls back to .npz)
  - ``actions_ep<NN>.npy``   shape ``(T, A)``
  - ``observations_ep<NN>.npy`` shape ``(T, ...)`` if --record-obs
  - one row per episode in ``rollouts.csv``     reward / success / wall
  - one row per step  in ``timing.csv``         step_idx, env_step_ms, plan_ms
  - ``summary.json``                            aggregate stats
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class StepRecord:
    """Captured at every env step inside an episode."""
    episode: int
    step: int
    env_step_ms: float
    plan_ms: float
    reward: float = 0.0
    done: bool = False


@dataclass
class EpisodeRecord:
    """Aggregated stats per episode."""
    episode: int
    steps: int
    total_reward: float
    success: bool
    wall_time_s: float
    extras: dict[str, Any] = field(default_factory=dict)


class RolloutRecorder:
    """Owns the eval_run directory and the row buffers.

    Typical usage:

        rec = RolloutRecorder(run_dir, record_obs=False)
        for ep in range(N):
            rec.begin_episode(ep)
            ...
            rec.log_step(env_step_ms=..., plan_ms=..., reward=r, done=d,
                         frame=rgb_frame, action=a, obs=...)
            ...
            rec.end_episode(success=True)
        rec.finalize()
    """

    def __init__(self, run_dir: Path, *, record_obs: bool = False, fps: int = 20):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.record_obs = record_obs
        self.fps = fps
        self._steps: list[StepRecord] = []
        self._eps: list[EpisodeRecord] = []
        self._cur_ep: int | None = None
        self._cur_frames: list[np.ndarray] = []
        self._cur_actions: list[np.ndarray] = []
        self._cur_obs: list[np.ndarray] = []
        self._cur_steps: int = 0
        self._cur_reward: float = 0.0
        self._cur_start: float | None = None

    # ----- per-episode lifecycle

    def begin_episode(self, ep: int) -> None:
        self._cur_ep = ep
        self._cur_frames = []
        self._cur_actions = []
        self._cur_obs = []
        self._cur_steps = 0
        self._cur_reward = 0.0
        self._cur_start = time.time()

    def log_step(self, *, env_step_ms: float, plan_ms: float = 0.0,
                  reward: float = 0.0, done: bool = False,
                  frame: np.ndarray | None = None,
                  action: np.ndarray | None = None,
                  obs: np.ndarray | None = None) -> None:
        assert self._cur_ep is not None, "begin_episode() not called"
        self._steps.append(StepRecord(
            episode=self._cur_ep, step=self._cur_steps,
            env_step_ms=env_step_ms, plan_ms=plan_ms,
            reward=float(reward), done=bool(done),
        ))
        if frame is not None:
            self._cur_frames.append(np.asarray(frame))
        if action is not None:
            self._cur_actions.append(np.asarray(action))
        if obs is not None and self.record_obs:
            self._cur_obs.append(np.asarray(obs))
        self._cur_steps += 1
        self._cur_reward += float(reward)

    def end_episode(self, *, success: bool, extras: dict[str, Any] | None = None) -> None:
        assert self._cur_ep is not None
        wall = time.time() - (self._cur_start or time.time())
        self._eps.append(EpisodeRecord(
            episode=self._cur_ep, steps=self._cur_steps,
            total_reward=self._cur_reward, success=bool(success),
            wall_time_s=wall, extras=extras or {},
        ))
        self._dump_episode(self._cur_ep)
        self._cur_ep = None

    # ----- per-episode dump

    def _dump_episode(self, ep: int) -> None:
        if self._cur_frames:
            self._write_video(self.run_dir / f"frames_ep{ep:02d}.mp4",
                              np.stack(self._cur_frames, axis=0))
        if self._cur_actions:
            np.save(self.run_dir / f"actions_ep{ep:02d}.npy",
                    np.stack(self._cur_actions, axis=0))
        if self.record_obs and self._cur_obs:
            np.save(self.run_dir / f"obs_ep{ep:02d}.npy",
                    np.stack(self._cur_obs, axis=0))

    def _write_video(self, path: Path, frames: np.ndarray) -> None:
        try:
            import imageio.v3 as iio
            iio.imwrite(str(path), frames.astype(np.uint8), fps=self.fps)
        except Exception:  # fall back if ffmpeg missing
            np.savez_compressed(path.with_suffix(".npz"), frames=frames)

    # ----- finalize

    def finalize(self) -> dict[str, Any]:
        timing_path = self.run_dir / "timing.csv"
        with timing_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "step", "env_step_ms", "plan_ms", "reward", "done"])
            for s in self._steps:
                w.writerow([s.episode, s.step, f"{s.env_step_ms:.3f}",
                            f"{s.plan_ms:.3f}", f"{s.reward:.6g}", int(s.done)])

        rollouts_path = self.run_dir / "rollouts.csv"
        with rollouts_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "steps", "total_reward", "success", "wall_s"])
            for e in self._eps:
                w.writerow([e.episode, e.steps, f"{e.total_reward:.6g}",
                            int(e.success), f"{e.wall_time_s:.3f}"])

        summary = self._summary()
        with (self.run_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        return summary

    def _summary(self) -> dict[str, Any]:
        if not self._eps:
            return {"episodes": 0}
        n = len(self._eps)
        succ = sum(1 for e in self._eps if e.success)
        return {
            "episodes": n,
            "success_rate": succ / n,
            "mean_reward": float(np.mean([e.total_reward for e in self._eps])),
            "mean_steps": float(np.mean([e.steps for e in self._eps])),
            "mean_wall_s": float(np.mean([e.wall_time_s for e in self._eps])),
            "mean_env_step_ms": float(np.mean([s.env_step_ms for s in self._steps])),
            "mean_plan_ms": float(np.mean([s.plan_ms for s in self._steps])),
        }
