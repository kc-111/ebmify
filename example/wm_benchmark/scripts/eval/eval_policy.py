"""Feed-forward GC-policy evaluation entrypoint.

Loads a trained policy checkpoint (saved by
``scripts/train/{gcbc,gciql,gcivl,hilp}.py``), wraps it with
``swm.policy.FeedForwardPolicy``, and runs ``world.evaluate_from_dataset(...)``
over N episodes sampled from a dataset. Writes the same artifact set as
``eval_wm.py`` under ``data/eval_runs/<run_id>/``.

Usage:

    python scripts/eval/eval_policy.py \\
        --checkpoint data/runs/gcbc/<id>/last.pt \\
        --env tworooms --dataset tworoom_expert --episodes 5 --eval-budget 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

_BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
if str(_BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn import preprocessing  # noqa: E402
from torchvision.transforms import v2 as transforms  # noqa: E402

import stable_pretraining as spt  # noqa: E402
import stable_worldmodel as swm  # noqa: E402

from _paths import DATA_DIR  # noqa: E402
from scripts.common import logging as logmod  # noqa: E402
from scripts.common import seeding  # noqa: E402
from scripts.eval._envs import get_world_kwargs  # noqa: E402
from scripts.eval.recorder import RolloutRecorder  # noqa: E402


def _img_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=img_size),
        transforms.CenterCrop(size=img_size),
    ])


def _episodes_length(dataset, episodes) -> np.ndarray:
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_idx = dataset.get_col_data(col)
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[ep_idx == ep]) + 1 for ep in episodes])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env", required=True,
                   help="benchmark env tag (pointmaze|antmaze|pusht|tworooms|cube)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--eval-budget", type=int, default=50)
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=None)
    ns = p.parse_args(argv)

    seeding.set_seed(ns.seed)
    run_id = logmod.make_run_id(prefix=f"eval-policy-{ns.env}")
    run_dir = ns.out_dir or (DATA_DIR / "eval_runs" / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    log = logmod.setup_text_logger(run_dir)
    log.info("eval_policy run_dir=%s ckpt=%s env=%s", run_dir, ns.checkpoint, ns.env)

    world_kwargs = get_world_kwargs(ns.env, max_episode_steps=2 * ns.eval_budget)
    world_kwargs["num_envs"] = ns.episodes
    world = swm.World(**world_kwargs)

    transform = {"pixels": _img_transform(ns.img_size),
                 "goal": _img_transform(ns.img_size)}

    dataset = swm.data.load_dataset(ns.dataset)
    process: dict = {}
    action_scaler = preprocessing.StandardScaler()
    action_scaler.fit(dataset.get_col_data("action"))
    process["action"] = action_scaler
    if "proprio" in dataset.column_names:
        proprio_scaler = preprocessing.StandardScaler()
        proprio_scaler.fit(dataset.get_col_data("proprio"))
        process["proprio"] = proprio_scaler
        process["goal_proprio"] = proprio_scaler

    model = swm.policy.AutoActionableModel(str(ns.checkpoint))
    model = model.to("cuda").eval()
    model.requires_grad_(False)

    policy = swm.policy.FeedForwardPolicy(
        model=model, process=process, transform=transform,
    )

    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col), return_index=True)
    episode_len = _episodes_length(dataset, ep_indices)
    max_start = episode_len - ns.goal_offset - 1
    max_start_by_ep = dict(zip(ep_indices, max_start))
    max_start_per_row = np.array([max_start_by_ep[e] for e in dataset.get_col_data(col)])
    valid = np.nonzero(dataset.get_col_data("step_idx") <= max_start_per_row)[0]
    rng = np.random.default_rng(ns.seed)
    chosen = np.sort(valid[rng.choice(len(valid) - 1, size=ns.episodes, replace=False)])
    eval_episodes = dataset.get_row_data(chosen)[col]
    eval_start = dataset.get_row_data(chosen)["step_idx"]

    world.set_policy(policy)
    rec = RolloutRecorder(run_dir)

    t0 = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start.tolist(),
        goal_offset_steps=ns.goal_offset,
        eval_budget=ns.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=None,
        video_path=run_dir,
    )
    wall = time.time() - t0

    for ep_id in range(ns.episodes):
        rec.begin_episode(ep_id)
        rec.log_step(env_step_ms=0.0, plan_ms=0.0, reward=0.0, done=True)
        ep_metric = {}
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                try:
                    ep_metric[k] = float(v[ep_id]) if hasattr(v, "__len__") else float(v)
                except (TypeError, ValueError):
                    pass
        success = bool(ep_metric.get("success", False))
        rec.end_episode(success=success, extras=ep_metric)

    summary = rec.finalize()
    summary["wall_time_s"] = wall
    summary["metrics_raw"] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                                for k, v in (metrics.items() if isinstance(metrics, dict) else [])}
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    log.info("eval_policy done in %.1fs; summary=%s", wall, summary)
    print(f"[eval] summary written to {run_dir/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
