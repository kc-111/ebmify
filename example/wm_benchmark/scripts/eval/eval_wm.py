"""World-model + planner evaluation entrypoint.

Loads a trained world-model checkpoint (saved by
``scripts/train/{lewm,pldm,prejepa}.py``), wraps it with one of the
``planners.py`` adapters, runs ``world.evaluate(...)`` over N episodes
sampled from a dataset, and writes frames + actions + per-step timings
+ aggregate stats to ``data/eval_runs/<run_id>/``.

Usage:

    python scripts/eval/eval_wm.py \\
        --checkpoint data/runs/lewm/<id>/last.pt \\
        --env tworooms --planner cem \\
        --episodes 5 --eval-budget 50 --goal-offset 25
"""
from __future__ import annotations

import argparse
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
from scripts.eval import planners as planners_mod  # noqa: E402
from scripts.eval._envs import get_world_kwargs  # noqa: E402
from scripts.eval.recorder import RolloutRecorder  # noqa: E402


def _img_transform(img_size: int, dtype=torch.float32):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(dtype, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=img_size),
    ])


def _episodes_length(dataset, episodes) -> np.ndarray:
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_idx = dataset.get_col_data(col)
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[ep_idx == ep]) + 1 for ep in episodes])


def _build_process(dataset, keys_to_cache: list[str]) -> dict:
    process: dict[str, preprocessing.StandardScaler] = {}
    for col in keys_to_cache:
        if col == "pixels":
            continue
        scaler = preprocessing.StandardScaler()
        data = dataset.get_col_data(col)
        data = data[~np.isnan(data).any(axis=1)]
        scaler.fit(data)
        process[col] = scaler
        if col != "action":
            process[f"goal_{col}"] = scaler
    return process


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="path to a world-model checkpoint, e.g. data/runs/lewm/<id>/last.pt")
    p.add_argument("--env", required=True,
                   help="benchmark env tag (pointmaze|antmaze|pusht|tworooms|cube)")
    p.add_argument("--dataset", required=True,
                   help="name of the evaluation dataset (passed to swm.data.load_dataset)")
    p.add_argument("--planner", default="cem",
                   choices=["cem", "icem", "mppi", "ebmify"])
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--eval-budget", type=int, default=50)
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--receding-horizon", type=int, default=5)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--keys-to-cache", nargs="*", default=["action", "proprio", "state"])
    p.add_argument("--out-dir", type=Path, default=None,
                   help="override run dir (default: data/eval_runs/<run_id>/)")
    ns = p.parse_args(argv)

    seeding.set_seed(ns.seed)

    run_id = logmod.make_run_id(prefix=f"eval-{ns.env}-{ns.planner}")
    run_dir = ns.out_dir or (DATA_DIR / "eval_runs" / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    log = logmod.setup_text_logger(run_dir)
    log.info("eval_wm run_dir=%s ckpt=%s env=%s planner=%s",
              run_dir, ns.checkpoint, ns.env, ns.planner)

    world_kwargs = get_world_kwargs(ns.env, max_episode_steps=2 * ns.eval_budget)
    world_kwargs["num_envs"] = ns.episodes
    world = swm.World(**world_kwargs)

    dtype = torch.bfloat16 if ns.bf16 else torch.float32
    transform = {
        "pixels": _img_transform(ns.img_size, dtype),
        "goal": _img_transform(ns.img_size, dtype),
    }

    dataset = swm.data.load_dataset(ns.dataset, keys_to_cache=list(ns.keys_to_cache))
    process = _build_process(dataset, list(ns.keys_to_cache))

    model = swm.wm.utils.load_pretrained(str(ns.checkpoint))
    if ns.bf16:
        model = model.to(torch.bfloat16)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    if ns.compile:
        attr = "backbone" if hasattr(model, "backbone") else "encoder"
        setattr(model, attr, torch.compile(getattr(model, attr)))
        model.predictor = torch.compile(model.predictor)

    plan_cfg = swm.PlanConfig(
        horizon=ns.horizon,
        receding_horizon=ns.receding_horizon,
        action_block=ns.action_block,
    )
    planner = planners_mod.get(
        ns.planner,
        num_samples=ns.num_samples, n_steps=ns.n_steps, topk=ns.topk, seed=ns.seed,
    ) if ns.planner != "ebmify" else planners_mod.get("ebmify")
    policy = planner.build_policy(model=model, plan_cfg=plan_cfg,
                                   transform=transform, process=process)

    # sample (episode, start_step) pairs from the dataset
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col), return_index=True)
    episode_len = _episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - ns.goal_offset - 1
    max_start_by_ep = dict(zip(ep_indices, max_start_idx))
    max_start_per_row = np.array([max_start_by_ep[e] for e in dataset.get_col_data(col)])
    valid = np.nonzero(dataset.get_col_data("step_idx") <= max_start_per_row)[0]
    rng = np.random.default_rng(ns.seed)
    chosen = np.sort(valid[rng.choice(len(valid) - 1, size=ns.episodes, replace=False)])
    eval_episodes = dataset.get_row_data(chosen)[col]
    eval_start = dataset.get_row_data(chosen)["step_idx"]

    world.set_policy(policy)
    rec = RolloutRecorder(run_dir)

    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                   enabled=ns.bf16)

    t0 = time.time()
    with autocast_ctx:
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start.tolist(),
            goal_offset=ns.goal_offset,
            eval_budget=ns.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=None,
            video=run_dir,
        )
    wall = time.time() - t0

    # We can't easily slot the RolloutRecorder mid-rollout into swm.World.evaluate
    # without monkey-patching, so seed it with the aggregate metrics so summary.json
    # still lands beside the videos written by swm itself.
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
    import json
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    log.info("eval_wm done in %.1fs; summary=%s", wall, summary)
    print(f"[eval] summary written to {run_dir/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
