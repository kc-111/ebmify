"""Random-baseline CIFAR-10 coreset builder.

A trivial baseline: pick ``k`` indices uniformly at random from the 50k
CIFAR-10 train set, no encoder required. Used as a sanity floor that any
informed selection (greedy / leverage / spectral_rank) should beat.

Layout mirrors ``cifar_build_coreset_{supervised,ssl}.py`` so the same
training scripts can consume the output by passing
``--algorithms random``:

    <artifacts_root>/random/indices.pt          (k,) int64
    <artifacts_root>/random/weights.pt          (k,) float32, all ones
    <artifacts_root>/random/stats.json          {seed, budget_k, N}

No aux targets are emitted -- ``discover_aux_targets`` silently skips
missing files, so training falls back to label-only / SSL-only losses.

Usage:
    python example/cifar/coreset/cifar_build_coreset_random.py
    python example/cifar/coreset/cifar_build_coreset_random.py \\
        --tag supervised_resnet18 --budget-k 5000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import REPO_ROOT  # noqa: F401, E402


def _select_random(n_total: int, budget_k: int, seed: int) -> np.ndarray:
    if budget_k > n_total:
        raise ValueError(f"budget_k ({budget_k}) > N ({n_total})")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=budget_k, replace=False))


def build_random_coreset(
    *,
    artifacts_root: Path,
    n_total: int,
    budget_k: int,
    seed: int,
) -> Path:
    algo_dir = artifacts_root / "random"
    algo_dir.mkdir(parents=True, exist_ok=True)

    idx = _select_random(n_total, budget_k, seed)
    indices = torch.from_numpy(idx).long()
    weights = torch.ones(budget_k, dtype=torch.float32)

    torch.save(indices, algo_dir / "indices.pt")
    torch.save(weights, algo_dir / "weights.pt")
    with open(algo_dir / "stats.json", "w") as f:
        json.dump({"seed": int(seed), "budget_k": int(budget_k),
                   "n_total": int(n_total)}, f, indent=2)
    return algo_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--tag", default="supervised_resnet18",
        help=("Artifacts tag (subdir under example/cifar/cache/coreset/). "
              "Use the same tag as the informed builder you want to compare "
              "against -- this places 'random/' alongside greedy/leverage/"
              "spectral_rank/ so the training scripts can pick it up."),
    )
    ap.add_argument(
        "--artifacts-root", default=None,
        help=("Override the artifacts root. Default: "
              "example/cifar/cache/coreset/<tag>/."),
    )
    ap.add_argument(
        "--budget-k", type=int, default=500, dest="budget_k",
        help="Coreset size k -- number of random samples to draw.",
    )
    ap.add_argument(
        "--n-total", type=int, default=50000, dest="n_total",
        help="Population size (CIFAR-10 train is 50000).",
    )
    ap.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for the uniform draw.",
    )
    args = ap.parse_args()

    if args.artifacts_root is not None:
        artifacts_root = Path(args.artifacts_root)
    else:
        artifacts_root = REPO_ROOT / "example" / "cifar" / "cache" / "coreset" / args.tag

    algo_dir = build_random_coreset(
        artifacts_root=artifacts_root,
        n_total=args.n_total,
        budget_k=args.budget_k,
        seed=args.seed,
    )
    print(f"[random] wrote {algo_dir}/indices.pt  "
          f"(k={args.budget_k}, N={args.n_total}, seed={args.seed})")


if __name__ == "__main__":
    main()
