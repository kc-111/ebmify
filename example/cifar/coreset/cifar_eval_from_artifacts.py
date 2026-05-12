"""Evaluate supervised ResNet18-CIFAR models trained per coreset algorithm.

Iterates over ``<artifacts_root>/<algo>/`` for every algorithm with an
``indices.pt`` and looks up the trained checkpoint at
``example/cifar/cache/coreset_models/<tag>/<algo>/model.pt`` (produced
by ``cifar_train_from_artifacts.py``). If the checkpoint is present it
evaluates on the CIFAR-10 test set; if it is missing the algorithm is
skipped with a log message.

Outputs:
    example/out/coreset/<tag>_coreset_eval_results.json
    example/out/coreset/<tag>_coreset_eval_accuracy.png

Usage:
    python example/cifar/coreset/cifar_eval_from_artifacts.py \\
        --artifacts example/cifar/cache/coreset/supervised_resnet18
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import REPO_ROOT  # noqa: F401, E402

from cifar_data import load_cifar_test  # noqa: E402
from cifar_resnet18_train import evaluate, make_resnet18_cifar  # noqa: E402

from _artifacts import (  # noqa: E402
    artifacts_help, default_artifacts, resolve_artifacts,
)

ALGO_CHOICES = ["greedy", "leverage", "spectral_rank"]
DEFAULT_TAG = "supervised_resnet18"  # paired with cifar_train_from_artifacts.py


class _Fmt(argparse.ArgumentDefaultsHelpFormatter,
           argparse.RawDescriptionHelpFormatter):
    """Keeps the docstring intact + always shows '(default: X)' inline."""

    def _get_help_string(self, action):
        h = action.help or ""
        if (action.default is not argparse.SUPPRESS
                and action.default is not None
                and "%(default)" not in h
                and not action.required):
            h = (h + " " if h else "") + "(default: %(default)s)"
        return h


_ALGO_STYLE = {
    "greedy":        ("C3", "Greedy max-variance"),
    "leverage":      ("C0", "Ridge leverage sample"),
    "spectral_rank": ("C2", "Spectral-rank coverage"),
}


def _discover_algos(art: Path) -> list[str]:
    return sorted(
        d.name for d in art.iterdir()
        if d.is_dir() and (d / "indices.pt").exists()
    )


def _bar_plot(results: list[dict], full_acc: float | None,
              out_path: Path, tag: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.arange(len(results))
    accs = [r["test_acc"] * 100 for r in results]
    colors = [_ALGO_STYLE.get(r["algorithm"], ("C5", r["algorithm"]))[0]
              for r in results]
    ax.bar(xs, accs, color=colors)
    for x, a in zip(xs, accs):
        ax.text(x, a + 0.3, f"{a:.2f}", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{r['algorithm']}\nk={r['budget']}" for r in results],
        rotation=0, fontsize=9,
    )
    ax.set_ylabel("test accuracy (%)")
    ax.set_title(f"{tag}: standalone test eval per coreset algorithm")
    if full_acc is not None:
        ax.axhline(full_acc * 100, color="gray", ls="--", lw=0.8,
                   label=f"full 50k = {full_acc * 100:.2f}%")
        ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=_Fmt)

    g = ap.add_argument_group("data")
    _art_default, _art_required = default_artifacts(DEFAULT_TAG)
    g.add_argument("--artifacts", default=_art_default, required=_art_required,
                   type=resolve_artifacts, metavar="TAG_OR_PATH",
                   help=artifacts_help(preferred=DEFAULT_TAG))
    g.add_argument("--algorithms", nargs="+", default=None,
                   choices=ALGO_CHOICES, metavar="ALGO",
                   help=("restrict to these algos (default: all under --artifacts). "
                         "Choices: "
                         "greedy=greedy max-variance, "
                         "leverage=ridge leverage sampling, "
                         "spectral_rank=greedy max-coverage on rank strata"))
    g.add_argument("--models-root", default=None, dest="models_root",
                   help="override coreset_models/<tag>/ lookup path")

    g = ap.add_argument_group("eval")
    g.add_argument("--batch",    type=int,   default=256)
    g.add_argument("--full-acc", type=float, default=None, dest="full_acc",
                   help="50k-train reference accuracy, drawn as a dashed line")

    args = ap.parse_args()

    art = args.artifacts  # already a validated Path
    tag = art.name
    algos = args.algorithms or _discover_algos(art)
    if not algos:
        raise RuntimeError(f"no <algo>/indices.pt under {art}")

    models_root = (Path(args.models_root) if args.models_root else
                   REPO_ROOT / "example" / "cifar" / "cache"
                   / "coreset_models" / tag)
    out_dir = REPO_ROOT / "example" / "out" / "coreset"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{tag}_coreset_eval_results.json"
    plot_path = out_dir / f"{tag}_coreset_eval_accuracy.png"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    X_te, y_te = load_cifar_test("cifar10")
    X_te_t = torch.as_tensor(X_te, dtype=torch.float32, device=device)
    y_te_t = torch.as_tensor(y_te, dtype=torch.long, device=device)

    results: list[dict] = []
    skipped: list[str] = []
    for algo in algos:
        model_path = models_root / algo / "model.pt"
        if not model_path.exists():
            print(f"[skip] {algo}: no checkpoint at {model_path}")
            skipped.append(algo)
            continue
        idx_path = art / algo / "indices.pt"
        budget = int(torch.load(idx_path).numel()) if idx_path.exists() else -1
        raw = torch.load(model_path, map_location=device, weights_only=False)
        cfg = raw.get("config", {})
        model = make_resnet18_cifar(num_classes=int(cfg.get("num_classes", 10)))
        model.load_state_dict(raw["state_dict"])
        model.to(device)
        acc = evaluate(model, X_te_t, y_te_t, batch_size=args.batch, device=device)
        print(f"[{algo}]  k={budget}  test_acc={acc*100:.2f}%  "
              f"(train-time best={raw.get('best_acc', float('nan'))*100:.2f}%)")
        results.append({
            "algorithm": algo,
            "budget": budget,
            "test_acc": float(acc),
            "train_best_acc": float(raw.get("best_acc", float("nan"))),
            "train_final_acc": float(raw.get("final_acc", float("nan"))),
            "epochs": raw.get("epochs"),
            "seed": raw.get("seed"),
            "checkpoint": str(model_path),
        })

    if not results:
        print("no evaluated algorithms (all checkpoints missing). exiting.")
        return

    with open(results_path, "w") as f:
        json.dump({"tag": tag, "results": results, "skipped": skipped},
                  f, indent=2)
    print(f"\nresults -> {results_path}")

    print("\n=== summary ===")
    print(f"  {'algorithm':<18} {'budget':>8} {'test':>9}")
    for r in results:
        print(f"  {r['algorithm']:<18} {r['budget']:>8} "
              f"{r['test_acc']*100:>8.2f}%")
    if skipped:
        print(f"  (skipped: {', '.join(skipped)})")

    _bar_plot(results, args.full_acc, plot_path, tag)
    print(f"plot -> {plot_path}")


if __name__ == "__main__":
    main()
