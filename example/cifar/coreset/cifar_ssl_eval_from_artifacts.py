"""Evaluate LeJEPA-recon SSL coreset runs by running their trained linear probe on test.

Iterates over ``<artifacts_root>/<algo>/`` for every algorithm with an
``indices.pt`` and looks up the Lightning ``last.ckpt`` written by
``cifar_ssl_train_from_artifacts.py`` at:

    example/cifar/logs/ssl_coreset_<tag>_<algo>_s<seed>/checkpoints/last.ckpt

The SSL training script attaches an ``spt.callbacks.OnlineProbe``
(`name="linear_probe"`, a ``nn.Linear(EMB_DIM, 10)``) which is trained
online during pretraining and persisted inside ``last.ckpt`` under
``state_dict["callbacks_modules.linear_probe.{weight,bias}"]``. This
script lifts that trained probe directly — no fresh probe is trained at
eval time — and reports its CIFAR-10 test top-1.

If the checkpoint is missing the algorithm is skipped with a log
message.

Outputs:
    example/out/coreset/<tag>_ssl_coreset_eval_results.json
    example/out/coreset/<tag>_ssl_coreset_eval_linprobe.png

Usage:
    python example/cifar/coreset/cifar_ssl_eval_from_artifacts.py \\
        --artifacts example/cifar/cache/coreset/ssl_resnet18_recon
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import stable_pretraining as spt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import REPO_ROOT  # noqa: F401, E402

from cifar_data import load_cifar_test  # noqa: E402

from _artifacts import (  # noqa: E402
    artifacts_help, default_artifacts, resolve_artifacts,
)

ALGO_CHOICES = ["greedy", "leverage", "spectral_rank"]
DEFAULT_TAG = "ssl_resnet18_recon_ema"  # paired with cifar_ssl_train_from_artifacts.py


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


EMB_DIM = 512
N_CLASSES = 10
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
PROBE_PREFIX = "callbacks_modules.linear_probe."

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


def _ckpt_path_for(tag: str, algo: str, seed: int) -> Path:
    run_name = f"ssl_coreset_{tag}_{algo}_s{seed}"
    return LOG_DIR / run_name / "checkpoints" / "last.ckpt"


def _load_backbone_and_probe(ckpt_path: Path, device: str) -> tuple[nn.Module, nn.Linear]:
    """Reconstruct (backbone, trained linear probe) from a Lightning checkpoint.

    The SSL Module's state_dict prefixes keys with the submodule name, so
    backbone weights live under ``backbone.*`` and the OnlineProbe's
    ``nn.Linear(512, 10)`` lives under ``callbacks_modules.linear_probe.*``.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    full_state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    backbone_state = {
        k.removeprefix("backbone."): v
        for k, v in full_state.items() if k.startswith("backbone.")
    }
    if not backbone_state:
        raise RuntimeError(f"no backbone.* keys in {ckpt_path}")
    backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)
    backbone.fc = nn.Identity()
    backbone.load_state_dict(backbone_state)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    probe_state = {
        k.removeprefix(PROBE_PREFIX): v
        for k, v in full_state.items() if k.startswith(PROBE_PREFIX)
    }
    if not probe_state or "weight" not in probe_state or "bias" not in probe_state:
        raise RuntimeError(
            f"no '{PROBE_PREFIX}weight/bias' in {ckpt_path}; "
            f"was the OnlineProbe callback attached at training time?"
        )
    w = probe_state["weight"]
    if w.shape != (N_CLASSES, EMB_DIM):
        raise RuntimeError(
            f"probe weight shape {tuple(w.shape)} != expected "
            f"({N_CLASSES}, {EMB_DIM}) in {ckpt_path}"
        )
    probe = nn.Linear(EMB_DIM, N_CLASSES)
    probe.load_state_dict(probe_state)
    probe.eval()
    for p in probe.parameters():
        p.requires_grad_(False)

    return backbone.to(device), probe.to(device)


@torch.no_grad()
def _eval_linprobe(backbone: nn.Module, probe: nn.Linear, X: np.ndarray,
                   y: np.ndarray, *, batch_size: int, device: str) -> float:
    """Test top-1 of ``probe(backbone(normalize(x)))``."""
    mean = CIFAR_MEAN.to(device); std = CIFAR_STD.to(device)
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)
    N = X_t.shape[0]
    correct = 0
    for i in range(0, N, batch_size):
        chunk = X_t[i:i + batch_size].to(device)
        chunk = (chunk - mean) / std
        logits = probe(backbone(chunk))
        correct += int((logits.argmax(-1) == y_t[i:i + batch_size]).sum())
    return correct / N


def _bar_plot(results: list[dict], full_acc: float | None,
              out_path: Path, tag: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.arange(len(results))
    accs = [r["linear_probe_top1"] * 100 for r in results]
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
    ax.set_ylabel("test top-1 (%)")
    ax.set_title(f"{tag}: SSL trained linear probe on test (per algorithm)")
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
    g.add_argument("--seed", type=int, default=0,
                   help="must match the seed used at SSL train time (run-name lookup)")

    g = ap.add_argument_group("eval")
    g.add_argument("--batch",    type=int,   default=512)
    g.add_argument("--full-acc", type=float, default=None, dest="full_acc",
                   help="50k-train reference accuracy, drawn as a dashed line")

    args = ap.parse_args()

    art = args.artifacts  # already a validated Path
    tag = art.name
    algos = args.algorithms or _discover_algos(art)
    if not algos:
        raise RuntimeError(f"no <algo>/indices.pt under {art}")

    out_dir = REPO_ROOT / "example" / "out" / "coreset"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{tag}_ssl_coreset_eval_results.json"
    plot_path = out_dir / f"{tag}_ssl_coreset_eval_linprobe.png"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    X_te, y_te = load_cifar_test("cifar10")

    results: list[dict] = []
    skipped: list[str] = []
    for algo in algos:
        ckpt_path = _ckpt_path_for(tag, algo, args.seed)
        if not ckpt_path.exists():
            print(f"[skip] {algo}: no checkpoint at {ckpt_path}")
            skipped.append(algo)
            continue

        idx_path = art / algo / "indices.pt"
        budget = int(torch.load(idx_path).numel()) if idx_path.exists() else -1
        print(f"\n=== {algo}  k={budget}  ({ckpt_path}) ===")
        backbone, probe = _load_backbone_and_probe(ckpt_path, device)
        acc = _eval_linprobe(backbone, probe, X_te, y_te,
                             batch_size=args.batch, device=device)
        print(f"  linear probe top-1 = {acc*100:.2f}%")
        results.append({
            "algorithm": algo,
            "budget": budget,
            "linear_probe_top1": float(acc),
            "seed": args.seed,
            "checkpoint": str(ckpt_path),
        })
        del backbone, probe
        if device == "cuda":
            torch.cuda.empty_cache()

    if not results:
        print("no evaluated algorithms (all checkpoints missing). exiting.")
        return

    with open(results_path, "w") as f:
        json.dump({"tag": tag, "results": results, "skipped": skipped},
                  f, indent=2)
    print(f"\nresults -> {results_path}")

    print("\n=== summary ===")
    print(f"  {'algorithm':<18} {'budget':>8} {'linprobe':>10}")
    for r in results:
        print(f"  {r['algorithm']:<18} {r['budget']:>8} "
              f"{r['linear_probe_top1']*100:>9.2f}%")
    if skipped:
        print(f"  (skipped: {', '.join(skipped)})")

    _bar_plot(results, args.full_acc, plot_path, tag)
    print(f"plot -> {plot_path}")


if __name__ == "__main__":
    main()
