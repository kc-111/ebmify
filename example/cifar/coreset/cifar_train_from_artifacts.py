"""Train ResNet18-CIFAR from scratch on coresets produced by ``coreset.cli``.

Reads the artifact layout written by ``cifar_build_coreset_supervised.py``
or ``cifar_build_coreset_ssl.py``:

    <artifacts_root>/
        greedy/indices.pt
        leverage/indices.pt
        spectral_rank/indices.pt
        ...

Trains one supervised ResNet18-CIFAR per algorithm with the same recipe
as ``cifar_resnet18_train.py``. Writes:

- ``example/out/coreset/<tag>_coreset_train_results.json``
- ``example/out/coreset/<tag>_coreset_train_accuracy.png``
- ``example/cifar/cache/coreset_models/<tag>/<algo>/model.pt`` (one per
  algorithm, consumed by ``cifar_eval_from_artifacts.py``).

Usage:
    python example/cifar/coreset/cifar_build_coreset_supervised.py
    python example/cifar/coreset/cifar_train_from_artifacts.py \\
        --artifacts example/cifar/cache/coreset/supervised_resnet18 \\
        --epochs 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import REPO_ROOT  # noqa: F401, E402

from cifar_data import load_cifar_test, load_cifar_train  # noqa: E402
from cifar_resnet18_train import make_resnet18_cifar, train  # noqa: E402

from _artifacts import (  # noqa: E402
    artifacts_help, default_artifacts, resolve_artifacts,
)
from _aux_losses import (  # noqa: E402
    build_aux_heads, collect_aux_lambdas, discover_aux_targets,
)

ALGO_CHOICES = ["greedy", "leverage", "spectral_rank", "random"]
DEFAULT_TAG = "supervised_resnet18"  # written by cifar_build_coreset_supervised.py
RESNET18_EMB_DIM = 512  # pre-fc avgpool output for CIFAR ResNet18


class _Fmt(argparse.ArgumentDefaultsHelpFormatter,
           argparse.RawDescriptionHelpFormatter):
    """Keeps the docstring intact + always shows '(default: X)' inline,
    even for flags without an explicit help string."""

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
    "random":        ("C7", "Random baseline"),
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
    # Plot the EMA-best test accuracy: that's the checkpoint that gets
    # saved as ``state_dict`` and consumed by cifar_eval_from_artifacts.py,
    # so it's the number that matches what downstream eval will see.
    accs = [r["best_ema_acc"] * 100 for r in results]
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
    ax.set_ylabel("EMA test accuracy (%)")
    ax.set_title(f"{tag}: ResNet18-CIFAR trained on coreset (per algorithm)")
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

    # Which coreset bundle to train on. Accepts either a bare <tag>
    # (resolved against example/cifar/cache/coreset/) or a full path.
    # resolve_artifacts() validates that <algo>/indices.pt exists.
    # Defaults to DEFAULT_TAG if that tag exists on disk; otherwise the
    # flag becomes required with a helpful error listing available tags.
    _art_default, _art_required = default_artifacts(DEFAULT_TAG)
    g.add_argument("--artifacts", default=_art_default, required=_art_required,
                   type=resolve_artifacts, metavar="TAG_OR_PATH",
                   help=artifacts_help(preferred=DEFAULT_TAG))

    # Restrict which algorithms to train. By default we discover every
    # <algo>/indices.pt under --artifacts and train one ResNet18 per algo.
    # The library-supported choices are:
    #   greedy        -- greedy max-variance pick (diverse-but-noisy)
    #   leverage      -- ridge leverage-score sampling (importance weighted)
    #   spectral_rank -- greedy max-coverage on rank strata (uniform-ish)
    #   random        -- random baseline
    g.add_argument("--algorithms", nargs="+", 
                   default=["leverage"], #["spectral_rank", "random"],
                #    default=["random"],
                   choices=ALGO_CHOICES, metavar="ALGO",
                   help=("restrict to these algos (default: all under --artifacts). "
                         "Choices: "
                         "greedy=greedy max-variance, "
                         "leverage=ridge leverage sampling, "
                         "spectral_rank=greedy max-coverage on rank strata, "
                         "random=random baseline"))

    # Mirrors the SOTA recipe in example/cifar/train/cifar_resnet18_train.py
    # so coreset / full-50k numbers are directly comparable: SGD+Nesterov,
    # warmup-cosine, label smoothing, Mixup/CutMix, cutout, affine + color
    # jitter, EMA, bfloat16 AMP, channels_last. Defaults match upstream.
    g = ap.add_argument_group("training (SOTA recipe, mirrors cifar_resnet18_train.py)")

    # Epochs per algorithm.
    # 10 times reduction in data means 10 times more steps.
    g.add_argument("--epochs", type=int, default=2000) 

    # Linear LR warmup length (steps = warmup_epochs * steps_per_epoch).
    # Same as upstream; cosine anneal kicks in after warmup ends.
    g.add_argument("--warmup-epochs", type=int, default=5, dest="warmup_epochs")

    # Mini-batch size. 128 is canonical for CIFAR ResNet18.
    g.add_argument("--batch", type=int, default=128)

    # Initial SGD learning rate; cosine-annealed inside train(). 0.1 is
    # the canonical CIFAR ResNet18 lr.
    g.add_argument("--lr", type=float, default=0.1, help="SGD lr (warmup -> cosine)")

    # SGD momentum -- standard 0.9, rarely touch.
    g.add_argument("--momentum", type=float, default=0.9)

    # Decoupled L2 weight decay on conv/linear weights only (BN scales
    # and biases get 0). 5e-4 is the upstream default.
    g.add_argument("--weight-decay", type=float, default=5e-4, dest="weight_decay")

    # Label smoothing for soft cross-entropy. 0 disables (== hard CE).
    g.add_argument("--label-smoothing", type=float, default=0.1,
                   dest="label_smoothing")

    # Mixup Beta(alpha, alpha) coefficient (gentle blends near 0.5).
    g.add_argument("--mixup-alpha", type=float, default=0.2, dest="mixup_alpha")

    # CutMix Beta(alpha, alpha) coefficient (alpha=1 -> uniform lam).
    g.add_argument("--cutmix-alpha", type=float, default=1.0, dest="cutmix_alpha")

    # Probability of applying any mix per step (then 50/50 Mixup vs CutMix).
    g.add_argument("--mix-prob", type=float, default=0.8, dest="mix_prob")

    # Cutout square side in pixels (0 disables).
    g.add_argument("--cutout-size", type=int, default=16, dest="cutout_size")

    # Per-sample random affine: rotation amplitude (degrees) and translation
    # fraction. 0 disables the corresponding axis of the affine.
    g.add_argument("--affine-rot-deg", type=float, default=15.0,
                   dest="affine_rot_deg")
    g.add_argument("--affine-translate", type=float, default=0.05,
                   dest="affine_translate")

    # Per-sample color jitter strengths (each in [0, 1]; 0 disables).
    g.add_argument("--color-brightness", type=float, default=0.2,
                   dest="color_brightness")
    g.add_argument("--color-contrast", type=float, default=0.2,
                   dest="color_contrast")
    g.add_argument("--color-saturation", type=float, default=0.2,
                   dest="color_saturation")

    # EMA decay for the model snapshot used at eval / checkpoint time.
    # The saved ``state_dict`` is the EMA weights (better generalization).
    g.add_argument("--ema-decay", type=float, default=0.999, dest="ema_decay")

    # RNG seed for model init / batch order / data aug. Set explicitly
    # so per-algo numbers are reproducible; results vary by ~0.3-0.5%
    # across seeds at this budget.
    g.add_argument("--seed", type=int, default=0)

    # Auxiliary loss weights. Each one attaches a separate nn.Linear
    # head on the 512-D pre-fc feature (captured via a forward hook on
    # avgpool) and adds the head's per-target loss to the cross-entropy
    # loss with the given lambda. 0.0 disables that head entirely (no
    # parameters trained). Targets are loaded from
    # <artifacts>/<algo>/aux_<name>.pt and indexed by coreset position.
    #
    # Clean-vs-clean: when any aux is active the training loop runs a
    # second forward through the model per step on the un-augmented,
    # un-Mixup'd batch and feeds *that* clean embedding to every aux
    # head. Targets are the on-disk rows indexed by coreset position
    # -- no lam/perm blending. This keeps the heads aligned with the
    # clean encoder state the targets were precomputed against, at
    # the cost of one extra backbone forward per step.
    g_aux = ap.add_argument_group("auxiliary losses (lambda per aux target; 0=off)")

    # Regress projections onto the top eigenvectors of Phi^T Phi.
    # Loss is per-coordinate MSE weighted by w_j = 1/sqrt(sigma2[j] + lam),
    # so noisy trailing coords contribute less. Head: 512 -> n_top_eigvecs.
    g_aux.add_argument("--aux-spectral-coords", type=float, default=0.0,
                       dest="aux_spectral_coords",
                       help="weight for spectral_coords aux (per-coord ridge-weighted MSE; 0=off)")

    # Regress per-bucket uniform ranks in [0, 1] (one rank per bucket).
    # Encourages the embedding to know its rank-position within every
    # bucket of equal-mass eigenvectors. Head: 512 -> n_buckets.
    g_aux.add_argument("--aux-bucket-ranks", type=float, default=0.0,
                       dest="aux_bucket_ranks",
                       help="weight for bucket_ranks aux (per-bucket uniform-rank MSE; 0=off)")

    # Regress scalar ridge leverage h_i; uniformly upweights samples that
    # carry unique information about the Gram matrix. Head: 512 -> 1.
    g_aux.add_argument("--aux-leverage-score", type=float, default=0.0,
                       dest="aux_leverage_score",
                       help="weight for leverage_score aux (scalar leverage MSE; 0=off)")

    # Classify each sample's home-bucket id b*_i = argmax_b S_{i,b}.
    # Acts like a coarse cluster label derived from the spectrum.
    # Head: 512 -> n_buckets, cross-entropy.
    g_aux.add_argument("--aux-home-bucket", type=float, default=0.0,
                       dest="aux_home_bucket",
                       help="weight for home_bucket aux (cross-entropy over bucket id; 0=off)")

    # In-batch kernel distillation from the encoder that built the
    # coreset: match the cosine Gram of the student head's projection
    # against the cosine Gram of the on-disk teacher features phi_i.
    # Rotation-invariant on both sides; loss is MSE over the (B,B)
    # Gram. Like the other aux heads, this rides on the shared
    # clean-vs-clean forward (see group preamble), so both sides of
    # the Gram are evaluated on clean inputs. Head: 512 -> D (the
    # kernel itself is never materialized on disk).
    g_aux.add_argument("--aux-feature-distill", type=float, default=0.0,
                       dest="aux_feature_distill",
                       help="weight for feature_distill aux (cosine-Gram MSE vs teacher phi_i; 0=off)")

    g = ap.add_argument_group("plot")

    # Optional reference line on the bar plot (e.g. your 50k-train
    # accuracy). Pass as a fraction in [0, 1], not a percentage.
    # Default None = no reference line drawn.
    g.add_argument("--full-acc", type=float, default=None, dest="full_acc",
                   help="50k-train reference accuracy, drawn as a dashed line")

    args = ap.parse_args()

    art = args.artifacts  # already a validated Path (resolve_artifacts)
    tag = art.name
    algos = args.algorithms or _discover_algos(art)
    if not algos:
        raise RuntimeError(f"no <algo>/indices.pt under {art}")

    out_dir = REPO_ROOT / "example" / "out" / "coreset"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{tag}_coreset_train_results.json"
    plot_path = out_dir / f"{tag}_coreset_train_accuracy.png"
    models_root = REPO_ROOT / "example" / "cifar" / "cache" / "coreset_models" / tag

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    X_tr, y_tr = load_cifar_train("cifar10")
    X_te, y_te = load_cifar_test("cifar10")
    X_te_t = torch.as_tensor(X_te, dtype=torch.float32, device=device)
    y_te_t = torch.as_tensor(y_te, dtype=torch.long, device=device)

    # Per-aux loss weights collected once for the whole run (same weights
    # applied to every algorithm). 0.0 means "head not attached".
    aux_lambdas = collect_aux_lambdas(args)
    any_aux_on = any(v > 0 for v in aux_lambdas.values())

    results: list[dict] = []
    for algo in algos:
        idx = torch.load(art / algo / "indices.pt").cpu().numpy().astype(np.int64)
        counts = np.bincount(y_tr[idx], minlength=10).tolist()
        print(f"\n=== {algo}  ({len(idx)} samples) ===")
        print(f"  class counts: {counts}")

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = make_resnet18_cifar(num_classes=10).to(device)
        # channels_last layout matches the upstream recipe; bfloat16 AMP
        # + channels_last gives the ~2x throughput that funds 200 epochs.
        model = model.to(memory_format=torch.channels_last)

        # Discover whichever aux targets this algorithm has on disk and
        # build a matching ModuleDict of linear heads on the 512-D feature.
        # Targets are aligned to coreset positions [0, k), so train()
        # indexes them by the permutation index `b` and blends them by
        # the same lam/perm as the Mixup/CutMix label mix.
        aux_heads = None
        aux_targets: dict = {}
        aux_specs: dict = {}
        if any_aux_on:
            aux_targets, aux_specs = discover_aux_targets(art / algo)
            active = [n for n in aux_specs if aux_lambdas.get(n, 0.0) > 0]
            missing = [n for n, w in aux_lambdas.items()
                       if w > 0 and n not in aux_specs]
            if missing:
                print(f"  [aux] requested but missing on disk: {missing}")
            if active:
                print(f"  [aux] active heads: {active}  "
                      f"lambdas={ {n: aux_lambdas[n] for n in active} }")
                aux_heads = build_aux_heads(RESNET18_EMB_DIM, aux_specs)

        t0 = time.time()
        out = train(
            model, X_tr[idx], y_tr[idx], X_te_t, y_te_t, device,
            epochs=args.epochs, warmup_epochs=args.warmup_epochs,
            batch_size=args.batch, lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha,
            mix_prob=args.mix_prob, cutout_size=args.cutout_size,
            affine_rot_deg=args.affine_rot_deg,
            affine_translate=args.affine_translate,
            color_brightness=args.color_brightness,
            color_contrast=args.color_contrast,
            color_saturation=args.color_saturation,
            ema_decay=args.ema_decay,
            aux_heads=aux_heads, aux_targets=aux_targets,
            aux_specs=aux_specs, aux_lambdas=aux_lambdas,
        )
        results.append({
            "algorithm": algo,
            "budget": int(len(idx)),
            "best_acc": float(out["best_acc"]),
            "best_ema_acc": float(out["best_ema_acc"]),
            "final_ema_acc": float(out["final_ema_acc"]),
            "epochs": args.epochs,
            "seed": args.seed,
            "class_counts": counts,
            "aux_lambdas": aux_lambdas,
            "aux_active": sorted(aux_specs.keys()) if aux_heads is not None else [],
            "runtime_sec": time.time() - t0,
        })

        model_path = models_root / algo / "model.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # state_dict = EMA weights (better generalization; matches the
        # upstream save convention so cifar_eval_from_artifacts.py loads
        # the same thing the downstream OOD eval would). raw_state_dict
        # kept as a fallback for ablations.
        torch.save({
            "state_dict": out["ema_state"],
            "raw_state_dict": model.state_dict(),
            "config": {"num_classes": 10, "arch": "resnet18_cifar"},
            "best_acc": float(out["best_acc"]),
            "best_ema_acc": float(out["best_ema_acc"]),
            "final_ema_acc": float(out["final_ema_acc"]),
            "epochs": args.epochs,
            "seed": args.seed,
            "algorithm": algo,
            "tag": tag,
        }, model_path)
        print(f"  saved model -> {model_path}")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults -> {results_path}")

    print("\n=== summary ===")
    print(f"  {'algorithm':<18} {'budget':>8} {'raw_best':>10} "
          f"{'ema_best':>10} {'ema_final':>10}")
    for r in results:
        print(f"  {r['algorithm']:<18} {r['budget']:>8} "
              f"{r['best_acc']*100:>9.2f}%  "
              f"{r['best_ema_acc']*100:>9.2f}%  "
              f"{r['final_ema_acc']*100:>9.2f}%")

    _bar_plot(results, args.full_acc, plot_path, tag)
    print(f"plot -> {plot_path}")


if __name__ == "__main__":
    main()
