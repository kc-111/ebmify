"""Test whether concatenating VAE and LeJEPA features recovers
BOTH pixel-statistics OOD (Gaussian noise) AND semantic OOD (cifar100).

Hypothesis: VAE alone catches noise but misses cifar100; LeJEPA alone
catches cifar100 but misses noise. If linear leverage on
phi(x) = [phi_vae(x) ; phi_lejepa(x)] catches both, the trade-off was a
feature-coverage problem, not a training-objective problem -- joint
training is unnecessary, you just deploy a concatenated-feature head at
inference.

The two pieces have different natural norms (VAE mu ~ 12, LeJEPA ~ 6),
so we normalize *per piece* before concatenating, otherwise the larger
piece dominates the leverage Gram. Treatments tested per piece:
  - block-L2          : each piece L2-normalized, then concat
  - block-centered+L2 : each piece (z - mu_piece) / ||z - mu_piece||,
                        then concat

A "raw concat" baseline is included to show how much the per-piece
preprocessing matters.

For comparison, we also run the same treatments on each backbone alone
so the concat row can be read directly against its components.

Probes: cifar10 test (memorization), cifar100 (semantic OOD), Gaussian
noise (pixel-stat OOD), inverted (mid-level OOD).

Usage:
    python example/cifar/diagnostics/cifar_concat_features_test.py
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import REPO_ROOT  # noqa: E402

from cifar_data import load_cifar_test, load_cifar_train  # noqa: E402
from cifar_memorization_scan import make_encode_fn  # noqa: E402
from cifar_resnet18_ood_threshold import auroc  # noqa: E402
from mnist_vae_langevin import build_ood_x_sources, build_phi_leverage  # noqa: E402


PROBE_NAMES = ["cifar10 test", "cifar100", "Gaussian", "inverted"]
TREATMENTS = ["raw concat", "block-L2", "block-centered+L2"]


def _l2(z: torch.Tensor) -> torch.Tensor:
    return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def apply_block(z: torch.Tensor, treatment: str,
                mu_train: torch.Tensor) -> torch.Tensor:
    """Apply per-block treatment. mu_train must be the train mean of THIS block."""
    if treatment == "raw":
        return z
    if treatment == "L2":
        return _l2(z)
    if treatment == "centered":
        return z - mu_train
    if treatment == "centered+L2":
        return _l2(z - mu_train)
    raise ValueError(treatment)


def encode_all_sources(name: str, args, X_tr_t, X_te_t, X_ot_t,
                       x_gauss, x_inv, device) -> dict:
    """Encode train + 4 probes through one backbone. Returns dict of features."""
    encode_fn, z_dim, model_obj = make_encode_fn(name, device, args)
    print(f"  [{name}] z_dim = {z_dim}, encoding ...")
    feats = {
        "train":        encode_fn(X_tr_t),
        "cifar10 test": encode_fn(X_te_t),
        "cifar100":     encode_fn(X_ot_t),
        "Gaussian":     encode_fn(x_gauss),
        "inverted":     encode_fn(x_inv),
    }
    print(f"  [{name}] ||z|| medians:  "
          + "  ".join(f"{k}={v.norm(dim=1).median().item():.3f}"
                      for k, v in feats.items()))
    del encode_fn, model_obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return feats


def run_treatment(feats_blocks: list[dict], block_treatment: str, *,
                  ridge: float, n_train: int) -> dict:
    """
    feats_blocks: list of dicts {name: features_tensor}, one per piece.
    block_treatment: "raw", "L2", "centered", "centered+L2".
    Returns AUROC per probe + dof/n + dim.
    """
    keys = ["train", *PROBE_NAMES]
    # Compute per-block train means.
    mus = [b["train"].mean(dim=0, keepdim=True) for b in feats_blocks]
    # Apply per-block treatment, then concat.
    processed = {}
    for k in keys:
        pieces = [apply_block(b[k], block_treatment, mu) for b, mu in zip(feats_blocks, mus)]
        processed[k] = torch.cat(pieces, dim=-1)
    Zt = processed["train"]
    h_fn, h_char, D = build_phi_leverage(lambda z: z, Zt, ridge=ridge)
    with torch.no_grad():
        h_tr = h_fn(Zt).cpu().numpy()
    au = {}
    for s in PROBE_NAMES:
        with torch.no_grad():
            h_p = h_fn(processed[s]).cpu().numpy()
        au[s] = auroc(h_p, h_tr)
    return dict(
        au=au, dof_per_n=float(h_tr.sum()) / n_train,
        D=D, dim=Zt.shape[1],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", nargs="+", default=["vae", "ssl"],
                    help="Backbones to concatenate (also reported individually).")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--n-train", type=int, default=8192, dest="n_train")
    ap.add_argument("--n-eval", type=int, default=2048, dest="n_eval")
    ap.add_argument("--batch", type=int, default=256, dest="batch_size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resnet-tag", type=str, default="")
    ap.add_argument("--ssl-tag", type=str, default="")
    ap.add_argument("--dinov2-variant", type=str, default="dinov2_vitb14")
    ap.add_argument("--dinov2-dtype", choices=["fp32", "bf16", "fp16"],
                    default="fp32")
    ap.add_argument("--vae-z", type=int, default=256)
    ap.add_argument("--vae-beta", type=float, default=1.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}  pieces = {args.pieces}")

    X_tr_np, _ = load_cifar_train("cifar10")
    X_te_np, _ = load_cifar_test("cifar10")
    X_other_np, _ = load_cifar_train("cifar100")

    rng = np.random.default_rng(0)
    tr_idx = rng.choice(len(X_tr_np), size=args.n_train, replace=False)
    te_idx = rng.choice(len(X_te_np), size=min(args.n_eval, len(X_te_np)),
                        replace=False)
    ot_idx = rng.choice(len(X_other_np), size=args.n_eval, replace=False)
    X_tr_t = torch.as_tensor(X_tr_np[tr_idx], dtype=torch.float32, device=device)
    X_te_t = torch.as_tensor(X_te_np[te_idx], dtype=torch.float32, device=device)
    X_ot_t = torch.as_tensor(X_other_np[ot_idx], dtype=torch.float32, device=device)
    base = build_ood_x_sources(X_tr_t, device, n_eval=args.n_eval,
                               seed=args.seed, in_name="cifar10")
    x_gauss, x_inv = base[3][1], base[5][1]

    # Encode through each piece backbone.
    feats_per_piece = []
    for piece in args.pieces:
        print(f"\n=== encoding piece: {piece} ===")
        feats_per_piece.append(
            encode_all_sources(piece, args, X_tr_t, X_te_t, X_ot_t,
                               x_gauss, x_inv, device)
        )

    # --- Individual-backbone references under each block treatment -----
    print("\n=== individual backbones (block treatment applied to single piece) ===")
    individual: dict[str, dict[str, dict]] = {}
    for piece, fb in zip(args.pieces, feats_per_piece):
        individual[piece] = {}
        for block_t in ["raw", "L2", "centered+L2"]:
            res = run_treatment([fb], block_t,
                                ridge=args.ridge, n_train=args.n_train)
            individual[piece][block_t] = res
            print(f"  {piece:<6} {block_t:<14}  "
                  + "  ".join(f"{s}={res['au'][s]:.3f}" for s in PROBE_NAMES)
                  + f"   | D={res['D']}  dof/n={res['dof_per_n']:.3f}")

    # --- Concat -------------------------------------------------------
    concat_label = "+".join(args.pieces)
    print(f"\n=== concat: {concat_label} ===")
    concat_results: dict[str, dict] = {}
    treatment_to_block = {
        "raw concat":         "raw",
        "block-L2":           "L2",
        "block-centered+L2":  "centered+L2",
    }
    for tname, block_t in treatment_to_block.items():
        res = run_treatment(feats_per_piece, block_t,
                            ridge=args.ridge, n_train=args.n_train)
        concat_results[tname] = res
        print(f"  {tname:<20}  "
              + "  ".join(f"{s}={res['au'][s]:.3f}" for s in PROBE_NAMES)
              + f"   | D={res['D']}  dof/n={res['dof_per_n']:.3f}")

    # --- Summary table -------------------------------------------------
    print("\n=== summary: AUROC (linear phi=z) under matched treatment ===")
    print(f"  {'rep':<28} {'treatment':<18} "
          + " ".join(f"{s:>13}" for s in PROBE_NAMES))
    for piece in args.pieces:
        for block_t in ["raw", "L2", "centered+L2"]:
            au = individual[piece][block_t]["au"]
            print(f"  {piece:<28} {block_t:<18} "
                  + " ".join(f"{au[s]:>13.3f}" for s in PROBE_NAMES))
    for tname, res in concat_results.items():
        print(f"  {concat_label:<28} {tname:<18} "
              + " ".join(f"{res['au'][s]:>13.3f}" for s in PROBE_NAMES))

    # --- Plot ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 5.5))
    # We'll show three treatments: individual pieces (block-centered+L2),
    # and concat (block-centered+L2). Plus concat block-L2 for ref.
    bar_specs: list[tuple[str, dict, str]] = []
    palette = ["#7f7f7f", "#9467bd", "#1f77b4", "#2ca02c", "#d62728"]
    color_iter = iter(palette)
    for piece in args.pieces:
        bar_specs.append(
            (f"{piece}  (centered+L2)",
             individual[piece]["centered+L2"]["au"],
             next(color_iter))
        )
    bar_specs.append(
        (f"{concat_label}  (block-L2)",
         concat_results["block-L2"]["au"],
         next(color_iter))
    )
    bar_specs.append(
        (f"{concat_label}  (block-centered+L2)",
         concat_results["block-centered+L2"]["au"],
         next(color_iter))
    )
    n_bars = len(bar_specs)
    x = np.arange(len(PROBE_NAMES))
    width = 0.8 / n_bars
    for i, (label, au, color) in enumerate(bar_specs):
        vals = [au[s] for s in PROBE_NAMES]
        ax.bar(x + (i - (n_bars - 1) / 2) * width, vals, width=width,
               color=color, edgecolor="black", lw=0.4, label=label)
    ax.axhline(0.5, color="black", lw=0.7, ls="-", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(PROBE_NAMES)
    ax.set_ylabel("AUROC vs cifar10 train")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=9, loc="lower right", ncols=1)
    ax.set_title(
        "Concat-features test: can a hybrid representation get both "
        "pixel-stat (Gaussian) AND semantic (cifar100) OOD?\n"
        f"(linear phi=z, Gram = cifar10 train n={args.n_train}, "
        f"ridge={args.ridge})",
        fontsize=10,
    )
    fig.tight_layout()
    out = (REPO_ROOT / "example" / "out"
           / f"cifar10_concat_{concat_label}.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
