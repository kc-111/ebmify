"""Compare feature-preprocessing treatments under linear leverage.

For each backbone (supervised ResNet18, SSL LeJEPA ResNet18, DINOv2 ViT-B/14,
VAE encoder mu) and each preprocessing treatment, build the Gram on the
*preprocessed* cifar10 train features and report AUROC of the linear
leverage score `phi = z` against several probe distributions.

Treatments (all using train statistics):
  - raw          : z
  - L2           : z / ||z||
  - centered     : z - mu_train
  - centered+L2  : (z - mu_train) / ||z - mu_train||

Probe distributions: cifar10 test (memorization signal), cifar100
(semantic OOD), Gaussian pixel noise (low-level OOD), inverted natural
images (mid-level OOD). Same indices as `mnist_vae_langevin.build_ood_x_sources`.

Output:
  - per-(backbone, treatment) row of AUROCs and a few geometry summaries
  - one figure: 4-panel grouped bar chart, one panel per backbone, four
    grouped bars per probe source

Usage:
    python example/cifar/diagnostics/cifar_centering_comparison.py
    python example/cifar/diagnostics/cifar_centering_comparison.py --backbones resnet18 vae
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
from cifar_memorization_scan import BACKBONE_STYLE, make_encode_fn  # noqa: E402
from cifar_resnet18_ood_threshold import auroc  # noqa: E402
from mnist_vae_langevin import build_ood_x_sources, build_phi_leverage  # noqa: E402


TREATMENTS = ["raw", "L2", "centered", "centered+L2"]
PROBE_NAMES = ["cifar10 test", "cifar100", "Gaussian", "inverted"]


def _l2(z: torch.Tensor) -> torch.Tensor:
    return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def apply_treatment(z: torch.Tensor, treatment: str,
                    mu_train: torch.Tensor) -> torch.Tensor:
    if treatment == "raw":
        return z
    if treatment == "L2":
        return _l2(z)
    if treatment == "centered":
        return z - mu_train
    if treatment == "centered+L2":
        return _l2(z - mu_train)
    raise ValueError(treatment)


def evaluate_backbone(name: str, encode_fn, z_dim: int,
                      X_tr_np: np.ndarray, X_te_np: np.ndarray,
                      X_other_np: np.ndarray, device: str, *,
                      n_train: int, n_eval: int, ridge: float, seed: int,
                      ) -> dict:
    rng = np.random.default_rng(0)
    tr_idx = rng.choice(len(X_tr_np), size=n_train, replace=False)
    te_idx = rng.choice(len(X_te_np), size=min(n_eval, len(X_te_np)),
                        replace=False)
    ot_idx = rng.choice(len(X_other_np), size=n_eval, replace=False)
    x_tr = torch.as_tensor(X_tr_np[tr_idx], dtype=torch.float32, device=device)
    x_te = torch.as_tensor(X_te_np[te_idx], dtype=torch.float32, device=device)
    x_ot = torch.as_tensor(X_other_np[ot_idx], dtype=torch.float32, device=device)

    base = build_ood_x_sources(x_tr, device, n_eval=n_eval, seed=seed,
                               in_name="cifar10")
    x_gauss = base[3][1]
    x_inv = base[5][1]

    print(f"  encoding train ({n_train}), test ({len(te_idx)}), "
          f"cifar100 ({n_eval}), Gaussian ({n_eval}), inverted ({n_eval}) ...")
    Z_tr = encode_fn(x_tr)
    probes_raw = {
        "cifar10 test": encode_fn(x_te),
        "cifar100":     encode_fn(x_ot),
        "Gaussian":     encode_fn(x_gauss),
        "inverted":     encode_fn(x_inv),
    }
    mu_train = Z_tr.mean(dim=0, keepdim=True)

    # Geometry summary (under raw z): median norms.
    raw_norms = {"train": Z_tr.norm(dim=1).median().item()}
    for k, v in probes_raw.items():
        raw_norms[k] = v.norm(dim=1).median().item()
    print("  raw ||z|| medians:  " + "  ".join(
        f"{k}={raw_norms[k]:.3f}" for k in ["train", *PROBE_NAMES]
    ))

    rows = {}  # treatment -> dict of source -> AUROC
    geo = {}   # treatment -> "norm" / "mu_norm" summary
    for treat in TREATMENTS:
        Zt = apply_treatment(Z_tr, treat, mu_train)
        probes_t = {k: apply_treatment(v, treat, mu_train)
                    for k, v in probes_raw.items()}

        h_fn, h_char, _ = build_phi_leverage(
            lambda z: z, Zt, ridge=ridge,
        )
        with torch.no_grad():
            h_tr = h_fn(Zt).cpu().numpy()
        au = {}
        for src, Zp in probes_t.items():
            with torch.no_grad():
                h_p = h_fn(Zp).cpu().numpy()
            au[src] = auroc(h_p, h_tr)
        rows[treat] = au
        geo[treat] = dict(
            train_norm=float(Zt.norm(dim=1).median().item()),
            mu_norm=float(Zt.mean(dim=0).norm().item()),
            dof_per_n=float(h_tr.sum()) / n_train,
        )
        print(f"  {treat:<13}  "
              + "  ".join(f"{s}={au[s]:.3f}" for s in PROBE_NAMES)
              + f"   |  train ||z||={geo[treat]['train_norm']:.3f}  "
              f"||mu||={geo[treat]['mu_norm']:.3e}  "
              f"dof/n={geo[treat]['dof_per_n']:.3f}")
    return dict(rows=rows, geo=geo, raw_norms=raw_norms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+",
                    default=["resnet18", "ssl", "dinov2", "vae"],
                    choices=list(BACKBONE_STYLE.keys()))
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--n-train", type=int, default=8192, dest="n_train")
    ap.add_argument("--n-eval", type=int, default=2048, dest="n_eval")
    ap.add_argument("--batch", type=int, default=256, dest="batch_size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resnet-tag", type=str, default="")
    ap.add_argument("--ssl-tag", type=str, default="recon")
    ap.add_argument("--dinov2-variant", type=str, default="dinov2_vitb14")
    ap.add_argument("--dinov2-dtype", choices=["fp32", "bf16", "fp16"],
                    default="fp32")
    ap.add_argument("--vae-z", type=int, default=256)
    ap.add_argument("--vae-beta", type=float, default=1.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    X_tr_np, _ = load_cifar_train("cifar10")
    X_te_np, _ = load_cifar_test("cifar10")
    X_other_np, _ = load_cifar_train("cifar100")

    results: dict[str, dict] = {}
    for name in args.backbones:
        print(f"\n=== {name} ===")
        try:
            encode_fn, z_dim, model_obj = make_encode_fn(name, device, args)
        except FileNotFoundError as e:
            print(f"  skipped: {e}")
            continue
        print(f"  z_dim = {z_dim}")
        results[name] = evaluate_backbone(
            name, encode_fn, z_dim, X_tr_np, X_te_np, X_other_np, device,
            n_train=args.n_train, n_eval=args.n_eval, ridge=args.ridge,
            seed=args.seed,
        )
        del encode_fn, model_obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not results:
        print("\nno results to plot.")
        return

    # Summary table.
    print("\n=== AUROC summary (linear phi=z under each treatment) ===")
    hdr = f"  {'backbone':<22} {'treatment':<14} " + " ".join(
        f"{s:>12}" for s in PROBE_NAMES
    )
    print(hdr)
    for name, res in results.items():
        for treat in TREATMENTS:
            au = res["rows"][treat]
            print(f"  {BACKBONE_STYLE[name][1]:<22} {treat:<14} "
                  + " ".join(f"{au[s]:>12.3f}" for s in PROBE_NAMES))
        print()

    # Plot: 2x2 grid of grouped bars, one panel per backbone.
    n_bb = len(results)
    ncols = 2
    nrows = (n_bb + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows),
                             squeeze=False)
    x = np.arange(len(PROBE_NAMES))
    width = 0.2
    treat_colors = {
        "raw":         "#999999",
        "L2":          "#1f77b4",
        "centered":    "#ff7f0e",
        "centered+L2": "#2ca02c",
    }
    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx // ncols][idx % ncols]
        for i, treat in enumerate(TREATMENTS):
            vals = [res["rows"][treat][s] for s in PROBE_NAMES]
            ax.bar(x + (i - 1.5) * width, vals, width=width,
                   color=treat_colors[treat], edgecolor="black", lw=0.4,
                   label=treat)
        ax.axhline(0.5, color="black", lw=0.7, ls="-", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(PROBE_NAMES, fontsize=9)
        ax.set_ylabel("AUROC vs cifar10 train")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(BACKBONE_STYLE[name][1], fontsize=10)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=8, loc="lower right", ncols=2)
    for j in range(n_bb, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(
        "Linear leverage AUROC across preprocessing treatments  "
        f"(Gram = cifar10 train, n={args.n_train}, ridge={args.ridge})",
        fontsize=12,
    )
    fig.tight_layout()
    out = REPO_ROOT / "example" / "out" / "cifar" / "cifar10_centering_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
