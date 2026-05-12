"""Bandwidth scan diagnostic for RFF leverage OOD on the resnet18 backbone.

Question this addresses: under --normalize, the ``cifar_resnet18_ood_threshold``
script's protocol-chosen ell* makes the RFF Gaussian kernel almost
constant on the unit sphere, so phi=RFF(z) AUROC for cifar100 falls
near 0.5 even though phi=z reaches 0.94. Does a different ell *exist*
that lets RFF match phi=z, or is the unit-sphere RFF kernel inherently
weaker than the linear kernel here? This sweeps ell over a wide log
range and plots AUROC(source vs cifar10 train) per OOD source for
phi = RFF(z), with the phi=z value shown as a horizontal reference.

Usage:
    python example/cifar/diagnostics/cifar_resnet18_bandwidth_scan.py --normalize
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import REPO_ROOT  # noqa: E402

from ebmify.models.fc import RFFLayer  # noqa: E402

from cifar_data import load_cifar_test, load_cifar_train  # noqa: E402
from cifar_resnet18_ood_threshold import (  # noqa: E402
    auroc, encode, load_trained_resnet18,
)
from cifar_resnet18_train import resnet18_ckpt_path  # noqa: E402
from mnist_vae_langevin import build_ood_x_sources, build_phi_leverage  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--M", type=int, default=2048, dest="M_rff")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--n-train", type=int, default=8192, dest="n_train")
    ap.add_argument("--n-eval", type=int, default=2048, dest="n_eval")
    ap.add_argument("--batch", type=int, default=512, dest="batch_size")
    ap.add_argument("--ell-min", type=float, default=0.01)
    ap.add_argument("--ell-max", type=float, default=500.0)
    ap.add_argument("--n-ell", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--normalize", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    ckpt_path = Path(args.ckpt) if args.ckpt else resnet18_ckpt_path(args.tag)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    print(f"loading ResNet18 from {ckpt_path} ...")
    model, z_dim = load_trained_resnet18(ckpt_path, device)

    dataset = "cifar10"
    other = "cifar100"
    X_tr, _ = load_cifar_train(dataset)
    X_other, _ = load_cifar_train(other)
    X_te, _ = load_cifar_test(dataset)

    rng = np.random.default_rng(0)
    sub_idx = rng.choice(len(X_tr), size=args.n_train, replace=False)
    X_sub_t = torch.as_tensor(X_tr[sub_idx], dtype=torch.float32, device=device)
    print(f"encoding {args.n_train} train images ...")
    Z_train = encode(model, X_sub_t,
                     batch_size=args.batch_size, device=device)

    base = build_ood_x_sources(
        X_sub_t, device, n_eval=args.n_eval, seed=args.seed,
        in_name=dataset,
    )
    cross_idx = rng.choice(len(X_other), size=args.n_eval, replace=False)
    x_cross = torch.as_tensor(
        X_other[cross_idx], dtype=torch.float32, device=device,
    )
    test_idx = rng.choice(len(X_te), size=min(args.n_eval, len(X_te)),
                          replace=False)
    x_test = torch.as_tensor(X_te[test_idx], dtype=torch.float32, device=device)

    # Curated subset for legibility: in-data, near-OOD (test, cifar100),
    # texture/structure-OOD (inverted), pure noise (Gaussian).
    x_sources = [
        ("cifar10",      base[0][1],  "C0"),
        ("cifar10 test", x_test,      "C9"),
        ("cifar100",     x_cross,     "C8"),
        # base layout: [(in,...), (uniform,...), (Bernoulli,...),
        #               (Gaussian,...), (shuffled,...), (inverted,...),
        #               (black,...), (white,...)]
        ("Gaussian",     base[3][1],  "C3"),
        ("inverted",     base[5][1],  "C5"),
    ]

    print("encoding sources ...")
    z_sources = []
    for name, x, color in x_sources:
        z = encode(model, x, batch_size=args.batch_size, device=device)
        z_sources.append((name, z, color))
        print(f"  {name:>14}: ||z|| median={z.norm(dim=1).median().item():.3f}")

    if args.normalize:
        def _l2(z: torch.Tensor) -> torch.Tensor:
            return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        Z_train = _l2(Z_train)
        z_sources = [(n, _l2(z), c) for n, z, c in z_sources]
        print("\n[normalize] features L2-normalized to unit norm.")

    # --- Linear (phi=z) reference AUROC --------------------------------
    print("\nphi=z reference AUROC:")
    h_fn_lin, h_char_lin, _ = build_phi_leverage(lambda z: z, Z_train,
                                                  ridge=args.ridge)
    with torch.no_grad():
        h_tr_lin = h_fn_lin(Z_train).cpu().numpy() / h_char_lin
    lin_aurocs = {}
    for name, z, _ in z_sources:
        with torch.no_grad():
            h = h_fn_lin(z).cpu().numpy() / h_char_lin
        lin_aurocs[name] = auroc(h, h_tr_lin)
        print(f"  {name:>14}: {lin_aurocs[name]:.4f}")

    # --- Sweep RFF bandwidth -------------------------------------------
    ells = np.logspace(np.log10(args.ell_min), np.log10(args.ell_max),
                       args.n_ell)
    aurocs: dict[str, list[float]] = {n: [] for n, _, _ in z_sources}
    print(f"\nsweeping {args.n_ell} bandwidths in [{args.ell_min}, {args.ell_max}] ...")
    print(f"  {'ell':>10}  " + "  ".join(f"{n:>14}" for n, _, _ in z_sources))
    for ell in ells:
        rff = RFFLayer(in_dim=z_dim, n_features=args.M_rff,
                       length_scale=[float(ell)], rff_seed=0).to(device)
        phi_fn = lambda z, _rff=rff: _rff(z)
        h_fn, h_char, _ = build_phi_leverage(phi_fn, Z_train, ridge=args.ridge)
        with torch.no_grad():
            h_tr = h_fn(Z_train).cpu().numpy() / h_char
        row = []
        for name, z, _ in z_sources:
            with torch.no_grad():
                h = h_fn(z).cpu().numpy() / h_char
            au = auroc(h, h_tr)
            aurocs[name].append(au)
            row.append(au)
        print(f"  {ell:>10.4f}  " + "  ".join(f"{a:>14.4f}" for a in row))

    # --- Plot -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, _, color in z_sources:
        ax.plot(ells, aurocs[name], "o-", color=color, lw=1.5, ms=4,
                label=f"phi=RFF: {name}  (lin: {lin_aurocs[name]:.3f})")
        ax.axhline(lin_aurocs[name], color=color, ls=":", lw=1.0, alpha=0.6)
    ax.axhline(0.5, color="black", ls="-", lw=0.8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("RFF length scale ell")
    ax.set_ylabel("AUROC (source vs cifar10 train baseline)")
    norm_tag = "  [L2-normalized features]" if args.normalize else ""
    ax.set_title(
        f"Bandwidth scan -- ResNet18 (supervised) leverage OOD{norm_tag}\n"
        f"solid = phi=RFF(z) vs bandwidth   dotted = phi=z linear reference",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper right", ncols=2)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.02, 1.05)
    fig.tight_layout()
    norm_suffix = "_norm" if args.normalize else ""
    out = (REPO_ROOT / "example" / "out" / "cifar"
           / f"cifar10_resnet18{norm_suffix}_bandwidth_scan.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
