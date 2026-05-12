"""RFF length-scale sweep under the inductive OOD protocol.

Calibrates on ``cifar10 train`` ONLY (literature-style: Setup A from
``cifar_vae_ood_threshold.py``). For each ``ell`` we measure:

 - med h(z) on train (cal), cifar10 test (held-out ID), cifar100 (OOD).
 - generalization gap: ``med(h_test) / med(h_train)``  (want ~1).
 - AUROC: cifar10 test vs cifar10 train (want ~0.5 -- ID indistinguishable).
 - AUROC: cifar100 vs cifar10 train      (want >> 0.5 -- OOD separable).
 - TPR @ tau = 0.95 quantile of h_train: test (want ~0.05) and
   cifar100 (want >> 0.05).

Hypothesis being tested: no ell simultaneously gives
``AUROC_test ~ 0.5`` AND ``AUROC_cifar100 >> 0.5``. That would mean the
encoder does not place cifar10 test closer to cifar10 train (under the
RFF kernel) than cifar100 is. If true, the only way our pipeline gets
>0.5 AUROC on cifar100 is by also rejecting cifar10 test -- the
"memorize" regime, which is the Setup B / transductive cheat.

Usage:
    python example/cifar/cifar_vae_ood_ell_sweep.py --dataset cifar10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "example" / "mnist"))

from ebmify.models.fc import RFFLayer  # noqa: E402

from cifar_data import cifar_ckpt_path, load_cifar_test, load_cifar_train  # noqa: E402
from cifar_vae_train import load_vae  # noqa: E402
from mnist_vae_langevin import build_phi_leverage  # noqa: E402


def auroc(s_pos: np.ndarray, s_neg: np.ndarray) -> float:
    """Mann-Whitney rank-based AUROC. Positive = predicted OOD (high h)."""
    a = np.concatenate([s_pos, s_neg])
    labels = np.concatenate([np.ones_like(s_pos), np.zeros_like(s_neg)])
    order = np.argsort(a)
    labels_sorted = labels[order]
    ranks = np.arange(1, len(a) + 1, dtype=np.float64)
    R_pos = ranks[labels_sorted == 1].sum()
    n_pos = float(len(s_pos)); n_neg = float(len(s_neg))
    U = R_pos - n_pos * (n_pos + 1) / 2
    return float(U / (n_pos * n_neg))


def median_pairwise_distance(Z: torch.Tensor, n_pairs: int = 200_000) -> float:
    N = Z.shape[0]
    g = torch.Generator(device=Z.device).manual_seed(0)
    i = torch.randint(0, N, (n_pairs,), generator=g, device=Z.device)
    j = torch.randint(0, N, (n_pairs,), generator=g, device=Z.device)
    mask = i != j
    d = (Z[i[mask]] - Z[j[mask]]).norm(dim=1)
    return float(d.median().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    ap.add_argument("--z", type=int, default=256, dest="z_dim")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--M", type=int, default=2048, dest="M_rff")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--n-train", type=int, default=8192, dest="n_train")
    ap.add_argument("--n-eval", type=int, default=2048, dest="n_eval")
    ap.add_argument("--ell-min", type=float, default=0.1)
    ap.add_argument("--ell-max", type=float, default=200.0)
    ap.add_argument("--n-ell", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    ckpt = cifar_ckpt_path(args.dataset, args.z_dim, args.beta)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No cached VAE at {ckpt}. Run cifar_vae_train.py --dataset "
            f"{args.dataset} first."
        )
    vae = load_vae(ckpt, device)
    print(f"loaded VAE from {ckpt}")

    other = "cifar100" if args.dataset == "cifar10" else "cifar10"
    X_tr, _ = load_cifar_train(args.dataset)
    X_te, _ = load_cifar_test(args.dataset)
    X_other, _ = load_cifar_train(other)

    rng = np.random.default_rng(0)
    tr_idx = rng.choice(len(X_tr), size=args.n_train, replace=False)
    te_idx = rng.choice(len(X_te), size=min(args.n_eval, len(X_te)),
                        replace=False)
    ot_idx = rng.choice(len(X_other), size=args.n_eval, replace=False)

    X_tr_t = torch.as_tensor(X_tr[tr_idx], dtype=torch.float32, device=device)
    X_te_t = torch.as_tensor(X_te[te_idx], dtype=torch.float32, device=device)
    X_ot_t = torch.as_tensor(X_other[ot_idx], dtype=torch.float32, device=device)

    with torch.no_grad():
        Z_train, _ = vae.encode(X_tr_t)
        Z_test, _ = vae.encode(X_te_t)
        Z_other, _ = vae.encode(X_ot_t)

    ell_median = median_pairwise_distance(Z_train)
    print(f"Z_train shape={tuple(Z_train.shape)}  "
          f"||z|| median={Z_train.norm(dim=1).median().item():.3f}")
    print(f"median pairwise dist in Z_train = {ell_median:.3f}  "
          f"(median-heuristic ell)")

    ells = np.logspace(
        np.log10(args.ell_min), np.log10(args.ell_max), args.n_ell,
    )

    results = []
    print()
    print(f"{'ell':>10} {'med_train':>12} {'med_test':>12} "
          f"{'med_other':>12} {'gap_test':>10} {'gap_other':>10} "
          f"{'AUROC_te':>10} {'AUROC_ot':>10} "
          f"{'TPR_te':>8} {'TPR_ot':>8}")
    for ell in ells:
        rff = RFFLayer(
            in_dim=args.z_dim, n_features=args.M_rff,
            length_scale=[float(ell)], rff_seed=0,
        ).to(device)
        with torch.no_grad():
            phi_fn = lambda z, _rff=rff: _rff(z)
            h_fn, _h_char, _D = build_phi_leverage(
                phi_fn, Z_train, ridge=args.ridge,
            )
            h_train = h_fn(Z_train).cpu().numpy()
            h_test = h_fn(Z_test).cpu().numpy()
            h_other = h_fn(Z_other).cpu().numpy()
        med_train = float(np.median(h_train))
        med_test = float(np.median(h_test))
        med_other = float(np.median(h_other))
        gap_test = med_test / max(med_train, 1e-12)
        gap_other = med_other / max(med_train, 1e-12)
        au_test = auroc(h_test, h_train)
        au_other = auroc(h_other, h_train)
        tau = float(np.quantile(h_train, 0.95))
        tpr_test = float((h_test > tau).mean())
        tpr_other = float((h_other > tau).mean())
        results.append(dict(
            ell=float(ell), med_train=med_train, med_test=med_test,
            med_other=med_other, gap_test=gap_test, gap_other=gap_other,
            au_test=au_test, au_other=au_other,
            tpr_test=tpr_test, tpr_other=tpr_other,
        ))
        print(f"{ell:>10.3f} {med_train:>12.3e} {med_test:>12.3e} "
              f"{med_other:>12.3e} {gap_test:>10.3f} {gap_other:>10.3f} "
              f"{au_test:>10.3f} {au_other:>10.3f} "
              f"{tpr_test:>8.3f} {tpr_other:>8.3f}")

    # ---- Plot ------------------------------------------------------------
    ells_arr = np.array([r["ell"] for r in results])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(ells_arr, [r["med_train"] for r in results],
            "C0o-", label="cifar10 train (cal)")
    ax.plot(ells_arr, [r["med_test"] for r in results],
            "C9s-", label="cifar10 test (held-out ID)")
    ax.plot(ells_arr, [r["med_other"] for r in results],
            "C8^-", label=f"{other} (OOD)")
    ax.axvline(ell_median, color="gray", ls=":", lw=1.0,
               label=f"median heuristic ({ell_median:.2f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("RFF length scale ell")
    ax.set_ylabel("median h(z)")
    ax.set_title("Median leverage vs ell")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ells_arr, [r["gap_test"] for r in results],
            "C9s-", label="med(h_test) / med(h_train)")
    ax.plot(ells_arr, [r["gap_other"] for r in results],
            "C8^-", label=f"med(h_{other}) / med(h_train)")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.axvline(ell_median, color="gray", ls=":", lw=1.0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("RFF length scale ell")
    ax.set_ylabel("median ratio")
    ax.set_title("Generalization gap: test ~1, OOD >> 1?")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(ells_arr, [r["au_test"] for r in results],
            "C9s-", label="AUROC: cifar10 test vs train (want 0.5)")
    ax.plot(ells_arr, [r["au_other"] for r in results],
            "C8^-", label=f"AUROC: {other} vs train (want >> 0.5)")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.axvline(ell_median, color="gray", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("RFF length scale ell")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("AUROC under Setup A (calibrate on cifar10 train only)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(ells_arr, [r["tpr_test"] for r in results],
            "C9s-", label="TPR cifar10 test (want ~0.05)")
    ax.plot(ells_arr, [r["tpr_other"] for r in results],
            "C8^-", label=f"TPR {other} (want ~1.0)")
    ax.axhline(0.05, color="gray", ls="--", lw=0.8,
               label="0.05 (calibrated FPR)")
    ax.axvline(ell_median, color="gray", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("RFF length scale ell")
    ax.set_ylabel("flag rate")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("TPR @ tau = 0.95 quantile of h_train")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(
        f"{args.dataset} OOD ell sweep -- inductive protocol "
        f"(calibrate on train only)  "
        f"z_dim={args.z_dim}  M={args.M_rff}  ridge={args.ridge}",
        fontsize=11,
    )
    fig.tight_layout()
    out = REPO_ROOT / "example" / "out" / f"{args.dataset}_vae_ood_ell_sweep.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
