"""Threshold OOD classifier on DINOv2 ViT-B/14 features (Meta AI, self-supervised).

Same protocol as ``cifar_resnet18_ood_threshold.py`` -- inductive:
 1. Kernel built from cifar10 train only.
 2. RFF bandwidth tuned on cifar10 test as held-out validation
    (pick smallest ell with AUROC(val, train) <= --tune-target).
 3. Threshold = 0.95 quantile of h on cifar10 train.
 4. Score under three phi maps: {z, RFF(z), [z; RFF(z)]}.

DINOv2 features (CLS token, 768-dim) are self-supervised and generally
the top-performing open visual features for OOD detection.

Usage:
    python example/cifar/ood/cifar_dinov2_ood_threshold.py --dataset cifar10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import REPO_ROOT  # noqa: E402

from ebmify.models.fc import RFFLayer  # noqa: E402

from cifar_data import load_cifar_test, load_cifar_train  # noqa: E402
from mnist_vae_langevin import (  # noqa: E402
    build_ood_x_sources,
    build_phi_leverage,
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# DINOv2 uses patch_size=14, so input must be a multiple of 14.
# 224x224 -> 16x16 = 256 patches, the standard preprocessing.
DINOV2_INPUT = 224


def make_feature_extractor(variant: str, device: str, dtype: torch.dtype,
                           ) -> tuple[nn.Module, int]:
    print(f"loading DINOv2 ({variant}) from torch.hub  (dtype={dtype}) ...")
    model = torch.hub.load("facebookresearch/dinov2", variant)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device=device, dtype=dtype)
    # Probe feature dim with a dummy input.
    with torch.no_grad():
        z = model(torch.zeros(1, 3, DINOV2_INPUT, DINOV2_INPUT,
                              device=device, dtype=dtype))
    z_dim = int(z.shape[-1])
    return model, z_dim


def encode(model: nn.Module, x: torch.Tensor, *, dtype: torch.dtype,
           batch_size: int, device: str) -> torch.Tensor:
    """Encode (B, 3, 32, 32) in [0,1] -> (B, z_dim) DINOv2 CLS features.

    Returns features in fp32 regardless of model dtype (so downstream
    leverage math runs in fp32).
    """
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    feats = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            chunk = x[i:i + batch_size]
            chunk = F.interpolate(chunk, size=(DINOV2_INPUT, DINOV2_INPUT),
                                  mode="bilinear", align_corners=False)
            chunk = ((chunk - mean) / std).to(dtype)
            feats.append(model(chunk).float())
    return torch.cat(feats, dim=0)


def auroc(s_pos: np.ndarray, s_neg: np.ndarray) -> float:
    a = np.concatenate([s_pos, s_neg])
    labels = np.concatenate([np.ones_like(s_pos), np.zeros_like(s_neg)])
    order = np.argsort(a)
    labels_sorted = labels[order]
    ranks = np.arange(1, len(a) + 1, dtype=np.float64)
    R_pos = ranks[labels_sorted == 1].sum()
    n_pos = float(len(s_pos)); n_neg = float(len(s_neg))
    U = R_pos - n_pos * (n_pos + 1) / 2
    return float(U / (n_pos * n_neg))


def roc_auroc(s_pos: np.ndarray, s_neg: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    all_s = np.unique(np.concatenate([s_pos, s_neg]))
    cands = np.concatenate(
        [[np.inf], 0.5 * (all_s[:-1] + all_s[1:])[::-1], [-np.inf]]
    )
    tpr = np.array([(s_pos > t).mean() for t in cands])
    fpr = np.array([(s_neg > t).mean() for t in cands])
    order = np.argsort(fpr)
    fpr_o = fpr[order]; tpr_o = tpr[order]
    au = float(np.trapezoid(tpr_o, fpr_o))
    return fpr_o, tpr_o, au


def tune_bandwidth(Z_train: torch.Tensor, Z_val: torch.Tensor, *,
                   z_dim: int, M_rff: int, ridge: float, device: str,
                   ells: np.ndarray, target_auroc: float,
                   ) -> tuple[float, list[dict]]:
    sweep = []
    print(f"\n  bandwidth sweep on held-out validation (target AUROC <= {target_auroc}):")
    print(f"  {'ell':>10} {'AUROC_val':>11} {'med_gap':>10}")
    for ell in ells:
        rff = RFFLayer(in_dim=z_dim, n_features=M_rff,
                       length_scale=[float(ell)], rff_seed=0).to(device)
        with torch.no_grad():
            phi_fn = lambda z, _rff=rff: _rff(z)
            h_fn, _h_char, _D = build_phi_leverage(phi_fn, Z_train, ridge=ridge)
            h_tr = h_fn(Z_train).cpu().numpy()
            h_va = h_fn(Z_val).cpu().numpy()
        au = auroc(h_va, h_tr)
        gap = float(np.median(h_va) / max(np.median(h_tr), 1e-12))
        sweep.append({"ell": float(ell), "auroc_val": au, "gap": gap})
        print(f"  {ell:>10.3f} {au:>11.3f} {gap:>10.3f}")
    qualifying = [s for s in sweep if s["auroc_val"] <= target_auroc]
    if qualifying:
        chosen = qualifying[0]
        why = f"smallest ell with AUROC_val <= {target_auroc}"
    else:
        chosen = min(sweep, key=lambda s: s["auroc_val"])
        why = f"no ell met target; using argmin AUROC_val = {chosen['auroc_val']:.3f}"
    print(f"  -> chose ell* = {chosen['ell']:.3f}  ({why})")
    return chosen["ell"], sweep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    ap.add_argument("--variant", type=str, default="dinov2_vitb14",
                    help="DINOv2 hub name (dinov2_vits14, dinov2_vitb14, "
                         "dinov2_vitl14, dinov2_vitg14).")
    ap.add_argument("--M", type=int, default=2048, dest="M_rff")
    ap.add_argument("--ell", type=float, default=None,
                    help="Skip tuning; use this ell directly.")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--n-train", type=int, default=8192, dest="n_train")
    ap.add_argument("--n-eval", type=int, default=2048, dest="n_eval")
    ap.add_argument("--batch", type=int, default=128, dest="batch_size")
    ap.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32",
                    help="Backbone dtype for inference; bf16 halves memory.")
    ap.add_argument("--ell-min", type=float, default=0.1)
    ap.add_argument("--ell-max", type=float, default=200.0)
    ap.add_argument("--n-ell", type=int, default=24)
    ap.add_argument("--tune-target", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--normalize", action="store_true",
                    help="L2-normalize features before building the Gram and "
                         "scoring. Removes ||z||^2 confound; leverage becomes "
                         "purely directional.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    model_dtype = dtype_map[args.dtype]
    model, z_dim = make_feature_extractor(args.variant, device, model_dtype)
    print(f"  feature dim = {z_dim}")

    # --- Load + encode --------------------------------------------------
    X_tr, _ = load_cifar_train(args.dataset)
    other = "cifar100" if args.dataset == "cifar10" else "cifar10"
    X_other, _ = load_cifar_train(other)
    X_te, _ = load_cifar_test(args.dataset)

    rng = np.random.default_rng(0)
    sub_idx = rng.choice(len(X_tr), size=args.n_train, replace=False)
    X_sub_t = torch.as_tensor(X_tr[sub_idx], dtype=torch.float32, device=device)
    print(f"encoding {args.n_train} training images ...")
    Z_train = encode(model, X_sub_t, dtype=model_dtype,
                     batch_size=args.batch_size, device=device)
    print(f"  Z_train: {tuple(Z_train.shape)}  "
          f"||z||_2 median={Z_train.norm(dim=1).median().item():.3f}")

    base = build_ood_x_sources(
        X_sub_t, device, n_eval=args.n_eval, seed=args.seed,
        in_name=args.dataset,
    )
    cross_idx = rng.choice(len(X_other), size=args.n_eval, replace=False)
    x_cross = torch.as_tensor(
        X_other[cross_idx], dtype=torch.float32, device=device,
    )
    test_idx = rng.choice(len(X_te), size=min(args.n_eval, len(X_te)),
                          replace=False)
    x_test = torch.as_tensor(X_te[test_idx], dtype=torch.float32, device=device)
    x_sources = (
        [base[0],
         (f"{args.dataset} test", x_test, "C9"),
         (other, x_cross, "C8")]
        + base[1:]
    )

    print("encoding all x sources through DINOv2 ...")
    z_sources: list[tuple[str, torch.Tensor, str]] = []
    for name, x, color in x_sources:
        z = encode(model, x, dtype=model_dtype,
                   batch_size=args.batch_size, device=device)
        z_sources.append((name, z, color))
        print(f"  {name:>22}: {tuple(z.shape)}  ||z|| median="
              f"{z.norm(dim=1).median().item():.3f}")

    if args.normalize:
        def _l2(z: torch.Tensor) -> torch.Tensor:
            return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        Z_train = _l2(Z_train)
        z_sources = [(n, _l2(z), c) for n, z, c in z_sources]
        print("\n[normalize] L2-normalized Z_train and all z sources "
              "to unit norm. Gram, RFF bandwidth tuning, and leverage "
              "now use unit-norm features.")

    test_src_idx = 1
    assert z_sources[test_src_idx][0] == f"{args.dataset} test"
    Z_val = z_sources[test_src_idx][1]

    # --- Tune ell on held-out cifar10 test ------------------------------
    if args.ell is None:
        ells = np.logspace(
            np.log10(args.ell_min), np.log10(args.ell_max), args.n_ell,
        )
        ell_star, sweep = tune_bandwidth(
            Z_train, Z_val, z_dim=z_dim, M_rff=args.M_rff,
            ridge=args.ridge, device=device,
            ells=ells, target_auroc=args.tune_target,
        )
    else:
        ell_star = args.ell
        sweep = None
        print(f"\n  skipping tuning; using --ell {ell_star}")

    rff = RFFLayer(in_dim=z_dim, n_features=args.M_rff,
                   length_scale=[float(ell_star)], rff_seed=0).to(device)
    print(f"\nfinal RFF length_scale = {rff.length_scale.tolist()}  "
          f"M = {args.M_rff}  ridge = {args.ridge}")

    specs = [
        ("phi = z",           lambda z: z),
        ("phi = RFF(z)",      lambda z: rff(z)),
        ("phi = [z; RFF(z)]", lambda z: torch.cat([z, rff(z)], dim=-1)),
    ]
    in_name = args.dataset
    n_phi = len(specs)
    base_label = args.dataset

    per_phi: list[dict] = []
    for spec_name, phi_fn in specs:
        h_fn, h_char, D = build_phi_leverage(phi_fn, Z_train, ridge=args.ridge)
        rows: list[tuple[str, np.ndarray, str]] = []
        with torch.no_grad():
            for src_name, z_src, color in z_sources:
                h_vals = h_fn(z_src).cpu().numpy() / h_char
                rows.append((src_name, h_vals, color))
            h_baseline = h_fn(Z_train).cpu().numpy() / h_char
        tau = float(np.quantile(h_baseline, 0.95))
        per_phi.append({
            "spec_name": spec_name, "D": D, "h_char": float(h_char),
            "tau": tau, "rows": rows, "baseline": h_baseline,
        })

    # --- Print tables ---------------------------------------------------
    print(f"\n=== Gram = {args.dataset} train  (DINOv2/{args.variant}, "
          f"ell* = {ell_star:.3f}) ===")
    print("median h(z) / h_char per source  (h_char = 95th pct of in-sample h):")
    header = f"  {'phi':<22} {'D':>5}"
    for name, _, _ in z_sources:
        header += f"  {name:>13}"
    print(header)
    for entry in per_phi:
        row_str = f"  {entry['spec_name']:<22} {entry['D']:>5}"
        for _, h, _ in entry["rows"]:
            row_str += f"  {np.median(h):>13.3e}"
        print(row_str)

    print(f"\nthreshold tau = 0.95 quantile of '{base_label}' h(z). "
          f"TPR (fraction flagged OOD) per source:")
    head = f"  {'phi':<22} {'D':>5}  {'tau':>10}"
    for name, _, _ in z_sources:
        head += f"  {name:>13}"
    print(head)
    for entry in per_phi:
        row_str = f"  {entry['spec_name']:<22} {entry['D']:>5}  {entry['tau']:>10.4e}"
        for _, h, _ in entry["rows"]:
            tpr_val = float((h > entry["tau"]).mean())
            row_str += f"  {tpr_val:>13.3f}"
        print(row_str)

    print(f"\nAUROC: each source vs '{base_label}' baseline:")
    head = f"  {'phi':<22} {'D':>5}"
    for name, _, _ in z_sources:
        head += f"  {name:>13}"
    print(head)
    for entry in per_phi:
        row_str = f"  {entry['spec_name']:<22} {entry['D']:>5}"
        for _, h, _ in entry["rows"]:
            _, _, au = roc_auroc(h, entry["baseline"])
            row_str += f"  {au:>13.4f}"
        print(row_str)

    # --- Plot -----------------------------------------------------------
    if args.no_plot:
        return
    fig = plt.figure(figsize=(6 * n_phi, 14))
    n_rows = 3 if sweep is not None else 2
    gs = fig.add_gridspec(n_rows, n_phi, height_ratios=[1.0] * n_rows)

    row_offset = 0
    if sweep is not None:
        ax = fig.add_subplot(gs[0, :])
        ells_arr = np.array([s["ell"] for s in sweep])
        ax.plot(ells_arr, [s["auroc_val"] for s in sweep],
                "C9s-", label="AUROC(cifar10 test vs cifar10 train)")
        ax.axhline(args.tune_target, color="gray", ls="--", lw=0.8,
                   label=f"target = {args.tune_target}")
        ax.axhline(0.5, color="black", ls=":", lw=0.6)
        ax.axvline(ell_star, color="red", ls="-", lw=1.0,
                   label=f"chosen ell* = {ell_star:.3f}")
        ax.set_xscale("log")
        ax.set_xlabel("RFF length scale ell")
        ax.set_ylabel("AUROC")
        ax.set_title(f"Bandwidth tuning on held-out cifar10 test (want AUROC ~ 0.5)",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        ax.set_ylim(0.4, 1.05)
        row_offset = 1

    for j, entry in enumerate(per_phi):
        ax = fig.add_subplot(gs[row_offset, j])
        rows = entry["rows"]
        tau = entry["tau"]
        all_vals = np.concatenate([r[1] for r in rows])
        log_vals = np.log10(np.clip(all_vals, 1e-6, None))
        log_tau = float(np.log10(max(tau, 1e-6)))
        lo = float(min(log_vals.min(), log_tau))
        hi = float(max(log_vals.max(), log_tau))
        pad = 0.03 * max(hi - lo, 1e-3)
        lo -= pad; hi += pad
        bins = np.linspace(lo, hi, 80)
        for src_name, h, color in rows:
            med = float(np.median(h))
            if h.std() < 1e-5 * (abs(med) + 1e-12):
                ax.axvline(
                    np.log10(med + 1e-6), color=color, lw=2.0, alpha=0.8,
                    ls="-", label=f"{src_name} ({med:.2e})",
                )
            else:
                ax.hist(
                    np.log10(h + 1e-6), bins=bins, alpha=0.45, density=True,
                    label=f"{src_name} ({med:.2e})", color=color,
                )
        ax.axvline(np.log10(tau + 1e-6), color="k", ls="--", lw=1.4,
                   label="tau (0.95 quantile of cal)")
        ax.set_xlim(lo, hi)
        ax.margins(x=0)
        ax.set_title(
            f"{entry['spec_name']}  (D={entry['D']})  tau={tau:.3e}",
            fontsize=9,
        )
        ax.set_xlabel("log10(h(z) / h_char)")
        ax.set_ylabel("density")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(alpha=0.3)

    for j, entry in enumerate(per_phi):
        ax = fig.add_subplot(gs[row_offset + 1, j])
        for src_name, h, color in entry["rows"]:
            if src_name == in_name:
                continue
            fpr_c, tpr_c, au = roc_auroc(h, entry["baseline"])
            ax.plot(fpr_c, tpr_c, color=color, lw=1.2,
                    label=f"{src_name} (AUC={au:.3f})")
        ax.plot([0, 1], [0, 1], "k:", lw=0.7, alpha=0.6)
        ax.set_title(f"ROC: {entry['spec_name']}", fontsize=9)
        ax.set_xlabel("FPR (on cifar10 train baseline)")
        ax.set_ylabel("TPR (each source as positive)")
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    norm_tag = "  [L2-normalized features]" if args.normalize else ""
    fig.suptitle(
        f"DINOv2/{args.variant} OOD on {in_name}:{norm_tag} "
        f"tuned bandwidth on held-out test, tau = 0.95-quantile of train  "
        f"z_dim={z_dim}  M_rff={args.M_rff}  ell*={ell_star:.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    norm_suffix = "_norm" if args.normalize else ""
    out = (REPO_ROOT / "example" / "out" / "cifar"
           / f"{args.dataset}_{args.variant}{norm_suffix}_ood_threshold.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
