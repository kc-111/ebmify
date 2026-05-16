"""Compare feature-preprocessing treatments under leverage OOD scoring.

For each backbone (supervised ResNet18, SSL LeJEPA ResNet18, DINOv2 ViT-B/14,
VAE encoder mu) and each treatment, build the Gram on cifar10 train features
and report AUROC of the leverage score against several probe distributions.

Preprocess treatments (phi = z on transformed features):
  - raw          : z
  - L2           : z / ||z||
  - centered     : z - mu_train
  - centered+L2  : (z - mu_train) / ||z - mu_train||

Phi-map treatments (on raw z, same maps as ``cifar_ssl_ood_threshold.py``):
  - norm-bias           : [norm([z;1]); ||z||]
  - cent-norm-bias      : [norm([z-mu;1]); ||z-mu||]
  - dual-norm-bias      : [norm([z;1]); ||z||; norm([z-mu;1]); ||z-mu||]
  - norm-bias+RFF       : [norm([z;1]); ||z||; RFF(z)]
  - cent-norm-bias+RFF  : [norm([z-mu;1]); ||z-mu||; RFF(z-mu)]
  - combined+RFF        : dual-norm-bias block + RFF(z-mu)

RFF specs share one bandwidth ell* tuned per backbone on cifar10 test vs train
(same protocol as the SSL OOD threshold script).

Probe distributions: cifar10 test, cifar100, Gaussian noise, inverted images.

Usage:
    python example/cifar/diagnostics/cifar_centering_comparison.py
    python example/cifar/diagnostics/cifar_centering_comparison.py --backbones ssl
    python example/cifar/diagnostics/cifar_centering_comparison.py --no-rff-phi
"""

from __future__ import annotations

import argparse
import gc
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import REPO_ROOT  # noqa: E402

from ebmify.models.fc import RFFLayer  # noqa: E402

from cifar_data import load_cifar_test, load_cifar_train  # noqa: E402
from cifar_memorization_scan import BACKBONE_STYLE, make_encode_fn  # noqa: E402
from cifar_resnet18_ood_threshold import auroc  # noqa: E402
from mnist_vae_langevin import build_ood_x_sources, build_phi_leverage  # noqa: E402
from ood.cifar_ssl_ood_threshold import (  # noqa: E402
    phi_combined_norm_centered_rff,
    phi_dual_norm_bias,
    phi_norm_with_norm,
    phi_norm_with_norm_rff,
    tune_bandwidth,
)

TREATMENTS_PREPROCESS = ["raw", "L2", "centered", "centered+L2"]
PHI_TREATMENTS_NO_RFF = ["norm-bias", "cent-norm-bias", "dual-norm-bias"]
PHI_TREATMENTS_RFF = [
    "norm-bias+RFF", "cent-norm-bias+RFF", "combined+RFF",
]
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


def _make_phi_fn(name: str, mu: torch.Tensor,
                 rff: RFFLayer | None) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "norm-bias":
        return lambda z: phi_norm_with_norm(z)
    if name == "cent-norm-bias":
        return lambda z, _mu=mu: phi_norm_with_norm(z, _mu)
    if name == "dual-norm-bias":
        return lambda z, _mu=mu: phi_dual_norm_bias(z, _mu)
    assert rff is not None
    if name == "norm-bias+RFF":
        return lambda z, _rff=rff: phi_norm_with_norm_rff(z, _rff)
    if name == "cent-norm-bias+RFF":
        return lambda z, _rff=rff, _mu=mu: phi_norm_with_norm_rff(z, _rff, _mu)
    if name == "combined+RFF":
        return lambda z, _rff=rff, _mu=mu: phi_combined_norm_centered_rff(
            z, _rff, _mu,
        )
    raise ValueError(name)


def _eval_leverage(phi_fn: Callable[[torch.Tensor], torch.Tensor],
                   Z_tr: torch.Tensor,
                   probes: dict[str, torch.Tensor], *,
                   ridge: float, n_train: int, bias: bool = True,
                   ) -> tuple[dict[str, float], dict[str, float]]:
    last_err: RuntimeError | None = None
    h_fn = None
    for ridge_mul in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        try:
            h_fn, _, _ = build_phi_leverage(
                phi_fn, Z_tr, ridge=ridge * ridge_mul, bias=bias,
            )
            break
        except RuntimeError as exc:
            last_err = exc
    if h_fn is None:
        raise last_err or RuntimeError("Cholesky failed")
    with torch.no_grad():
        h_tr = h_fn(Z_tr).cpu().numpy()
    au: dict[str, float] = {}
    for src, Zp in probes.items():
        with torch.no_grad():
            h_p = h_fn(Zp).cpu().numpy()
        au[src] = auroc(h_p, h_tr)
    with torch.no_grad():
        Zt = phi_fn(Z_tr)
    geo = dict(
        feat_dim=float(Zt.shape[1]),
        train_norm=float(Zt.norm(dim=1).median().item()),
        mu_norm=float(Zt.mean(dim=0).norm().item()),
        dof_per_n=float(h_tr.sum()) / n_train,
    )
    return au, geo


def evaluate_backbone(name: str, encode_fn, z_dim: int,
                      X_tr_np: np.ndarray, X_te_np: np.ndarray,
                      X_other_np: np.ndarray, device: str, *,
                      n_train: int, n_eval: int, ridge: float, seed: int,
                      include_rff_phi: bool, M_rff: int,
                      ell_min: float, ell_max: float, n_ell: int,
                      tune_target: float,
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

    raw_norms = {"train": Z_tr.norm(dim=1).median().item()}
    for k, v in probes_raw.items():
        raw_norms[k] = v.norm(dim=1).median().item()
    print("  raw ||z|| medians:  " + "  ".join(
        f"{k}={raw_norms[k]:.3f}" for k in ["train", *PROBE_NAMES]
    ))

    rows: dict[str, dict[str, float]] = {}
    geo: dict[str, dict[str, float]] = {}

    for treat in TREATMENTS_PREPROCESS:
        Zt = apply_treatment(Z_tr, treat, mu_train)
        probes_t = {k: apply_treatment(v, treat, mu_train)
                    for k, v in probes_raw.items()}
        au, g = _eval_leverage(lambda z: z, Zt, probes_t,
                               ridge=ridge, n_train=n_train, bias=True)
        rows[treat] = au
        geo[treat] = g
        print(f"  {treat:<18}  "
              + "  ".join(f"{s}={au[s]:.3f}" for s in PROBE_NAMES)
              + f"   |  D={g['feat_dim']:.0f}  train ||phi||={g['train_norm']:.3f}  "
              f"||mu||={g['mu_norm']:.3e}  dof/n={g['dof_per_n']:.3f}")

    for treat in PHI_TREATMENTS_NO_RFF:
        phi_fn = _make_phi_fn(treat, mu_train, rff=None)
        au, g = _eval_leverage(phi_fn, Z_tr, probes_raw,
                               ridge=ridge, n_train=n_train, bias=False)
        rows[treat] = au
        geo[treat] = g
        print(f"  {treat:<18}  "
              + "  ".join(f"{s}={au[s]:.3f}" for s in PROBE_NAMES)
              + f"   |  D={g['feat_dim']:.0f}  train ||phi||={g['train_norm']:.3f}  "
              f"||mu||={g['mu_norm']:.3e}  dof/n={g['dof_per_n']:.3f}")

    if include_rff_phi:
        Z_val = probes_raw["cifar10 test"]
        ells = np.logspace(np.log10(ell_min), np.log10(ell_max), n_ell)
        print(f"  tuning RFF bandwidth on cifar10 test (target AUROC <= "
              f"{tune_target}) ...")
        ell_star, _ = tune_bandwidth(
            Z_tr, Z_val, z_dim=z_dim, M_rff=M_rff, ridge=ridge,
            device=device, ells=ells, target_auroc=tune_target,
        )
        rff = RFFLayer(
            in_dim=z_dim, n_features=M_rff,
            length_scale=[float(ell_star)], rff_seed=0,
        ).to(device)
        print(f"  ell* = {ell_star:.3f}  (shared by RFF phi maps)")
        for treat in PHI_TREATMENTS_RFF:
            phi_fn = _make_phi_fn(treat, mu_train, rff)
            au, g = _eval_leverage(phi_fn, Z_tr, probes_raw,
                                   ridge=ridge, n_train=n_train, bias=False)
            rows[treat] = au
            geo[treat] = g
            print(f"  {treat:<18}  "
                  + "  ".join(f"{s}={au[s]:.3f}" for s in PROBE_NAMES)
                  + f"   |  D={g['feat_dim']:.0f}  train ||phi||="
                  f"{g['train_norm']:.3f}  dof/n={g['dof_per_n']:.3f}")

    return dict(rows=rows, geo=geo, raw_norms=raw_norms)


def _all_treatments(include_rff_phi: bool) -> list[str]:
    out = list(TREATMENTS_PREPROCESS) + list(PHI_TREATMENTS_NO_RFF)
    if include_rff_phi:
        out.extend(PHI_TREATMENTS_RFF)
    return out


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
    ap.add_argument("--no-rff-phi", action="store_true",
                    help="Skip norm-bias+RFF / cent-norm-bias+RFF / combined+RFF.")
    ap.add_argument("--M", type=int, default=2048, dest="M_rff")
    ap.add_argument("--ell-min", type=float, default=0.1)
    ap.add_argument("--ell-max", type=float, default=200.0)
    ap.add_argument("--n-ell", type=int, default=24)
    ap.add_argument("--tune-target", type=float, default=0.55)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    include_rff_phi = not args.no_rff_phi
    treatments = _all_treatments(include_rff_phi)

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
            seed=args.seed, include_rff_phi=include_rff_phi,
            M_rff=args.M_rff, ell_min=args.ell_min, ell_max=args.ell_max,
            n_ell=args.n_ell, tune_target=args.tune_target,
        )
        del encode_fn, model_obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not results:
        print("\nno results to plot.")
        return

    print("\n=== AUROC summary ===")
    hdr = f"  {'backbone':<22} {'treatment':<20} " + " ".join(
        f"{s:>12}" for s in PROBE_NAMES
    )
    print(hdr)
    for name, res in results.items():
        for treat in treatments:
            if treat not in res["rows"]:
                continue
            au = res["rows"][treat]
            print(f"  {BACKBONE_STYLE[name][1]:<22} {treat:<20} "
                  + " ".join(f"{au[s]:>12.3f}" for s in PROBE_NAMES))
        print()

    n_bb = len(results)
    ncols = 2
    nrows = (n_bb + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows),
                             squeeze=False)
    x = np.arange(len(PROBE_NAMES))
    n_treat = len(treatments)
    width = min(0.8 / n_treat, 0.12)
    cmap = plt.cm.tab20(np.linspace(0, 1, n_treat))
    treat_colors = {t: cmap[i] for i, t in enumerate(treatments)}

    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx // ncols][idx % ncols]
        for i, treat in enumerate(treatments):
            if treat not in res["rows"]:
                continue
            vals = [res["rows"][treat][s] for s in PROBE_NAMES]
            offset = (i - (n_treat - 1) / 2) * width
            ax.bar(x + offset, vals, width=width,
                   color=treat_colors[treat], edgecolor="black", lw=0.3,
                   label=treat)
        ax.axhline(0.5, color="black", lw=0.7, ls="-", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(PROBE_NAMES, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel("AUROC vs cifar10 train")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(BACKBONE_STYLE[name][1], fontsize=10)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=5.5, loc="upper right", ncols=2,
                  framealpha=0.9)
    for j in range(n_bb, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    rff_note = "" if include_rff_phi else "  (RFF phi maps skipped)"
    fig.suptitle(
        "Leverage AUROC across preprocessing + norm-bias phi maps  "
        f"(Gram = cifar10 train, n={args.n_train}, ridge={args.ridge})"
        f"{rff_note}",
        fontsize=11,
    )
    fig.tight_layout()
    out = REPO_ROOT / "example" / "out" / "cifar" / "cifar10_centering_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
