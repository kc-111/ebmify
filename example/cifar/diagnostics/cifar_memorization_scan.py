"""Memorization-capacity scan across all CIFAR feature backbones.

For each backbone (supervised ResNet18, SSL ResNet18, DINOv2 ViT-B/14,
VAE encoder mu) we encode cifar10 train, cifar10 test, cifar100,
L2-normalize the features, then sweep the RFF length scale ell. At each
ell we record:

  - AUROC(cifar10 test vs cifar10 train)
        Memorization-capacity signal. High = the kernel can tell train
        from a held-out same-distribution sample = features have
        train-specific structure not shared with test = memorization.
  - AUROC(cifar100      vs cifar10 train)
        OOD-discrimination signal. Tells us whether the same kernel
        actually separates a different-distribution sample.
  - eff_dof = trace(hat matrix) / n_train = mean_i h(z_i^train)
        Effective degrees of freedom. ~1 = the model interpolates each
        train point (memorization-capable). ~0 = total smoothing.
  - h_char = 95th-pctile of h on train (the scaling used in OOD scripts)

We also report the phi=z linear-kernel values per backbone as a
bandwidth-free reference (linear has no ell knob; its memorization
capacity is set entirely by feature geometry + ridge).

Usage:
    python example/cifar/diagnostics/cifar_memorization_scan.py
    python example/cifar/diagnostics/cifar_memorization_scan.py --backbones resnet18 ssl
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

from ebmify.models.fc import RFFLayer  # noqa: E402

from cifar_data import (  # noqa: E402
    cifar_ckpt_path, load_cifar_test, load_cifar_train,
)
from cifar_dinov2_ood_threshold import (  # noqa: E402
    encode as encode_dinov2_raw,
    make_feature_extractor as make_dinov2,
)
from cifar_resnet18_ood_threshold import (  # noqa: E402
    auroc,
    encode as encode_cifar_normed,
    load_trained_resnet18,
)
from cifar_resnet18_train import resnet18_ckpt_path  # noqa: E402
from cifar_vae_train import load_vae  # noqa: E402
from mnist_vae_langevin import build_phi_leverage  # noqa: E402
from ssl_linear_probe import load_ssl_backbone  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

# (name, color, label).
BACKBONE_STYLE = {
    "resnet18": ("C0", "ResNet18 (supervised)"),
    "ssl":      ("C1", "LeJEPA ResNet18"),
    "dinov2":   ("C2", "DINOv2 ViT-B/14"),
    "vae":      ("C3", "VAE (mu, z=256)"),
}


def make_encode_fn(name: str, device: str, args) -> tuple[callable, int, object]:
    """Build (encode_fn(x_in_01) -> features, z_dim, model_obj)."""
    bs = args.batch_size
    if name == "resnet18":
        ckpt = resnet18_ckpt_path(args.resnet_tag)
        if not ckpt.exists():
            raise FileNotFoundError(f"No resnet18 ckpt at {ckpt}")
        model, z_dim = load_trained_resnet18(ckpt, device)
        def fn(x):
            return encode_cifar_normed(model, x, batch_size=bs, device=device)
        return fn, z_dim, model

    if name == "ssl":
        suffix = f"_{args.ssl_tag}" if args.ssl_tag else ""
        ckpt = CACHE_DIR / f"cifar10_ssl_resnet18{suffix}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"No SSL ckpt at {ckpt}")
        model, z_dim = load_ssl_backbone(ckpt, device)
        def fn(x):
            return encode_cifar_normed(model, x, batch_size=bs, device=device)
        return fn, z_dim, model

    if name == "dinov2":
        dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16,
                     "fp16": torch.float16}
        dtype = dtype_map[args.dinov2_dtype]
        model, z_dim = make_dinov2(args.dinov2_variant, device, dtype)
        def fn(x):
            return encode_dinov2_raw(model, x, dtype=dtype,
                                     batch_size=bs, device=device)
        return fn, z_dim, model

    if name == "vae":
        ckpt = cifar_ckpt_path("cifar10", args.vae_z, args.vae_beta)
        if not ckpt.exists():
            raise FileNotFoundError(f"No VAE ckpt at {ckpt}")
        vae = load_vae(ckpt, device)
        z_dim = args.vae_z
        def fn(x):
            outs = []
            with torch.no_grad():
                for i in range(0, x.shape[0], bs):
                    mu, _ = vae.encode(x[i:i + bs])
                    outs.append(mu)
            return torch.cat(outs, dim=0)
        return fn, z_dim, vae

    raise ValueError(name)


def scan_backbone(name: str, encode_fn, z_dim: int,
                  X_tr_np: np.ndarray, X_te_np: np.ndarray,
                  X_other_np: np.ndarray, device: str, *,
                  n_train: int, n_eval: int, M_rff: int, ridge: float,
                  ells: np.ndarray, seed: int,
                  ) -> dict:
    rng = np.random.default_rng(0)
    tr_idx = rng.choice(len(X_tr_np), size=n_train, replace=False)
    te_idx = rng.choice(len(X_te_np), size=min(n_eval, len(X_te_np)),
                        replace=False)
    ot_idx = rng.choice(len(X_other_np), size=n_eval, replace=False)
    x_tr = torch.as_tensor(X_tr_np[tr_idx], dtype=torch.float32, device=device)
    x_te = torch.as_tensor(X_te_np[te_idx], dtype=torch.float32, device=device)
    x_ot = torch.as_tensor(X_other_np[ot_idx], dtype=torch.float32, device=device)

    print(f"  encoding train ({n_train}), test ({len(te_idx)}), cifar100 ({n_eval}) ...")
    Z_tr = encode_fn(x_tr); Z_te = encode_fn(x_te); Z_ot = encode_fn(x_ot)
    print(f"  pre-norm ||z|| median train={Z_tr.norm(dim=1).median().item():.3f}  "
          f"test={Z_te.norm(dim=1).median().item():.3f}  "
          f"cifar100={Z_ot.norm(dim=1).median().item():.3f}")

    def _l2(z): return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    Z_tr, Z_te, Z_ot = _l2(Z_tr), _l2(Z_te), _l2(Z_ot)

    h_fn_lin, h_char_lin, _ = build_phi_leverage(
        lambda z: z, Z_tr, ridge=ridge,
    )
    with torch.no_grad():
        h_tr_lin = h_fn_lin(Z_tr).cpu().numpy()
        h_te_lin = h_fn_lin(Z_te).cpu().numpy()
        h_ot_lin = h_fn_lin(Z_ot).cpu().numpy()
    linear = dict(
        au_test=auroc(h_te_lin, h_tr_lin),
        au_other=auroc(h_ot_lin, h_tr_lin),
        dof_per_n=float(h_tr_lin.sum()) / n_train,
        h_char=float(h_char_lin),
    )
    print(f"  phi=z:  AUROC(test vs train)={linear['au_test']:.3f}  "
          f"AUROC(cifar100 vs train)={linear['au_other']:.3f}  "
          f"dof/n={linear['dof_per_n']:.3f}")

    sweep = []
    print(f"  sweeping {len(ells)} bandwidths ...")
    print(f"    {'ell':>10} {'AU(test)':>10} {'AU(c100)':>10} {'dof/n':>10}")
    for ell in ells:
        rff = RFFLayer(in_dim=z_dim, n_features=M_rff,
                       length_scale=[float(ell)], rff_seed=0).to(device)
        phi_fn = lambda z, _rff=rff: _rff(z)
        h_fn, h_char, _ = build_phi_leverage(phi_fn, Z_tr, ridge=ridge)
        with torch.no_grad():
            h_tr = h_fn(Z_tr).cpu().numpy()
            h_te = h_fn(Z_te).cpu().numpy()
            h_ot = h_fn(Z_ot).cpu().numpy()
        rec = dict(
            ell=float(ell),
            au_test=auroc(h_te, h_tr),
            au_other=auroc(h_ot, h_tr),
            dof_per_n=float(h_tr.sum()) / n_train,
            h_char=float(h_char),
        )
        sweep.append(rec)
        print(f"    {ell:>10.4f} {rec['au_test']:>10.4f} "
              f"{rec['au_other']:>10.4f} {rec['dof_per_n']:>10.4f}")
    return dict(linear=linear, sweep=sweep, z_dim=z_dim)


def integrate_metrics(sweep: list[dict], *,
                      dof_lo: float, dof_hi: float) -> dict | None:
    """Trapezoid-integrate AUROCs over log(ell), restricted to the band
    where dof/n is in [dof_lo, dof_hi] -- i.e. the kernel has resolution
    but isn't a delta. Returns per-band means and the band endpoints.
    """
    in_band = [s for s in sweep if dof_lo <= s["dof_per_n"] <= dof_hi]
    if len(in_band) < 2:
        return None
    log_ells = np.log(np.array([s["ell"] for s in in_band]))
    au_test = np.array([s["au_test"] for s in in_band])
    au_other = np.array([s["au_other"] for s in in_band])
    width = float(log_ells[-1] - log_ells[0])
    if width <= 0:
        return None
    return dict(
        ell_lo=in_band[0]["ell"], ell_hi=in_band[-1]["ell"],
        n_pts=len(in_band),
        mean_au_test=float(np.trapezoid(au_test, log_ells) / width),
        mean_au_other=float(np.trapezoid(au_other, log_ells) / width),
        mean_gap=float(np.trapezoid(au_other - au_test, log_ells) / width),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+",
                    default=["resnet18", "ssl", "dinov2", "vae"],
                    choices=list(BACKBONE_STYLE.keys()))
    ap.add_argument("--M", type=int, default=2048, dest="M_rff")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--n-train", type=int, default=8192, dest="n_train")
    ap.add_argument("--n-eval", type=int, default=2048, dest="n_eval")
    ap.add_argument("--batch", type=int, default=256, dest="batch_size")
    ap.add_argument("--ell-min", type=float, default=0.01)
    ap.add_argument("--ell-max", type=float, default=500.0)
    ap.add_argument("--n-ell", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    # Per-backbone knobs (defaults match the OOD threshold scripts).
    ap.add_argument("--resnet-tag", type=str, default="")
    ap.add_argument("--ssl-tag", type=str, default="")
    ap.add_argument("--dinov2-variant", type=str, default="dinov2_vitb14")
    ap.add_argument("--dinov2-dtype", choices=["fp32", "bf16", "fp16"],
                    default="fp32")
    ap.add_argument("--vae-z", type=int, default=256)
    ap.add_argument("--vae-beta", type=float, default=1.0)
    ap.add_argument("--dof-lo", type=float, default=0.001,
                    help="Lower dof/n bound of the integration band.")
    ap.add_argument("--dof-hi", type=float, default=0.10,
                    help="Upper dof/n bound of the integration band. "
                         "(0.25 is the delta-kernel ceiling for our M/n.)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    X_tr_np, _ = load_cifar_train("cifar10")
    X_te_np, _ = load_cifar_test("cifar10")
    X_other_np, _ = load_cifar_train("cifar100")
    ells = np.logspace(np.log10(args.ell_min), np.log10(args.ell_max),
                       args.n_ell)

    results: dict[str, dict] = {}
    for name in args.backbones:
        print(f"\n=== {name} ===")
        try:
            encode_fn, z_dim, model_obj = make_encode_fn(name, device, args)
        except FileNotFoundError as e:
            print(f"  skipped: {e}")
            continue
        print(f"  z_dim = {z_dim}")
        results[name] = scan_backbone(
            name, encode_fn, z_dim, X_tr_np, X_te_np, X_other_np, device,
            n_train=args.n_train, n_eval=args.n_eval, M_rff=args.M_rff,
            ridge=args.ridge, ells=ells, seed=args.seed,
        )
        results[name]["integrated"] = integrate_metrics(
            results[name]["sweep"], dof_lo=args.dof_lo, dof_hi=args.dof_hi,
        )
        ig = results[name]["integrated"]
        if ig is not None:
            print(f"  band [dof/n in {args.dof_lo}..{args.dof_hi}]: "
                  f"ell in [{ig['ell_lo']:.3f}, {ig['ell_hi']:.3f}], "
                  f"n_pts={ig['n_pts']}")
            print(f"    mean AUROC(test)={ig['mean_au_test']:.3f}  "
                  f"mean AUROC(c100)={ig['mean_au_other']:.3f}  "
                  f"mean gap={ig['mean_gap']:.3f}")
        else:
            print(f"  band [{args.dof_lo}..{args.dof_hi}] empty; no integration.")
        del encode_fn, model_obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not results:
        print("\nno results to plot.")
        return

    print("\n=== integrated metrics in the resolution band "
          f"(dof/n in [{args.dof_lo}, {args.dof_hi}]) ===")
    print(f"  {'backbone':<24} {'ell range':>16} "
          f"{'mean AU(test)':>13} {'mean AU(c100)':>13} {'mean gap':>9}  "
          f"{'lin AU(test)':>12} {'lin AU(c100)':>12} {'lin gap':>8}")
    for name, res in results.items():
        ig = res["integrated"]; lin = res["linear"]
        lin_gap = lin["au_other"] - lin["au_test"]
        if ig is None:
            print(f"  {name:<24} {'(empty band)':>16} "
                  f"{'-':>13} {'-':>13} {'-':>9}  "
                  f"{lin['au_test']:>12.3f} {lin['au_other']:>12.3f} "
                  f"{lin_gap:>8.3f}")
        else:
            rng = f"{ig['ell_lo']:.2f}..{ig['ell_hi']:.1f}"
            print(f"  {name:<24} {rng:>16} "
                  f"{ig['mean_au_test']:>13.3f} {ig['mean_au_other']:>13.3f} "
                  f"{ig['mean_gap']:>9.3f}  "
                  f"{lin['au_test']:>12.3f} {lin['au_other']:>12.3f} "
                  f"{lin_gap:>8.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11.5))
    ax_mem, ax_ood = axes[0]
    ax_dof, ax_sc = axes[1]

    for name, res in results.items():
        color, label = BACKBONE_STYLE[name]
        sweep = res["sweep"]
        ells_arr = np.array([s["ell"] for s in sweep])
        au_test = np.array([s["au_test"] for s in sweep])
        au_other = np.array([s["au_other"] for s in sweep])
        dof = np.array([s["dof_per_n"] for s in sweep])

        ax_mem.plot(ells_arr, au_test, "o-", color=color, lw=1.5, ms=3.5,
                    label=f"{label}  (lin: {res['linear']['au_test']:.3f})")
        ax_mem.axhline(res["linear"]["au_test"], color=color, ls=":",
                       lw=1.0, alpha=0.55)

        ax_ood.plot(ells_arr, au_other, "o-", color=color, lw=1.5, ms=3.5,
                    label=f"{label}  (lin: {res['linear']['au_other']:.3f})")
        ax_ood.axhline(res["linear"]["au_other"], color=color, ls=":",
                       lw=1.0, alpha=0.55)

        ax_dof.plot(ells_arr, dof, "o-", color=color, lw=1.5, ms=3.5,
                    label=f"{label}  (lin: {res['linear']['dof_per_n']:.3f})")
        ax_dof.axhline(res["linear"]["dof_per_n"], color=color, ls=":",
                       lw=1.0, alpha=0.55)

    for ax in (ax_mem, ax_ood):
        ax.axhline(0.5, color="black", ls="-", lw=0.7, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("RFF length scale ell")
        ax.set_ylabel("AUROC")
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8, loc="upper right")
    ax_mem.set_title("Memorization capacity\nAUROC(cifar10 test vs cifar10 train)",
                     fontsize=10)
    ax_ood.set_title("OOD discrimination\nAUROC(cifar100 vs cifar10 train)",
                     fontsize=10)

    ax_dof.set_xscale("log")
    ax_dof.set_xlabel("RFF length scale ell")
    ax_dof.set_ylabel("eff. dof / n_train")
    ax_dof.set_yscale("log")
    ax_dof.axhspan(args.dof_lo, args.dof_hi, color="gray", alpha=0.12,
                   label=f"integration band  [{args.dof_lo}, {args.dof_hi}]")
    ax_dof.grid(alpha=0.3)
    ax_dof.set_title("Effective dof  (0.25 = delta-kernel ceiling for "
                     f"M={args.M_rff}, n={args.n_train})", fontsize=10)
    ax_dof.legend(fontsize=8, loc="upper right")

    # 2D scatter: (memorization, OOD-discrimination) per backbone.
    ax_sc.plot([0.5, 1.0], [0.5, 1.0], "k:", lw=0.9, alpha=0.6,
               label="gap = 0  (no useful OOD over generalization)")
    for name, res in results.items():
        color, label = BACKBONE_STYLE[name]
        lin = res["linear"]
        ax_sc.plot(lin["au_test"], lin["au_other"],
                   "X", color=color, ms=14, mew=1.5, mec="black",
                   label=f"{label}  linear")
        ig = res["integrated"]
        if ig is not None:
            ax_sc.plot(ig["mean_au_test"], ig["mean_au_other"],
                       "o", color=color, ms=14, mec="black", mew=0.5,
                       label=f"{label}  RFF band-mean")
            # Connect linear and RFF points with a thin line.
            ax_sc.plot([lin["au_test"], ig["mean_au_test"]],
                       [lin["au_other"], ig["mean_au_other"]],
                       "-", color=color, lw=0.8, alpha=0.5)
    ax_sc.set_xlabel(
        "AUROC(cifar10 test vs cifar10 train)  -- lower = better generalization"
    )
    ax_sc.set_ylabel(
        "AUROC(cifar100 vs cifar10 train)  -- higher = better OOD discrim."
    )
    ax_sc.set_title("Feature-quality landscape\nX = linear phi=z   "
                    "o = RFF band-mean (integrated over log ell)", fontsize=10)
    ax_sc.grid(alpha=0.3)
    ax_sc.set_xlim(0.48, 1.02); ax_sc.set_ylim(0.48, 1.02)
    ax_sc.set_aspect("equal")
    ax_sc.legend(fontsize=7, loc="lower right", ncols=1)

    fig.suptitle(
        "Memorization-capacity scan on L2-normalized features  "
        "(Gram = cifar10 train, ridge = 1e-3, M_rff = {})".format(args.M_rff),
        fontsize=12,
    )
    fig.tight_layout()
    out = REPO_ROOT / "example" / "out" / "cifar10_memorization_scan.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
