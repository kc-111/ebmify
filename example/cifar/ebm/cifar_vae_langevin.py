"""z-space Langevin sampling on a cached CIFAR beta-VAE.

Mirrors ``mnist_vae_langevin.py``: builds leverage on
``phi(z) = [z; RFF(z)]`` over a Z_train sample from the cached VAE, then
runs SamAdams overdamped Langevin in z-space. Decoded samples land on
the data manifold; the same h(z) doubles as an OOD detector.

Run (after training is cached):
    python example/cifar/ebm/cifar_vae_langevin.py --dataset cifar10
    python example/cifar/ebm/cifar_vae_langevin.py --dataset cifar100 --steps 50000
"""

from __future__ import annotations

import argparse
import sys
from argparse import BooleanOptionalAction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import REPO_ROOT  # noqa: E402

from ebmify.models.fc import RFFLayer  # noqa: E402
from ebmify.sampler import SamAdamsConfig, samadams_sample  # noqa: E402

from cifar_data import cifar_ckpt_path, load_cifar_train  # noqa: E402
from cifar_vae_train import load_vae  # noqa: E402
from hetero_demo_2d_ood_checkerboard_langevin import geometric_anneal  # noqa: E402
from mnist_vae_langevin import build_phi_leverage, build_z_leverage  # noqa: E402


def imgrid_rgb(ax, imgs: np.ndarray, title: str, n: int = 8) -> None:
    """imgs: (B, 3, H, W) float in [0, 1]."""
    B, C, H, W = imgs.shape
    rows = (B + n - 1) // n
    canvas = np.ones((rows * H, n * W, 3), dtype=np.float32)
    for i in range(B):
        r, c = divmod(i, n)
        canvas[r * H : (r + 1) * H, c * W : (c + 1) * W] = np.transpose(imgs[i], (1, 2, 0))
    ax.imshow(np.clip(canvas, 0, 1))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    ap.add_argument("--z", type=int, default=512, dest="z_dim")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--M", type=int, default=1024, dest="M_rff")
    ap.add_argument("--ell", type=float, default=0.5,
                    help="RFF length scale; default = median heuristic")
    ap.add_argument("--raw-z", action=BooleanOptionalAction, default=True,
                    dest="include_raw_z")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--n-part", type=int, default=64, dest="n_part")
    ap.add_argument("--dtau", type=float, default=5e-3)
    ap.add_argument("--dtau-lo", type=float, default=1e-4, dest="dtau_lo")
    ap.add_argument("--T", type=float, default=1e-2)
    ap.add_argument("--T-lo", type=float, default=1e-8, dest="T_lo")
    ap.add_argument("--hot-frac", type=float, default=0.0, dest="hot_frac")
    ap.add_argument("--anneal", choices=("on", "off"), default="on")
    ap.add_argument("--tag", type=str, default="")
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
    print(f"Loaded VAE from {ckpt}")

    X_tr, _ = load_cifar_train(args.dataset)
    rng = np.random.default_rng(0)
    sub_idx = rng.choice(len(X_tr), size=8192, replace=False)
    X_sub_t = torch.as_tensor(X_tr[sub_idx], dtype=torch.float32, device=device)
    with torch.no_grad():
        Z_train, _ = vae.encode(X_sub_t)
    print(f"  Z_train: {tuple(Z_train.shape)}  "
          f"||z||_2 median={Z_train.norm(dim=1).median().item():.3f}")

    if args.ell is None:
        rff = RFFLayer(
            in_dim=args.z_dim, n_features=args.M_rff,
            length_scale="median", rff_seed=0,
        ).to(device)
        with torch.no_grad():
            rff.init_bandwidth(Z_train)
    else:
        rff = RFFLayer(
            in_dim=args.z_dim, n_features=args.M_rff,
            length_scale=[args.ell], rff_seed=0,
        ).to(device)
    print(f"  RFF length_scale = {rff.length_scale.tolist()}, M = {args.M_rff}")

    h_fn, h_char, D = build_z_leverage(
        rff, Z_train, ridge=1e-3, bias=True, include_raw_z=args.include_raw_z,
    )
    print(f"  phi-dim D = {D}, include_raw_z = {args.include_raw_z}")
    print(f"  h_char (95th pct of in-data h(z)) = {h_char:.4e}")

    with torch.no_grad():
        z_prior = torch.randn(2048, args.z_dim, device=device)
        z_far = 5.0 * torch.randn(2048, args.z_dim, device=device)
        h_in = h_fn(Z_train).cpu().numpy() / h_char
        h_prior = h_fn(z_prior).cpu().numpy() / h_char
        h_far = h_fn(z_far).cpu().numpy() / h_char
    print(f"  h/h_char  in-data  median = {np.median(h_in):.3f}")
    print(f"  h/h_char  N(0,I)   median = {np.median(h_prior):.3f}")
    print(f"  h/h_char  N(0,25)  median = {np.median(h_far):.3f}")

    def E(z: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(h_fn(z) / h_char + 1.0)

    z0 = torch.randn(args.n_part, args.z_dim, device=device)
    z_ref = z0.clone().detach().requires_grad_(True)
    g_ref = torch.autograd.grad(E(z_ref).sum(), z_ref)[0]
    omega = float(g_ref.pow(2).sum(dim=-1).median().item())
    print(f"  Omega (median ||grad E||^2 at prior) = {omega:.4e}")

    dtau_hi = args.dtau
    dtau_lo = args.dtau_lo if args.dtau_lo is not None else args.dtau
    if args.anneal == "on":
        T_hi = args.T
        T_lo = args.T_lo if args.T_lo is not None else args.T * 1e-3
        hot_steps = int(args.hot_frac * args.steps)
        cool_steps = max(1, args.steps - hot_steps)
        T_cool = geometric_anneal(T_hi, T_lo, cool_steps)
        d_cool = geometric_anneal(dtau_hi, dtau_lo, cool_steps)
        def temperature(t: int) -> float:
            return T_hi if t < hot_steps else T_cool(t - hot_steps)
        def dtau_schedule(t: int) -> float:
            return dtau_hi if t < hot_steps else d_cool(t - hot_steps)
    else:
        T_hi = T_lo = args.T
        temperature = float(args.T)
        dtau_schedule = float(args.dtau)
        hot_steps = args.steps

    cfg = SamAdamsConfig(
        dtau=args.dtau, alpha=1.0, s=2.0, Omega=omega,
        m=0.05, M=10.0, r=0.5, kernel="psi1", grad_clip=10.0,
    )
    print(f"  SamAdams n_part={args.n_part}  steps={args.steps}  "
          f"dtau:{dtau_hi}->{dtau_lo}  T:{T_hi}->{T_lo}  "
          f"hot_steps={hot_steps if args.anneal == 'on' else 'all'}")

    out = samadams_sample(
        E, z0, n_steps=args.steps, temperature=temperature,
        dtau_schedule=dtau_schedule, config=cfg,
        record_every=50, zeta_init="g", log_every=1000,
    )
    z_final = out["x_final"].to(device)
    energies = out["energies"]
    print(f"  E start median = {np.median(energies[0]):.4f}")
    print(f"  E end   median = {np.median(energies[-1]):.4f}")

    with torch.no_grad():
        x_final = vae.decode(z_final).cpu().numpy()
        x_init = vae.decode(z0).cpu().numpy()
        ref_idx = rng.choice(len(X_tr), size=args.n_part, replace=False)
        X_ref_t = torch.as_tensor(X_tr[ref_idx], dtype=torch.float32, device=device)
        mu_ref, _ = vae.encode(X_ref_t)
        x_ref = vae.decode(mu_ref).cpu().numpy()

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
    ax_e = fig.add_subplot(gs[0, :])
    rec_steps = np.linspace(0, args.steps, energies.shape[0])
    ax_e.plot(rec_steps, energies[:, :16], color="gray", alpha=0.3, lw=0.8)
    ax_e.plot(rec_steps, np.median(energies, axis=1),
              color="C3", lw=2.0, label="median")
    ax_e.set_xlabel("step"); ax_e.set_ylabel("E(z)")
    ax_e.set_title(
        f"z-space SamAdams ({args.dataset})  z_dim={args.z_dim}  "
        f"M_rff={args.M_rff}  ell={rff.length_scale.tolist()}  "
        f"raw_z={args.include_raw_z}  T:{T_hi}->{T_lo}"
    )
    ax_e.legend(); ax_e.grid(alpha=0.3)

    ax1 = fig.add_subplot(gs[1, 0])
    imgrid_rgb(ax1, x_init, "decode(z0 ~ N(0,I))")
    ax2 = fig.add_subplot(gs[1, 1])
    imgrid_rgb(ax2, x_final, "decode(z_final) — after Langevin")
    ax3 = fig.add_subplot(gs[2, 0])
    imgrid_rgb(ax3, x_ref, "decode(encode(CIFAR)) — reconstruction ceiling")
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.hist(np.log10(h_in + 1e-6), bins=60, alpha=0.5,
             label="in-data", color="C0")
    ax4.hist(np.log10(h_prior + 1e-6), bins=60, alpha=0.5,
             label="z~N(0,I)", color="C1")
    ax4.set_xlabel("log10(h(z)/h_char)"); ax4.legend()
    ax4.set_title("Leverage separation")

    fig.tight_layout()
    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = REPO_ROOT / "example" / "out" / "cifar"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_vae_langevin{suffix}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
