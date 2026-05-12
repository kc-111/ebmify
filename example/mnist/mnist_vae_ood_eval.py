"""Tune x -> z OOD classification on the cached β-VAE.

Loads the cached VAE, encodes a Z_train subsample, and evaluates leverage
separation for several OOD x distributions under three phi maps:

    phi = z
    phi = RFF(z)
    phi = [z; RFF(z)]

OOD x sources: MNIST (in-data), uniform, Bernoulli, Gaussian noise,
pixel-shuffled MNIST, inverted MNIST, all-black, all-white.

Examples:
    # default tuning sweep (M=2048, median bandwidth)
    python example/mnist/mnist_vae_ood_eval.py

    # tighter kernel
    python example/mnist/mnist_vae_ood_eval.py --M 2048 --ell 0.8

    # bigger Z_train, lower ridge
    python example/mnist/mnist_vae_ood_eval.py --n-train 16384 --ridge 1e-4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ebmify.models.fc import RFFLayer

from mnist_vae_train import load_mnist_train, load_vae, vae_ckpt_path
from mnist_vae_langevin import plot_x_to_z_leverage_separation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=int, default=64, dest="z_dim")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--M", type=int, default=1024, dest="M_rff",
                     help="number of RFF features (default 2048)")
    ap.add_argument("--ell", type=float, default=0.1,
                     help="RFF length scale; default = median heuristic")
    ap.add_argument("--ridge", type=float, default=1e-3,
                     help="ridge added to Phi^T Phi before Cholesky")
    ap.add_argument("--n-train", type=int, default=8192, dest="n_train",
                     help="number of MNIST samples used to fit leverage")
    ap.add_argument("--n-eval", type=int, default=2048, dest="n_eval",
                     help="number of x's per OOD source for evaluation")
    ap.add_argument("--tag", type=str, default="",
                     help="suffix for output plot filename")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    ckpt = vae_ckpt_path(args.z_dim, args.beta)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No cached VAE at {ckpt}. Run mnist_vae_train.py first."
        )
    vae = load_vae(ckpt, device)
    print(f"Loaded VAE from {ckpt}")

    X_tr, _ = load_mnist_train()
    rng = np.random.default_rng(args.seed)
    sub_idx = rng.choice(len(X_tr), size=args.n_train, replace=False)
    X_sub_t = torch.as_tensor(
        X_tr[sub_idx].reshape(-1, 1, 28, 28),
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        Z_train, _ = vae.encode(X_sub_t)
    print(f"  Z_train: {tuple(Z_train.shape)}  "
          f"||z||_2 median={Z_train.norm(dim=1).median().item():.3f}")

    if args.ell is None:
        rff = RFFLayer(
            in_dim=args.z_dim, n_features=args.M_rff,
            length_scale="median", rff_seed=args.seed,
        ).to(device)
        with torch.no_grad():
            rff.init_bandwidth(Z_train)
    else:
        rff = RFFLayer(
            in_dim=args.z_dim, n_features=args.M_rff,
            length_scale=[args.ell], rff_seed=args.seed,
        ).to(device)
    print(f"  RFF length_scale = {rff.length_scale.tolist()}  M = {args.M_rff}")
    print(f"  ridge = {args.ridge}")

    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = Path(__file__).resolve().parent.parent / "out" / "mnist"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mnist_vae_ood_eval{suffix}.png"

    plot_x_to_z_leverage_separation(
        vae, X_sub_t, Z_train, rff, device, out_path,
        M_rff=args.M_rff, n_eval=args.n_eval, ridge=args.ridge,
        seed=args.seed + 1,
    )


if __name__ == "__main__":
    main()
