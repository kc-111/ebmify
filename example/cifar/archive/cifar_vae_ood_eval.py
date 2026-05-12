"""x -> z OOD evaluation on a cached CIFAR beta-VAE.

Mirrors ``mnist_vae_ood_eval.py``: encodes a Z_train subsample from the
trained dataset, then measures leverage separation for several OOD x
sources under phi in {z, RFF(z), [z; RFF(z)]}. Adds a **cross-dataset**
column: the CIFAR-10 VAE scoring CIFAR-100 inputs and vice versa --
the standard hard case in the OOD-detection literature.

Examples:
    python example/cifar/cifar_vae_ood_eval.py --dataset cifar10
    python example/cifar/cifar_vae_ood_eval.py --dataset cifar100 --M 2048
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "example" / "mnist"))

from ebmify.models.fc import RFFLayer  # noqa: E402

from cifar_data import cifar_ckpt_path, load_cifar_train  # noqa: E402
from cifar_vae_train import load_vae  # noqa: E402
from mnist_vae_langevin import (  # noqa: E402
    build_ood_x_sources,
    plot_x_to_z_leverage_separation,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    ap.add_argument("--z", type=int, default=256, dest="z_dim")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--M", type=int, default=2048, dest="M_rff")
    ap.add_argument("--ell", type=float, default=0.1,
                    help="RFF length scale; default = median heuristic")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--n-train", type=int, default=8192, dest="n_train")
    ap.add_argument("--n-eval", type=int, default=2048, dest="n_eval")
    ap.add_argument("--tag", type=str, default="")
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
    print(f"Loaded VAE from {ckpt}")

    X_tr, _ = load_cifar_train(args.dataset)
    other = "cifar100" if args.dataset == "cifar10" else "cifar10"
    X_other, _ = load_cifar_train(other)

    rng = np.random.default_rng(0)
    sub_idx = rng.choice(len(X_tr), size=args.n_train, replace=False)
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
    print(f"  RFF length_scale = {rff.length_scale.tolist()}  M = {args.M_rff}")
    print(f"  ridge = {args.ridge}")

    # Build the standard OOD set from X_sub_t, then insert a cross-dataset
    # source between the in-data and the synthetic OOD sources.
    base = build_ood_x_sources(
        X_sub_t, device, n_eval=args.n_eval, seed=args.seed,
        in_name=args.dataset,
    )
    cross_idx = rng.choice(len(X_other), size=args.n_eval, replace=False)
    x_cross = torch.as_tensor(
        X_other[cross_idx], dtype=torch.float32, device=device,
    )
    x_sources = [base[0], (other, x_cross, "C8")] + base[1:]

    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = REPO_ROOT / "example" / "out"
    out_dir.mkdir(exist_ok=True)
    plot_x_to_z_leverage_separation(
        vae, X_sub_t, Z_train, rff, device,
        out_dir / f"{args.dataset}_vae_ood_eval{suffix}.png",
        M_rff=args.M_rff, n_eval=args.n_eval, ridge=args.ridge,
        seed=args.seed, in_name=args.dataset,
        x_sources=x_sources,
    )


if __name__ == "__main__":
    main()
