"""Quick sanity grid for a cached MNIST VAE.

Self-contained -- does not depend on the legacy/fix recon_compare script.
Plots:

1. ``train`` reconstruction grid (8 random training images vs recon).
2. ``test``  reconstruction grid (8 random held-out images vs recon).
3. ``prior`` samples: decode ``z ~ N(0, I)`` to surface decoder bias /
   posterior collapse independently of the encoder.

Usage:
    python example/mnist/mnist_vae_sanity.py
    python example/mnist/mnist_vae_sanity.py --z 64 --beta 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mnist_vae_train import load_mnist_idx, load_mnist_train, load_vae, vae_ckpt_path


def _load_mnist_test() -> tuple[np.ndarray, np.ndarray]:
    repo_root = Path(__file__).resolve().parent.parent.parent
    mnist_dir = repo_root / "MNIST" / "raw"
    return load_mnist_idx(
        mnist_dir / "t10k-images-idx3-ubyte",
        mnist_dir / "t10k-labels-idx1-ubyte",
    )


def _imgrid(ax, imgs: np.ndarray, title: str, *, n: int = 8) -> None:
    """imgs: (B, 1, 28, 28) float; clipped to [0,1] for display."""
    B = imgs.shape[0]
    rows = (B + n - 1) // n
    canvas = np.ones((rows * 28, n * 28), dtype=np.float32)
    for i in range(B):
        r, c = divmod(i, n)
        canvas[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = imgs[i, 0]
    canvas = np.clip(canvas, 0.0, 1.0)
    ax.imshow(canvas, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=int, default=64, dest="z_dim")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--tag", type=str, default="",
                    help="Optional tag suffix on the checkpoint filename.")
    ap.add_argument("--n-show", type=int, default=8, dest="n_show")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    ckpt = vae_ckpt_path(args.z_dim, args.beta, tag=args.tag)
    if not ckpt.exists():
        raise FileNotFoundError(f"No cached VAE at {ckpt}. Run mnist_vae_train.py first.")
    vae = load_vae(ckpt, device)
    print(f"loaded {ckpt.name}  z_dim={vae.z_dim}")

    X_tr, _ = load_mnist_train()
    X_te, _ = _load_mnist_test()
    print(f"  train: {X_tr.shape}   test: {X_te.shape}")

    rng = np.random.default_rng(args.seed)
    idx_tr = rng.choice(X_tr.shape[0], size=args.n_show, replace=False)
    idx_te = rng.choice(X_te.shape[0], size=args.n_show, replace=False)
    x_tr = torch.as_tensor(
        X_tr[idx_tr].reshape(-1, 1, 28, 28), dtype=torch.float32, device=device,
    )
    x_te = torch.as_tensor(
        X_te[idx_te].reshape(-1, 1, 28, 28), dtype=torch.float32, device=device,
    )

    with torch.no_grad():
        mu_tr, _ = vae.encode(x_tr)
        mu_te, _ = vae.encode(x_te)
        r_tr = vae.decode(mu_tr).cpu().numpy()
        r_te = vae.decode(mu_te).cpu().numpy()
        z_prior = torch.randn(
            args.n_show, vae.z_dim, device=device,
            generator=torch.Generator(device=device).manual_seed(args.seed),
        )
        r_pr = vae.decode(z_prior).cpu().numpy()

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1])
    _imgrid(fig.add_subplot(gs[0, 0]), x_tr.cpu().numpy(), "train inputs")
    _imgrid(fig.add_subplot(gs[0, 1]), r_tr, "train reconstructions (mu)")
    _imgrid(fig.add_subplot(gs[1, 0]), x_te.cpu().numpy(), "test inputs")
    _imgrid(fig.add_subplot(gs[1, 1]), r_te, "test reconstructions (mu)")
    _imgrid(fig.add_subplot(gs[2, :]), r_pr,
            "prior samples: x_hat = decode(z), z ~ N(0, I)")

    fig.suptitle(
        f"MNIST VAE sanity: train recon | test recon | prior samples "
        f"(z_dim={vae.z_dim})",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "out" / "mnist_vae_sanity.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
