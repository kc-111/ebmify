"""Train a small β-VAE on MNIST and cache the weights.

Usage:
    python example/mnist/mnist_vae_train.py             # default config
    python example/mnist/mnist_vae_train.py --z 32      # different z_dim
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ebmify.models import ConvResVAE


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def load_mnist_idx(images_path: Path, labels_path: Path):
    with open(images_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    with open(labels_path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return images.astype(np.float32) / 255.0, labels.astype(np.int64)


def load_mnist_train():
    repo_root = Path(__file__).resolve().parent.parent.parent
    mnist_dir = repo_root / "MNIST" / "raw"
    return load_mnist_idx(
        mnist_dir / "train-images-idx3-ubyte",
        mnist_dir / "train-labels-idx1-ubyte",
    )


# ----------------------------------------------------------------------
# beta-VAE: residual conv + residual MLP trunks (see ebmify.models.conv)
# ----------------------------------------------------------------------

class VAE(ConvResVAE):
    """MNIST beta-VAE: 28x28 input, two stride-2 residual conv blocks
    (32 -> 64), then a single-hidden-layer residual MLP trunk to
    ``z_dim``. Mirror decoder.
    """

    def __init__(
        self,
        z_dim: int = 64,
        *,
        channels: tuple[int, ...] = (32, 64),
        fc_hidden: tuple[int, ...] = (256,),
        activation: str = "silu",
    ) -> None:
        super().__init__(
            input_shape=(1, 28, 28),
            z_dim=z_dim,
            channels=channels,
            fc_hidden=fc_hidden,
            activation=activation,
            sigmoid_out=True,
        )


def vae_loss(x_recon, x, mu, logvar, beta: float = 1.0):
    bce = F.binary_cross_entropy(x_recon, x, reduction="sum") / x.shape[0]
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    return bce + beta * kld, bce, kld


def train_vae(
    vae: VAE, X_tr: np.ndarray, device: str,
    *, epochs: int, batch_size: int, lr: float, beta: float,
):
    X_t = torch.as_tensor(
        X_tr.reshape(-1, 1, 28, 28), dtype=torch.float32, device=device,
    )
    n = X_t.shape[0]
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        idx = rng.permutation(n)
        vae.train()
        bce_sum = kld_sum = 0.0
        nb = 0
        for s in range(0, n, batch_size):
            b = idx[s : s + batch_size]
            xb = X_t[b]
            x_recon, mu, logvar = vae(xb)
            loss, bce, kld = vae_loss(x_recon, xb, mu, logvar, beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bce_sum += float(bce.detach())
            kld_sum += float(kld.detach())
            nb += 1
        print(f"  epoch {epoch+1:3d}/{epochs}  "
              f"bce={bce_sum/nb:.3f}  kld={kld_sum/nb:.3f}")
    vae.eval()


def vae_ckpt_path(z_dim: int, beta: float) -> Path:
    cache_dir = Path(__file__).resolve().parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"mnist_vae_z{z_dim}_beta{beta}.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=int, default=64, dest="z_dim")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    X_tr, _ = load_mnist_train()
    print(f"  train: {X_tr.shape}")

    ckpt = vae_ckpt_path(args.z_dim, args.beta)
    vae = VAE(z_dim=args.z_dim).to(device)
    print(f"Training β-VAE (z={args.z_dim}, β={args.beta}, "
          f"epochs={args.epochs}) ...")
    train_vae(
        vae, X_tr, device,
        epochs=args.epochs, batch_size=args.batch, lr=args.lr, beta=args.beta,
    )
    torch.save(vae.state_dict(), ckpt)
    print(f"  saved {ckpt}")


if __name__ == "__main__":
    main()
