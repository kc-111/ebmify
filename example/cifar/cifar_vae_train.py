"""Train a beta-VAE on CIFAR-10 or CIFAR-100 and cache the weights.

Architecture: a 4-stage residual conv encoder/decoder
(3 -> 32 -> 64 -> 128 -> 256, ending at a 2x2 feature map) bracketed by
residual MLP trunks. All building blocks come from ``ebmify.models``
and follow the same pre-norm + residual + SiLU/odd-piecewise activation
conventions as :class:`FCNet`. Default ``z_dim=64`` to match CIFAR's
higher intrinsic dimension.

Usage:
    python example/cifar/cifar_vae_train.py --dataset cifar10
    python example/cifar/cifar_vae_train.py --dataset cifar100 --epochs 80
    python example/cifar/cifar_vae_train.py --dataset cifar10 --gaussian-loss
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ebmify.models import ConvResVAE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cifar_data import cifar_ckpt_path, load_cifar_train  # noqa: E402


class CifarVAE(ConvResVAE):
    """CIFAR beta-VAE: 32x32x3 input, four stride-2 residual conv blocks
    (32 -> 64 -> 128 -> 256), then a single-hidden-layer residual MLP
    trunk to ``z_dim``. Mirror decoder.
    """

    def __init__(
        self,
        z_dim: int = 64,
        *,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        fc_hidden: tuple[int, ...] = (512,),
        activation: str = "silu",
        sigmoid_out: bool = True,
    ) -> None:
        super().__init__(
            input_shape=(3, 32, 32),
            z_dim=z_dim,
            channels=channels,
            fc_hidden=fc_hidden,
            activation=activation,
            sigmoid_out=sigmoid_out,
        )


def vae_loss(x_recon, x, mu, logvar, beta: float, gaussian_loss: bool):
    if gaussian_loss:
        recon = F.mse_loss(x_recon, x, reduction="sum") / x.shape[0]
    else:
        recon = F.binary_cross_entropy(x_recon, x, reduction="sum") / x.shape[0]
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    return recon + beta * kld, recon, kld


def train_vae(
    vae: CifarVAE, X_tr: np.ndarray, device: str,
    *, epochs: int, batch_size: int, lr: float, beta: float,
    gaussian_loss: bool,
):
    X_t = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
    n = X_t.shape[0]
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        idx = rng.permutation(n)
        vae.train()
        rec_sum = kld_sum = 0.0
        nb = 0
        for s in range(0, n, batch_size):
            b = idx[s : s + batch_size]
            xb = X_t[b]
            x_recon, mu, logvar = vae(xb)
            loss, rec, kld = vae_loss(
                x_recon, xb, mu, logvar, beta, gaussian_loss,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            rec_sum += float(rec.detach())
            kld_sum += float(kld.detach())
            nb += 1
        tag = "mse" if gaussian_loss else "bce"
        print(f"  epoch {epoch+1:3d}/{epochs}  "
              f"{tag}={rec_sum/nb:.3f}  kld={kld_sum/nb:.3f}")
    vae.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    ap.add_argument("--z", type=int, default=128, dest="z_dim")
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--gaussian-loss", action="store_true",
                    help="MSE on RGB instead of BCE.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    X_tr, _ = load_cifar_train(args.dataset)
    print(f"  {args.dataset} train: {X_tr.shape}")

    ckpt = cifar_ckpt_path(args.dataset, args.z_dim, args.beta)
    vae = CifarVAE(
        z_dim=args.z_dim,
        sigmoid_out=not args.gaussian_loss,
    ).to(device)
    print(f"Training beta-VAE on {args.dataset} (z={args.z_dim}, "
          f"beta={args.beta}, epochs={args.epochs}) ...")
    train_vae(
        vae, X_tr, device,
        epochs=args.epochs, batch_size=args.batch, lr=args.lr,
        beta=args.beta, gaussian_loss=args.gaussian_loss,
    )
    torch.save(vae.state_dict(), ckpt)
    print(f"  saved {ckpt}")


if __name__ == "__main__":
    main()
