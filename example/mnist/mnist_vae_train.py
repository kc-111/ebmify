"""Train a small beta-VAE on MNIST and cache the weights.

Architecture: residual conv encoder/decoder + residual MLP trunks from
:mod:`ebmify.models.conv`, with optional RFF feature lifts on the MLP
trunks, inside the conv blocks, and on the decoder's pre-readout
features. RFF args parallel :class:`FCNet` -- see :class:`ConvResVAE`.

Stability and posterior-collapse mitigations:

* **KL warmup** (default 10 epochs): beta ramps linearly from 0 to
  ``--beta``. Without it, the z=64 model collapses 43/64 latent dims
  at beta=1.0.
* **Gradient clipping** (default 5.0): cheap insurance against
  outlier-batch gradient spikes.
* **Free bits** is OFF by default. A KL floor forces every dim to
  carry posterior noise, which the decoder then has to average over --
  blurring exactly the textures we want sharp. Dims that naturally
  collapse pay no cost; the surviving active dims stay sharp.

Usage:
    python example/mnist/mnist_vae_train.py
    python example/mnist/mnist_vae_train.py --z 32 --kl-warmup-epochs 20
    python example/mnist/mnist_vae_train.py --mlp-output-rff 256
"""

from __future__ import annotations

import argparse
import json
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
    ``z_dim``. Mirror decoder. All RFF kwargs are forwarded to
    :class:`ConvResVAE`.
    """

    def __init__(
        self,
        z_dim: int = 64,
        *,
        channels: tuple[int, ...] = (32, 64),
        fc_hidden: tuple[int, ...] = (256,),
        activation: str = "silu",
        **rff_kwargs,
    ) -> None:
        super().__init__(
            input_shape=(1, 28, 28),
            z_dim=z_dim,
            channels=channels,
            fc_hidden=fc_hidden,
            activation=activation,
            sigmoid_out=True,
            **rff_kwargs,
        )


def vae_loss(
    x_recon, x, mu, logvar,
    *, beta: float, free_bits: float = 0.0,
):
    """Beta-VAE objective with optional free-bits KL floor.

    Free-bits: per-dim KL is clamped from below at ``free_bits`` nats
    before summing. With ``free_bits=0`` this reduces to the standard
    beta-VAE loss.
    """
    bce = F.binary_cross_entropy(x_recon, x, reduction="sum") / x.shape[0]
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    if free_bits > 0.0:
        kl_per_dim = kl_per_dim.clamp(min=free_bits)
    kld = kl_per_dim.sum(dim=-1).mean()
    return bce + beta * kld, bce, kld


def train_vae(
    vae: VAE, X_tr: np.ndarray, device: str,
    *, epochs: int, batch_size: int, lr: float, beta: float,
    kl_warmup_epochs: int = 0, free_bits: float = 0.0,
    grad_clip: float = 5.0,
):
    X_t = torch.as_tensor(
        X_tr.reshape(-1, 1, 28, 28), dtype=torch.float32, device=device,
    )
    n = X_t.shape[0]
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        # Linear warmup: beta=0 -> beta over the first kl_warmup_epochs.
        if kl_warmup_epochs > 0:
            beta_t = beta * min(1.0, (epoch + 1) / kl_warmup_epochs)
        else:
            beta_t = beta
        idx = rng.permutation(n)
        vae.train()
        bce_sum = kld_sum = 0.0
        nb = 0
        for s in range(0, n, batch_size):
            b = idx[s : s + batch_size]
            xb = X_t[b]
            x_recon, mu, logvar = vae(xb)
            loss, bce, kld = vae_loss(
                x_recon, xb, mu, logvar,
                beta=beta_t, free_bits=free_bits,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(vae.parameters(), grad_clip)
            opt.step()
            bce_sum += float(bce.detach())
            kld_sum += float(kld.detach())
            nb += 1
        print(f"  epoch {epoch+1:3d}/{epochs}  beta={beta_t:.3f}  "
              f"bce={bce_sum/nb:.3f}  kld={kld_sum/nb:.3f}")
    vae.eval()


def vae_ckpt_path(z_dim: int, beta: float, tag: str = "") -> Path:
    cache_dir = Path(__file__).resolve().parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    return cache_dir / f"mnist_vae_z{z_dim}_beta{beta}{suffix}.pt"


def load_vae(path: Path, device: str) -> VAE:
    """Load a checkpoint produced by this script.

    The new format is ``{'state_dict': ..., 'config': {...}}``. The legacy
    format was a bare state_dict; that path is detected and falls back to
    default architecture args.
    """
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw and "config" in raw:
        cfg = raw["config"]
        vae = VAE(**cfg).to(device)
        vae.load_state_dict(raw["state_dict"])
    else:
        # Legacy: bare state_dict, assume default ConvResVAE arch.
        vae = VAE().to(device)
        vae.load_state_dict(raw)
    vae.eval()
    return vae


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=int, default=64, dest="z_dim")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    # Posterior-collapse mitigations
    ap.add_argument("--kl-warmup-epochs", type=int, default=10, dest="kl_warmup_epochs",
                    help="Linear KL warmup window (epochs). 0 disables.")
    ap.add_argument("--free-bits", type=float, default=0.0, dest="free_bits",
                    help="Per-dim KL floor (nats). 0 disables (default). Off because "
                         "forcing every dim to carry posterior noise blurs recons; "
                         "naturally-active dims are sharper without the floor.")
    ap.add_argument("--grad-clip", type=float, default=5.0, dest="grad_clip",
                    help="Max grad norm. Defense against single-batch grad spikes.")
    # RFF placements (default off; mirror ConvResVAE's surface)
    ap.add_argument("--mlp-input-rff", type=int, default=None, dest="mlp_input_rff")
    ap.add_argument("--mlp-output-rff", type=int, default=None, dest="mlp_output_rff")
    ap.add_argument("--mlp-block-type", choices=("linear", "rff"), default="linear",
                    dest="mlp_block_type")
    ap.add_argument("--mlp-block-rff-features", type=int, default=None,
                    dest="mlp_block_rff_features")
    ap.add_argument("--conv-block-type", choices=("linear", "rff"), default="linear",
                    dest="conv_block_type")
    ap.add_argument("--conv-block-rff-features", type=int, default=None,
                    dest="conv_block_rff_features")
    ap.add_argument("--dec-pre-readout-rff", type=int, default=None,
                    dest="dec_pre_readout_rff")
    ap.add_argument("--rff-seed", type=int, default=0, dest="rff_seed")
    ap.add_argument("--tag", type=str, default="",
                    help="Suffix for the cache filename.")
    ap.add_argument("--resume", action="store_true",
                    help="If the target checkpoint already exists, load its weights and "
                         "continue training. KL warmup is skipped since the loaded model "
                         "is already past it.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    X_tr, _ = load_mnist_train()
    print(f"  train: {X_tr.shape}")

    config = dict(
        z_dim=args.z_dim,
        mlp_input_rff=args.mlp_input_rff,
        mlp_output_rff=args.mlp_output_rff,
        mlp_block_type=args.mlp_block_type,
        mlp_block_rff_features=args.mlp_block_rff_features,
        conv_block_type=args.conv_block_type,
        conv_block_rff_features=args.conv_block_rff_features,
        dec_pre_readout_rff=args.dec_pre_readout_rff,
        rff_seed=args.rff_seed,
    )
    vae = VAE(**config).to(device)
    # Calibrate "median" RFF bandwidths from a real input sample.
    cal_idx = np.random.default_rng(0).choice(X_tr.shape[0], size=2048, replace=False)
    cal_x = torch.as_tensor(
        X_tr[cal_idx].reshape(-1, 1, 28, 28), dtype=torch.float32, device=device,
    )
    vae.init_rff_bandwidths(cal_x)

    ckpt = vae_ckpt_path(args.z_dim, args.beta, tag=args.tag)
    n_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"  parameters: {n_params:,}")

    warmup = args.kl_warmup_epochs
    if args.resume and ckpt.exists():
        print(f"  resuming from {ckpt}")
        raw = torch.load(ckpt, map_location=device, weights_only=False)
        sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        vae.load_state_dict(sd)
        if warmup > 0:
            print(f"  (resume: skipping KL warmup; was --kl-warmup-epochs {warmup})")
            warmup = 0

    print(f"Training beta-VAE (z={args.z_dim}, beta={args.beta}, "
          f"epochs={args.epochs}, warmup={warmup}, "
          f"free_bits={args.free_bits}) ...")
    train_vae(
        vae, X_tr, device,
        epochs=args.epochs, batch_size=args.batch, lr=args.lr, beta=args.beta,
        kl_warmup_epochs=warmup, free_bits=args.free_bits,
        grad_clip=args.grad_clip,
    )
    torch.save({"state_dict": vae.state_dict(), "config": config}, ckpt)
    print(f"  saved {ckpt}")
    print(f"  config: {json.dumps(config)}")


if __name__ == "__main__":
    main()
