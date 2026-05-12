"""Train a beta-VAE on CIFAR-10 or CIFAR-100 and cache the weights.

Architecture: a 4-stage residual conv encoder/decoder
(3 -> 32 -> 64 -> 128 -> 256, ending at a 2x2 feature map) bracketed by
residual MLP trunks from :mod:`ebmify.models.conv`. Optional RFF feature
lifts (MLP trunks, inside conv blocks, decoder pre-readout) -- see
:class:`ConvResVAE`.

The decoder always uses Gaussian NLL (BCE on RGB is mis-specified).
The output is unconstrained (no sigmoid) so the proper Gaussian
likelihood can be applied directly.

Stability and posterior-collapse mitigations:

* **KL warmup** (default 15 epochs): beta ramps linearly from 0 to
  ``--beta``. Without it, the z=128 model collapses ~78/128 latent dims.
* **Gradient clipping** (default 5.0): critical with ``--learn-sigma``,
  since Gaussian-NLL gradients scale as ``1/sigma^2`` and an outlier
  batch at sigma ~ 0.07 can yank the encoder logvar head into a
  degenerate region in one step.
* **log_sigma clamp** (``[-7, 2]``, sigma in ``[9e-4, 7.4]``): defends
  against pathological learn_sigma trajectories.
* **Encoder logvar clamp** to ``[-10, 10]`` (built into ConvResVAE).
* **Free bits** is OFF by default (forces every dim to carry posterior
  noise, over-smoothing recons). KL warmup + grad clipping alone is
  enough; surviving active dims are sharper than free-bits would give.
* **Learned output sigma** (``--learn-sigma``, on by default): the
  Gaussian decoder's variance is a single learnable scalar. Replaces
  the implicit fixed-variance assumption of plain MSE; the model trades
  reconstruction precision against latent rate without external beta
  tuning.

Usage:
    python example/cifar/train/cifar_vae_train.py --dataset cifar10
    python example/cifar/train/cifar_vae_train.py --dataset cifar10 --no-learn-sigma
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _paths  # noqa: F401, E402
from ebmify.models import ConvResVAE  # noqa: E402
from cifar_data import cifar_ckpt_path, load_cifar_train  # noqa: E402


class CifarVAE(ConvResVAE):
    """CIFAR beta-VAE: 32x32x3 input, four stride-2 residual conv blocks
    (32 -> 64 -> 128 -> 256), then a single-hidden-layer residual MLP
    trunk to ``z_dim``. Mirror decoder.

    All RFF kwargs are forwarded to :class:`ConvResVAE`. ``learn_sigma``
    adds a learnable scalar ``log_sigma`` to the module (used by
    :func:`vae_loss` for the Gaussian decoder NLL).
    """

    def __init__(
        self,
        z_dim: int = 64,
        *,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        fc_hidden: tuple[int, ...] = (512,),
        activation: str = "silu",
        sigmoid_out: bool = True,
        learn_sigma: bool = False,
        **rff_kwargs,
    ) -> None:
        super().__init__(
            input_shape=(3, 32, 32),
            z_dim=z_dim,
            channels=channels,
            fc_hidden=fc_hidden,
            activation=activation,
            sigmoid_out=sigmoid_out,
            **rff_kwargs,
        )
        self.learn_sigma = bool(learn_sigma)
        if self.learn_sigma:
            # log_sigma starts at 0 -> sigma=1. The optimizer will pull it
            # down toward the true per-pixel residual scale.
            self.log_sigma = nn.Parameter(torch.zeros(()))


# Bound log_sigma to a sane range so a transient pathological update
# can't drive sigma to 0 (inv_var -> inf) or sigma -> inf (NLL -> inf).
# Range [-7, 2] corresponds to sigma in [~9e-4, ~7.4], generous in both
# directions for image data in [0, 1].
LOG_SIGMA_MIN = -7.0
LOG_SIGMA_MAX = 2.0


def vae_loss(
    x_recon, x, mu, logvar,
    *, beta: float, free_bits: float = 0.0,
    log_sigma: torch.Tensor | None = None,
):
    """beta-VAE Gaussian-NLL loss with optional learned scalar sigma.

    Reconstruction term is the proper Gaussian NLL
    ``0.5 * ((x - x_recon)^2 / sigma^2 + log(2 pi sigma^2))`` summed
    per sample. ``log_sigma`` (scalar tensor) is the learnable
    parameter; when ``None``, ``sigma=1`` is implied (plain MSE).
    log_sigma is clamped to [LOG_SIGMA_MIN, LOG_SIGMA_MAX] for stability;
    clamp is non-differentiable at the bounds but in practice log_sigma
    sits well inside the range during training.
    """
    if log_sigma is not None:
        log_sigma_eff = log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        inv_var = (-2.0 * log_sigma_eff).exp()
        const = 2.0 * log_sigma_eff + math.log(2.0 * math.pi)
        per_px = 0.5 * ((x - x_recon).pow(2) * inv_var + const)
    else:
        per_px = 0.5 * (x - x_recon).pow(2)
    recon = per_px.sum(dim=(1, 2, 3)).mean()
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    if free_bits > 0.0:
        kl_per_dim = kl_per_dim.clamp(min=free_bits)
    kld = kl_per_dim.sum(dim=-1).mean()
    return recon + beta * kld, recon, kld


def train_vae(
    vae: CifarVAE, X_tr: np.ndarray, device: str,
    *, epochs: int, batch_size: int, lr: float, beta: float,
    kl_warmup_epochs: int = 0, free_bits: float = 0.0,
    grad_clip: float = 5.0,
):
    """Train ``vae`` in place under Gaussian-NLL.

    ``grad_clip`` clips the total gradient norm before each optimizer
    step. With ``--learn-sigma`` the recon term scales as ``1/sigma^2``,
    so a hard batch at sigma ~ 0.07 amplifies weight gradients ~200x.
    Without clipping the encoder's logvar head can yank the posterior
    into a degenerate region in a single step and KLD explodes by
    100x+ -- standard fix is gradient clipping.
    """
    X_t = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
    n = X_t.shape[0]
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    log_sigma = getattr(vae, "log_sigma", None)
    for epoch in range(epochs):
        if kl_warmup_epochs > 0:
            beta_t = beta * min(1.0, (epoch + 1) / kl_warmup_epochs)
        else:
            beta_t = beta
        idx = rng.permutation(n)
        vae.train()
        rec_sum = kld_sum = 0.0
        nb = 0
        for s in range(0, n, batch_size):
            b = idx[s : s + batch_size]
            xb = X_t[b]
            x_recon, mu, logvar = vae(xb)
            loss, rec, kld = vae_loss(
                x_recon, xb, mu, logvar,
                beta=beta_t, free_bits=free_bits, log_sigma=log_sigma,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(vae.parameters(), grad_clip)
            opt.step()
            rec_sum += float(rec.detach())
            kld_sum += float(kld.detach())
            nb += 1
        sigma_str = (f"  sigma={log_sigma.exp().item():.4f}"
                     if log_sigma is not None else "")
        print(f"  epoch {epoch+1:3d}/{epochs}  beta={beta_t:.3f}  "
              f"nll={rec_sum/nb:.3f}  kld={kld_sum/nb:.3f}{sigma_str}")
    vae.eval()


def load_vae(path: Path, device: str) -> CifarVAE:
    """Load a checkpoint produced by this script (new or legacy format)."""
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw and "config" in raw:
        cfg = raw["config"]
        vae = CifarVAE(**cfg).to(device)
        vae.load_state_dict(raw["state_dict"])
    else:
        # Legacy: bare state_dict, assume default ConvResVAE arch + z_dim=128
        # (the cached default). Caller may need to pass z_dim explicitly for
        # other shapes.
        vae = CifarVAE(z_dim=128).to(device)
        vae.load_state_dict(raw)
    vae.eval()
    return vae


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    ap.add_argument("--z", type=int, default=32, dest="z_dim")
    ap.add_argument("--channels", type=int, nargs="+",
                    default=[64, 128, 256, 512],
                    help="Conv channel widths per stage. Defaults to (32,64,128,256). "
                         "Use --channels 64 128 256 512 for a 2x-wider decoder.")
    ap.add_argument("--fc-hidden", type=int, nargs="+", default=[512],
                    dest="fc_hidden",
                    help="MLP trunk hidden widths. Defaults to (512,).")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    # Posterior-collapse mitigations
    ap.add_argument("--kl-warmup-epochs", type=int, default=15,
                    dest="kl_warmup_epochs")
    ap.add_argument("--free-bits", type=float, default=0.0, dest="free_bits",
                    help="Per-dim KL floor (nats). 0 disables (default). Free-bits forces "
                         "every dim to carry posterior noise, which over-smooths recons. "
                         "Use KL warmup + grad clipping instead; if any dims do collapse, "
                         "the remaining active dims are still sharper than free-bits would give.")
    ap.add_argument("--grad-clip", type=float, default=5.0, dest="grad_clip",
                    help="Max grad norm. Critical for Gaussian-NLL + learn_sigma stability "
                         "since 1/sigma^2 amplifies gradients ~200x at sigma~0.07.")
    ap.add_argument("--learn-sigma", action=argparse.BooleanOptionalAction,
                    default=True, dest="learn_sigma",
                    help="Learnable scalar log_sigma for the Gaussian decoder "
                         "(on by default; use --no-learn-sigma to pin sigma=1).")
    # RFF placements
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
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--resume", action="store_true",
                    help="If the target checkpoint already exists, load its weights and "
                         "continue training. KL warmup is skipped since the loaded model "
                         "is already past it.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    X_tr, _ = load_cifar_train(args.dataset)
    print(f"  {args.dataset} train: {X_tr.shape}")

    config = dict(
        z_dim=args.z_dim,
        channels=tuple(args.channels),
        fc_hidden=tuple(args.fc_hidden),
        sigmoid_out=False,
        learn_sigma=args.learn_sigma,
        mlp_input_rff=args.mlp_input_rff,
        mlp_output_rff=args.mlp_output_rff,
        mlp_block_type=args.mlp_block_type,
        mlp_block_rff_features=args.mlp_block_rff_features,
        conv_block_type=args.conv_block_type,
        conv_block_rff_features=args.conv_block_rff_features,
        dec_pre_readout_rff=args.dec_pre_readout_rff,
        rff_seed=args.rff_seed,
    )
    vae = CifarVAE(**config).to(device)
    cal_idx = np.random.default_rng(0).choice(X_tr.shape[0], size=2048, replace=False)
    cal_x = torch.as_tensor(X_tr[cal_idx], dtype=torch.float32, device=device)
    vae.init_rff_bandwidths(cal_x)

    ckpt = cifar_ckpt_path(args.dataset, args.z_dim, args.beta)
    if args.tag:
        ckpt = ckpt.with_name(ckpt.stem + f"_{args.tag}" + ckpt.suffix)
    warmup = args.kl_warmup_epochs
    if args.resume and ckpt.exists():
        print(f"  resuming from {ckpt}")
        raw = torch.load(ckpt, map_location=device, weights_only=False)
        sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        vae.load_state_dict(sd)
        if warmup > 0:
            print(f"  (resume: skipping KL warmup; was --kl-warmup-epochs {warmup})")
            warmup = 0
    n_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"  parameters: {n_params:,}")
    print(f"Training beta-VAE on {args.dataset} (z={args.z_dim}, "
          f"beta={args.beta}, epochs={args.epochs}, warmup={warmup}, "
          f"free_bits={args.free_bits}, learn_sigma={config['learn_sigma']}) ...")
    train_vae(
        vae, X_tr, device,
        epochs=args.epochs, batch_size=args.batch, lr=args.lr,
        beta=args.beta,
        kl_warmup_epochs=warmup, free_bits=args.free_bits,
        grad_clip=args.grad_clip,
    )
    torch.save({"state_dict": vae.state_dict(), "config": config}, ckpt)
    print(f"  saved {ckpt}")
    print(f"  config: {json.dumps(config)}")


if __name__ == "__main__":
    main()
