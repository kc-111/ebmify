"""Quick sanity grid for cached CIFAR VAEs.

Self-contained -- does not depend on the legacy/fix recon_compare script.
For each available checkpoint (CIFAR-10, CIFAR-100), plots:

1. ``train`` reconstruction grid (8 random training images vs recon).
2. ``test``  reconstruction grid (8 random held-out images vs recon).
3. ``prior`` samples: decode ``z ~ N(0, I)`` to surface decoder bias /
   posterior collapse independently of the encoder.

Usage:
    python example/cifar/probes/cifar_vae_sanity.py
    python example/cifar/probes/cifar_vae_sanity.py --z 256 --beta 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _paths  # noqa: F401, E402
from cifar_data import cifar_ckpt_path, load_cifar_test, load_cifar_train  # noqa: E402
from cifar_vae_train import LOG_SIGMA_MAX, LOG_SIGMA_MIN, load_vae  # noqa: E402


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------

def _imgrid(ax, imgs: np.ndarray, title: str, *, n: int = 8, upscale: int = 4) -> None:
    """imgs: (B, 3, 32, 32) float; clipped to [0,1] for display."""
    B, C, H, W = imgs.shape
    rows = (B + n - 1) // n
    canvas = np.ones((rows * H, n * W, 3), dtype=np.float32)
    for i in range(B):
        r, c = divmod(i, n)
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = np.transpose(imgs[i], (1, 2, 0))
    canvas = np.clip(canvas, 0.0, 1.0)
    canvas = np.repeat(np.repeat(canvas, upscale, axis=0), upscale, axis=1)
    ax.imshow(canvas, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


# ----------------------------------------------------------------------
# Plot one sanity figure per VAE
# ----------------------------------------------------------------------

def plot_sanity(
    vae, dataset: str, X_tr: np.ndarray, X_te: np.ndarray,
    device: str, out_path: Path, *, n_show: int = 8, seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    idx_tr = rng.choice(X_tr.shape[0], size=n_show, replace=False)
    idx_te = rng.choice(X_te.shape[0], size=n_show, replace=False)
    x_tr = torch.as_tensor(X_tr[idx_tr], dtype=torch.float32, device=device)
    x_te = torch.as_tensor(X_te[idx_te], dtype=torch.float32, device=device)

    with torch.no_grad():
        # Use mu (no reparam noise) so recons are deterministic.
        mu_tr, _ = vae.encode(x_tr)
        mu_te, _ = vae.encode(x_te)
        r_tr = vae.decode(mu_tr).cpu().numpy()
        r_te = vae.decode(mu_te).cpu().numpy()
        z_prior = torch.randn(n_show, vae.z_dim, device=device,
                               generator=torch.Generator(device=device).manual_seed(seed))
        r_pr = vae.decode(z_prior).cpu().numpy()

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1])
    _imgrid(fig.add_subplot(gs[0, 0]), x_tr.cpu().numpy(),
            f"train inputs ({dataset})")
    _imgrid(fig.add_subplot(gs[0, 1]), r_tr, "train reconstructions (mu)")
    _imgrid(fig.add_subplot(gs[1, 0]), x_te.cpu().numpy(),
            f"test inputs ({dataset})")
    _imgrid(fig.add_subplot(gs[1, 1]), r_te, "test reconstructions (mu)")
    _imgrid(fig.add_subplot(gs[2, :]), r_pr,
            "prior samples: x_hat = decode(z), z ~ N(0, I)")

    log_sigma = getattr(vae, "log_sigma", None)
    sigma_str = ""
    if log_sigma is not None:
        sigma = float(log_sigma.detach().clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX).exp().item())
        sigma_str = f", decoder sigma={sigma:.4f}"
    fig.suptitle(
        f"{dataset} VAE sanity: train recon | test recon | prior samples "
        f"(z_dim={vae.z_dim}{sigma_str})",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _load_if_exists(dataset: str, z_dim: int, beta: float, device: str):
    ckpt = cifar_ckpt_path(dataset, z_dim, beta)
    if not ckpt.exists():
        print(f"  [skip] no checkpoint at {ckpt}")
        return None, ckpt
    return load_vae(ckpt, device), ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=int, default=512, dest="z_dim")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--n-show", type=int, default=8, dest="n_show")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    out_dir = Path(__file__).resolve().parent.parent / "out"

    any_found = False
    for ds in ("cifar10", "cifar100"):
        print(f"\n[{ds}] loading checkpoint ...")
        vae, ckpt = _load_if_exists(ds, args.z_dim, args.beta, device)
        if vae is None:
            continue
        any_found = True
        print(f"  loaded {ckpt.name}  z_dim={vae.z_dim}")

        X_tr, _ = load_cifar_train(ds)
        X_te, _ = load_cifar_test(ds)
        print(f"  train: {X_tr.shape}   test: {X_te.shape}")
        plot_sanity(
            vae, ds, X_tr, X_te, device,
            out_dir / f"{ds}_vae_sanity.png",
            n_show=args.n_show, seed=args.seed,
        )

    if not any_found:
        print("\nNo checkpoints found; nothing to do.")


if __name__ == "__main__":
    main()
