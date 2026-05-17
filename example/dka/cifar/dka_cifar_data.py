"""CIFAR patchify + additive Gaussian noise utilities for the DKA demo.

Wraps ``example/cifar/cifar_data.py`` for the raw arrays and reshapes
``(N, 3, 32, 32)`` images into token sequences of fixed-size patches:

    patchify(images, patch_size=4) -> (N, L=64, patch_dim=48)   # CIFAR
    depatchify(patches, patch_size=4) -> (N, 3, 32, 32)

The DKA model consumes the per-image patch sequence as if it were a
sentence; the denoising task is to recover clean patches from a noisy
copy added in patch space.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import REPO_ROOT  # noqa: F401, E402

from cifar_data import load_cifar_test, load_cifar_train  # noqa: E402


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(N, 3, H, W) -> (N, L, patch_dim) with L = (H/p)*(W/p), patch_dim = 3*p*p."""
    if images.ndim != 4:
        raise ValueError(f"expected (N, 3, H, W); got shape {tuple(images.shape)}")
    n, c, h, w = images.shape
    p = patch_size
    if h % p != 0 or w % p != 0:
        raise ValueError(f"image size {(h, w)} not divisible by patch_size {p}")
    # (N, C, H/p, p, W/p, p) -> (N, H/p, W/p, C, p, p) -> (N, L, C*p*p)
    x = images.reshape(n, c, h // p, p, w // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.reshape(n, (h // p) * (w // p), c * p * p)


def depatchify(patches: torch.Tensor, patch_size: int, image_size: int = 32) -> torch.Tensor:
    """Inverse of patchify."""
    n, l, pdim = patches.shape
    p = patch_size
    c = pdim // (p * p)
    side = image_size // p
    if side * side != l:
        raise ValueError(f"L={l} incompatible with image_size={image_size} patch_size={p}")
    x = patches.reshape(n, side, side, c, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.reshape(n, c, image_size, image_size)


def add_gaussian_noise(x: torch.Tensor, sigma: float,
                       generator: torch.Generator | None = None) -> torch.Tensor:
    if generator is None:
        noise = torch.randn_like(x) * sigma
    else:
        noise = torch.randn(x.shape, generator=generator, device=x.device,
                            dtype=x.dtype) * sigma
    return x + noise


def get_arrays(dataset: str, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (images_float [N,3,32,32], labels [N]) as torch tensors on CPU."""
    if split == "train":
        x_np, y_np = load_cifar_train(dataset)
    elif split in ("test", "val"):
        x_np, y_np = load_cifar_test(dataset)
    else:
        raise ValueError(f"split must be train|test; got {split!r}")
    return torch.from_numpy(np.ascontiguousarray(x_np)), torch.from_numpy(y_np)


def get_noisy_clean_batch(
    images: torch.Tensor,
    batch_size: int,
    patch_size: int,
    sigma: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of images, patchify, add noise. Returns
    ``(x_noisy_patches, x_clean_patches)`` both on ``device``.
    """
    n = images.shape[0]
    if generator is None:
        idx = torch.randint(0, n, (batch_size,))
    else:
        idx = torch.randint(0, n, (batch_size,), generator=generator)
    batch = images[idx].to(device)                          # (B, 3, 32, 32)
    clean = patchify(batch, patch_size)                     # (B, L, patch_dim)
    noisy = add_gaussian_noise(clean, sigma)
    return noisy, clean
