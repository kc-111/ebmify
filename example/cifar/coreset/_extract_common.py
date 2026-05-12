"""Shared helpers for the CIFAR coreset-building wrappers.

Provides the encode-CIFAR-with-backbone loop, the cache-aware feature save,
the coreset-cli invocation, and the diagnostics-plot orchestration. Each
backbone-specific script (``cifar_build_coreset_supervised.py`` /
``cifar_build_coreset_ssl.py``) loads its model and calls
:func:`extract_and_select` to do everything else.

This is the layer where the three coreset algorithms actually run — the
top-level scripts only handle backbone construction; this helper feeds
``Phi`` into ``coreset.cli.run`` which executes ``greedy_max_variance``,
``ridge_leverage_sample``, and ``spectral_rank_coverage``.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import REPO_ROOT  # noqa: F401, E402

from cifar_data import load_cifar_train  # noqa: E402

from coreset.cli import Config, run  # noqa: E402
from coreset.leverage import compute_leverage  # noqa: E402
from coreset.preprocess import fit_view  # noqa: E402
from coreset.eig import compute_eig  # noqa: E402
from coreset.spectral_rank import (  # noqa: E402
    bucket_assignment, per_bucket_alignment, _ranks_per_column,
)
from coreset.plots import plot_selection_diagnostics, plot_spectrum  # noqa: E402


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


@torch.no_grad()
def encode_all(
    model: nn.Module,
    X: np.ndarray,
    *,
    norm: str,
    batch_size: int,
    device: str,
    resize: int | None = None,
) -> torch.Tensor:
    """Encode (N, 3, 32, 32) in [0, 1] -> (N, D) features on CPU."""
    if norm == "imagenet":
        mean = IMAGENET_MEAN.to(device); std = IMAGENET_STD.to(device)
    elif norm == "cifar":
        mean = CIFAR_MEAN.to(device); std = CIFAR_STD.to(device)
    else:
        raise ValueError(norm)
    X_t = torch.as_tensor(X, dtype=torch.float32)
    feats = []
    N = X_t.shape[0]
    for i in range(0, N, batch_size):
        chunk = X_t[i:i + batch_size].to(device)
        if resize is not None and resize != chunk.shape[-1]:
            chunk = F.interpolate(chunk, size=resize, mode="bilinear",
                                  align_corners=False)
        chunk = (chunk - mean) / std
        feats.append(model(chunk).cpu())
        done = min(i + batch_size, N)
        print(f"  encoded {done}/{N}", end="\r")
    print()
    return torch.cat(feats, dim=0)


def save_phi_pt(Phi: torch.Tensor, y: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"Phi": Phi.to(torch.float32), "y": torch.from_numpy(y).long()}, path)


def write_yaml_config(out_path: Path, cfg_dict: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore
        out_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False))
    except ImportError:
        out_path.write_text(json.dumps(cfg_dict, indent=2))


def extract_and_select(
    *,
    tag: str,
    model: nn.Module,
    norm: str,
    resize: int | None,
    batch_size: int,
    device: str,
    budget_k: int,
    ridge_lambda: float = 1e-3,
    n_buckets: int = 16,
    n_top_eigvecs: int = 64,
    standardize: str = "center_l2",
    seed: int = 0,
    low_rank_eig: bool = False,
    low_rank_r: int = 512,
    cache_dir: Path | None = None,
    artifacts_root: Path | None = None,
    plot_dir: Path | None = None,
) -> dict:
    """Encode CIFAR-10 train, run all three selection algorithms, plot diagnostics.

    Returns a dict with all artifact paths.
    """
    cache_dir = cache_dir or (REPO_ROOT / "example" / "cifar" / "cache")
    artifacts_root = artifacts_root or (cache_dir / "coreset" / tag)
    plot_dir = plot_dir or (REPO_ROOT / "example" / "out" / "coreset")
    plot_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_cifar_train("cifar10")
    print(f"[{tag}] encoding {X.shape[0]} CIFAR-10 train images")
    t0 = time.time()
    Phi = encode_all(model, X, norm=norm, batch_size=batch_size,
                     device=device, resize=resize)
    print(f"[{tag}] encoded in {time.time() - t0:.1f}s -> Phi {tuple(Phi.shape)}")

    phi_path = cache_dir / f"phi_{tag}.pt"
    save_phi_pt(Phi, y, phi_path)
    print(f"[{tag}] saved {phi_path}")

    cfg_dict = {
        "phi_path": str(phi_path),
        "phi_key": "Phi",
        "budget_k": int(budget_k),
        "ridge_lambda": float(ridge_lambda),
        "n_buckets": int(n_buckets),
        "n_top_eigvecs": int(n_top_eigvecs),
        "standardize": standardize,
        "algorithms": ["greedy", "leverage", "spectral_rank"],
        "aux_targets": ["spectral_coords", "bucket_ranks",
                        "leverage_score", "home_bucket", "feature_distill"],
        "seed": int(seed),
        "device": device,
        "chunk_size": 8192,
        "low_rank_eig": bool(low_rank_eig),
        "low_rank_r": int(low_rank_r),
    }
    yaml_path = artifacts_root / "config.yaml"
    write_yaml_config(yaml_path, cfg_dict)
    print(f"[{tag}] wrote config -> {yaml_path}")

    cfg = Config.from_dict(cfg_dict)
    print(f"[{tag}] running coreset.cli ...")
    stats = run(cfg, artifacts_root)
    with open(artifacts_root / "summary.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    # ---- Diagnostics: cache full-N stats next to artifacts so the plot script
    # has population-level info (lev, S, ranks, top-2 coords) without the user
    # having to redo any heavy work.
    print(f"[{tag}] computing population diagnostics ...")
    view = fit_view(Phi, mode=standardize, chunk_size=8192,
                    device=device, out_path=None)
    sigma2, V = compute_eig(view, lam=ridge_lambda,
                             low_rank=low_rank_eig, r=low_rank_r, seed=seed)
    pop_h = compute_leverage(view, sigma2, V, lam=ridge_lambda).cpu().numpy()
    bucket_of = bucket_assignment(sigma2, n_buckets, mode="equal_mass")
    pop_S = per_bucket_alignment(view, V, bucket_of, n_buckets).cpu().numpy()
    pop_ranks = _ranks_per_column(torch.from_numpy(pop_S).to(device)).cpu().numpy()
    pop_coords = None  # cheap to compute; do top-2 only
    V2 = V[:, :2].to(device)
    coords_chunks = []
    for _, chunk in view.stream():
        coords_chunks.append((chunk @ V2).cpu())
    pop_coords = torch.cat(coords_chunks, dim=0).numpy()

    plot_path = plot_dir / f"{tag}_coreset_selection_diagnostics.png"
    plot_selection_diagnostics(
        artifacts_root,
        population_leverage=pop_h,
        population_S=pop_S,
        population_ranks=pop_ranks,
        population_coords=pop_coords,
        out_path=plot_path,
        suptitle=(f"{tag}: coreset selection diagnostics  "
                  f"(N={X.shape[0]}, D={Phi.shape[1]}, k={budget_k})"),
    )
    print(f"[{tag}] diagnostics -> {plot_path}")

    spectrum_path = plot_dir / f"{tag}_coreset_spectrum.png"
    plot_spectrum(artifacts_root, out_path=spectrum_path)
    print(f"[{tag}] spectrum -> {spectrum_path}")

    return {
        "phi_path": str(phi_path),
        "artifacts": str(artifacts_root),
        "yaml": str(yaml_path),
        "diagnostics": str(plot_path),
        "spectrum": str(spectrum_path),
    }
