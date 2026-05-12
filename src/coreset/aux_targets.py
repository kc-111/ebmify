"""Auxiliary targets attached to selected indices.

All targets are derived from the global eigendecomposition; no raw samples
are stored. Each target produces a file in the algorithm's output directory.

The four target families are:

- ``spectral_coords``: top eigenvector projections ``c[i, j] = V[:, j]^T phi_i``
  paired with frequency weights ``1 / sqrt(sigma2[j] + lam)``. Use these as
  a regression target where the model predicts the leading spectral
  coordinates of each row (ridge-weighted to emphasize stable directions).
- ``bucket_ranks``: per-bucket uniform-in-``[0, 1]`` rank of each selected
  sample, capturing where it sits in each bucket's "easy-to-hard" axis.
- ``leverage_score``: scalar ridge leverage ``h_i``, useful as a difficulty /
  importance auxiliary head.
- ``home_bucket``: integer id of the bucket the sample aligns most with;
  works as a coarse cluster label aux head.
- ``feature_distill``: the raw post-standardization feature vector
  ``phi_i`` for the selected samples. Drop-in regression target for a
  student-of-the-backbone aux head.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from coreset.preprocess import StandardizedView
from coreset.spectral_rank import bucket_assignment, per_bucket_alignment, _ranks_per_column
from coreset.leverage import compute_leverage


def spectral_coords(
    view: StandardizedView,
    V: torch.Tensor,
    selected_idx: torch.Tensor,
    n_top: int,
) -> torch.Tensor:
    """Top-eigenvector coordinates for selected rows.

    Computes ``c[i, j] = V[:, j]^T phi_i`` for ``i`` in ``selected_idx``
    and ``j`` in ``[0, n_top)``, streaming the selected rows in blocks of
    ``view.chunk_size``.

    Args:
        view: Standardized view over ``Phi``.
        V: ``(D, r)`` eigenvectors of ``Phi^T Phi`` (columns sorted by
            descending eigenvalue).
        selected_idx: Long tensor of selected global row indices.
        n_top: Number of leading eigenvectors to project onto. Capped at
            ``V.shape[1]``.

    Returns:
        ``(len(selected_idx), n_top)`` float32 CPU tensor of coordinates.
    """
    device = view.device
    n_top = min(n_top, V.shape[1])
    V_top = V[:, :n_top].to(device)
    sel = selected_idx.to(device)
    out = torch.empty(sel.numel(), n_top, dtype=torch.float32, device=device)
    chunk = view.chunk_size
    for start in range(0, sel.numel(), chunk):
        block = sel[start:start + chunk]
        phi = view(block)
        out[start:start + block.numel()] = phi @ V_top
    return out.cpu()


def spectral_weights(sigma2: torch.Tensor, lam: float, n_top: int) -> torch.Tensor:
    """Frequency weights ``1 / sqrt(sigma2[j] + lam)`` for the top eigvecs.

    Used as per-coordinate weights when fitting models to the
    ``spectral_coords`` target. They downweight coordinates along
    high-variance directions (which are easy to predict) and upweight
    rare-direction coordinates, matching ridge-leverage geometry.

    Args:
        sigma2: ``(r,)`` eigenvalues (without ridge).
        lam: Ridge ``lambda``.
        n_top: Number of leading coordinates to weight. Capped at
            ``sigma2.numel()``.

    Returns:
        ``(n_top,)`` float32 CPU tensor of weights.
    """
    n_top = min(n_top, int(sigma2.numel()))
    return (1.0 / torch.sqrt(sigma2[:n_top] + lam)).cpu()


def bucket_ranks_for_selected(
    view: StandardizedView,
    V: torch.Tensor,
    selected_idx: torch.Tensor,
    n_buckets: int,
    ranks_cache: torch.Tensor | None = None,
    sigma2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-bucket uniform ranks for the selected samples.

    Args:
        view: Standardized view over ``Phi``.
        V: ``(D, r)`` eigenvectors.
        selected_idx: Long tensor of global row indices.
        n_buckets: Number of contiguous eigvec buckets to use when
            ``ranks_cache`` is not provided.
        ranks_cache: Optional ``(N, n_buckets)`` precomputed ranks (e.g.
            from a prior call to :func:`coreset.spectral_rank._ranks_per_column`).

    Returns:
        ``(len(selected_idx), n_buckets)`` float32 CPU tensor of ranks
        in ``[0, 1]``.
    """
    if ranks_cache is not None:
        return ranks_cache[selected_idx].cpu().contiguous()
    if sigma2 is None:
        raise ValueError("sigma2 is required when ranks_cache is not provided "
                         "(equal-mass bucketing depends on the spectrum)")
    bucket_of = bucket_assignment(sigma2, n_buckets, mode="equal_mass")
    S = per_bucket_alignment(view, V, bucket_of, n_buckets)
    ranks = _ranks_per_column(S)
    return ranks[selected_idx.to(view.device)].cpu().contiguous()


def home_bucket_for_selected(
    view: StandardizedView,
    V: torch.Tensor,
    selected_idx: torch.Tensor,
    n_buckets: int,
    S_cache: torch.Tensor | None = None,
    sigma2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Home-bucket id (``argmax_b S[i, b]``) for each selected sample.

    Args:
        view: Standardized view over ``Phi``.
        V: ``(D, r)`` eigenvectors.
        selected_idx: Long tensor of global row indices.
        n_buckets: Number of contiguous eigvec buckets to use when
            ``S_cache`` is not provided.
        S_cache: Optional ``(N, n_buckets)`` precomputed alignment matrix
            (from :func:`coreset.spectral_rank.per_bucket_alignment`).

    Returns:
        ``(len(selected_idx),)`` long CPU tensor of bucket ids.
    """
    if S_cache is None:
        if sigma2 is None:
            raise ValueError("sigma2 is required when S_cache is not provided "
                             "(equal-mass bucketing depends on the spectrum)")
        bucket_of = bucket_assignment(sigma2, n_buckets, mode="equal_mass")
        S_cache = per_bucket_alignment(view, V, bucket_of, n_buckets)
    return S_cache.argmax(dim=1)[selected_idx.to(view.device)].cpu().contiguous()


def feature_distill_for_selected(
    view: StandardizedView,
    selected_idx: torch.Tensor,
) -> torch.Tensor:
    """Standardized feature vectors phi_i for the selected samples.

    Streams the selected rows through ``view`` to materialize them on CPU
    as a single ``(k, D)`` float32 tensor, suitable for use as a
    feature-distillation regression target.

    Args:
        view: Standardized view over ``Phi``.
        selected_idx: Long tensor of selected global row indices.

    Returns:
        ``(len(selected_idx), D)`` float32 CPU tensor of features.
    """
    device = view.device
    sel = selected_idx.to(device)
    chunk = view.chunk_size
    parts: list[torch.Tensor] = []
    for start in range(0, sel.numel(), chunk):
        block = sel[start:start + chunk]
        parts.append(view(block).to(torch.float32).cpu())
    return torch.cat(parts, dim=0) if parts else torch.empty((0, 0), dtype=torch.float32)


def leverage_for_selected(
    view: StandardizedView,
    sigma2: torch.Tensor,
    V: torch.Tensor,
    lam: float,
    selected_idx: torch.Tensor,
    leverage_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ridge leverage ``h_i`` for the selected samples.

    Args:
        view: Standardized view over ``Phi``.
        sigma2: ``(r,)`` eigenvalues (without ridge).
        V: ``(D, r)`` eigenvectors.
        lam: Ridge ``lambda``.
        selected_idx: Long tensor of global row indices.
        leverage_cache: Optional ``(N,)`` precomputed leverage tensor
            (from :func:`coreset.leverage.compute_leverage`).

    Returns:
        ``(len(selected_idx),)`` float32 CPU tensor of leverage scores.
    """
    if leverage_cache is None:
        leverage_cache = compute_leverage(view, sigma2, V, lam)
    return leverage_cache[selected_idx.to(view.device)].cpu().contiguous()


def compute_aux_targets(
    view: StandardizedView,
    sigma2: torch.Tensor,
    V: torch.Tensor,
    lam: float,
    selected_idx: torch.Tensor,
    targets: Iterable[str],
    *,
    n_buckets: int,
    n_top_eigvecs: int,
    out_dir: Path,
    leverage_cache: torch.Tensor | None = None,
    ranks_cache: torch.Tensor | None = None,
    S_cache: torch.Tensor | None = None,
) -> dict[str, str]:
    """Compute the requested aux targets and persist them under ``out_dir``.

    For each name in ``targets``, computes the corresponding tensor and
    writes it via ``torch.save`` to a file named ``aux_<target>.pt`` in
    ``out_dir``. ``spectral_coords`` also writes a companion
    ``aux_spectral_weights.pt``.

    Args:
        view: Standardized view over ``Phi``.
        sigma2: ``(r,)`` eigenvalues (without ridge).
        V: ``(D, r)`` eigenvectors.
        lam: Ridge ``lambda``.
        selected_idx: Long tensor of selected global row indices.
        targets: Iterable of target names. Recognized values are
            ``"spectral_coords"``, ``"bucket_ranks"``, ``"leverage_score"``,
            ``"home_bucket"``. Unknown names are silently ignored.
        n_buckets: Number of buckets used by ``bucket_ranks`` /
            ``home_bucket`` (unused when caches are supplied).
        n_top_eigvecs: Number of leading eigvecs for ``spectral_coords``.
        out_dir: Directory to write outputs into; created if missing.
        leverage_cache: Optional ``(N,)`` leverage tensor to reuse.
        ranks_cache: Optional ``(N, n_buckets)`` ranks tensor to reuse.
        S_cache: Optional ``(N, n_buckets)`` alignment tensor to reuse.

    Returns:
        Map from target name (e.g. ``"spectral_coords"``,
        ``"spectral_weights"``, ``"bucket_ranks"``, ``"leverage_score"``,
        ``"home_bucket"``) to the path string of the saved file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    targets = set(targets)
    if "spectral_coords" in targets:
        coords = spectral_coords(view, V, selected_idx, n_top_eigvecs)
        wts = spectral_weights(sigma2, lam, n_top_eigvecs)
        torch.save(coords, out_dir / "aux_spectral_coords.pt")
        torch.save(wts, out_dir / "aux_spectral_weights.pt")
        saved["spectral_coords"] = str(out_dir / "aux_spectral_coords.pt")
        saved["spectral_weights"] = str(out_dir / "aux_spectral_weights.pt")
    if "bucket_ranks" in targets:
        br = bucket_ranks_for_selected(
            view, V, selected_idx, n_buckets, ranks_cache, sigma2=sigma2,
        )
        torch.save(br, out_dir / "aux_bucket_ranks.pt")
        saved["bucket_ranks"] = str(out_dir / "aux_bucket_ranks.pt")
    if "leverage_score" in targets:
        h = leverage_for_selected(view, sigma2, V, lam, selected_idx, leverage_cache)
        torch.save(h, out_dir / "aux_leverage_score.pt")
        saved["leverage_score"] = str(out_dir / "aux_leverage_score.pt")
    if "home_bucket" in targets:
        hb = home_bucket_for_selected(
            view, V, selected_idx, n_buckets, S_cache, sigma2=sigma2,
        )
        torch.save(hb, out_dir / "aux_home_bucket.pt")
        saved["home_bucket"] = str(out_dir / "aux_home_bucket.pt")
    if "feature_distill" in targets:
        fd = feature_distill_for_selected(view, selected_idx)
        torch.save(fd, out_dir / "aux_feature_distill.pt")
        saved["feature_distill"] = str(out_dir / "aux_feature_distill.pt")
    return saved
