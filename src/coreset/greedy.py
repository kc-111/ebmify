"""Greedy max-variance coreset selection.

We maintain on device the inverse of the (regularized) selected Gram

    A_inv = (lam I + sum_{i in S} phi_i phi_i^T)^{-1}      (D x D)

and a length-``N`` vector of per-sample leverage scores

    h_i = phi_i^T A_inv phi_i.

Each iteration:

1. Pick ``i* = argmax_i h_i`` (ties broken by lowest index).
2. Apply the Sherman-Morrison rank-1 update to ``A_inv``:
       u = A_inv @ phi_{i*}
       beta = 1 + phi_{i*}^T u
       A_inv := A_inv - (u u^T) / beta
3. Apply the matching streamed update to ``h``:
       for each chunk:  proj = chunk @ u;   h_chunk -= proj^2 / beta
4. Mark ``h[i*] = -inf`` so the picked sample is never re-selected.

Every ``refactor_every`` iterations we rebuild ``A_inv`` from scratch via
``Phi_S^T Phi_S + lam I -> Cholesky -> inverse`` to control round-off
drift. Seeding uses farthest-point sampling in feature space so the
initial subset isn't dominated by one direction.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from coreset.preprocess import StandardizedView


def _initial_leverage(view: StandardizedView, lam: float) -> torch.Tensor:
    """Leverage under the all-ridge prior: ``h_i = ||phi_i||^2 / lam``.

    Args:
        view: Standardized view.
        lam: Ridge ``lambda``.

    Returns:
        ``(N,)`` float32 leverage tensor on ``view.device``.
    """
    N = view.N
    h = torch.empty(N, dtype=torch.float32, device=view.device)
    for idx, chunk in view.stream():
        h[idx.to(view.device)] = (chunk * chunk).sum(dim=1) / lam
    return h


def _row_norms_sq(view: StandardizedView) -> torch.Tensor:
    """Stream ``||phi_i||^2`` for all rows.

    Args:
        view: Standardized view.

    Returns:
        ``(N,)`` float32 squared-norm tensor on ``view.device``.
    """
    N = view.N
    out = torch.empty(N, dtype=torch.float32, device=view.device)
    for idx, chunk in view.stream():
        out[idx.to(view.device)] = (chunk * chunk).sum(dim=1)
    return out


def _farthest_point_seed(
    view: StandardizedView,
    seed_size: int,
    norms_sq: torch.Tensor,
) -> list[int]:
    """Pick ``seed_size`` indices by farthest-point sampling in feature space.

    Starts from the row of largest norm, then repeatedly adds
    ``argmax_i min_{s in S} ||phi_i - phi_s||^2``. This is the standard
    Gonzalez seed for k-medoids / coreset constructions; it spreads the
    initial set out instead of clumping along one eigendirection.

    Args:
        view: Standardized view.
        seed_size: Target number of seed indices to return.
        norms_sq: Pre-computed ``(N,)`` squared norms (avoids a re-pass).

    Returns:
        ``list[int]`` of length ``min(seed_size, view.N)`` -- selected
        indices in pick order.
    """
    N = view.N
    device = view.device
    selected: list[int] = []
    i0 = int(torch.argmax(norms_sq).item())
    selected.append(i0)
    if seed_size <= 1:
        return selected
    min_dist = torch.full((N,), float("inf"), dtype=torch.float32, device=device)
    last_phi = view(torch.tensor([i0], dtype=torch.long))[0]
    last_norm_sq = float(norms_sq[i0].item())
    while len(selected) < seed_size:
        for idx, chunk in view.stream():
            idx_dev = idx.to(device)
            dot = chunk @ last_phi
            d = norms_sq[idx_dev] + last_norm_sq - 2.0 * dot
            d.clamp_(min=0.0)
            cur = min_dist[idx_dev]
            torch.minimum(cur, d, out=cur)
            min_dist[idx_dev] = cur
        sel_t = torch.tensor(selected, dtype=torch.long, device=device)
        min_dist[sel_t] = -1.0
        i_next = int(torch.argmax(min_dist).item())
        selected.append(i_next)
        last_phi = view(torch.tensor([i_next], dtype=torch.long))[0]
        last_norm_sq = float(norms_sq[i_next].item())
    return selected


def _refactor_A_inv(
    view: StandardizedView,
    selected_idx: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Rebuild ``A_inv = (lam I + Phi_S^T Phi_S)^{-1}`` from scratch.

    Streams the selected rows ``Phi[S]`` in chunks of ``view.chunk_size``,
    accumulates the rank-update, and Cholesky-inverts. Called periodically
    in the main loop to control floating-point drift in the
    Sherman-Morrison maintenance.

    Args:
        view: Standardized view.
        selected_idx: Long tensor of currently selected row indices.
        lam: Ridge ``lambda``.

    Returns:
        ``(D, D)`` symmetric float32 ``A_inv`` on ``view.device``.
    """
    D = view.D
    device = view.device
    A = torch.eye(D, device=device, dtype=torch.float32) * lam
    chunk = view.chunk_size
    s = selected_idx
    for start in range(0, s.numel(), chunk):
        block = s[start:start + chunk]
        phi_block = view(block)
        A.addmm_(phi_block.T, phi_block)
    A = 0.5 * (A + A.T)
    L = torch.linalg.cholesky(A)
    A_inv = torch.cholesky_inverse(L)
    A_inv = 0.5 * (A_inv + A_inv.T)
    return A_inv


def _recompute_h(
    view: StandardizedView,
    A_inv: torch.Tensor,
) -> torch.Tensor:
    """Recompute ``h_i = phi_i^T A_inv phi_i`` for all rows.

    Args:
        view: Standardized view.
        A_inv: Current ``(D, D)`` selected-Gram inverse.

    Returns:
        ``(N,)`` float32 leverage tensor on ``view.device``.
    """
    N = view.N
    h = torch.empty(N, dtype=torch.float32, device=view.device)
    for idx, chunk in view.stream():
        Aphi = chunk @ A_inv
        h[idx.to(view.device)] = (chunk * Aphi).sum(dim=1)
    return h


def greedy_max_variance(
    view: StandardizedView,
    lam: float,
    k: int,
    *,
    seed_size: int = 32,
    refactor_every: int = 500,
    seed: int = 0,
    device: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Greedy max-variance coreset of size ``k``.

    Maintains the selected-Gram inverse via Sherman-Morrison and picks the
    largest-leverage sample each step (with farthest-point seeding and
    periodic Cholesky refactor). The trajectory of ``max h`` after every
    pick is recorded and should trend monotonically downward.

    Args:
        view: Standardized view of the feature matrix.
        lam: Ridge ``lambda``. Smaller ``lambda`` makes the early picks
            more selective along data-poor directions; larger ``lambda``
            smooths the score toward uniform.
        k: Target coreset size.
        seed_size: Number of farthest-point seeds taken before the
            Sherman-Morrison loop starts (capped to ``k``). Larger
            seed_size means more diverse start.
        refactor_every: How many picks between fresh Cholesky refactors
            of ``A_inv``. Smaller means more numerical safety, slower.
        seed: Manual seed for ``torch.manual_seed`` (the farthest-point
            seeding is deterministic given ``view`` so the seed only
            affects tiebreaking in argmax with NaN values, which we
            avoid by clamping; included for completeness).
        device: Optional device override. Defaults to ``view.device``.

    Returns:
        ``(indices, stats)``: ``indices`` is a ``(k,)`` CPU long tensor of
        selected row indices in pick order. ``stats`` is a dict with keys
        ``"max_h_trajectory"`` (list of floats, one per non-seed pick),
        ``"runtime_sec"``, ``"seed_size"``, ``"refactors"``.
    """
    torch.manual_seed(seed)
    if device is None:
        device = view.device
    t0 = time.time()

    norms_sq = _row_norms_sq(view)
    h = norms_sq.clone() / lam
    selected_set: set[int] = set()
    selected: list[int] = []
    max_h_traj: list[float] = []

    seed_picks = _farthest_point_seed(view, seed_size=min(seed_size, k), norms_sq=norms_sq)
    selected.extend(seed_picks)
    selected_set.update(seed_picks)

    A_inv = _refactor_A_inv(view, torch.tensor(selected, dtype=torch.long, device=device), lam)
    h = _recompute_h(view, A_inv)
    for s_idx in selected:
        h[s_idx] = -float("inf")

    while len(selected) < k:
        i_star = int(torch.argmax(h).item())
        if i_star in selected_set:
            h[i_star] = -float("inf")
            continue
        max_h_traj.append(float(h[i_star].item()))
        phi = view(torch.tensor([i_star], dtype=torch.long, device=device))[0]
        u = A_inv @ phi
        beta = 1.0 + float(phi @ u)
        if beta <= 0.0 or not torch.isfinite(torch.tensor(beta)):
            selected.append(i_star)
            selected_set.add(i_star)
            h[i_star] = -float("inf")
            A_inv = _refactor_A_inv(view, torch.tensor(selected, dtype=torch.long, device=device), lam)
            h = _recompute_h(view, A_inv)
            for s_idx in selected:
                h[s_idx] = -float("inf")
            continue

        A_inv.addr_(u, u, alpha=-1.0 / beta)
        A_inv = 0.5 * (A_inv + A_inv.T)

        for idx, chunk in view.stream():
            proj = chunk @ u
            idx_dev = idx.to(device)
            h[idx_dev] -= (proj * proj) / beta

        selected.append(i_star)
        selected_set.add(i_star)
        h[i_star] = -float("inf")

        if len(selected) % refactor_every == 0 and len(selected) < k:
            A_inv = _refactor_A_inv(
                view, torch.tensor(selected, dtype=torch.long, device=device), lam
            )
            h = _recompute_h(view, A_inv)
            for s_idx in selected:
                h[s_idx] = -float("inf")

    runtime = time.time() - t0
    indices = torch.tensor(selected, dtype=torch.long)
    stats = {
        "max_h_trajectory": max_h_traj,
        "runtime_sec": runtime,
        "seed_size": int(min(seed_size, k)),
        "refactors": int((len(selected) - len(seed_picks)) // refactor_every),
    }
    return indices, stats
