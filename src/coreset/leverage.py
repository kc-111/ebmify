"""Ridge leverage sampling.

Per-sample ridge leverage

    h_i = phi_i^T (Phi^T Phi + lam I)^{-1} phi_i
        = sum_j (V[:, j]^T phi_i)^2 / (sigma2[j] + lam)

is streamed via the eigendecomposition of ``Phi^T Phi``. We then sample
``k`` indices by either

- **with replacement**: ``torch.multinomial`` on the mixed distribution
  ``p_i = (1 - alpha) h_i / sum(h) + alpha / N``. This keeps a uniform
  tail so samples with ``h_i = 0`` can still be picked, important when
  ``Phi`` has near-redundant rows;
- **without replacement**: independent Bernoulli inclusion with
  probability ``q_i = min(1, c * h_i)`` for ``c`` chosen via binary
  search so ``sum(q_i) ~= k`` within 1%. The actual selected set is then
  padded or truncated to land at exactly ``k``.

Importance-sampling weights ``w_i = 1 / sqrt(k * p_i_used)`` are returned
so downstream estimators (e.g. weighted least squares on the coreset) are
unbiased.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from coreset.preprocess import StandardizedView


def compute_leverage(
    view: StandardizedView,
    sigma2: torch.Tensor,
    V: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Streamed ridge leverage for every row of ``Phi``.

    Args:
        view: Standardized view over ``Phi``.
        sigma2: ``(r,)`` eigenvalues of ``Phi^T Phi`` (without ``lam``).
            May come from ``compute_eig`` with either the full or
            low-rank path.
        V: ``(D, r)`` eigenvectors matching ``sigma2`` as columns.
        lam: Ridge ``lambda``.

    Returns:
        ``(N,)`` float32 tensor on ``view.device``. Entry ``i`` is
        ``h_i = sum_j (V[:, j]^T phi_i)^2 / (sigma2[j] + lam)``.
    """
    N = view.N
    device = view.device
    w = 1.0 / (sigma2.to(device) + lam)
    h = torch.empty(N, dtype=torch.float32, device=device)
    V_dev = V.to(device)
    for idx, chunk in view.stream():
        proj = chunk @ V_dev
        h[idx.to(device)] = (proj * proj * w[None, :]).sum(dim=1)
    return h


def _bernoulli_scale(h: torch.Tensor, k: int, tol: float = 0.01,
                     max_iter: int = 60) -> float:
    """Binary-search the scale ``c`` so that ``sum(min(1, c * h)) ~= k``.

    Args:
        h: ``(N,)`` non-negative leverage tensor.
        k: Target expected count.
        tol: Relative tolerance on the sum (with absolute floor of 1).
        max_iter: Cap on bisection rounds.

    Returns:
        Scalar ``c`` solving the constraint within tolerance, or its
        midpoint estimate if iterations run out.
    """
    if h.sum() <= 0.0:
        return 0.0
    lo = 0.0
    hi = float(k) / float(h.mean().item())
    hi = max(hi, 1.0)
    for _ in range(20):
        s = torch.clamp(hi * h, max=1.0).sum().item()
        if s >= k:
            break
        hi *= 2.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s = torch.clamp(mid * h, max=1.0).sum().item()
        if abs(s - k) <= max(tol * k, 1.0):
            return mid
        if s < k:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ridge_leverage_sample(
    view: StandardizedView,
    sigma2: torch.Tensor,
    V: torch.Tensor,
    lam: float,
    k: int,
    *,
    alpha: float = 0.2,
    replace: bool = False,
    seed: int = 0,
    leverage: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Sample a size-``k`` coreset by mixed ridge-leverage importance sampling.

    Args:
        view: Standardized view over ``Phi``.
        sigma2: ``(r,)`` eigenvalues from :func:`coreset.eig.compute_eig`.
        V: ``(D, r)`` matching eigenvectors.
        lam: Ridge ``lambda``.
        k: Coreset size to return.
        alpha: Mixing weight in ``[0, 1]`` on the uniform prior used in
            the replacement path. ``0`` = pure leverage; ``1`` = uniform.
            Default ``0.2`` follows standard leverage-sampling practice
            and keeps the tail covered.
        replace: ``True`` uses multinomial with replacement and returns
            duplicates if any; ``False`` uses Bernoulli without
            replacement padded/truncated to exactly ``k``.
        seed: RNG seed.
        leverage: Optional pre-computed ``(N,)`` leverage tensor (e.g.
            cached by the CLI). If ``None``, computed via
            :func:`compute_leverage`.

    Returns:
        ``(indices, weights, stats)``:

        - ``indices``: ``(k,)`` CPU long tensor of selected row indices
          (sorted in the no-replacement path; pick order in the
          replacement path).
        - ``weights``: ``(k,)`` CPU float32 IS weights
          ``w_i = 1 / sqrt(k * p_i_used)`` where ``p_i_used`` is the
          actual sampling probability used per sample (``p_i`` for
          replacement, ``q_i`` for Bernoulli).
        - ``stats``: dict with ``effective_dim = sum(sigma2 / (sigma2 + lam))``,
          ``alpha``, ``mode``, ``runtime_sec``; plus ``c`` and
          ``n_bernoulli`` for the no-replacement path.

    Raises:
        RuntimeError: If every leverage is zero (degenerate ``Phi``).
    """
    torch.manual_seed(seed)
    t0 = time.time()
    device = view.device
    N = view.N

    h = leverage if leverage is not None else compute_leverage(view, sigma2, V, lam)
    h_sum = float(h.sum().item())
    if h_sum <= 0.0:
        raise RuntimeError("leverage sum is zero -- features may be all-zero")
    p = (1.0 - alpha) * (h / h_sum) + alpha / float(N)
    p = p / p.sum()

    eff_dim = float((sigma2.to(device) / (sigma2.to(device) + lam)).sum().item())

    if replace:
        g = torch.Generator(device=device).manual_seed(seed)
        idx = torch.multinomial(p, k, replacement=True, generator=g)
        p_used = p[idx]
        weights = 1.0 / torch.sqrt(float(k) * p_used.clamp(min=1e-30))
        stats = {
            "effective_dim": eff_dim,
            "alpha": alpha,
            "mode": "with_replacement",
            "runtime_sec": time.time() - t0,
        }
        return idx.cpu(), weights.cpu(), stats

    c = _bernoulli_scale(h, k)
    q = torch.clamp(c * h, max=1.0)
    g = torch.Generator(device=device).manual_seed(seed)
    rnd = torch.rand(N, device=device, generator=g)
    sampled_mask = rnd < q
    sampled_idx = torch.nonzero(sampled_mask, as_tuple=False).flatten()
    cur = int(sampled_idx.numel())

    if cur > k:
        q_s = q[sampled_idx]
        order = torch.argsort(q_s, descending=True)
        sampled_idx = sampled_idx[order[:k]]
    elif cur < k:
        not_mask = ~sampled_mask
        unsampled_idx = torch.nonzero(not_mask, as_tuple=False).flatten()
        h_u = h[unsampled_idx]
        topk = torch.topk(h_u, k - cur, largest=True).indices
        addn = unsampled_idx[topk]
        sampled_idx = torch.cat([sampled_idx, addn], dim=0)

    sampled_idx, _ = torch.sort(sampled_idx)
    q_used = q[sampled_idx].clamp(min=1e-30)
    weights = 1.0 / torch.sqrt(float(k) * q_used)
    stats = {
        "effective_dim": eff_dim,
        "alpha": alpha,
        "c": float(c),
        "n_bernoulli": int(cur),
        "mode": "without_replacement",
        "runtime_sec": time.time() - t0,
    }
    return sampled_idx.cpu(), weights.cpu(), stats
