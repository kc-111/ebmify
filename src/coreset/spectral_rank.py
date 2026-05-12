"""Spectral-rank coverage coreset selection.

Algorithm:

1. **Bucket eigenvectors.** The columns of ``V`` are already sorted by
   descending ``sigma2``. Equal-mass bucketing splits ``[0, D)`` into
   ``n_buckets`` contiguous groups whose summed ``sigma2`` is roughly
   ``total / n_buckets`` apiece, so the leading bucket is narrow but
   high-variance and trailing buckets are wider.

2. **Per-sample, per-bucket alignment.** For each row,
   ``S[i, b] = sum_{j in bucket_b} (V[:, j]^T phi_i)^2``. Streamed chunk
   by chunk so we never materialize the full projection.

3. **Per-bucket rank.** For each bucket, rank samples by their
   ``S[:, b]`` value, normalized to ``[0, 1]``. The resulting matrix
   ``R in [0, 1]^{N x B}`` is the *rank vector* of each sample.

4. **Greedy maximum-coverage of rank strata.** Partition each bucket's
   ``[0, 1]`` rank axis into ``k`` equal strata of width ``1 / k``.
   Each sample becomes a length-``B`` set, one stratum per bucket, and
   picking ``k`` samples is the classical maximum-coverage problem on
   a universe of ``B * k`` cells. The textbook greedy -- pick the
   sample covering the most currently-uncovered cells -- attains the
   ``(1 - 1/e)`` approximation (Hochbaum 1996). When ``N >> k`` and the
   rank distribution is non-degenerate, the optimum hits every
   stratum exactly once per bucket, so the per-bucket marginal of the
   selected ranks is exactly ``Uniform[0, 1]`` -- mean ``~ 0.5`` and
   std ``~ 1/sqrt(12)`` in every bucket. This avoids the corner-bias
   that ``L_inf`` farthest-first develops as ``B`` grows.

The ``home_bucket`` aux target ``b_star_i = argmax_b S[i, b]`` is still
computed and exported -- it remains a useful coarse cluster label
downstream -- but it is no longer used for selection.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from coreset.preprocess import StandardizedView


def bucket_assignment(
    sigma2: torch.Tensor | int,
    n_buckets: int,
    *,
    mode: str = "equal_mass",
) -> torch.Tensor:
    """Contiguous bucket id for each eigenvector index.

    Eigenvectors are assumed sorted by descending ``sigma2``, so bucket 0
    holds the leading eigenvectors and the last bucket the trailing ones.

    Two partitioning modes are supported:

    - ``"equal_mass"`` (default): each bucket holds a contiguous range
      whose ``sum(sigma2[j])`` is roughly equal to ``total / n_buckets``.
      Boundaries are placed where the cumulative variance crosses each
      ``b * total / n_buckets`` threshold. This is the recommended mode:
      because the top eigenvalues typically dwarf the tail, equal-*count*
      bucketing concentrates almost all of ``S[i, b]`` in bucket 0 and
      makes ``argmax_b S[i, b]`` degenerate. Equal-mass bucketing puts
      each bucket on the same expected alignment scale.
    - ``"equal_count"``: contiguous slices of equal width (``D / B``).
      Mostly retained for diagnostics / backward compatibility.

    Args:
        sigma2: ``(r,)`` descending eigenvalues (without ridge). For
            ``mode="equal_count"`` you may pass an int ``D`` instead.
        n_buckets: Desired number of contiguous buckets.
        mode: ``"equal_mass"`` or ``"equal_count"``.

    Returns:
        ``(D,)`` CPU long tensor mapping eigvec index -> bucket id.
        Buckets are 0-indexed and contiguous; some buckets in the tail
        may be empty when the spectrum is concentrated -- that's fine.

    Raises:
        ValueError: If ``n_buckets`` is outside ``[1, D]`` or ``mode`` is
            unknown.
    """
    if isinstance(sigma2, int):
        if mode != "equal_count":
            raise ValueError("int sigma2 only supported for mode='equal_count'")
        D = int(sigma2)
        sigma2_cpu = None
    else:
        sigma2_cpu = sigma2.detach().to("cpu", dtype=torch.float64)
        D = int(sigma2_cpu.numel())
    if n_buckets <= 0 or n_buckets > D:
        raise ValueError(f"n_buckets must be in [1, D]; got {n_buckets}")

    if mode == "equal_count":
        sizes = [D // n_buckets] * n_buckets
        for i in range(D % n_buckets):
            sizes[i] += 1
        out = torch.empty(D, dtype=torch.long)
        start = 0
        for b, s in enumerate(sizes):
            out[start:start + s] = b
            start += s
        return out

    if mode != "equal_mass":
        raise ValueError(f"unknown bucketing mode: {mode}")

    assert sigma2_cpu is not None
    sigma2_pos = sigma2_cpu.clamp(min=0.0)
    cum = torch.cumsum(sigma2_pos, dim=0)
    total = float(cum[-1].item())
    out = torch.empty(D, dtype=torch.long)
    if total <= 0.0:
        # Degenerate spectrum: fall back to equal_count.
        return bucket_assignment(D, n_buckets, mode="equal_count")
    # Place B - 1 boundary thresholds at b * total / B (b = 1, ..., B - 1).
    # Use searchsorted on the cumulative sum to find the first eigvec whose
    # cumulative mass exceeds each threshold; that's the bucket boundary.
    thresholds = torch.linspace(total / n_buckets,
                                total * (n_buckets - 1) / n_buckets,
                                n_buckets - 1, dtype=torch.float64)
    edges = torch.searchsorted(cum, thresholds, right=True).tolist()
    # Force each bucket to be non-empty (when possible): require strictly
    # increasing edges, and pad with the trailing index when the spectrum
    # is so concentrated that consecutive thresholds land on the same eigvec.
    fixed: list[int] = []
    prev = 0
    for b, e in enumerate(edges):
        lo = max(prev + 1, e)
        lo = min(lo, D - (n_buckets - 1 - b))
        fixed.append(lo)
        prev = lo
    start = 0
    for b, stop in enumerate([*fixed, D]):
        out[start:stop] = b
        start = stop
    return out


def per_bucket_alignment(
    view: StandardizedView,
    V: torch.Tensor,
    bucket_of: torch.Tensor,
    n_buckets: int,
) -> torch.Tensor:
    """Streamed ``S[i, b] = sum_{j in bucket_b} (V[:, j]^T phi_i)^2``.

    Args:
        view: Standardized view over ``Phi``.
        V: ``(D, r)`` eigenvector matrix.
        bucket_of: ``(r,)`` long tensor mapping eigvec index -> bucket id
            (typically from :func:`bucket_assignment`).
        n_buckets: Total number of buckets (must equal
            ``bucket_of.max().item() + 1``).

    Returns:
        ``(N, n_buckets)`` float32 alignment matrix on ``view.device``.
    """
    N = view.N
    device = view.device
    V_dev = V.to(device)
    bucket_of_dev = bucket_of.to(device)
    S = torch.zeros(N, n_buckets, dtype=torch.float32, device=device)
    for idx, chunk in view.stream():
        proj = chunk @ V_dev
        sq = proj * proj
        S[idx.to(device)] = (
            torch.zeros(chunk.shape[0], n_buckets, dtype=torch.float32, device=device)
            .index_add_(1, bucket_of_dev, sq)
        )
    return S


def _ranks_per_column(S: torch.Tensor) -> torch.Tensor:
    """Per-column uniform-in-``[0, 1]`` ranks via two argsorts.

    Args:
        S: ``(N, B)`` tensor.

    Returns:
        ``(N, B)`` float32 tensor of ranks per column. Largest value in
        each column maps to ``1.0``, smallest to ``1/N``.
    """
    N = S.shape[0]
    order = torch.argsort(S, dim=0)
    ranks = torch.empty_like(order)
    arange = torch.arange(1, N + 1, device=S.device)
    ranks.scatter_(0, order, arange.unsqueeze(1).expand_as(order))
    return ranks.float() / float(N)


def _stratified_rank_select(
    ranks: torch.Tensor, k: int, *, seed: int = 0,
) -> torch.Tensor:
    """Greedy max-coverage of rank strata.

    For each bucket ``b``, partition ``[0, 1]`` into ``k`` equal strata
    of width ``1 / k``. The stratum of sample ``i`` in bucket ``b`` is
    ``s_b(i) = min(floor(R[i, b] * k), k - 1) in {0, ..., k - 1}``. Each
    sample is therefore a length-``B`` set, one stratum per bucket, and
    picking samples is exactly the classical *maximum coverage* problem:
    pick ``k`` sets out of ``N`` to maximize the size of their union
    over a universe of ``B * k`` cells.

    The textbook greedy -- repeatedly pick the sample covering the most
    currently-uncovered cells -- attains the standard ``(1 - 1/e)``
    approximation (Hochbaum 1996). When ``N >> k`` and the rank
    distribution is non-degenerate, the optimal solution hits *every*
    stratum exactly once per bucket, which means the per-bucket
    marginal of the selected ranks is exactly ``Uniform[0, 1]`` -- so
    mean ``~ 0.5`` and std ``~ 1/sqrt(12)`` in *every* bucket, not just
    on average. The greedy gets close to this; in particular it does
    not exhibit the corner-bias L_inf farthest-first develops as ``B``
    grows.

    A small per-sample jitter breaks score ties deterministically given
    ``seed``.

    Args:
        ranks: ``(N, B)`` float tensor with values in ``[0, 1]``.
        k: Number of indices to pick. Must satisfy ``k <= N``.
        seed: RNG seed for tie-breaking jitter only.

    Returns:
        ``(k,)`` long tensor of selected indices on ``ranks.device``.

    Raises:
        ValueError: If ``k > N``.
    """
    N, B = ranks.shape
    if k > N:
        raise ValueError(f"k={k} > N={N}")
    device = ranks.device

    strata = (ranks * k).long().clamp_(min=0, max=k - 1)  # (N, B) in [0, k-1]
    bucket_idx = torch.arange(B, device=device).expand(N, B)  # (N, B)

    covered = torch.zeros(B, k, dtype=torch.bool, device=device)
    selected = torch.empty(k, dtype=torch.long, device=device)
    available = torch.ones(N, dtype=torch.bool, device=device)
    bb = torch.arange(B, device=device)

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    jitter = (torch.rand(N, generator=gen) * 1e-6).to(
        device=device, dtype=ranks.dtype)

    NEG = torch.tensor(-1.0, device=device, dtype=ranks.dtype)
    for step in range(k):
        # hits[i, b] = covered[b, strata[i, b]]
        hits = covered[bucket_idx, strata]  # (N, B) bool
        score = (~hits).sum(dim=1).to(ranks.dtype)
        score = torch.where(available, score, NEG)
        next_idx = int((score + jitter).argmax())
        selected[step] = next_idx
        available[next_idx] = False
        covered[bb, strata[next_idx]] = True

    return selected


def spectral_rank_coverage(
    view: StandardizedView,
    V: torch.Tensor,
    sigma2: torch.Tensor,
    lam: float,
    k: int,
    n_buckets: int,
    *,
    seed: int = 0,
    return_full_ranks: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Spectral-rank coverage coreset (greedy max-coverage of rank strata).

    Computes the per-bucket rank matrix ``R in [0, 1]^{N x B}`` and runs
    :func:`_stratified_rank_select` to pick ``k`` samples whose ranks,
    aggregated per bucket, cover as many of the ``B * k`` equal-width
    strata as possible. The per-bucket marginals of selected ranks are
    therefore approximately ``Uniform[0, 1]`` (mean ``~ 0.5``, std
    ``~ 1/sqrt(12)``), without the corner-bias ``L_inf`` farthest-first
    develops as ``B`` grows.

    Args:
        view: Standardized view over ``Phi``.
        V: ``(D, r)`` eigenvectors.
        sigma2: ``(r,)`` eigenvalues (without ``lam``) -- used only for
            the ``effective_dim`` stat and equal-mass bucketing.
        lam: Ridge ``lambda``.
        k: Coreset size to return.
        n_buckets: Number of contiguous eigvec buckets.
        seed: Tie-breaking seed for the greedy argmax (algorithm is
            otherwise deterministic given ``view``, ``V``, and
            ``n_buckets``).
        return_full_ranks: If ``True``, also return the
            ``(k, n_buckets)`` matrix of per-bucket ranks at the selected
            samples. Useful as an aux target downstream.

    Returns:
        ``(indices, ranks_at_selected, stats)``:

        - ``indices``: ``(k,)`` CPU long tensor, sorted ascending.
        - ``ranks_at_selected``: ``(k, n_buckets)`` CPU float32 tensor of
          per-bucket ranks for the selected samples (when
          ``return_full_ranks`` is True), else ``None``.
        - ``stats``: dict with ``home_bucket_histogram`` (list of int),
          ``mean_rank_per_bucket`` (list of float, target 0.5),
          ``std_rank_per_bucket`` (list of float, target ``1/sqrt(12)
          ~= 0.2887``), ``effective_dim``, ``n_buckets``, ``runtime_sec``.
    """
    t0 = time.time()
    device = view.device

    bucket_of = bucket_assignment(sigma2, n_buckets, mode="equal_mass")
    S = per_bucket_alignment(view, V, bucket_of, n_buckets)
    ranks = _ranks_per_column(S)
    b_star = S.argmax(dim=1)
    home_hist = torch.bincount(b_star, minlength=n_buckets).cpu().tolist()
    eff_dim = float((sigma2.to(device) / (sigma2.to(device) + lam)).sum().item())

    selected = _stratified_rank_select(ranks, k, seed=seed)
    selected_sorted, _ = torch.sort(selected)
    sel_cpu = selected_sorted.cpu()

    sel_ranks = ranks[selected_sorted]
    mean_rank = sel_ranks.mean(dim=0).cpu().tolist()
    std_rank = sel_ranks.std(dim=0).cpu().tolist()
    ranks_at_sel = sel_ranks.cpu() if return_full_ranks else None

    stats = {
        "home_bucket_histogram": home_hist,
        "mean_rank_per_bucket": mean_rank,
        "std_rank_per_bucket": std_rank,
        "effective_dim": eff_dim,
        "n_buckets": int(n_buckets),
        "runtime_sec": time.time() - t0,
    }
    return sel_cpu, ranks_at_sel, stats
