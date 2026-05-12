"""Feature-covariance eigendecomposition.

Computes the ``D x D`` feature covariance ``C = Phi^T Phi`` by streaming row
chunks via a :class:`~coreset.preprocess.StandardizedView` and accumulating
``chunk.T @ chunk`` on device. We return ``sigma2 = eigvals(C)`` (sorted
descending) and ``V`` (matching eigenvectors as columns).

Conventions about the ridge ``lambda``
--------------------------------------
The returned ``sigma2`` does **not** include the ridge. Adding ``lambda`` to
the diagonal of ``C`` just shifts every eigenvalue by ``lambda`` without
touching the eigenvectors, so we report ``sigma2 = eigvals(Phi^T Phi)`` and
let every downstream formula use ``sigma2 + lambda`` as the regularized
eigenvalue. This matches the standard ridge-leverage form

    h_i = phi_i^T (Phi^T Phi + lambda I)^{-1} phi_i
        = sum_j (V[:, j]^T phi_i)^2 / (sigma2[j] + lambda).

For numerical robustness when ``lambda`` is large relative to ``sigma2`` we
still ``eigh`` the ridged matrix and subtract ``lambda`` afterwards.

Low-rank path
-------------
For ``D > ~4096`` you can pass ``low_rank=True``: instead of building the
full ``D x D`` covariance and ``eigh``-ing it (``O(D^3)``), we run
randomized subspace iteration on ``Phi^T (Phi @ Q)``, returning only the
top ``r`` eigenpairs.
"""

from __future__ import annotations

from pathlib import Path

import torch

from coreset.preprocess import StandardizedView


def _gram_full(view: StandardizedView) -> torch.Tensor:
    """Stream ``C = Phi^T Phi`` on device.

    Args:
        view: A standardized view over the feature matrix.

    Returns:
        ``(D, D)`` float32 tensor on ``view.device`` containing
        ``Phi^T Phi``.
    """
    D = view.D
    device = view.device
    C = torch.zeros(D, D, dtype=torch.float32, device=device)
    for _, chunk in view.stream():
        C.addmm_(chunk.T, chunk)
    return C


def _eig_full(view: StandardizedView, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Full eigendecomposition path.

    Args:
        view: Standardized view.
        lam: Ridge ``lambda`` (added to the diagonal pre-``eigh`` only for
            numerical robustness; subtracted from the returned eigenvalues).

    Returns:
        ``(sigma2, V)``: descending eigenvalues of ``Phi^T Phi`` (with the
        ridge subtracted back out and clamped to non-negative) and the
        matching ``(D, D)`` eigenvector matrix as columns.
    """
    C = _gram_full(view)
    C.diagonal().add_(lam)
    eigvals, V = torch.linalg.eigh(C)
    eigvals = eigvals.flip(0)
    V = V.flip(1)
    sigma2 = (eigvals - lam).clamp(min=0.0)
    return sigma2.contiguous(), V.contiguous()


def _eig_low_rank(
    view: StandardizedView,
    r: int,
    n_iter: int = 4,
    oversample: int = 16,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomized subspace iteration for the top-``r`` eigenpairs.

    Iterates ``Y = Phi^T (Phi @ Q); Q, _ = qr(Y)`` for ``n_iter`` rounds to
    align ``Q`` with the dominant eigenspace, then forms ``B = Q^T C Q``
    and eigendecomposes the small ``(r+oversample, r+oversample)`` matrix.

    Args:
        view: Standardized view.
        r: Number of top eigenpairs to return.
        n_iter: Subspace iteration rounds. 4 is usually enough; bump for
            heavy-tailed spectra.
        oversample: Extra columns held during iteration for stability.
        seed: RNG seed for the starting ``Q``.

    Returns:
        ``(sigma2, V)``: ``(r,)`` descending eigenvalues (no ridge added,
        clamped non-negative) and ``(D, r)`` eigenvectors as columns.
    """
    D = view.D
    device = view.device
    rr = min(D, r + oversample)
    g = torch.Generator(device=device).manual_seed(seed)
    Q = torch.randn(D, rr, device=device, generator=g)
    Q, _ = torch.linalg.qr(Q)

    for _ in range(n_iter):
        Y = torch.zeros(D, rr, device=device, dtype=torch.float32)
        for _, chunk in view.stream():
            proj = chunk @ Q
            Y.addmm_(chunk.T, proj)
        Q, _ = torch.linalg.qr(Y)

    B = torch.zeros(rr, rr, device=device, dtype=torch.float32)
    for _, chunk in view.stream():
        proj = chunk @ Q
        B.addmm_(proj.T, proj)
    B = 0.5 * (B + B.T)
    eigvals, V_small = torch.linalg.eigh(B)
    eigvals = eigvals.flip(0)
    V_small = V_small.flip(1)
    sigma2 = eigvals[:r].clamp(min=0.0)
    V = Q @ V_small[:, :r]
    return sigma2.contiguous(), V.contiguous()


def compute_eig(
    view: StandardizedView,
    lam: float,
    *,
    low_rank: bool = False,
    r: int = 512,
    n_iter: int = 4,
    seed: int = 0,
    out_path: Path | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigendecompose ``Phi^T Phi`` and optionally persist the result.

    Args:
        view: Standardized view of ``Phi``.
        lam: Ridge ``lambda`` used downstream. Only affects numerical
            stability of the ``eigh`` itself; the returned ``sigma2``
            does **not** include it (callers add it back as needed).
        low_rank: If ``True``, run randomized subspace iteration and
            return only the top ``r`` eigenpairs. Recommended for
            ``D > ~4096``.
        r: Number of top eigenpairs to return in the low-rank path.
        n_iter: Subspace-iteration rounds in the low-rank path.
        seed: RNG seed for the low-rank path.
        out_path: Optional file path to persist ``{"sigma2", "V", "lam",
            "low_rank"}`` via ``torch.save``. Parents are created.

    Returns:
        ``(sigma2, V)`` -- ``sigma2`` is a ``(D,)`` or ``(r,)`` float32
        tensor sorted descending; ``V`` is the matching ``(D, D)`` or
        ``(D, r)`` eigenvector matrix with eigenvectors as columns. Both
        live on ``view.device``.
    """
    if low_rank:
        sigma2, V = _eig_low_rank(view, r=r, n_iter=n_iter, seed=seed)
    else:
        sigma2, V = _eig_full(view, lam=lam)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"sigma2": sigma2.cpu(), "V": V.cpu(), "lam": float(lam), "low_rank": low_rank},
            out_path,
        )
    return sigma2, V
