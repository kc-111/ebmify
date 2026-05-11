"""Composable, invertible per-feature transforms.

Each transform is an ``nn.Module`` that learns per-feature parameters via
``fit(X)``, applies them in ``forward(X)``, and exactly reverses them in
``inverse(Z)``. ``TransformPipeline`` chains transforms left-to-right (and
inverts in reverse order).

Default pipeline for the regressors:

* inputs use ``["quantile_gpd"]`` — ``RandomizedQuantileGPD`` maps each
  feature exactly to N(0, 1) marginals via empirical CDF body + GPD tails,
  with stochastic PIT for tied / atomic data. No hyperparameter tuning;
  handles arbitrarily heavy tails and discrete spikes uniformly.
* outputs use ``["robust", "yeo_johnson"]`` — ``RobustScale`` first centers
  and scales by median/IQR (so the YJ ``(x+1)^λ`` term operates on data near
  unit scale), then ``YeoJohnson`` removes residual skew. The YJ lambda is
  selected per feature by minimizing the **Wasserstein-2 distance to N(0,1)**
  (compares the full empirical CDF to the standard-normal CDF — far more
  robust to outliers than classical MLE). YJ is preferred over RQT-GPD on
  the output side because its inverse is a smooth analytic function called
  on every ``predict``, while RQT-GPD's inverse uses GPD extrapolation that
  can amplify noise when predicted z values land in extrapolated tail
  regions.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import scipy.optimize
import scipy.special
import scipy.stats
import torch
import torch.nn as nn


_EPS = 1e-12


# ----------------------------------------------------------------------
# Pure-torch helpers used by RandomizedQuantileGPD's hot path
# ----------------------------------------------------------------------


def _interp_torch(
    x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor
) -> torch.Tensor:
    """1-D piecewise-linear interpolation, ``np.interp`` semantics, in torch.

    Args:
        x:  Query points, any shape ``(...)`` with the same dtype as ``xp``.
        xp: Reference x's, ``(M,)``, sorted ascending.
        fp: Reference y's, ``(M,)``.

    Returns:
        Interpolated values with the same shape as ``x``. Queries below
        ``xp[0]`` clamp to ``fp[0]``; queries above ``xp[-1]`` clamp to
        ``fp[-1]`` (matching ``np.interp``).
    """
    M = xp.shape[0]
    if M == 0:
        return torch.zeros_like(x)
    if M == 1:
        return fp[0].expand_as(x).clone()
    idx = torch.searchsorted(xp, x).clamp(1, M - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]
    denom = (x1 - x0).clamp_min(_EPS)
    out = y0 + (x - x0) / denom * (y1 - y0)
    out = torch.where(x <= xp[0], fp[0].expand_as(out), out)
    out = torch.where(x >= xp[-1], fp[-1].expand_as(out), out)
    return out


def _gpd_cdf_scalar(
    y: torch.Tensor, xi: float, sigma: float
) -> torch.Tensor:
    """GPD CDF ``F(y; xi, sigma)`` for ``y >= 0`` and scalar ``(xi, sigma)``.

    Hot-path closed form (no autograd path required). The companion
    ``_gpd_cdf_torch`` below is the same math but with batched tensor params
    and a Taylor branch for autograd stability — kept separate because that
    overhead is wasted in the inference hot path called from
    ``_forward_one`` / ``_inverse_one``.

    Args:
        y:     Non-negative exceedance values, any shape.
        xi:    Shape parameter (Python float).
        sigma: Scale parameter (Python float, ``> 0``).

    Returns:
        ``F(y)`` with the same shape as ``y``. Closed form:

        * ``xi != 0``: ``1 - (1 + xi*y/sigma)^(-1/xi)``.
        * ``xi == 0``: ``1 - exp(-y/sigma)`` (continuous limit).
    """
    z = y / max(sigma, _EPS)
    if abs(xi) < 1e-10:
        return -torch.expm1(-z)
    arg = (1.0 + xi * z).clamp_min(_EPS)
    return 1.0 - arg.pow(-1.0 / xi)


def _gpd_ppf_scalar(
    p: torch.Tensor, xi: float, sigma: float
) -> torch.Tensor:
    """GPD inverse CDF (quantile function), closed form, scalar params.

    Args:
        p:     Probabilities in ``(0, 1)``, any shape.
        xi:    Shape parameter (Python float).
        sigma: Scale parameter (Python float, ``> 0``).

    Returns:
        ``Q(p)`` with the same shape as ``p``. Closed form:

        * ``xi != 0``: ``sigma * ((1-p)^(-xi) - 1) / xi``.
        * ``xi == 0``: ``-sigma * log(1-p)`` (continuous limit).
    """
    if abs(xi) < 1e-10:
        return -sigma * torch.log1p(-p)
    return sigma * ((1.0 - p).pow(-xi) - 1.0) / xi


def _body_bin_index(
    u: torch.Tensor, n_bins: int, q_lo: float, q_hi: float
) -> torch.Tensor:
    """Equiprobable body-bin index in ``[0, n_bins - 1]`` (long).

    The body interval ``[q_lo, q_hi]`` is partitioned into ``n_bins`` equal-width
    sub-intervals in ``u``-space. Values of ``u`` outside the body are clamped
    to the boundary bins (``0`` for the lower tail, ``n_bins - 1`` for the
    upper tail) — that's the **Option A** label policy: tail samples fold
    into the boundary classes.

    Args:
        u:       PIT outputs in ``[0, 1]``, any shape.
        n_bins:  Number of body bins ``K >= 2``.
        q_lo:    Lower tail threshold (``q_downarrow``).
        q_hi:    Upper tail threshold (``q_uparrow``).

    Returns:
        Long-tensor bin indices in ``[0, n_bins - 1]`` with the same shape as ``u``.
    """
    width = max(q_hi - q_lo, _EPS)
    raw = ((u - q_lo) / width * n_bins).floor()
    return raw.clamp(min=0.0, max=float(n_bins - 1)).long()


def _body_snap_u_to_midpoint(
    u: torch.Tensor, n_bins: int, q_lo: float, q_hi: float
) -> torch.Tensor:
    """Snap body-region ``u`` to the nearest of K equiprobable body-bin midpoints.

    Body bin midpoints in ``u``-space:
    ``u_k* = q_lo + (k + 0.5) / K * (q_hi - q_lo),  k = 0, ..., K - 1``.

    Tail-region ``u`` (``u < q_lo`` or ``u > q_hi``) is **left unchanged** —
    binning is body-only, so the GPD tails stay continuous. This matches the
    "snap body, continuous tail" hybrid: regression targets retain full tail
    information; only the body is quantized for denoising.

    Args:
        u:       PIT outputs in ``[0, 1]``, any shape.
        n_bins:  Number of body bins ``K >= 2``.
        q_lo:    Lower tail threshold (``q_downarrow``).
        q_hi:    Upper tail threshold (``q_uparrow``).

    Returns:
        Hybrid ``u``: body values snapped to bin midpoints, tail values unchanged.
    """
    width = max(q_hi - q_lo, _EPS)
    in_body = (u >= q_lo) & (u <= q_hi)
    idx = ((u - q_lo) / width * n_bins).floor().clamp(min=0.0, max=float(n_bins - 1))
    u_snap = q_lo + (idx + 0.5) / n_bins * width
    return torch.where(in_body, u_snap, u)


# ----------------------------------------------------------------------
# Base
# ----------------------------------------------------------------------


class _Transform(nn.Module):
    """Base class for invertible per-feature transforms.

    Subclasses store fitted parameters via ``register_buffer`` so they persist
    through ``state_dict`` / ``.to(device)``.
    """

    def fit(self, X: torch.Tensor) -> "_Transform":
        """Estimate per-feature parameters from a training sample.

        Args:
            X: Training data ``[N, d]``.

        Returns:
            ``self``.
        """
        raise NotImplementedError

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Apply the (fitted) transform.

        Args:
            X: Inputs ``[N, d]``.

        Returns:
            Transformed tensor ``[N, d]``.
        """
        raise NotImplementedError

    def inverse(self, Z: torch.Tensor) -> torch.Tensor:
        """Invert the transform.

        Args:
            Z: Transformed values ``[N, d]``.

        Returns:
            Reconstructed inputs ``[N, d]``.
        """
        raise NotImplementedError

    def is_stochastic_in_train(self) -> bool:
        """Whether ``forward`` produces different outputs across calls in train mode.

        Used by training loops to decide whether the pipeline can be applied
        once up-front (deterministic) or must run per-batch (e.g. randomized
        PIT acting as data augmentation). Default is False; override to True
        for transforms whose forward depends on a per-call RNG draw.
        """
        return False


# ----------------------------------------------------------------------
# Concrete transforms
# ----------------------------------------------------------------------


class Identity(_Transform):
    """Pass-through transform; useful as a placeholder slot in pipelines."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = d

    def fit(self, X: torch.Tensor) -> "Identity":
        """No-op fit; returns ``self``."""
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Return ``X`` unchanged."""
        return X

    def inverse(self, Z: torch.Tensor) -> torch.Tensor:
        """Return ``Z`` unchanged."""
        return Z


class StandardScale(_Transform):
    """Z-score per feature: ``(x - mean) / std``."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = d
        self.register_buffer("mean", torch.zeros(d))
        self.register_buffer("std", torch.ones(d))

    def fit(self, X: torch.Tensor) -> "StandardScale":
        """Fit per-feature mean and (biased) std from ``X``.

        ``unbiased=False`` matches ``np.std`` semantics; the ``clamp(min=_EPS)``
        guards against zero-variance columns producing NaNs in ``forward``.
        """
        self.mean.copy_(X.mean(dim=0))
        self.std.copy_(X.std(dim=0, unbiased=False).clamp(min=_EPS))
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Z-score each column with the fitted mean/std."""
        return (X - self.mean) / self.std

    def inverse(self, Z: torch.Tensor) -> torch.Tensor:
        """Reverse the z-score: ``Z * std + mean``."""
        return Z * self.std + self.mean


class RobustScale(_Transform):
    """Robust scaler per feature: ``(x - median) / IQR``.

    Resistant to outliers and heavy tails; preferred over ``StandardScale``
    as the first step of a robust preprocessing pipeline.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = d
        self.register_buffer("median", torch.zeros(d))
        self.register_buffer("iqr", torch.ones(d))

    def fit(self, X: torch.Tensor) -> "RobustScale":
        """Compute per-feature median and inter-quartile range.

        IQR is clamped to ``_EPS`` so a constant column (zero IQR) stays
        finite under division.
        """
        qs = torch.tensor([0.25, 0.5, 0.75], device=X.device, dtype=X.dtype)
        q = torch.quantile(X, qs, dim=0)  # [3, d]
        self.median.copy_(q[1])
        self.iqr.copy_((q[2] - q[0]).clamp(min=_EPS))
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Center by median and rescale by IQR."""
        return (X - self.median) / self.iqr

    def inverse(self, Z: torch.Tensor) -> torch.Tensor:
        """Reverse: ``Z * IQR + median``."""
        return Z * self.iqr + self.median


class MinMaxScale(_Transform):
    """Linearly map each feature's data range to ``[low, high]``."""

    def __init__(self, d: int, low: float = -1.0, high: float = 1.0) -> None:
        super().__init__()
        if not high > low:
            raise ValueError(f"MinMaxScale requires high > low, got {low=}, {high=}")
        self.d = d
        self.low = float(low)
        self.high = float(high)
        self.register_buffer("data_min", torch.zeros(d))
        self.register_buffer("data_max", torch.ones(d))

    def fit(self, X: torch.Tensor) -> "MinMaxScale":
        """Record per-column min and max of ``X``."""
        self.data_min.copy_(X.min(dim=0).values)
        self.data_max.copy_(X.max(dim=0).values)
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Map ``[data_min, data_max]`` linearly to ``[low, high]``."""
        rng = (self.data_max - self.data_min).clamp(min=_EPS)
        return (X - self.data_min) / rng * (self.high - self.low) + self.low

    def inverse(self, Z: torch.Tensor) -> torch.Tensor:
        """Reverse the linear map back to the original range."""
        rng = (self.data_max - self.data_min).clamp(min=_EPS)
        return (Z - self.low) / (self.high - self.low) * rng + self.data_min


# ----------------------------------------------------------------------
# Yeo-Johnson power transform
#
# Two parallel implementations of the same formula exist below:
#
#   * ``_yj_forward_np``: 1-D numpy version, called by the
#     ``scipy.optimize.minimize_scalar`` inner loop in ``_fit_yj_lambda_w2``
#     (scipy needs a numpy-returning objective; converting to torch every
#     iteration would be much slower than just keeping numpy here).
#   * ``_yj_forward_torch`` / ``_yj_inverse_torch``: batched, per-feature
#     ``lambdas``, autograd-friendly. Used at fit-finalize and at
#     ``forward`` / ``inverse`` time.
#
# The two implementations are intentionally redundant; they serve different
# call sites with different performance and gradient requirements.
# ----------------------------------------------------------------------


def _yj_forward_np(x: np.ndarray, lam: float) -> np.ndarray:
    """Yeo-Johnson forward on a 1-D numpy array (used during ``lambda`` fitting).

    The transform has two singular values for ``lam``:

    * ``lam == 0`` for the positive branch — the ``(x+1)^lam - 1) / lam``
      formula degenerates to the limit ``log(x+1)``.
    * ``lam == 2`` for the negative branch — the ``-((-x+1)^(2-lam) - 1) / (2-lam)``
      formula degenerates to ``-log(-x+1)``.

    Args:
        x:   1-D float array (one feature column).
        lam: Yeo-Johnson lambda parameter.

    Returns:
        Transformed array, same shape and float64 dtype.
    """
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos

    if abs(lam) < 1e-7:
        out[pos] = np.log1p(x[pos])
    else:
        out[pos] = (np.power(x[pos] + 1.0, lam) - 1.0) / lam

    if abs(2.0 - lam) < 1e-7:
        out[neg] = -np.log1p(-x[neg])
    else:
        out[neg] = -(np.power(-x[neg] + 1.0, 2.0 - lam) - 1.0) / (2.0 - lam)

    return out


def _yj_forward_torch(x: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
    """Yeo-Johnson forward in torch, broadcasting per-feature ``lambdas``.

    Args:
        x:       Inputs ``[..., d]``.
        lambdas: Per-feature lambdas ``[d]``.

    Returns:
        Transformed tensor ``[..., d]`` with the same dtype/device as ``x``.
    """
    # Detect the two singular-lambda branches up front so we can route them
    # to their continuous-limit formula instead of the degenerate division.
    is_zero = lambdas.abs() < 1e-7        # positive branch: log(x+1) limit
    is_two = (lambdas - 2.0).abs() < 1e-7  # negative branch: -log(-x+1) limit

    pos_mask = x >= 0

    # Positive branch ((x+1)^lam - 1) / lam, with the log(x+1) limit at lam=0.
    # ``lam_safe`` substitutes 1.0 in the singular slot so the dead branch in
    # the where-blend is finite (we don't actually use its value, but torch
    # still has to evaluate both sides).
    x_plus_1 = (x + 1.0).clamp(min=_EPS)
    lam_safe = torch.where(is_zero, torch.ones_like(lambdas), lambdas)
    pos_pow = (torch.pow(x_plus_1, lam_safe) - 1.0) / lam_safe
    pos_log = torch.log(x_plus_1)
    pos_val = torch.where(is_zero, pos_log, pos_pow)

    # Negative branch -((-x+1)^(2-lam) - 1) / (2-lam), with -log(-x+1) at lam=2.
    neg_x_plus_1 = (-x + 1.0).clamp(min=_EPS)
    lam2 = 2.0 - lambdas
    lam2_safe = torch.where(is_two, torch.ones_like(lambdas), lam2)
    neg_pow = -(torch.pow(neg_x_plus_1, lam2_safe) - 1.0) / lam2_safe
    neg_log = -torch.log(neg_x_plus_1)
    neg_val = torch.where(is_two, neg_log, neg_pow)

    return torch.where(pos_mask, pos_val, neg_val)


def _yj_inverse_torch(y: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
    """Yeo-Johnson inverse, broadcasting per-feature ``lambdas`` over rows.

    Sign of ``y`` matches sign of the original ``x`` (the forward transform
    preserves sign), so the masking key is ``y >= 0``.

    Args:
        y:       Transformed values ``[..., d]``.
        lambdas: Per-feature lambdas ``[d]``.

    Returns:
        Reconstructed inputs ``[..., d]``.
    """
    # Same singular-lambda routing as the forward direction: lam=0 collapses
    # the positive branch to ``exp(y) - 1``; lam=2 collapses the negative
    # branch to ``1 - exp(-y)``.
    is_zero = lambdas.abs() < 1e-7
    is_two = (lambdas - 2.0).abs() < 1e-7

    pos_mask = y >= 0

    # Positive branch: lam != 0: x = (y*lam + 1)^(1/lam) - 1; lam == 0: x = exp(y) - 1
    lam_safe = torch.where(is_zero, torch.ones_like(lambdas), lambdas)
    pos_arg = (y * lam_safe + 1.0).clamp(min=_EPS)
    pos_pow = torch.pow(pos_arg, 1.0 / lam_safe) - 1.0
    pos_exp = torch.exp(y) - 1.0
    pos_val = torch.where(is_zero, pos_exp, pos_pow)

    # Negative branch: lam != 2: x = 1 - (1 - y*(2-lam))^(1/(2-lam)); lam == 2: x = 1 - exp(-y)
    lam2 = 2.0 - lambdas
    lam2_safe = torch.where(is_two, torch.ones_like(lambdas), lam2)
    neg_arg = (1.0 - y * lam2_safe).clamp(min=_EPS)
    neg_pow = 1.0 - torch.pow(neg_arg, 1.0 / lam2_safe)
    neg_exp = 1.0 - torch.exp(-y)
    neg_val = torch.where(is_two, neg_exp, neg_pow)

    return torch.where(pos_mask, pos_val, neg_val)


def _fit_yj_lambda_w2(x: np.ndarray) -> float:
    """Choose Yeo-Johnson ``lambda`` minimizing Wasserstein-2 distance to N(0,1).

    Compares the standardized transformed sample (sorted) against the
    standard-normal quantile function ``Phi^{-1}((i + 0.5) / n)``. Bounded
    1-D scalar optimization on ``lambda in [-2, 2]``.

    Args:
        x: 1-D feature column.

    Returns:
        Best ``lambda`` (returns ``1.0`` — identity — if fewer than 2
        finite values are available).
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 1.0  # identity
    n = x.size
    # Hazen plotting positions ``(i + 0.5) / n`` map order statistics to
    # interior probabilities, avoiding 0/1 which would blow up ``ndtri``.
    quantile_pos = (np.arange(n) + 0.5) / n
    theoretical = scipy.special.ndtri(quantile_pos)

    def objective(lam: float) -> float:
        try:
            z = _yj_forward_np(x, float(lam))
        except (FloatingPointError, OverflowError):
            return np.inf
        if not np.all(np.isfinite(z)):
            return np.inf
        std = z.std()
        if std < _EPS:
            return np.inf
        # Re-standardize the transformed sample inside the objective. The
        # Yeo-Johnson lambda controls *shape*; comparing scale-different
        # samples to N(0,1) would conflate scale and shape distortion. By
        # standardizing first, the W2 score is invariant to the input's
        # scale and tracks only the shape mismatch — this is what makes
        # the criterion robust to outlier-driven scale shifts.
        z = (z - z.mean()) / std
        z_sorted = np.sort(z)
        return float(np.mean((z_sorted - theoretical) ** 2))

    res = scipy.optimize.minimize_scalar(
        objective, bounds=(-2.0, 2.0), method="bounded", options={"xatol": 1e-4}
    )
    return float(res.x)


def _fit_yj_lambda_mle(x: np.ndarray) -> float:
    """Classical MLE-based Yeo-Johnson lambda (scipy fallback).

    Args:
        x: 1-D feature column.

    Returns:
        Best ``lambda`` from ``scipy.stats.yeojohnson``; returns ``1.0``
        when ``x`` has fewer than 2 finite values.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 1.0
    _, lam = scipy.stats.yeojohnson(x)
    return float(lam)


class YeoJohnson(_Transform):
    """Yeo-Johnson power transform (handles positive AND negative inputs).

    Lambda is fit per feature at ``fit`` time then frozen as a buffer.
    Output is standardized (zero-mean unit-std) so the transform plays nicely
    in pipelines that follow it.

    Args:
        d:         Number of features.
        criterion: ``"w2"`` (robust, default) or ``"mle"`` (classical scipy).
    """

    def __init__(self, d: int, criterion: str = "w2") -> None:
        super().__init__()
        if criterion not in {"w2", "mle"}:
            raise ValueError(f"YeoJohnson criterion must be 'w2' or 'mle', got {criterion!r}")
        self.d = d
        self.criterion = criterion
        self.register_buffer("lambdas", torch.ones(d))
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("sigma", torch.ones(d))

    def fit(self, X: torch.Tensor) -> "YeoJohnson":
        """Fit per-feature ``lambda`` then freeze a post-transform mean/std.

        Args:
            X: ``[N, d]`` training sample.

        Returns:
            ``self``.

        Raises:
            ValueError: if ``X.shape[-1] != self.d``.
        """
        if X.shape[-1] != self.d:
            raise ValueError(f"YeoJohnson(d={self.d}) got X with last dim {X.shape[-1]}")
        X_np = X.detach().cpu().numpy().astype(np.float64)
        d = X_np.shape[1]
        lambdas = np.empty(d, dtype=np.float64)
        for j in range(d):
            col = X_np[:, j]
            if self.criterion == "w2":
                lambdas[j] = _fit_yj_lambda_w2(col)
            else:
                lambdas[j] = _fit_yj_lambda_mle(col)

        self.lambdas.copy_(torch.from_numpy(lambdas).to(self.lambdas))

        # Freeze the post-YJ mean/std so ``forward`` always emits unit-scale
        # outputs even when the chosen lambda doesn't naturally do so.
        with torch.no_grad():
            Y = _yj_forward_torch(X, self.lambdas)
            self.mu.copy_(Y.mean(dim=0))
            self.sigma.copy_(Y.std(dim=0, unbiased=False).clamp(min=_EPS))
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply Yeo-Johnson then z-score using the frozen mean/std."""
        Y = _yj_forward_torch(X, self.lambdas)
        return (Y - self.mu) / self.sigma

    def inverse(self, Z: torch.Tensor) -> torch.Tensor:
        """Reverse: un-z-score then apply the analytic Yeo-Johnson inverse."""
        Y = Z * self.sigma + self.mu
        return _yj_inverse_torch(Y, self.lambdas)


# ----------------------------------------------------------------------
# GPD helpers — autograd-friendly, vectorized across features
# ----------------------------------------------------------------------


def _gpd_cdf_torch(
    y: torch.Tensor, xi: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """Generalized-Pareto CDF on positive support ``y >= 0``.

    ``F(y) = 1 - (1 + xi*y/sigma)^(-1/xi)`` for ``xi != 0``,
    ``F(y) = 1 - exp(-y/sigma)`` for ``xi = 0`` (continuous limit).

    Implemented via ``log(1-F) = -log1p(xi*z)/xi`` with a Taylor expansion
    when ``|xi|`` is small, so the gradient flows smoothly through the
    ``xi → 0`` exponential limit (a hard ``where`` switch would zero out the
    gradient at exactly ``xi = 0``, stalling the optimizer at its init).
    All inputs broadcastable.
    """
    z = y / sigma
    u = xi * z
    log1p_u = torch.log1p(u.clamp(min=-1.0 + 1e-12))

    # Branch A (exact): -log1p(xi*z) / xi
    xi_safe = torch.where(xi.abs() < 1e-12, torch.full_like(xi, 1e-12), xi)
    log_one_minus_F_exact = -log1p_u / xi_safe
    # Branch B (Taylor near xi=0): -z + xi*z²/2 - xi²*z³/3 (keeps grad alive at xi=0)
    z2 = z * z
    log_one_minus_F_taylor = -z + 0.5 * xi * z2 - (1.0 / 3.0) * xi * xi * z2 * z

    is_small = xi.abs() < 1e-3
    log_one_minus_F = torch.where(
        is_small.expand_as(log_one_minus_F_exact),
        log_one_minus_F_taylor,
        log_one_minus_F_exact,
    )
    return -torch.expm1(log_one_minus_F)


def _fit_gpd_w2_batched(
    excess: torch.Tensor,
    mask: torch.Tensor,
    n_steps: int = 200,
    lr: float = 0.05,
    xi_min: float = -0.49,
    xi_max: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit GPD ``(xi, sigma)`` per row by minimizing W2² to ``Uniform(0, 1)``.

    Vectorized across features: a single Adam optimization runs on all rows
    simultaneously, sorting and W2² done with batched tensor ops. Replaces the
    per-feature ``scipy.stats.genpareto.fit`` loop. Empirical speedup at
    n=4000 samples on CPU: ~4x at d=100, ~9x at d=1000 (grows linearly in d).

    Args:
        excess: ``[d, M]`` non-negative exceedances per feature, padded to ``M``.
        mask:   ``[d, M]`` bool, ``True`` on valid entries (the rest is padding).
        n_steps: Adam steps.
        lr:      Adam learning rate.
        xi_min:  Lower clamp for ``xi`` (``-0.5`` is the MLE-consistency floor).
        xi_max:  Upper clamp for ``xi``.

    Returns:
        ``(xi [d], sigma [d])`` as detached tensors.
    """
    device = excess.device
    dtype = excess.dtype if excess.is_floating_point() else torch.float64
    excess = excess.to(dtype)
    d, M = excess.shape

    valid_count = mask.sum(dim=1).clamp(min=1).to(dtype)  # [d]
    excess_safe = torch.where(mask, excess, torch.zeros_like(excess))

    # Adam needs autograd; force-enable so this works inside ``torch.no_grad``
    # contexts (e.g. ``TransformPipeline.fit``).
    with torch.enable_grad():
        # MoM-style init: sigma ≈ mean of valid exceedances; xi = 0 (exponential).
        sigma_init = (excess_safe.sum(dim=1) / valid_count).clamp(min=1e-6)
        log_sigma = torch.log(sigma_init).clone().detach().requires_grad_(True)
        xi_raw = torch.zeros(d, dtype=dtype, device=device, requires_grad=True)

        opt = torch.optim.Adam([log_sigma, xi_raw], lr=lr)

        # Theoretical Uniform(0,1) order statistics at Hazen plotting
        # positions ``(rank + 0.5) / k``: avoids exact 0/1 endpoints (which
        # would cause -inf/inf under log(1-F)) and uses each feature's own
        # exceedance count ``k`` (some features have shorter tails).
        ranks = torch.arange(M, device=device, dtype=dtype).unsqueeze(0)  # [1, M]
        theoretical = (ranks + 0.5) / valid_count.unsqueeze(1)  # [d, M]
        INF = torch.tensor(1e30, dtype=dtype, device=device)

        for _ in range(n_steps):
            opt.zero_grad()
            sigma = log_sigma.exp().unsqueeze(1)  # [d, 1]
            xi = xi_raw.clamp(min=xi_min, max=xi_max).unsqueeze(1)  # [d, 1]

            F = _gpd_cdf_torch(excess_safe, xi, sigma)  # [d, M]

            # Sort ascending; route padded entries to the end via a detached key.
            F_for_sort = torch.where(mask, F.detach(), INF.expand_as(F))
            sort_idx = torch.argsort(F_for_sort, dim=1)
            F_sorted = F.gather(1, sort_idx)
            mask_sorted = mask.gather(1, sort_idx).to(dtype)

            diff = (F_sorted - theoretical) * mask_sorted
            W2_per_feat = (diff * diff).sum(dim=1) / valid_count
            loss = W2_per_feat.sum()
            loss.backward()
            opt.step()

        xi_final = xi_raw.detach().clamp(min=xi_min, max=xi_max)
        sigma_final = log_sigma.detach().exp()
    return xi_final, sigma_final


# ----------------------------------------------------------------------
# Randomized Quantile Transform with GPD tails (RQT-GPD)
# ----------------------------------------------------------------------


class RandomizedQuantileGPD(_Transform):
    """Robust transform to N(0,1) via empirical CDF body + GPD tails.

    Body
    ----
    Empirical CDF using Hazen plotting positions ``(rank + 0.5) / n``. Tied
    training values are handled via the **randomized probability integral
    transform**: at training-mode forward, each tied input is mapped to a
    uniform draw within its rank interval ``[q_low, q_high]``. The
    transformed sample's marginal is then exactly N(0,1) regardless of
    atomic mass in the input (e.g., a 50% spike at zero).

    Tails
    -----
    Generalized Pareto Distribution fit on exceedances above ``tail_upper_q``
    and below ``tail_lower_q``. Pickands–Balkema–de Haan: exceedances over a
    high threshold from any distribution converge to GPD, so this is the
    principled extrapolation. Out-of-sample extreme inputs receive distinct,
    ordered z values instead of saturating at the rank boundary like a plain
    empirical quantile transform.

    Default fitter (``gpd_method="w2"``) minimizes Wasserstein-2² between the
    GPD-mapped exceedances and Uniform(0,1), batched across all features in a
    single PyTorch Adam run. The classical ``scipy.stats.genpareto.fit`` MLE
    path is available via ``gpd_method="mle"``.

    Fallback
    --------
    If a tail has fewer than ``min_tail_size`` exceedances, that tail falls
    back to clipped empirical extrapolation (extreme values saturate at the
    boundary rank).

    Invertibility
    -------------
    ``forward(x) -> inverse(z)`` round-trips exactly for both unique and
    tied training values (the inverse snaps z within a tie's rank range
    back to the tied anchor). The reverse direction
    ``inverse(z) -> forward(x)`` is exact for unique anchors but
    re-randomizes for z values that map back to a tied anchor — that's
    fundamental: the forward map is non-injective in the tied region.

    Composition
    -----------
    This transform already outputs ~ N(0,1); chaining it with a downstream
    Gaussianizer (``YeoJohnson``, ``StandardScale``) is unnecessary and
    just costs accuracy. Use it standalone:
    ``PreprocessConfig(input_transforms=["quantile_gpd"])``.

    Binned mode (optional, hybrid body-snap / continuous-tail)
    ----------------------------------------------------------
    With ``n_bins=K``, the **body** interval ``[q_downarrow, q_uparrow]``
    of ``u``-space is partitioned into K equiprobable sub-intervals; PIT
    outputs that land in the body are snapped to the nearest body-bin
    midpoint
    ``u_k* = q_downarrow + (k + 0.5) / K * (q_uparrow - q_downarrow)``
    before Φ⁻¹. PIT outputs that land in the **GPD tails** are left
    continuous — so extreme inputs keep their full information instead
    of all collapsing onto a single boundary z-level.

    Concretely, the transformed-z marginal is a comb of K delta spikes
    inside ``[Φ⁻¹(q_downarrow), Φ⁻¹(q_uparrow)]`` flanked by two
    continuous GPD tails. This is the right regression-target view: the
    body (where label noise dominates within-bin spread) gets the
    quantization regularizer, while rare extreme samples keep their
    distinct continuous z so the model can fit the tail magnitude.

    Two complementary outputs share this fitted state:

    * :meth:`forward` returns the **regression target** — body-snapped
      and tail-continuous z. :meth:`inverse` mirrors it.
    * :meth:`bin_indices` returns the **classification target** — K-class
      integer labels in ``[0, K - 1]``. Body samples land in their owning
      bin; tail samples fold into the boundary class (``0`` for lower,
      ``K - 1`` for upper) under the **Option A** policy. Use
      :meth:`is_body` to filter or distinguish tail samples from body
      samples that genuinely live in boundary bins.
    * :meth:`from_bin_indices` inverts a label to its body-bin midpoint
      original-scale value (always interior body, never GPD-inv).

    Bins are equiprobable on the *training* body distribution: each body
    bin holds ``(q_uparrow - q_downarrow) / K`` of the mass; the boundary
    bins absorb the corresponding tail mass. Balanced for any input
    shape — heavy-tailed, bimodal-with-gap, or atomic — modulo Binomial
    sampling noise.

    Args:
        d:                 Number of features.
        tail_upper_q:      Upper-tail threshold quantile. ``"auto"`` (default)
                           selects per-fit from the sample count via the
                           ``k = max(min_tail_size, ceil(N**auto_alpha))`` rule
                           (Hall 1990 / de Haan-style scaling) so the tail has
                           a sensible number of exceedances regardless of ``N``.
                           Pass a float in ``(0.5, 1.0)`` to override.
        tail_lower_q:      Lower-tail threshold quantile, mirror of the upper.
                           ``"auto"`` (default) or float in ``(0.0, 0.5)``.
        min_tail_size:     Floor on exceedances for the GPD fit. Tails with
                           fewer than this fall back to clipped empirical
                           extrapolation. Also the ``k`` floor in ``"auto"``.
        auto_alpha:        Exponent in the auto-scaling rule
                           ``k = ceil(N ** auto_alpha)``. Default ``2/3`` is
                           the EVT-standard choice (e.g. Hall 1990); ``0.5``
                           is more conservative for very large ``N``.
        randomize_ties:    If True, randomize tied inputs in training mode.
                           Eval-mode forward is always deterministic.
        gpd_method:        ``"w2"`` (default — minimize W2² to Uniform, batched
                           in PyTorch across all features) or ``"mle"``
                           (classical scipy MLE, per-feature loop). The W2 path
                           is ~9x faster at d=1000 (n=4000 samples, CPU) with
                           comparable marginal goodness-of-fit; the criterion
                           is also robust to outliers in the exceedance set.
        gpd_w2_steps:      Number of Adam steps for the batched W2 fit
                           (only used when ``gpd_method == "w2"``).
        n_bins:            Optional. If set to ``K >= 2``, snap the PIT
                           output ``u`` to one of K equiprobable bin
                           midpoints in both ``forward`` and ``inverse``,
                           and expose ``bin_indices(X)`` /
                           ``from_bin_indices(idx)``. ``None`` (default)
                           keeps the continuous PIT behavior.
    """

    def __init__(
        self,
        d: int,
        tail_upper_q: float | str = "auto",
        tail_lower_q: float | str = "auto",
        min_tail_size: int = 30,
        auto_alpha: float = 2.0 / 3.0,
        randomize_ties: bool = True,
        gpd_method: str = "w2",
        gpd_w2_steps: int = 100,
        n_bins: int | None = None,
    ) -> None:
        super().__init__()
        self._tail_upper_q_spec = self._validate_tail_q(
            tail_upper_q, name="tail_upper_q", upper=True
        )
        self._tail_lower_q_spec = self._validate_tail_q(
            tail_lower_q, name="tail_lower_q", upper=False
        )
        if min_tail_size < 5:
            raise ValueError(f"min_tail_size should be >= 5, got {min_tail_size}")
        if not 0.0 < auto_alpha < 1.0:
            raise ValueError(f"auto_alpha must be in (0, 1), got {auto_alpha}")
        if gpd_method not in {"w2", "mle"}:
            raise ValueError(f"gpd_method must be 'w2' or 'mle', got {gpd_method!r}")
        if n_bins is not None and (not isinstance(n_bins, int) or n_bins < 2):
            raise ValueError(f"n_bins must be an int >= 2 or None, got {n_bins!r}")
        self.d = d
        self.min_tail_size = int(min_tail_size)
        self.auto_alpha = float(auto_alpha)
        self.randomize_ties = bool(randomize_ties)
        self.gpd_method = gpd_method
        self.gpd_w2_steps = int(gpd_w2_steps)
        self.n_bins = n_bins

        # Resolved per-fit threshold quantiles, persisted via state_dict.
        init_upper = (
            self._tail_upper_q_spec if isinstance(self._tail_upper_q_spec, float) else 0.95
        )
        init_lower = (
            self._tail_lower_q_spec if isinstance(self._tail_lower_q_spec, float) else 0.05
        )
        self.register_buffer(
            "tail_upper_q", torch.tensor(init_upper, dtype=torch.float64)
        )
        self.register_buffer(
            "tail_lower_q", torch.tensor(init_lower, dtype=torch.float64)
        )

        # Variable-length per-feature state stored as concatenated tensors
        # plus per-feature offsets (feature j owns indices [offsets[j], offsets[j+1])).
        # These three buffers are concatenated across features (each feature
        # owns ``offsets[j+1] - offsets[j]`` slots) so their total length
        # depends on training-data uniqueness counts. Initialised here as
        # zero-length placeholders; ``fit()`` rewrites them, and
        # ``_load_from_state_dict`` resizes them to match a checkpoint.
        self.register_buffer("sorted_unique", torch.zeros(0))
        self.register_buffer("q_low", torch.zeros(0))
        self.register_buffer("q_high", torch.zeros(0))
        self.register_buffer("offsets", torch.zeros(d + 1, dtype=torch.long))
        # Per-feature scalars.
        self.register_buffer("u_upper", torch.zeros(d))
        self.register_buffer("u_lower", torch.zeros(d))
        self.register_buffer("xi_upper", torch.zeros(d))
        self.register_buffer("sigma_upper", torch.ones(d))
        self.register_buffer("xi_lower", torch.zeros(d))
        self.register_buffer("sigma_lower", torch.ones(d))
        self.register_buffer("has_upper", torch.zeros(d, dtype=torch.bool))
        self.register_buffer("has_lower", torch.zeros(d, dtype=torch.bool))

    @staticmethod
    def _validate_tail_q(val: float | str, name: str, upper: bool) -> float | str:
        """Validate a tail-quantile spec.

        Args:
            val:   ``"auto"`` or a float; the user-facing value.
            name:  Argument name (for error messages).
            upper: ``True`` to validate against ``(0.5, 1.0)``,
                   ``False`` to validate against ``(0.0, 0.5)``.

        Returns:
            ``"auto"`` if that string was passed in, else a validated float.

        Raises:
            ValueError: if ``val`` is neither ``"auto"`` nor a float in the
                appropriate half of ``(0, 1)``.
        """
        if isinstance(val, str):
            if val != "auto":
                raise ValueError(f"{name} must be 'auto' or a float, got {val!r}")
            return "auto"
        v = float(val)
        if upper and not 0.5 < v < 1.0:
            raise ValueError(f"{name} must be in (0.5, 1.0), got {v}")
        if not upper and not 0.0 < v < 0.5:
            raise ValueError(f"{name} must be in (0.0, 0.5), got {v}")
        return v

    def _resolve_tail_quantiles(self, n: int) -> tuple[float, float]:
        """Resolve any ``"auto"`` thresholds from sample count ``n``.

        Uses the classical EVT scaling ``k = max(min_tail_size, ceil(n ** alpha))``
        with ``alpha = self.auto_alpha`` (default 2/3, Hall 1990). The resulting
        ``k`` is also capped at ``n // 2 - 1`` to leave a non-trivial body.

        Args:
            n: Number of training samples.

        Returns:
            ``(upper_q, lower_q)`` as floats. Either may already have been
            user-overridden (in which case ``"auto"`` is bypassed for that
            side).
        """
        k_auto = max(self.min_tail_size, int(math.ceil(n ** self.auto_alpha)))
        k_auto = min(k_auto, max(1, n // 2 - 1))
        q_low_auto = k_auto / n
        q_up_auto = 1.0 - q_low_auto
        upper = (
            q_up_auto if self._tail_upper_q_spec == "auto" else self._tail_upper_q_spec
        )
        lower = (
            q_low_auto if self._tail_lower_q_spec == "auto" else self._tail_lower_q_spec
        )
        return float(upper), float(lower)

    # ------------------------------------------------------------------

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # Body buffers are concatenated per-feature and only get their final
        # shape/dtype during fit() (zero-length float32 -> length-N float64).
        # Resize them to match the checkpoint before PyTorch does its
        # default in-place copy_, otherwise load_state_dict raises a shape
        # mismatch.
        for name in ("sorted_unique", "q_low", "q_high"):
            key = prefix + name
            if key in state_dict:
                target = state_dict[key]
                current = self._buffers.get(name)
                if current is None:
                    continue
                if current.shape != target.shape or current.dtype != target.dtype:
                    self._buffers[name] = torch.empty(
                        target.shape, dtype=target.dtype, device=current.device
                    )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def fit(self, X: torch.Tensor) -> "RandomizedQuantileGPD":
        """Fit the empirical-CDF body and per-feature GPD tails.

        Per-feature state is stored in a *flat* layout so it can live on a
        single device buffer despite each feature having a different number
        of unique training values:

            ``sorted_unique[offsets[j] : offsets[j+1]]`` — sorted unique
            values for feature ``j``, with companion ``q_low`` / ``q_high``
            arrays giving each value's rank-interval bounds.

        This avoids ragged tensors / per-feature ``ParameterList`` and keeps
        the hot path indexable with simple slice arithmetic.

        Args:
            X: ``[N, d]`` training sample.

        Returns:
            ``self``.

        Raises:
            ValueError: if ``X.shape[-1] != self.d``.
        """
        if X.shape[-1] != self.d:
            raise ValueError(f"Expected d={self.d}, got X with last dim {X.shape[-1]}")
        X_np = X.detach().cpu().numpy().astype(np.float64)

        # Resolve "auto" tail thresholds from sample count.
        n_total = X_np.shape[0]
        q_up, q_lo_thresh = self._resolve_tail_quantiles(n_total)
        self.tail_upper_q.fill_(q_up)
        self.tail_lower_q.fill_(q_lo_thresh)

        # Variable-length per-feature buffers are accumulated as Python
        # lists of numpy arrays, then concatenated to a flat tensor at the
        # end (offsets[j] points at where feature j's slice starts).
        sorted_unique_list, q_low_list, q_high_list = [], [], []
        offsets = [0]
        upper_excess_per_feat: list[np.ndarray] = []
        lower_excess_per_feat: list[np.ndarray] = []

        for j in range(self.d):
            col = X_np[:, j]
            col = col[np.isfinite(col)]
            if col.size < 2:
                v = float(col.mean()) if col.size else 0.0
                sorted_unique_list.append(np.array([v], dtype=np.float64))
                q_low_list.append(np.array([0.5], dtype=np.float64))
                q_high_list.append(np.array([0.5], dtype=np.float64))
                offsets.append(offsets[-1] + 1)
                upper_excess_per_feat.append(np.empty(0, dtype=np.float64))
                lower_excess_per_feat.append(np.empty(0, dtype=np.float64))
                self.has_upper[j] = False
                self.has_lower[j] = False
                continue

            n = col.size
            sorted_col = np.sort(col)
            unique, first_idx, counts = np.unique(
                sorted_col, return_index=True, return_counts=True
            )
            q_lo = (first_idx + 0.5) / n
            q_hi = (first_idx + counts - 1 + 0.5) / n
            sorted_unique_list.append(unique)
            q_low_list.append(q_lo)
            q_high_list.append(q_hi)
            offsets.append(offsets[-1] + unique.size)

            u_hi = float(np.quantile(sorted_col, q_up))
            u_lo = float(np.quantile(sorted_col, q_lo_thresh))
            self.u_upper[j] = u_hi
            self.u_lower[j] = u_lo

            upper_excess_per_feat.append(sorted_col[sorted_col > u_hi] - u_hi)
            lower_excess_per_feat.append(u_lo - sorted_col[sorted_col < u_lo])

        device = X.device
        self.sorted_unique = torch.from_numpy(
            np.concatenate(sorted_unique_list)
        ).to(device=device, dtype=torch.float64)
        self.q_low = torch.from_numpy(np.concatenate(q_low_list)).to(
            device=device, dtype=torch.float64
        )
        self.q_high = torch.from_numpy(np.concatenate(q_high_list)).to(
            device=device, dtype=torch.float64
        )
        self.offsets = torch.tensor(offsets, dtype=torch.long, device=device)

        # Tail fits: choose path.
        if self.gpd_method == "w2":
            self._fit_gpd_tails_w2_batched(upper_excess_per_feat, upper=True, device=device)
            self._fit_gpd_tails_w2_batched(lower_excess_per_feat, upper=False, device=device)
        else:  # "mle"
            for j in range(self.d):
                if upper_excess_per_feat[j].size == 0 and lower_excess_per_feat[j].size == 0:
                    continue
                self.has_upper[j] = self._fit_gpd_tail(upper_excess_per_feat[j], j, upper=True)
                self.has_lower[j] = self._fit_gpd_tail(lower_excess_per_feat[j], j, upper=False)
        return self

    def _fit_gpd_tails_w2_batched(
        self,
        excess_per_feat: list[np.ndarray],
        upper: bool,
        device: torch.device,
    ) -> None:
        """Vectorized W2-based GPD fit for one tail across all eligible features.

        Eligibility: a feature is included only when it has at least
        ``self.min_tail_size`` exceedances on this side. The eligible
        features are right-padded into a single ``[len(eligible), max_M]``
        excess matrix with a companion mask, so a single batched Adam run
        in ``_fit_gpd_w2_batched`` fits all of them simultaneously.

        Args:
            excess_per_feat: One numpy array per feature of non-negative
                exceedances.
            upper:           Which tail this is — selects which buffers
                (``xi_upper``/``has_upper`` vs ``xi_lower``/``has_lower``)
                receive the fitted parameters.
            device:          Device to run the fit on.
        """
        eligible = [j for j, e in enumerate(excess_per_feat) if e.size >= self.min_tail_size]

        # Mark fit/non-fit for every feature.
        has_flag = self.has_upper if upper else self.has_lower
        xi_buf = self.xi_upper if upper else self.xi_lower
        sigma_buf = self.sigma_upper if upper else self.sigma_lower
        for j in range(self.d):
            has_flag[j] = j in set(eligible)

        if not eligible:
            return

        max_M = max(excess_per_feat[j].size for j in eligible)
        excess_pad = np.zeros((len(eligible), max_M), dtype=np.float64)
        mask_pad = np.zeros((len(eligible), max_M), dtype=bool)
        for i, j in enumerate(eligible):
            e = excess_per_feat[j]
            excess_pad[i, : e.size] = e
            mask_pad[i, : e.size] = True

        excess_t = torch.from_numpy(excess_pad).to(device=device)
        mask_t = torch.from_numpy(mask_pad).to(device=device)

        xi_batch, sigma_batch = _fit_gpd_w2_batched(
            excess_t, mask_t, n_steps=self.gpd_w2_steps
        )
        for i, j in enumerate(eligible):
            xi_buf[j] = float(xi_batch[i].item())
            sigma_buf[j] = float(sigma_batch[i].item())

    def _fit_gpd_tail(self, excess: np.ndarray, j: int, upper: bool) -> bool:
        """Fit GPD on positive ``excess`` values via scipy MLE.

        Args:
            excess: Non-negative exceedances for one tail of one feature.
            j:      Feature index (writes into the per-feature buffer slot).
            upper:  ``True`` for the upper tail, ``False`` for the lower.

        Returns:
            ``True`` if a finite, positive-sigma fit was obtained and
            written; ``False`` otherwise (caller treats as fallback to
            clipped empirical extrapolation).
        """
        if excess.size < self.min_tail_size:
            return False
        try:
            xi, _, sig = scipy.stats.genpareto.fit(excess, floc=0.0)
        except Exception:  # noqa: BLE001
            return False
        if not (np.isfinite(xi) and np.isfinite(sig) and sig > 0):
            return False
        if upper:
            self.xi_upper[j] = float(xi)
            self.sigma_upper[j] = float(sig)
        else:
            self.xi_lower[j] = float(xi)
            self.sigma_lower[j] = float(sig)
        return True

    # ------------------------------------------------------------------

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply the empirical-CDF + GPD-tail PIT, returning standard-normal scores.

        Args:
            X: ``[N, d]`` inputs in the original scale.

        Returns:
            ``[N, d]`` standard-normal scores (``Φ⁻¹`` of the PIT outputs).
        """
        return self._apply_pit(X, inverse=False)

    def is_stochastic_in_train(self) -> bool:
        """Train-mode forward draws a fresh per-call uniform on tied rows when
        ``randomize_ties`` is True (acts as data augmentation)."""
        return self.randomize_ties

    def inverse(self, Z: torch.Tensor) -> torch.Tensor:
        """Invert the PIT: standard-normal scores back to original-scale values.

        Args:
            Z: ``[N, d]`` standard-normal scores.

        Returns:
            ``[N, d]`` reconstructed inputs.
        """
        return self._apply_pit(Z, inverse=True)

    def _apply_pit(self, A: torch.Tensor, inverse: bool) -> torch.Tensor:
        """Pure-torch hot path; per-feature dispatch over flat-stored state.

        Stays on ``A.device``; promotes to float64 internally for
        empirical-CDF / inverse-Φ stability and casts back to the input
        dtype on return. Per-feature scalar parameters that don't change
        across features (``tail_upper_q`` / ``tail_lower_q``) are extracted
        *once* up front rather than per call into the inner loop, and the
        offsets buffer is converted to a Python list once to avoid repeated
        GPU→CPU syncs inside the loop.

        Randomization rule: ties in the body are randomized only on a
        forward pass that runs while ``self.training`` is True (so that
        ``predict`` and ``inverse`` are deterministic).

        Args:
            A:       Inputs ``[N, d]`` in original scale (forward) or
                     standard-normal scores (inverse).
            inverse: Whether to apply the inverse PIT.

        Returns:
            ``[N, d]`` transformed tensor with ``A``'s original dtype.
        """
        in_dtype = A.dtype
        device = A.device
        A64 = A.detach().to(dtype=torch.float64)
        N = A64.shape[0]
        out = torch.empty((N, self.d), dtype=torch.float64, device=device)

        randomize = self.randomize_ties and self.training and not inverse

        # Hoist feature-invariant scalars out of the inner loop. ``offsets``
        # gets one CPU sync (.tolist) instead of 2*d (.item per index).
        offs = self.offsets.tolist()
        tail_upper_q = float(self.tail_upper_q.item())
        tail_lower_q = float(self.tail_lower_q.item())

        for j in range(self.d):
            lo, hi = offs[j], offs[j + 1]
            uniq = self.sorted_unique[lo:hi]
            qlo = self.q_low[lo:hi]
            qhi = self.q_high[lo:hi]
            col = A64[:, j].contiguous()
            if inverse:
                out[:, j] = self._inverse_one(
                    col, j, uniq, qlo, qhi, tail_upper_q, tail_lower_q
                )
            else:
                out[:, j] = self._forward_one(
                    col, j, uniq, qlo, qhi, randomize, tail_upper_q, tail_lower_q
                )
        return out.to(dtype=in_dtype)

    def _x_to_u(
        self,
        x: torch.Tensor,
        j: int,
        uniq: torch.Tensor,
        qlo: torch.Tensor,
        qhi: torch.Tensor,
        randomize: bool,
        tail_upper_q: float,
        tail_lower_q: float,
    ) -> torch.Tensor:
        """PIT body+tail for one feature column, returning ``u`` in ``[0, 1]``.

        Three regions are handled independently:

        * **Body** (between the upper / lower thresholds): empirical CDF
          via linear interpolation on the ``(unique_value, qmid)`` curve,
          with tied training values patched to a random (or midpoint) draw
          within their rank interval ``[qlo, qhi]``.
        * **Lower tail** (``x < u_lower``): closed-form GPD CDF mapped
          onto ``[0, tail_lower_q]``.
        * **Upper tail** (``x > u_upper``): closed-form GPD CDF mapped
          onto ``[tail_upper_q, 1]``.

        Stops just before the optional bin-snap and ``Φ⁻¹`` so that
        :meth:`bin_indices` and :meth:`_forward_one` both reuse it without
        duplicating the body/tail logic.

        Args:
            x:            ``[N]`` feature column (float64).
            j:            Feature index (used to read per-feature GPD params).
            uniq:         Sorted unique training values for this feature.
            qlo, qhi:     Hazen-style rank intervals for each unique value
                          (``qlo == qhi`` for unique values, strict for ties).
            randomize:    Whether to draw uniform-in-interval for ties (else
                          midpoint).
            tail_upper_q: Pre-computed ``self.tail_upper_q`` scalar.
            tail_lower_q: Pre-computed ``self.tail_lower_q`` scalar.

        Returns:
            ``[N]`` PIT output ``u`` in ``[0, 1]`` (un-clamped, un-snapped).
        """
        qmid = 0.5 * (qlo + qhi)
        u_hi = float(self.u_upper[j].item())
        u_lo = float(self.u_lower[j].item())

        # Body via linear interp on (uniq, qmid). Ties are then patched up.
        if uniq.shape[0] >= 2:
            u = _interp_torch(x, uniq, qmid)
        elif uniq.shape[0] == 1:
            u = qmid[0].expand_as(x).clone()
        else:
            u = torch.full_like(x, 0.5)

        # Tied-value patch. Each tied atom k owns a rank interval
        # [qlo[k], qhi[k]]. For x values that exactly match a tied atom we
        # replace u with either a uniform draw within that interval
        # (training-mode randomized PIT) or the interval midpoint
        # (deterministic PIT). The match is vectorized via searchsorted on
        # the tied-atom subset — much cheaper than an O(K) Python loop.
        tie_mask = qhi > qlo + 1e-15
        if tie_mask.any():
            tie_uniq = uniq[tie_mask]
            tie_qlo = qlo[tie_mask]
            tie_qhi = qhi[tie_mask]
            K = tie_uniq.shape[0]
            idx = torch.searchsorted(tie_uniq, x).clamp_max(K - 1)
            cand = tie_uniq[idx]
            matches = cand == x
            if matches.any():
                m_qlo = tie_qlo[idx]
                m_qhi = tie_qhi[idx]
                if randomize:
                    v = torch.rand(x.shape, dtype=x.dtype, device=x.device)
                    new_u = m_qlo + v * (m_qhi - m_qlo)
                else:
                    new_u = 0.5 * (m_qlo + m_qhi)
                u = torch.where(matches, new_u, u)

        # Tails: GPD CDF on the exceedance, mapped to the tail probability range.
        # Since `u` may be the result of `torch.where`/`expand_as` above, force
        # one fresh allocation here — then we can write in place for both tails.
        below = x < u_lo
        above = x > u_hi
        if below.any() or above.any():
            u = u.clone()
        if below.any():
            if bool(self.has_lower[j].item()):
                xi = float(self.xi_lower[j].item())
                sig = float(self.sigma_lower[j].item())
                excess = u_lo - x[below]
                F = _gpd_cdf_scalar(excess, xi, sig)
                u[below] = tail_lower_q * (1.0 - F)
            else:
                # Fallback when GPD fit was not possible: saturate at the
                # lowest body-rank value (clipped empirical extrapolation).
                u[below] = float(qlo[0].item()) if qlo.shape[0] > 0 else 0.0
        if above.any():
            if bool(self.has_upper[j].item()):
                xi = float(self.xi_upper[j].item())
                sig = float(self.sigma_upper[j].item())
                excess = x[above] - u_hi
                F = _gpd_cdf_scalar(excess, xi, sig)
                u[above] = tail_upper_q + (1.0 - tail_upper_q) * F
            else:
                u[above] = float(qhi[-1].item()) if qhi.shape[0] > 0 else 1.0
        return u

    def _forward_one(
        self,
        x: torch.Tensor,
        j: int,
        uniq: torch.Tensor,
        qlo: torch.Tensor,
        qhi: torch.Tensor,
        randomize: bool,
        tail_upper_q: float,
        tail_lower_q: float,
    ) -> torch.Tensor:
        """Forward PIT for a single feature column → standard-normal scores.

        Computes ``u = _x_to_u(x)`` (body + tied-atom + GPD tails),
        optionally snaps ``u`` to the nearest equiprobable bin midpoint
        when ``self.n_bins`` is set, then maps to a standard-normal score
        via ``Φ⁻¹``. After the snap, ``u`` already lies strictly inside
        ``(0, 1)``, but the clamp is kept as a no-op safety net for the
        continuous (un-snapped) path.

        Args:
            x:            ``[N]`` feature column (float64).
            j:            Feature index.
            uniq:         Sorted unique training values.
            qlo, qhi:     Hazen rank intervals.
            randomize:    Whether to randomize ties (training mode).
            tail_upper_q: Pre-computed ``self.tail_upper_q``.
            tail_lower_q: Pre-computed ``self.tail_lower_q``.

        Returns:
            ``[N]`` standard-normal scores.
        """
        u = self._x_to_u(
            x, j, uniq, qlo, qhi, randomize, tail_upper_q, tail_lower_q
        )
        if self.n_bins is not None:
            u = _body_snap_u_to_midpoint(
                u, self.n_bins, tail_lower_q, tail_upper_q
            )
        # Φ⁻¹ blows up at exactly 0 / 1, so clamp just inside.
        u = u.clamp(_EPS, 1.0 - _EPS)
        return torch.special.ndtri(u)

    def _inverse_one(
        self,
        z: torch.Tensor,
        j: int,
        uniq: torch.Tensor,
        qlo: torch.Tensor,
        qhi: torch.Tensor,
        tail_upper_q: float,
        tail_lower_q: float,
    ) -> torch.Tensor:
        """Inverse PIT for a single feature column.

        Mirrors ``_forward_one``: maps ``z`` through ``Φ`` to a uniform
        ``u``, then inverts each region (body / lower tail / upper tail)
        using the same model as the forward direction.

        Round-trip note: in the body, the forward map is many-to-one on
        tied training atoms (a uniform interval ``[qlo, qhi]`` collapses
        to one anchor ``uniq[k]``). The inverse here detects when ``u``
        falls inside any tied interval and snaps the result to the
        corresponding ``uniq[k]`` so that ``forward(x) → inverse(z) → x``
        round-trips exactly for tied training values.

        Args:
            z:            ``[N]`` standard-normal scores.
            j:            Feature index.
            uniq:         Sorted unique training values.
            qlo, qhi:     Hazen rank intervals for each unique value.
            tail_upper_q: Pre-computed ``self.tail_upper_q`` scalar.
            tail_lower_q: Pre-computed ``self.tail_lower_q`` scalar.

        Returns:
            ``[N]`` reconstructed values in original scale.
        """
        qmid = 0.5 * (qlo + qhi)
        u = torch.special.ndtr(z)
        if self.n_bins is not None:
            u = _body_snap_u_to_midpoint(
                u, self.n_bins, tail_lower_q, tail_upper_q
            )

        below = u < tail_lower_q
        above = u > tail_upper_q
        body = ~(below | above)

        # Body inverse: linear interp on (qmid, uniq); the tied-atom snap
        # below makes the round-trip exact.
        if uniq.shape[0] >= 2:
            x_body = _interp_torch(u, qmid, uniq)
        elif uniq.shape[0] == 1:
            x_body = uniq[0].expand_as(u).clone()
        else:
            x_body = torch.zeros_like(u)
        tie_mask = qhi > qlo + 1e-15
        if tie_mask.any():
            tie_uniq = uniq[tie_mask]
            tie_qlo = qlo[tie_mask]
            tie_qhi = qhi[tie_mask]
            # For each u, find the rightmost tied interval whose qlo <= u,
            # then snap u back to that atom only if u also <= qhi (so we
            # don't accidentally snap a value from the body between ties).
            K = tie_uniq.shape[0]
            idx = torch.searchsorted(tie_qlo, u, right=True).clamp(1, K) - 1
            in_tie = (u >= tie_qlo[idx]) & (u <= tie_qhi[idx])
            x_body = torch.where(in_tie, tie_uniq[idx], x_body)

        x = torch.empty_like(u)
        x[body] = x_body[body]

        if below.any():
            if bool(self.has_lower[j].item()):
                xi = float(self.xi_lower[j].item())
                sig = float(self.sigma_lower[j].item())
                # Re-scale u from [0, tail_lower_q] to GPD argument in (0, 1).
                arg = (1.0 - u[below] / tail_lower_q).clamp(_EPS, 1.0 - _EPS)
                excess = _gpd_ppf_scalar(arg, xi, sig)
                x[below] = float(self.u_lower[j].item()) - excess
            else:
                x[below] = float(uniq[0].item()) if uniq.shape[0] > 0 else 0.0
        if above.any():
            if bool(self.has_upper[j].item()):
                xi = float(self.xi_upper[j].item())
                sig = float(self.sigma_upper[j].item())
                arg = (
                    (u[above] - tail_upper_q) / (1.0 - tail_upper_q)
                ).clamp(_EPS, 1.0 - _EPS)
                excess = _gpd_ppf_scalar(arg, xi, sig)
                x[above] = float(self.u_upper[j].item()) + excess
            else:
                x[above] = float(uniq[-1].item()) if uniq.shape[0] > 0 else 0.0
        return x

    # ------------------------------------------------------------------
    # Binned-mode auxiliary API
    # ------------------------------------------------------------------

    def bin_indices(self, X: torch.Tensor) -> torch.Tensor:
        """Per-feature body-equiprobable bin index in ``[0, n_bins - 1]``.

        Bins partition the **body interval** ``[q_downarrow, q_uparrow]``
        of ``u``-space into ``K = n_bins`` equal-width sub-intervals;
        body samples land in their owning bin, tail samples fold into
        the boundary class (``0`` for ``u < q_downarrow``, ``K - 1`` for
        ``u > q_uparrow``). This is the **Option A** label policy.

        Use :meth:`is_body` if you want to distinguish tail samples from
        body samples that genuinely live in the boundary bins (Option C).

        Under the training distribution each body bin holds
        ``(q_uparrow - q_downarrow) / K`` of the mass, plus the boundary
        bins absorb the corresponding tail mass — so every bin is
        balanced modulo Binomial sampling noise.

        Tied-atom handling mirrors :meth:`forward`: in training mode each
        tied sample draws uniformly within its rank interval (so labels
        are balanced across the bins the interval covers); in eval mode
        all ties collapse to the interval's midpoint bin.

        Args:
            X: ``[N, d]`` inputs in original scale.

        Returns:
            ``[N, d]`` ``torch.long`` bin indices.

        Raises:
            RuntimeError: if ``n_bins`` is not set on this transform.
            ValueError:   if ``X.shape[-1] != self.d``.
        """
        if self.n_bins is None:
            raise RuntimeError(
                "bin_indices requires n_bins to be set on this transform."
            )
        if X.shape[-1] != self.d:
            raise ValueError(
                f"Expected d={self.d}, got X with last dim {X.shape[-1]}"
            )
        device = X.device
        A64 = X.detach().to(dtype=torch.float64)
        N = A64.shape[0]
        out = torch.empty((N, self.d), dtype=torch.long, device=device)

        offs = self.offsets.tolist()
        tail_upper_q = float(self.tail_upper_q.item())
        tail_lower_q = float(self.tail_lower_q.item())
        K = self.n_bins
        randomize = self.randomize_ties and self.training

        for j in range(self.d):
            lo, hi = offs[j], offs[j + 1]
            uniq = self.sorted_unique[lo:hi]
            qlo = self.q_low[lo:hi]
            qhi = self.q_high[lo:hi]
            col = A64[:, j].contiguous()
            u = self._x_to_u(
                col, j, uniq, qlo, qhi,
                randomize=randomize,
                tail_upper_q=tail_upper_q,
                tail_lower_q=tail_lower_q,
            )
            out[:, j] = _body_bin_index(u, K, tail_lower_q, tail_upper_q)
        return out

    def is_body(self, X: torch.Tensor) -> torch.Tensor:
        """Per-feature mask: ``True`` where the sample lies in the body region.

        A sample is "body" iff its PIT ``u = F(x)`` lies in
        ``[q_downarrow, q_uparrow]`` — i.e., it does **not** trigger the
        GPD tail extrapolation in :meth:`forward`. Tail samples carry an
        extreme :meth:`bin_indices` label (``0`` or ``K - 1``) under the
        Option A policy and are distinguishable from genuine body
        boundary-bin samples only via this mask. Use it to filter
        training data for a classification head when you don't want
        tail samples conflated with body extremes.

        Tied-atom randomization mirrors :meth:`forward` (training mode
        randomizes within rank intervals; eval mode is deterministic).

        Args:
            X: ``[N, d]`` inputs in original scale.

        Returns:
            ``[N, d]`` ``torch.bool`` body mask.

        Raises:
            ValueError: if ``X.shape[-1] != self.d``.
        """
        if X.shape[-1] != self.d:
            raise ValueError(
                f"Expected d={self.d}, got X with last dim {X.shape[-1]}"
            )
        device = X.device
        A64 = X.detach().to(dtype=torch.float64)
        N = A64.shape[0]
        out = torch.empty((N, self.d), dtype=torch.bool, device=device)

        offs = self.offsets.tolist()
        tail_upper_q = float(self.tail_upper_q.item())
        tail_lower_q = float(self.tail_lower_q.item())
        randomize = self.randomize_ties and self.training

        for j in range(self.d):
            lo, hi = offs[j], offs[j + 1]
            uniq = self.sorted_unique[lo:hi]
            qlo = self.q_low[lo:hi]
            qhi = self.q_high[lo:hi]
            col = A64[:, j].contiguous()
            u = self._x_to_u(
                col, j, uniq, qlo, qhi,
                randomize=randomize,
                tail_upper_q=tail_upper_q,
                tail_lower_q=tail_lower_q,
            )
            out[:, j] = (u >= tail_lower_q) & (u <= tail_upper_q)
        return out

    def from_bin_indices(self, idx: torch.Tensor) -> torch.Tensor:
        """Convert per-feature body-bin indices to original-scale values.

        Each bin maps to the original-scale value at its body-midpoint
        ``u_k* = q_downarrow + (k + 0.5) / K * (q_uparrow - q_downarrow)``,
        which is **always inside the body** by construction — so this
        routes through body interpolation, never GPD-inverse. The
        outermost bins (``k = 0`` and ``k = K - 1``) therefore represent
        body extremes, *not* tail-extrapolated values.
        ``from_bin_indices(bin_indices(X))`` round-trips to one of K
        representative body-scale values per feature.

        Args:
            idx: ``[N, d]`` integer bin indices in ``[0, n_bins - 1]``.

        Returns:
            ``[N, d]`` reconstructed values in original scale (float64).

        Raises:
            RuntimeError: if ``n_bins`` is not set.
            ValueError:   if ``idx.shape[-1] != self.d``.
        """
        if self.n_bins is None:
            raise RuntimeError(
                "from_bin_indices requires n_bins to be set on this transform."
            )
        if idx.shape[-1] != self.d:
            raise ValueError(
                f"Expected d={self.d}, got idx with last dim {idx.shape[-1]}"
            )
        K = self.n_bins
        q_lo = float(self.tail_lower_q.item())
        q_hi = float(self.tail_upper_q.item())
        width = max(q_hi - q_lo, _EPS)
        u_mid = q_lo + (idx.to(dtype=torch.float64) + 0.5) / K * width
        z = torch.special.ndtri(u_mid.clamp(_EPS, 1.0 - _EPS))
        return self.inverse(z)


# ----------------------------------------------------------------------
# KDE-smoothed quantile transform
# ----------------------------------------------------------------------


def _interp_extrap_torch(
    x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor
) -> torch.Tensor:
    """1-D piecewise-linear interpolation with linear extrapolation.

    Same as :func:`_interp_torch` but does not clamp at the grid boundaries —
    queries outside ``[xp[0], xp[-1]]`` are extrapolated using the slope of
    the nearest interior segment, so the mapping stays monotonic and
    unbounded. Used by :class:`KDEQuantile` so extreme inputs receive
    distinct z values instead of saturating at the grid edge.
    """
    M = xp.shape[0]
    if M == 0:
        return torch.zeros_like(x)
    if M == 1:
        return fp[0].expand_as(x).clone()
    x_search = x.contiguous()
    idx = torch.searchsorted(xp, x_search).clamp(1, M - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]
    denom = (x1 - x0).clamp_min(_EPS)
    return y0 + (x - x0) / denom * (y1 - y0)



class KDEQuantile(_Transform):
    """KDE-smoothed quantile transform with GPD tails.

    Per-feature map ``T(x) = Φ⁻¹(F̂(x))`` where ``F̂`` is built piecewise:

    * **Body** (``x ∈ [x_↓, x_↑]``): Gaussian-kernel-smoothed CDF
      ``F_kde(x; h) = (1/N) Σ Φ((x − X_i) / h)``, rescaled to span exactly
      ``[q_↓, q_↑]`` so it joins the tails continuously.
    * **Upper tail** (``x > x_↑``): GPD extrapolation
      ``F̂(x) = q_↑ + (1 − q_↑) · G(x − x_↑; ξ_↑, σ_↑)``.
    * **Lower tail** (``x < x_↓``): mirror of the upper case.

    Compared to :class:`RandomizedQuantileGPD`:

    * The body is **smoothed**, so geometric gaps in the input survive
      preprocessing — empirical CDF makes the body flat in any zero-mass
      region and ``Φ⁻¹`` collapses it to a point.
    * The tails are **identical** (GPD extrapolation, Pickands-Balkema-de Haan
      universal limit), so extreme out-of-sample inputs receive distinct,
      ordered z values rather than saturating at the body boundary.
    * Atomic spikes are handled implicitly: the kernel smooths a tied value
      into a narrow Gaussian peak in z-space rather than collapsing it.
      RQT-GPD's randomized PIT is exact for atomic data; KDE-GPD trades a
      hair of marginal accuracy for continuous, deterministic z's.

    Bandwidth control
    -----------------
    The bandwidth is ``h = 1.06 · std · N^(−1/5) · bandwidth_factor``
    (Silverman's rule scaled by a user-supplied multiplier). Larger values
    preserve geometric gaps between clusters; smaller values pull the marginal
    closer to N(0,1). Default ``1.0`` is plain Silverman.

    Speed
    -----
    Fit cost is dominated by the per-feature ``O(N · n_grid)`` KDE-CDF
    evaluation plus the same batched GPD W2 fit RQT-GPD uses, so it sits
    within ~2x of RQT-GPD's fit time at typical sizes (N=4000, d=1, CPU:
    ~25 ms vs ~12 ms). Forward / inverse are ``O(log n_grid)`` per query,
    same complexity class as RQT-GPD's body interpolation.

    Args:
        d:                Number of features.
        bandwidth_factor: Multiplier on Silverman's rule (positive float).
                          Default ``1.0`` (plain Silverman). Larger values
                          widen kernels for stronger gap preservation;
                          smaller values tighten toward exact N(0,1) marginals.
        n_grid:           Body x-grid size per feature for the KDE-CDF
                          lookup. Default 1024.
        tail_upper_q:     Upper-tail threshold quantile or ``"auto"``
                          (Hall scaling, same as RQT-GPD).
        tail_lower_q:     Lower-tail threshold quantile or ``"auto"``.
        min_tail_size:    Floor on exceedance count; below this, that side
                          falls back to clipped body extrapolation.
        auto_alpha:       Hall exponent ``k = ⌈N^α⌉`` for ``"auto"``.
        gpd_w2_steps:     Adam steps for the batched GPD-W2 fit.
        n_bins:           Optional. If set to ``K >= 2``, the body PIT
                          output ``u`` is snapped to the nearest of K
                          equiprobable bin midpoints inside
                          ``[q_lo[j], q_up[j]]`` (per feature); tail
                          values fold into the boundary bins ``0`` /
                          ``K - 1``. Unlike RQT-GPD, the snap is fully
                          deterministic — no per-batch randomization —
                          because the body PIT is already built from a
                          fixed ``(x_grid, u_grid)`` lookup. ``None``
                          (default) leaves the transform continuous.
                          When set, :meth:`bin_indices` /
                          :meth:`from_bin_indices` become available.
    """

    def __init__(
        self,
        d: int,
        bandwidth_factor: float = 1.0,
        n_grid: int = 1024,
        tail_upper_q: float | str = "auto",
        tail_lower_q: float | str = "auto",
        min_tail_size: int = 30,
        auto_alpha: float = 2.0 / 3.0,
        gpd_w2_steps: int = 100,
        n_bins: int | None = None,
    ) -> None:
        super().__init__()
        if bandwidth_factor <= 0:
            raise ValueError(
                f"bandwidth_factor must be > 0, got {bandwidth_factor}"
            )
        if n_grid < 16:
            raise ValueError(f"n_grid must be >= 16, got {n_grid}")
        if min_tail_size < 5:
            raise ValueError(f"min_tail_size must be >= 5, got {min_tail_size}")
        if not 0.0 < auto_alpha < 1.0:
            raise ValueError(f"auto_alpha must be in (0, 1), got {auto_alpha}")
        if n_bins is not None and (not isinstance(n_bins, int) or n_bins < 2):
            raise ValueError(f"n_bins must be an int >= 2 or None, got {n_bins!r}")

        self.d = d
        self.bandwidth_factor = bandwidth_factor
        self.n_grid = int(n_grid)
        self._tail_upper_q_spec = RandomizedQuantileGPD._validate_tail_q(
            tail_upper_q, name="tail_upper_q", upper=True
        )
        self._tail_lower_q_spec = RandomizedQuantileGPD._validate_tail_q(
            tail_lower_q, name="tail_lower_q", upper=False
        )
        self.min_tail_size = int(min_tail_size)
        self.auto_alpha = float(auto_alpha)
        self.gpd_w2_steps = int(gpd_w2_steps)
        self.n_bins = n_bins

        # Per-feature body lookup tables (x → u, u in [q_lo, q_up]).
        self.register_buffer("x_body_grid", torch.zeros(d, n_grid))
        self.register_buffer("u_body_grid", torch.zeros(d, n_grid))
        # Per-feature body / tail thresholds.
        self.register_buffer("q_lo", torch.zeros(d, dtype=torch.float64))
        self.register_buffer("q_up", torch.ones(d, dtype=torch.float64))
        self.register_buffer("x_lo", torch.zeros(d))
        self.register_buffer("x_up", torch.zeros(d))
        # Per-feature bandwidth (Silverman * factor * mode shrink).
        self.register_buffer("h", torch.ones(d))
        # GPD parameters per feature, mirroring RQT-GPD.
        self.register_buffer("xi_upper", torch.zeros(d))
        self.register_buffer("sigma_upper", torch.ones(d))
        self.register_buffer("xi_lower", torch.zeros(d))
        self.register_buffer("sigma_lower", torch.ones(d))
        self.register_buffer("has_upper", torch.zeros(d, dtype=torch.bool))
        self.register_buffer("has_lower", torch.zeros(d, dtype=torch.bool))

    def _resolve_tail_quantiles(self, n: int) -> tuple[float, float]:
        """Same Hall-style auto rule as :class:`RandomizedQuantileGPD`."""
        k_auto = max(self.min_tail_size, int(math.ceil(n ** self.auto_alpha)))
        k_auto = min(k_auto, max(1, n // 2 - 1))
        q_low_auto = k_auto / n
        q_up_auto = 1.0 - q_low_auto
        upper = (
            q_up_auto if self._tail_upper_q_spec == "auto" else self._tail_upper_q_spec
        )
        lower = (
            q_low_auto if self._tail_lower_q_spec == "auto" else self._tail_lower_q_spec
        )
        return float(upper), float(lower)

    def _resolve_bandwidth(self, samples: np.ndarray, n: int) -> float:
        """Silverman's bandwidth × bandwidth_factor."""
        std = float(np.std(samples))
        h_silv = max(1.06 * std * n ** (-1.0 / 5.0), 1e-8)
        return h_silv * self.bandwidth_factor

    def fit(self, X: torch.Tensor) -> "KDEQuantile":
        """Fit per-feature bandwidth, KDE-CDF body grid, and GPD tails.

        Steps per feature:

        1. Pick bandwidth ``h_j`` (Silverman × ``bandwidth_factor``).
        2. Resolve tail quantiles ``q_↑, q_↓`` and corresponding x-thresholds.
        3. Build a uniform x-grid over ``[x_↓, x_↑]`` of size ``n_grid``,
           evaluate ``F_kde`` on it, and rescale linearly so the grid u
           values span exactly ``[q_↓, q_↑]``. The rescale closes the small
           gap between the empirical-quantile-defined body endpoints and
           the KDE-CDF values there, so body and tail join continuously.
        4. Collect exceedances, fit GPDs across all features in a single
           batched Adam W2² run (reusing :func:`_fit_gpd_w2_batched`).
        """
        X_np = np.asarray(X.detach().cpu().numpy(), dtype=np.float64)
        n, d = X_np.shape
        if d != self.d:
            raise ValueError(f"Expected d={self.d}, got X.shape[1]={d}")
        if n < max(2 * self.min_tail_size, 4):
            raise ValueError(
                f"Need at least {max(2 * self.min_tail_size, 4)} samples, got n={n}"
            )

        q_up, q_lo = self._resolve_tail_quantiles(n)
        # Per-feature x-thresholds via empirical quantiles (same as RQT-GPD).
        x_up_arr = np.quantile(X_np, q_up, axis=0)
        x_lo_arr = np.quantile(X_np, q_lo, axis=0)

        h_arr = np.empty(d, dtype=np.float64)
        x_body_grid = np.empty((d, self.n_grid), dtype=np.float64)
        u_body_grid = np.empty((d, self.n_grid), dtype=np.float64)

        # Build per-feature body grids and KDE-CDF lookups.
        for j in range(d):
            col = X_np[:, j]
            h_j = self._resolve_bandwidth(col, n)
            h_arr[j] = h_j

            x_lo_j = float(x_lo_arr[j])
            x_up_j = float(x_up_arr[j])
            if not x_up_j > x_lo_j:
                # Pathological: feature is constant on the body. Widen by
                # bandwidth so we still produce a non-degenerate grid.
                x_up_j = x_lo_j + max(1e-6, h_j)
                x_up_arr[j] = x_up_j

            grid_j = np.linspace(x_lo_j, x_up_j, self.n_grid)
            # KDE-CDF eval is the dominant fit cost (a [n_grid, N] matrix
            # of normal CDFs per feature). torch.special.ndtr in float32 is
            # ~5x faster than scipy.special.ndtr in float64; the float32
            # precision is below the body-grid lookup error already absorbed
            # by the inference path (buffers are float32).
            grid_t = torch.from_numpy(grid_j).to(torch.float32)
            col_t = torch.from_numpy(col).to(torch.float32)
            diffs_t = (grid_t[:, None] - col_t[None, :]) / float(h_j)
            F_kde_grid = (
                torch.special.ndtr(diffs_t).mean(dim=1).to(torch.float64).numpy()
            )

            # Linearly rescale so the body grid u-values span exactly
            # [q_lo, q_up]. This closes the (small) discrepancy between the
            # KDE-CDF at the empirical-quantile thresholds and (q_lo, q_up).
            f_at_lo = float(F_kde_grid[0])
            f_at_up = float(F_kde_grid[-1])
            denom = max(f_at_up - f_at_lo, 1e-12)
            u_grid = q_lo + (F_kde_grid - f_at_lo) / denom * (q_up - q_lo)
            u_grid = np.clip(u_grid, q_lo, q_up)
            u_grid[0] = q_lo
            u_grid[-1] = q_up

            x_body_grid[j] = grid_j
            u_body_grid[j] = u_grid

        # Tail GPD fits, batched across features.
        max_excess = 0
        per_feat_upper: list[np.ndarray] = []
        per_feat_lower: list[np.ndarray] = []
        for j in range(d):
            col = X_np[:, j]
            up_e = col[col > x_up_arr[j]] - x_up_arr[j]
            lo_e = x_lo_arr[j] - col[col < x_lo_arr[j]]
            per_feat_upper.append(up_e)
            per_feat_lower.append(lo_e)
            max_excess = max(max_excess, up_e.size, lo_e.size)
        max_excess = max(max_excess, 1)

        # Helper to pad ragged exceedances into [d, M] with mask.
        def _pad(per_feat: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            buf = np.zeros((d, max_excess), dtype=np.float64)
            mask = np.zeros((d, max_excess), dtype=bool)
            for j, arr in enumerate(per_feat):
                if arr.size > 0:
                    buf[j, : arr.size] = arr
                    mask[j, : arr.size] = True
            return buf, mask

        up_buf, up_mask = _pad(per_feat_upper)
        lo_buf, lo_mask = _pad(per_feat_lower)

        device = self.h.device
        up_t = torch.from_numpy(up_buf).to(device)
        up_m = torch.from_numpy(up_mask).to(device)
        lo_t = torch.from_numpy(lo_buf).to(device)
        lo_m = torch.from_numpy(lo_mask).to(device)

        # Gate features with too few exceedances out of the fit (their GPD
        # params will stay at the default ``(0, 1)`` and ``has_*`` will be
        # False, so the forward path saturates them).
        has_upper = up_m.sum(dim=1) >= self.min_tail_size
        has_lower = lo_m.sum(dim=1) >= self.min_tail_size

        if has_upper.any():
            xi_u, sig_u = _fit_gpd_w2_batched(
                up_t, up_m & has_upper.unsqueeze(1), n_steps=self.gpd_w2_steps
            )
        else:
            xi_u = torch.zeros(d, device=device, dtype=up_t.dtype)
            sig_u = torch.ones(d, device=device, dtype=up_t.dtype)
        if has_lower.any():
            xi_l, sig_l = _fit_gpd_w2_batched(
                lo_t, lo_m & has_lower.unsqueeze(1), n_steps=self.gpd_w2_steps
            )
        else:
            xi_l = torch.zeros(d, device=device, dtype=lo_t.dtype)
            sig_l = torch.ones(d, device=device, dtype=lo_t.dtype)

        # Persist everything as registered buffers.
        self.h.copy_(torch.tensor(h_arr, dtype=self.h.dtype))
        self.x_body_grid.copy_(torch.tensor(x_body_grid, dtype=self.x_body_grid.dtype))
        self.u_body_grid.copy_(torch.tensor(u_body_grid, dtype=self.u_body_grid.dtype))
        self.q_lo.fill_(float(q_lo))
        self.q_up.fill_(float(q_up))
        self.x_lo.copy_(torch.tensor(x_lo_arr, dtype=self.x_lo.dtype))
        self.x_up.copy_(torch.tensor(x_up_arr, dtype=self.x_up.dtype))
        self.xi_upper.copy_(xi_u.to(dtype=self.xi_upper.dtype))
        self.sigma_upper.copy_(sig_u.to(dtype=self.sigma_upper.dtype))
        self.xi_lower.copy_(xi_l.to(dtype=self.xi_lower.dtype))
        self.sigma_lower.copy_(sig_l.to(dtype=self.sigma_lower.dtype))
        self.has_upper.copy_(has_upper)
        self.has_lower.copy_(has_lower)
        return self

    # ------------------------------------------------------------------
    # Forward / inverse: per-feature region routing.
    # ------------------------------------------------------------------

    def _kde_u_one(self, x_col: torch.Tensor, j: int) -> torch.Tensor:
        """Compute body+tail PIT ``u = F̂_j(x_col)`` without the optional
        bin snap or ``Φ⁻¹``.

        Shared by :meth:`_forward_one` and :meth:`bin_indices` so the
        body / GPD-tail routing lives in one place.
        """
        x_lo = float(self.x_lo[j].item())
        x_up = float(self.x_up[j].item())
        q_lo = float(self.q_lo[j].item())
        q_up = float(self.q_up[j].item())

        u_body = _interp_extrap_torch(
            x_col, self.x_body_grid[j], self.u_body_grid[j]
        )
        u = u_body.clamp(q_lo, q_up)

        if bool(self.has_upper[j].item()):
            mask_up = x_col > x_up
            if mask_up.any():
                xi_u = float(self.xi_upper[j].item())
                sig_u = float(self.sigma_upper[j].item())
                excess = (x_col[mask_up] - x_up).clamp_min(0.0)
                G_up = _gpd_cdf_scalar(excess, xi_u, sig_u)
                u_up = q_up + (1.0 - q_up) * G_up
                u = u.clone()
                u[mask_up] = u_up.to(u.dtype)
        if bool(self.has_lower[j].item()):
            mask_lo = x_col < x_lo
            if mask_lo.any():
                xi_l = float(self.xi_lower[j].item())
                sig_l = float(self.sigma_lower[j].item())
                excess = (x_lo - x_col[mask_lo]).clamp_min(0.0)
                G_lo = _gpd_cdf_scalar(excess, xi_l, sig_l)
                u_lo = q_lo * (1.0 - G_lo)
                u = u.clone()
                u[mask_lo] = u_lo.to(u.dtype)
        return u

    def _forward_one(self, x_col: torch.Tensor, j: int) -> torch.Tensor:
        """Map column ``j`` from x-space to z-space."""
        u = self._kde_u_one(x_col, j)
        if self.n_bins is not None:
            q_lo = float(self.q_lo[j].item())
            q_up = float(self.q_up[j].item())
            u = _body_snap_u_to_midpoint(u, self.n_bins, q_lo, q_up)

        # Clamp in float64 — _EPS=1e-12 is below float32 resolution at 1.0,
        # so a float32 clamp is a no-op for u==1.0 and ndtri(1.0) = inf.
        u64 = u.to(torch.float64).clamp(_EPS, 1.0 - _EPS)
        return torch.special.ndtri(u64).to(x_col.dtype)

    def _inverse_one(self, z_col: torch.Tensor, j: int) -> torch.Tensor:
        """Map column ``j`` from z-space back to x-space."""
        x_lo = float(self.x_lo[j].item())
        x_up = float(self.x_up[j].item())
        q_lo = float(self.q_lo[j].item())
        q_up = float(self.q_up[j].item())

        u = 0.5 * (1.0 + torch.erf(z_col.to(torch.float64) / math.sqrt(2.0)))
        u = u.clamp(_EPS, 1.0 - _EPS)
        if self.n_bins is not None:
            # Snap before tail routing: post-snap u always lies strictly
            # inside (q_lo, q_up), so the tail-inverse branches below are
            # bypassed and inverse round-trips one of K body midpoints.
            u = _body_snap_u_to_midpoint(u, self.n_bins, q_lo, q_up)

        # Body: interp back x = F̂⁻¹(u) using the inverted lookup table.
        x = _interp_extrap_torch(
            u.to(z_col.dtype), self.u_body_grid[j].to(z_col.dtype), self.x_body_grid[j]
        )

        if bool(self.has_upper[j].item()):
            mask_up = u > q_up
            if mask_up.any():
                xi_u = float(self.xi_upper[j].item())
                sig_u = float(self.sigma_upper[j].item())
                # Keep p in float64 — for heavy tails (xi>0), (1-p)^(-xi)
                # overflows in float32 when p is within ~1e-7 of 1.
                p = ((u[mask_up] - q_up) / max(1.0 - q_up, _EPS)).clamp(_EPS, 1.0 - _EPS)
                excess = _gpd_ppf_scalar(p, xi_u, sig_u)
                x = x.clone()
                x[mask_up] = (x_up + excess).to(x.dtype)
        if bool(self.has_lower[j].item()):
            mask_lo = u < q_lo
            if mask_lo.any():
                xi_l = float(self.xi_lower[j].item())
                sig_l = float(self.sigma_lower[j].item())
                p = (1.0 - u[mask_lo] / max(q_lo, _EPS)).clamp(_EPS, 1.0 - _EPS)
                excess = _gpd_ppf_scalar(p, xi_l, sig_l)
                x = x.clone()
                x[mask_lo] = (x_lo - excess).to(x.dtype)
        return x

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Map each column ``x_j`` to ``z_j``: KDE body + GPD tails + Φ⁻¹."""
        out = torch.empty_like(X)
        for j in range(self.d):
            out[:, j] = self._forward_one(X[:, j].contiguous(), j)
        return out

    def inverse(self, Z: torch.Tensor) -> torch.Tensor:
        """Inverse map ``z_j → x_j``."""
        out = torch.empty_like(Z)
        for j in range(self.d):
            out[:, j] = self._inverse_one(Z[:, j].contiguous(), j)
        return out

    # ------------------------------------------------------------------
    # Binned-mode auxiliary API (parallels RandomizedQuantileGPD)
    # ------------------------------------------------------------------

    def bin_indices(self, X: torch.Tensor) -> torch.Tensor:
        """Per-feature body-equiprobable bin index in ``[0, n_bins - 1]``.

        Mirrors :meth:`RandomizedQuantileGPD.bin_indices`: partitions
        ``[q_lo[j], q_up[j]]`` into ``K`` equal-width sub-intervals; body
        samples land in their owning bin, tail samples fold into the
        boundary classes (``0`` for ``u < q_lo``, ``K - 1`` for
        ``u > q_up``). Fully deterministic — no randomization branch
        — because the body PIT comes from a fixed lookup.
        """
        if self.n_bins is None:
            raise RuntimeError(
                "bin_indices requires n_bins to be set on this transform."
            )
        if X.shape[-1] != self.d:
            raise ValueError(
                f"Expected d={self.d}, got X with last dim {X.shape[-1]}"
            )
        K = self.n_bins
        out = torch.empty((X.shape[0], self.d), dtype=torch.long, device=X.device)
        for j in range(self.d):
            x_col = X[:, j].contiguous()
            q_lo_j = float(self.q_lo[j].item())
            q_up_j = float(self.q_up[j].item())
            u = self._kde_u_one(x_col, j)
            out[:, j] = _body_bin_index(u, K, q_lo_j, q_up_j)
        return out

    def from_bin_indices(self, idx: torch.Tensor) -> torch.Tensor:
        """Convert per-feature body-bin indices back to original-scale values.

        Each bin maps to its body midpoint
        ``u_k* = q_lo[j] + (k + 0.5)/K * (q_up[j] - q_lo[j])`` (always
        strictly inside the body), then routes through the body inverse
        — never the GPD-tail inverse — so the K boundary bins represent
        body extremes, not tail extrapolations. Round-trips
        ``from_bin_indices(bin_indices(X))`` to one of K representative
        body-scale values per feature.
        """
        if self.n_bins is None:
            raise RuntimeError(
                "from_bin_indices requires n_bins to be set on this transform."
            )
        if idx.shape[-1] != self.d:
            raise ValueError(
                f"Expected d={self.d}, got idx with last dim {idx.shape[-1]}"
            )
        K = self.n_bins
        # q_lo / q_up are float64 buffers; broadcast over [N, d].
        q_lo = self.q_lo.to(idx.device)
        q_up = self.q_up.to(idx.device)
        width = (q_up - q_lo).clamp_min(_EPS)
        u_mid = q_lo + (idx.to(torch.float64) + 0.5) / K * width
        z = torch.special.ndtri(u_mid.clamp(_EPS, 1.0 - _EPS))
        return self.inverse(z)


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------


class TransformPipeline(_Transform):
    """Compose multiple transforms; ``forward`` left-to-right, ``inverse`` reversed.

    ``fit(X)`` fits each transform on the running output of the previous one
    (so e.g. a ``["robust", "yeo_johnson"]`` pipeline fits ``YeoJohnson`` on
    the *robust-scaled* sample, not on the raw inputs).
    """

    def __init__(self, transforms: Sequence[_Transform]) -> None:
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def fit(self, X: torch.Tensor) -> "TransformPipeline":
        """Fit each transform on the running output of the previous stage."""
        with torch.no_grad():
            for t in self.transforms:
                t.fit(X)
                X = t(X)
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply each transform left-to-right."""
        for t in self.transforms:
            X = t(X)
        return X

    def inverse(self, Z: torch.Tensor) -> torch.Tensor:
        """Apply each inverse in reverse order."""
        for t in reversed(self.transforms):
            Z = t.inverse(Z)
        return Z

    def is_stochastic_in_train(self) -> bool:
        """True if any constituent transform is stochastic in train mode."""
        return any(t.is_stochastic_in_train() for t in self.transforms)


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


_TRANSFORM_REGISTRY = {
    "identity": Identity,
    "standard": StandardScale,
    "robust": RobustScale,
    "minmax": MinMaxScale,
    "yeo_johnson": YeoJohnson,
    "quantile_gpd": RandomizedQuantileGPD,
    "kde_quantile": KDEQuantile,
}


def make_transform(
    name: str,
    d: int,
    minmax_low: float = -1.0,
    minmax_high: float = 1.0,
    yeo_johnson_criterion: str = "w2",
    kde_bandwidth_factor: float = 1.0,
    n_bins: int | None = None,
) -> _Transform:
    """Build a single transform by string name.

    Recognized names:

    * ``"identity"``     — :class:`Identity` (pass-through)
    * ``"standard"``     — :class:`StandardScale` (z-score)
    * ``"robust"``       — :class:`RobustScale` (median / IQR)
    * ``"minmax"``       — :class:`MinMaxScale` (uses ``minmax_low/high``)
    * ``"yeo_johnson"``  — :class:`YeoJohnson` (uses
      ``yeo_johnson_criterion``: ``"w2"`` or ``"mle"``)
    * ``"quantile_gpd"`` — :class:`RandomizedQuantileGPD` (uses ``n_bins``)
    * ``"kde_quantile"`` — :class:`KDEQuantile` (uses
      ``kde_bandwidth_factor`` and ``n_bins``)

    Args:
        name:                  One of the strings above.
        d:                     Feature dimension.
        minmax_low:            Used only by ``"minmax"``.
        minmax_high:           Used only by ``"minmax"``.
        yeo_johnson_criterion: Used only by ``"yeo_johnson"``.
        kde_bandwidth_factor:  Used only by ``"kde_quantile"``; multiplier on
                               Silverman's bandwidth.
        n_bins:                Used only by ``"quantile_gpd"`` and
                               ``"kde_quantile"``. ``None`` (default) keeps
                               the transform continuous; ``K >= 2`` switches
                               on body-equiprobable binning (see each class).

    Returns:
        A freshly-constructed (un-fitted) transform instance.

    Raises:
        ValueError: if ``name`` is not in the registry.
    """
    if name not in _TRANSFORM_REGISTRY:
        valid = ", ".join(sorted(_TRANSFORM_REGISTRY))
        raise ValueError(f"Unknown transform {name!r}. Valid: {valid}.")
    cls = _TRANSFORM_REGISTRY[name]
    if name == "minmax":
        return cls(d=d, low=minmax_low, high=minmax_high)
    if name == "yeo_johnson":
        return cls(d=d, criterion=yeo_johnson_criterion)
    if name == "quantile_gpd":
        return cls(d=d, n_bins=n_bins)
    if name == "kde_quantile":
        return cls(d=d, bandwidth_factor=kde_bandwidth_factor, n_bins=n_bins)
    return cls(d=d)


def make_pipeline(
    spec: list[str] | TransformPipeline,
    d: int,
    minmax_range: tuple[float, float] = (-1.0, 1.0),
    yeo_johnson_criterion: str = "w2",
    kde_bandwidth_factor: float = 1.0,
    n_bins: int | None = None,
) -> TransformPipeline:
    """Build a ``TransformPipeline`` from a list of transform names.

    A pre-built pipeline is returned unchanged so callers can pass either
    a recipe (``["robust", "yeo_johnson"]``) or a fully-configured pipeline.

    Args:
        spec:                  Either a list of names accepted by
                               :func:`make_transform`, or an existing
                               :class:`TransformPipeline`.
        d:                     Feature dimension shared by all transforms.
        minmax_range:          ``(low, high)`` for any ``"minmax"`` step.
        yeo_johnson_criterion: ``"w2"`` or ``"mle"`` for any ``"yeo_johnson"`` step.
        kde_bandwidth_factor:  Bandwidth multiplier for any ``"kde_quantile"`` step.
        n_bins:                Forwarded to any ``"quantile_gpd"`` /
                               ``"kde_quantile"`` step.

    Returns:
        A :class:`TransformPipeline` ready to be ``fit`` on data.
    """
    if isinstance(spec, TransformPipeline):
        return spec
    transforms = [
        make_transform(
            name,
            d=d,
            minmax_low=minmax_range[0],
            minmax_high=minmax_range[1],
            yeo_johnson_criterion=yeo_johnson_criterion,
            kde_bandwidth_factor=kde_bandwidth_factor,
            n_bins=n_bins,
        )
        for name in spec
    ]
    return TransformPipeline(transforms)
