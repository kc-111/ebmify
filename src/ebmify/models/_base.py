"""Shared base class for the regularized regressors.

``_RegularizedRegressor`` holds the input/output transform pipelines, runs a
configurable training loop (Adam/SGD/LBFGS, loss choice, L1/L2/elastic-net
penalties, optional L1-select-then-L2 two-phase fit, per-batch noise
injection on inputs **and** outputs, n-fold augmentation per epoch), and
implements ``predict`` with inverse-transformed outputs in original scale.

Subclasses (``LinearModel``, ``GAM``) only need to define how to map a
transformed input ``X_t`` to a feature matrix.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from ._config import FitConfig, NoiseConfig, PreprocessConfig, RegConfig
from ._scaler import TransformPipeline, make_pipeline


_LOSSES = {
    "mse": nn.functional.mse_loss,
    "mae": nn.functional.l1_loss,
    "huber": nn.functional.smooth_l1_loss,
}


class _ProxFistaState:
    """Per-tensor proximal-gradient state for L1 with Adam descent.

    Each :meth:`step` does, after the optimizer's descent step:

      1. Soft-threshold the current ``param.data`` to get
         ``W_{k+1} = sign(·) · relu(|·| − η · λ)``.
      2. Optionally apply a Nesterov / FISTA look-ahead mix and write
         ``Z_{k+1}`` to ``param.data`` (otherwise write ``W_{k+1}``
         directly, i.e. plain ISTA).

    L1 is pulled OUT of the loss in proximal mode; the smooth gradient
    never sees it. The threshold uses the nominal lr (``η``) — the
    AdamW-style "decoupled" scaling, not the per-parameter
    ``η / (√v̂ + ε)`` form. One consequence: a given ``λ`` produces less
    sparsity here than under a pure proximal-gradient solver with
    ``η = 1/L``, because Adam's effective per-step displacement is
    bounded by ``η`` (not by gradient magnitude), so the soft-threshold
    only kills a coordinate when its accumulated drift is below ``η · λ``.
    Use a noticeably larger ``λ`` than the legacy "L1 in loss" mode.

    **Why FISTA momentum defaults off.** FISTA's O(1/k²) rate requires
    the descent step to be contractive — typically ``η = 1/L`` for a
    Lipschitz-``L`` smooth gradient. Adam's adaptive step size is data
    -driven and not bounded by ``1/L``, so layering Nesterov momentum on
    top of an Adam descent step lets the iterate run away (we observed
    ~6× weight blow-up on a synthetic Lasso vs the same ``λ`` with ISTA-
    only). Plain ISTA + Adam — i.e. ``fista=False`` — is the safe and
    correct default for this code path. The standalone proximal-gradient
    solver in ``linear_stock_example/fit_factor_models.py`` uses a power-
    iteration estimate of ``L`` and DOES use FISTA momentum; that one is
    sound because the descent step is Lipschitz.

    :meth:`finalize` copies the prox-clean ``W_K`` (not the look-ahead
    ``Z_K``) back into the parameter, so ``predict`` runs on the actually
    -sparse weights. No-op when FISTA mode is off (``W_K == Z_K``).
    """

    def __init__(self, params: list[torch.Tensor]):
        self.params = list(params)
        # Each param holds whatever the caller initialized it to (zeros for
        # LinearModel / GAM / FCNet). Treat that as W_0 = Z_0 = prev.
        self.prev: list[torch.Tensor] = [p.detach().clone() for p in self.params]
        self.t: float = 1.0

    @torch.no_grad()
    def step(self, lr: float, l1: float, *, fista: bool = False) -> None:
        """Apply prox + (optional) Nesterov mix to every tracked parameter.

        Call AFTER ``optimizer.step()``. No-op if ``l1 <= 0``.

        Args:
            lr:    Effective learning rate (used as the prox threshold scale).
                   Pass ``fit_config.lr``.
            l1:    L1 coefficient (the prox threshold is ``lr * l1``).
            fista: Apply Nesterov momentum on top of the prox step. Defaults
                   to ``False`` because FISTA + Adam is unstable (see
                   class docstring); enable only for diagnostics or if you
                   know the descent step is Lipschitz-bounded by ``1/lr``.
        """
        if l1 <= 0:
            return
        thresh = float(lr) * float(l1)
        if fista:
            t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * self.t * self.t))
            ratio = (self.t - 1.0) / t_new
        else:
            t_new = 1.0
            ratio = 0.0
        for i, p in enumerate(self.params):
            # Soft-threshold the Adam-descended value (which currently
            # lives in p.data after optimizer.step()).
            sign = torch.sign(p.data)
            mag = (p.data.abs() - thresh).clamp_min_(0.0)
            W_new = sign * mag
            if fista:
                p.data.copy_(W_new + ratio * (W_new - self.prev[i]))
            else:
                p.data.copy_(W_new)
            self.prev[i] = W_new.clone()
        self.t = t_new

    @torch.no_grad()
    def finalize(self) -> None:
        """Restore each parameter to its last prox-clean ``W_K``.

        Without this, ``param.data`` holds the FISTA look-ahead ``Z_K``,
        which has nonzero entries that the prox step would have killed.
        Call this after training so ``predict`` sees the sparse weights.
        No-op when ISTA-only mode was used (``W_K == Z_K``).
        """
        for i, p in enumerate(self.params):
            p.data.copy_(self.prev[i])


def _resolve_loss(loss: str | Callable) -> Callable:
    """Resolve a loss spec to a callable ``(pred, target) -> scalar``.

    Args:
        loss: Either a key in ``_LOSSES`` (``"mse"`` | ``"mae"`` | ``"huber"``)
            or any callable with the same signature as ``F.mse_loss``.

    Returns:
        The loss callable.

    Raises:
        ValueError: ``loss`` is a string but not a known key.
    """
    if callable(loss):
        return loss
    if loss not in _LOSSES:
        valid = ", ".join(sorted(_LOSSES))
        raise ValueError(f"Unknown loss {loss!r}. Valid: {valid} or a callable.")
    return _LOSSES[loss]


def _weighted_loss_value(
    loss_spec: str | Callable,
    base_loss: Callable,
    pred: torch.Tensor,
    target: torch.Tensor,
    sw: torch.Tensor | None,
) -> torch.Tensor:
    """Compute a scalar loss, optionally with per-sample (or per-cell) weights.

    With ``sw=None`` this is just ``base_loss(pred, target)`` (mean reduction
    over all elements, as before). With ``sw`` provided:

    * ``sw`` shape ``(B,)`` — broadcast across output dims. Per-sample loss is
      the mean of the elementwise loss over non-batch dims, then
      ``sum_i sw[i] * loss[i] / sum_i sw[i]``. Use this when every output
      column shares the same weighting (e.g. one-target-at-a-time fitting).
    * ``sw`` shape equal to ``pred`` (e.g. ``(B, n_outputs)``) — per-cell
      weights. Each output column is weighted independently; the result is
      ``sum_{i,j} sw[i,j] * loss[i,j] / sum_{i,j} sw[i,j]``. Use this when
      different output targets have different concurrency / overlap (the
      ``SampleSet`` case where ``sample_weights`` is ``(N, A)``).

    The caller is responsible for normalising ``sw`` to a useful scale before
    calling — typically mean 1.0 globally for the 1-D case, or mean 1.0 per
    column for the 2-D case (which keeps each output's loss contribution at
    the same scale as the unweighted loss).

    Args:
        loss_spec: Original ``self.loss_spec``. String keys (``"mse"``,
                   ``"mae"``, ``"huber"``) hit fast elementwise paths;
                   callables are invoked with ``reduction="none"`` and
                   must accept that kwarg.
        base_loss: Result of ``_resolve_loss(loss_spec)`` — used as the
                   fast scalar path when ``sw is None``.
        pred:      Predictions, shape ``(B, ...)``.
        target:    Targets, same shape as ``pred``.
        sw:        ``(B,)`` per-sample weights, weights broadcastable to the
                   shape of ``pred`` (typically ``(B, n_outputs)``), or
                   ``None``.

    Returns:
        Scalar tensor.
    """
    if sw is None:
        return base_loss(pred, target)
    if isinstance(loss_spec, str):
        if loss_spec == "mse":
            per_elem = (pred - target).pow(2)
        elif loss_spec == "mae":
            per_elem = (pred - target).abs()
        elif loss_spec == "huber":
            per_elem = nn.functional.smooth_l1_loss(pred, target, reduction="none")
        else:
            valid = ", ".join(sorted(_LOSSES))
            raise ValueError(f"Unknown loss {loss_spec!r}. Valid: {valid}.")
    else:
        try:
            per_elem = loss_spec(pred, target, reduction="none")
        except TypeError as exc:
            raise ValueError(
                "Custom loss callables must accept reduction='none' to be "
                "used with sample_weight."
            ) from exc

    sw = sw.to(per_elem.dtype)
    if sw.ndim == 1:
        # 1-D: broadcast across output dims by reducing per-elem over them
        # first, then weighted mean over the batch.
        if per_elem.ndim > 1:
            per_sample = per_elem.mean(dim=tuple(range(1, per_elem.ndim)))
        else:
            per_sample = per_elem
        sw = sw.reshape(-1)
        return (sw * per_sample).sum() / sw.sum().clamp_min(1e-12)
    # 2-D (or higher): per-cell weights; must match per_elem shape.
    if sw.shape != per_elem.shape:
        raise ValueError(
            f"sample_weight shape {tuple(sw.shape)} must broadcast to prediction "
            f"shape {tuple(per_elem.shape)}; pass a 1-D ``(B,)`` tensor for "
            "per-sample weights or a tensor matching ``pred`` for per-cell "
            "weights."
        )
    return (sw * per_elem).sum() / sw.sum().clamp_min(1e-12)


def feature_leverage(
    Phi_train: torch.Tensor,
    Phi_query: torch.Tensor,
    ridge: float = 1e-3,
    *,
    bias: bool = True,
) -> torch.Tensor:
    """Last-layer leverage diagonal for precomputed feature matrices.

    Returns the diagonal of

    .. math::

        h(x^*) = \\phi(x^*)^\\top
                 (\\Phi_{train}^\\top \\Phi_{train} + r I)^{-1}
                 \\phi(x^*)

    With ``bias=True`` (the default) the formula is augmented with an
    all-ones column so it covers the linear head's bias term as well as
    the weight matrix — this is the correct posterior variance for the
    head ``feats @ W + b`` under uncentered features. The bias entry of
    the diagonal load is left at zero (improper prior on ``b``), matching
    how ``b`` is trained without an L2 penalty.

    The formula does NOT depend on the trained weights or any observed
    ``Y``; it is a purely geometric property of the train/query design
    matrices plus the ridge prior. ``Y`` only enters later through
    ``sigma_n`` if you want absolute calibration of ``sigma_epi``.

    This is the single source of truth used by
    ``_RegularizedRegressor.feature_leverage`` and
    ``FCNet.feature_leverage``. Call it directly when you already hold
    ``Phi_train`` / ``Phi_query`` (e.g. from ``model.features(X)``) and
    want to avoid re-running the forward pass that the model methods
    would do internally.

    Args:
        Phi_train: ``[N, F]`` training features.
        Phi_query: ``[M, F]`` query features.
        ridge:     Diagonal load on ``Phi_train^T Phi_train``. Acts as
                   the prior precision on the readout weights
                   (``r = sigma_n^2 / sigma_W^2``).
        bias:      Whether to augment with an all-ones column to cover
                   the head's bias. Set to ``False`` only if your linear
                   head has no bias term, or if you want the
                   centered-features variant that ignores ``b``.

    Returns:
        ``[M]`` non-negative tensor — leverage at each query point.
    """
    if bias:
        ones_tr = torch.ones(
            Phi_train.shape[0], 1,
            device=Phi_train.device, dtype=Phi_train.dtype,
        )
        ones_qy = torch.ones(
            Phi_query.shape[0], 1,
            device=Phi_query.device, dtype=Phi_query.dtype,
        )
        Phi_train = torch.cat([Phi_train, ones_tr], dim=1)
        Phi_query = torch.cat([Phi_query, ones_qy], dim=1)
    F = Phi_train.shape[1]
    pen = ridge * torch.eye(
        F, device=Phi_train.device, dtype=Phi_train.dtype,
    )
    if bias:
        pen[-1, -1] = 0.0
    A = Phi_train.T @ Phi_train + pen
    sol = torch.linalg.solve(A, Phi_query.T)
    return (Phi_query * sol.T).sum(dim=1)


def _to_tensor(x, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Coerce numpy/torch/sequence input to a tensor on ``device`` with ``dtype``.

    Args:
        x:      Input array. ``np.ndarray``, ``torch.Tensor``, or anything
                ``torch.as_tensor`` accepts (lists, tuples, etc.).
        device: Target device for the returned tensor.
        dtype:  Target dtype (default ``float32``).

    Returns:
        A tensor on ``device`` with the requested dtype. Numpy arrays are
        adopted via ``from_numpy`` (zero-copy when possible), existing
        tensors are passed through, anything else goes through
        ``as_tensor``.
    """
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(x)
    return t.to(device=device, dtype=dtype)


def _normalise_sample_weight(
    sw,
    n: int,
    n_outputs: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    """Coerce/normalise an optional ``sample_weight`` argument.

    Returns ``None`` if ``sw is None``. Otherwise:

    * 1-D input of length ``n`` → returned 1-D tensor rescaled to mean 1.0
      (one weight per sample, shared across output columns).
    * 2-D input of shape ``(n, n_outputs)`` → returned 2-D tensor with each
      column rescaled to mean 1.0 (one weight per (sample, target column),
      as produced by ``SampleSet.to_tensors`` / ``to_matrix`` for label-
      overlap-corrected losses).
    * 2-D input of shape ``(n, 1)`` is treated as 1-D after reshaping.

    Per-column normalisation in the 2-D case is critical: different target
    columns have different concurrency patterns and therefore different
    raw weight scales. Globally normalising to mean 1 would let one
    column's larger absolute weights dominate the loss; per-column
    normalisation keeps each target's contribution at the same scale as
    the unweighted loss.

    Negative weights are rejected (they would invert the loss); zero is
    allowed (a zero-weighted cell is a no-op).
    """
    if sw is None:
        return None
    t = _to_tensor(sw, device, dtype)
    if t.ndim == 2 and t.shape[1] == 1:
        t = t.reshape(-1)
    if t.ndim == 1:
        if t.shape[0] != n:
            raise ValueError(
                f"sample_weight length {t.shape[0]} does not match n={n}"
            )
        if torch.any(t < 0):
            raise ValueError("sample_weight must be non-negative")
        mean = t.mean()
        if float(mean) <= 0:
            raise ValueError("sample_weight has zero total mass")
        return t / mean
    if t.ndim == 2:
        if t.shape != (n, n_outputs):
            raise ValueError(
                f"2-D sample_weight shape {tuple(t.shape)} does not match "
                f"(n, n_outputs)=({n}, {n_outputs})"
            )
        if torch.any(t < 0):
            raise ValueError("sample_weight must be non-negative")
        col_means = t.mean(dim=0, keepdim=True)
        if torch.any(col_means <= 0):
            raise ValueError(
                "sample_weight has zero total mass for at least one target column"
            )
        return t / col_means
    raise ValueError(
        f"sample_weight must be 1-D (per-sample) or 2-D (per-sample-per-target), "
        f"got shape {tuple(t.shape)}"
    )


class _RegularizedRegressor(nn.Module):
    """Base class for regularized linear-in-features regressors.

    Args:
        n_inputs:        Number of input features.
        n_outputs:       Number of output targets.
        fit_config:      Training-loop settings (defaults to ``FitConfig()``).
        reg_config:      Regularization settings.
        noise_config:    Per-batch noise injection settings.
        preprocess:      Input/output preprocessing pipelines.
        loss:            ``"mse"`` | ``"mae"`` | ``"huber"`` or a callable
                         ``(pred, target) -> scalar``.
        device:          Torch device.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        fit_config: FitConfig | None = None,
        reg_config: RegConfig | None = None,
        noise_config: NoiseConfig | None = None,
        preprocess: PreprocessConfig | None = None,
        loss: str | Callable = "mse",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.fit_config = fit_config or FitConfig()
        self.reg_config = reg_config or RegConfig()
        self.noise_config = noise_config or NoiseConfig()
        self.preprocess = preprocess or PreprocessConfig()
        self.loss_spec = loss
        self.device = torch.device(device) if isinstance(device, str) else device

        self.input_pipeline: TransformPipeline = make_pipeline(
            self.preprocess.input_transforms,
            d=n_inputs,
            minmax_range=self.preprocess.minmax_range,
            yeo_johnson_criterion=self.preprocess.yeo_johnson_criterion,
            kde_bandwidth_factor=self.preprocess.kde_bandwidth_factor,
            n_bins=self.preprocess.n_bins,
        )
        self.output_pipeline: TransformPipeline = make_pipeline(
            self.preprocess.output_transforms,
            d=n_outputs,
            minmax_range=self.preprocess.minmax_range,
            yeo_johnson_criterion=self.preprocess.yeo_johnson_criterion,
            kde_bandwidth_factor=self.preprocess.kde_bandwidth_factor,
            n_bins=self.preprocess.n_bins,
        )

        # Lazily initialized in fit() once subclass _fit_features sets feature dim.
        self.W: nn.Parameter | None = None
        self.b: nn.Parameter | None = None
        self.register_buffer(
            "active_mask", torch.empty(0), persistent=True
        )  # populated only for L1-then-L2

        self._fitted = False
        self.to(self.device)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _fit_features(self, X_t: torch.Tensor) -> None:
        """Optional one-time hook to fit data-dependent feature builders.

        Called from ``fit`` once on a deterministic eval-mode preprocessed
        copy of the inputs, before ``W`` and ``b`` are allocated. Subclasses
        like ``GAM`` use this to place spline knots at training-data
        quantiles. ``LinearModel`` doesn't need it (default no-op).

        Args:
            X_t: Eval-mode preprocessed inputs ``[N, n_inputs]``.
        """

    def _features(self, X_t: torch.Tensor) -> torch.Tensor:
        """Map preprocessed inputs to the feature matrix the linear head consumes.

        Args:
            X_t: Preprocessed inputs ``[N, n_inputs]``.

        Returns:
            Feature matrix ``[N, F]`` where ``F = self._n_features()``.

        Raises:
            NotImplementedError: subclasses must override.
        """
        raise NotImplementedError

    def _n_features(self) -> int:
        """Return the feature dimension ``F`` produced by ``_features``.

        ``W`` is allocated as ``[F, n_outputs]``, so the value returned here
        determines the shape of the trainable weight matrix.

        Raises:
            NotImplementedError: subclasses must override.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, Y, sample_weight=None) -> "_RegularizedRegressor":
        """Fit preprocessing pipelines, subclass features, and weights end-to-end.

        Pipeline of work:
          1. Fit input/output transform pipelines on the raw data.
          2. Run a single deterministic (eval-mode) preprocessed pass so the
             subclass can place data-dependent feature parameters (e.g. GAM
             spline knots). Eval mode disables randomized PIT so the knots
             are stable across fits and across runs.
          3. Allocate ``W`` / ``b`` now that the feature dimension is known.
          4. Run the training loop. With ``RegConfig.l1_then_l2=True`` we do
             a two-phase fit: an L1 phase to pick a sparse support, then an
             L2 phase that freezes the unselected rows via ``active_mask``.

        Args:
            X:             Input matrix ``[N, n_inputs]`` (numpy or torch).
            Y:             Target matrix ``[N, n_outputs]`` (numpy or torch).
            sample_weight: Optional loss weights. Pass ``[N]`` (or ``[N, 1]``)
                           for per-sample weights shared across all output
                           columns; pass ``[N, n_outputs]`` for per-target
                           weights (e.g. ``SampleSet.to_tensors`` /
                           ``to_matrix`` returns this shape, with one weight
                           per (sample, target column) reflecting that
                           target's label-overlap concurrency). 1-D inputs
                           are normalised to global mean 1.0; 2-D inputs are
                           normalised per-column to mean 1.0 — different
                           target columns have different concurrency scales
                           and must not be collapsed together. Mean-1
                           normalisation keeps the loss magnitude comparable
                           to the unweighted case so previously-tuned L1/L2
                           coefficients transfer without re-tuning. ``None``
                           keeps the unweighted loss.

        Returns:
            ``self`` (so calls can be chained as ``model.fit(X, Y).predict(X)``).

        Raises:
            ValueError: if ``X`` or ``Y`` have wrong rank/columns, row counts
                disagree, or ``sample_weight`` has the wrong shape.
        """
        fc = self.fit_config
        if fc.seed is not None:
            torch.manual_seed(fc.seed)
            np.random.seed(fc.seed)

        X_t = _to_tensor(X, self.device)
        Y_t = _to_tensor(Y, self.device)
        if X_t.ndim != 2 or X_t.shape[1] != self.n_inputs:
            raise ValueError(f"Expected X shape [N, {self.n_inputs}], got {tuple(X_t.shape)}")
        if Y_t.ndim != 2 or Y_t.shape[1] != self.n_outputs:
            raise ValueError(f"Expected Y shape [N, {self.n_outputs}], got {tuple(Y_t.shape)}")
        if X_t.shape[0] != Y_t.shape[0]:
            raise ValueError(f"X and Y row count mismatch: {X_t.shape[0]} vs {Y_t.shape[0]}")
        sw_t = _normalise_sample_weight(
            sample_weight, X_t.shape[0], self.n_outputs, self.device
        )

        # Step 1: fit preprocessing pipelines on the raw inputs and targets.
        with torch.no_grad():
            self.input_pipeline.fit(X_t)
            self.output_pipeline.fit(Y_t)

        # Step 2: deterministic preprocessed pass for subclass feature setup.
        # Eval mode is required so randomized PIT (RandomizedQuantileGPD) does
        # not draw fresh ranks here — knot placement / etc. must be stable.
        X_proc_det, _ = self._preprocess_eval(X_t, Y_t)
        self._fit_features(X_proc_det)

        # Step 3: allocate weights now that the feature dimension is known.
        F = self._n_features()
        self.W = nn.Parameter(torch.zeros(F, self.n_outputs, device=self.device))
        self.b = nn.Parameter(torch.zeros(self.n_outputs, device=self.device))
        self.active_mask = torch.ones(F, device=self.device)

        rc = self.reg_config

        # Step 4: training. _train re-applies the input/output pipelines per
        # minibatch in train mode, so randomized PIT for tied/atom inputs
        # draws fresh ranks each step — the model integrates over the rank
        # ambiguity rather than overfitting to one frozen draw.
        if rc.l1_then_l2:
            # Phase 1: L1-only fit to pick a sparse support.
            self._train(X_t, Y_t, l1=rc.l1, l2=0.0, sample_weight=sw_t)
            # Build the active-feature mask: a feature row "survives" if any
            # of its output coefficients exceeded the threshold. The mask is
            # multiplicative in `predict` and `_train` so phase 2 only fits
            # the surviving rows; pruned rows are zeroed and held there.
            with torch.no_grad():
                survives = self.W.detach().abs().max(dim=1).values > rc.l1_select_threshold
                mask = survives.to(self.W.dtype)
                self.active_mask = mask
                self.W.data = self.W.data * mask.unsqueeze(1)
            # Phase 2: L2-only fit over the surviving features.
            self._train(X_t, Y_t, l1=0.0, l2=rc.l2, sample_weight=sw_t)
        else:
            self._train(X_t, Y_t, l1=rc.l1, l2=rc.l2, sample_weight=sw_t)

        self._fitted = True
        return self

    def features(self, X) -> torch.Tensor:
        """Return the design matrix the linear head consumes for ``X``.

        This is the same ``Φ * active_mask`` that ``predict`` would form
        before the readout multiplication. Mirrors ``FCNet.features`` so
        the same downstream recipe (last-layer leverage / Bayesian-linear
        readout / nearest-neighbor over learned features) works uniformly
        across model classes.

        For ``LinearModel`` this is the preprocessed input; for ``GAM``
        it is the centered B-spline basis evaluated at the preprocessed
        input. ``active_mask`` zeros out columns that were pruned in the
        L1-then-L2 phase, so leverage / kernel computations only see
        live features.

        Useful when the training set is too large to materialize the full
        Gram matrix ``Φᵀ Φ`` in memory: the caller can subsample, batch,
        or stream rows of ``Φ`` themselves rather than asking the model
        to build a posterior over the readout internally.

        Args:
            X: Input matrix ``[N, n_inputs]`` (numpy or torch).

        Returns:
            ``[N, F]`` design matrix in the trained feature space, where
            ``F == self._n_features()``.

        Raises:
            RuntimeError: if called before ``fit``.
        """
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call .fit(X, Y) first.")
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                X_t = _to_tensor(X, self.device)
                X_proc = self.input_pipeline(X_t)
                feats = self._features(X_proc) * self.active_mask
        finally:
            if was_training:
                self.train()
        return feats

    def feature_leverage(
        self,
        X_train,
        X_query,
        ridge: float = 1e-3,
        *,
        bias: bool = True,
    ) -> torch.Tensor:
        """Last-layer leverage / Bayesian-linear posterior variance scaling.

        Returns the diagonal of

        .. math::

            h(x^*) = \\phi(x^*)^\\top (\\Phi_{train}^\\top \\Phi_{train}
                     + r I)^{-1} \\phi(x^*)

        where ``Phi = self.features(X)``. ``h`` is the unitless "leverage"
        — the standard last-layer epistemic head. Calibrate to a noise
        scale via ``sigma_epi = lambda * sqrt(h)`` with
        ``lambda = sigma_n / sqrt(h_p95(train))``.

        The formula does NOT use ``self.W`` or any observed ``Y``: it is
        a purely geometric property of the train/query design matrices
        plus the ridge prior. Two different fits on the same ``Phi_train``
        produce identical leverage. ``Y`` only enters later, through
        ``sigma_n`` for absolute calibration of ``sigma_epi``.

        Args:
            X_train: Training inputs ``[N, n_inputs]`` whose features
                define the data manifold the model was identified on.
                Typically the same ``X`` passed to ``.fit``.
            X_query: Query inputs ``[M, n_inputs]`` to score for
                uncertainty.
            ridge: Diagonal load on ``Phi^T Phi``. Acts as the prior
                precision on the readout weights (``r = sigma_n^2 /
                sigma_W^2``). Small values give a tight, geometry-driven
                signal; larger values smooth the signal and prevent
                near-singular inverses when ``F`` is close to ``N``.
            bias: If True (default), augment ``Phi`` with an all-ones
                column so the leverage formula covers ``self.b`` as
                well as ``self.W`` — this is the correct posterior
                variance for the linear head ``feats @ W + b`` and is
                what you want for *uncentered* features. The bias entry
                is left un-penalized in the diagonal load (improper
                prior on ``b``), matching how ``b`` is trained. If False,
                the formula covers only ``W`` and ignores ``b``; for
                centered features and large ``N`` this differs by a
                roughly constant ``sigma_n^2 / N`` and matches the
                recipe in ``example/hetero_demo_2d_ood.py``.

        Returns:
            ``[M]`` tensor ``h(X_query)``. Always ``>= 0``.

        Raises:
            RuntimeError: if called before ``fit``.
        """
        Phi_train = self.features(X_train)
        Phi_query = self.features(X_query)
        return feature_leverage(Phi_train, Phi_query, ridge, bias=bias)

    def predict(self, X) -> torch.Tensor:
        """Predict targets in the original (un-transformed) output scale.

        The forward pass is: input pipeline (eval mode, deterministic) →
        feature map → ``feats @ W + b`` → inverse output pipeline. Eval mode
        ensures any stochastic transforms (randomized PIT) act
        deterministically at prediction time.

        Args:
            X: Input matrix ``[N, n_inputs]`` (numpy or torch).

        Returns:
            ``[N, n_outputs]`` point prediction in original scale.

        Raises:
            RuntimeError: if called before ``fit``.
        """
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call .fit(X, Y) first.")
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                X_t = _to_tensor(X, self.device)
                X_proc = self.input_pipeline(X_t)
                feats = self._features(X_proc) * self.active_mask
                Y_proc = feats @ self.W + self.b
                Y = self.output_pipeline.inverse(Y_proc)
        finally:
            if was_training:
                self.train()
        return Y

    def save(self, path: str | Path) -> None:
        """Serialize full model state (shape config + buffers + parameters) to disk.

        Args:
            path: Destination path for ``torch.save``.
        """
        torch.save(
            {
                "n_inputs": self.n_inputs,
                "n_outputs": self.n_outputs,
                "n_features": self._n_features() if self._fitted else None,
                "fitted": self._fitted,
                "state_dict": self.state_dict(),
            },
            path,
        )

    def load_state(self, path: str | Path) -> "_RegularizedRegressor":
        """Restore weights + buffers from a file produced by ``save``.

        The regressor must already be constructed with the same shape config
        (matching ``n_inputs`` / ``n_outputs``). If the saved model was
        fitted, ``W`` / ``b`` are pre-allocated to the saved feature
        dimension so ``load_state_dict`` can populate them.

        Args:
            path: Source file produced by a prior ``save()`` call.

        Returns:
            ``self``, with state restored.
        """
        state = torch.load(path, map_location=self.device, weights_only=True)
        if state.get("fitted", False):
            F = state["n_features"]
            self.W = nn.Parameter(torch.zeros(F, self.n_outputs, device=self.device))
            self.b = nn.Parameter(torch.zeros(self.n_outputs, device=self.device))
            self.active_mask = torch.ones(F, device=self.device)
        self.load_state_dict(state["state_dict"])
        self._fitted = state.get("fitted", False)
        return self

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _inject(x: torch.Tensor, add_std: float, mul_std: float) -> torch.Tensor:
        """Inject additive + multiplicative Gaussian noise.

        Used as a per-batch data-augmentation step. Returns ``x`` unchanged
        when both ``add_std`` and ``mul_std`` are ``0``.

        Args:
            x:       Tensor to perturb.
            add_std: Stddev of the additive noise (``x + add_std * N(0, 1)``).
            mul_std: Stddev of the multiplicative noise
                     (``x * (1 + mul_std * N(0, 1))``).

        Returns:
            The (possibly noised) tensor with the same shape as ``x``.
        """
        if add_std > 0:
            x = x + add_std * torch.randn_like(x)
        if mul_std > 0:
            x = x * (1.0 + mul_std * torch.randn_like(x))
        return x

    def _preprocess_eval(
        self, X_raw: torch.Tensor, Y_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the input + output pipelines in eval mode under ``no_grad``.

        Eval mode disables stochastic per-step behavior such as the
        randomized-tie PIT in ``RandomizedQuantileGPD``, giving a stable,
        reproducible preprocessed copy. Used for: subclass feature setup
        (knot placement), the validation split, the LBFGS full-batch
        objective, and verbose train-loss readouts.

        Args:
            X_raw: Raw inputs ``[N, n_inputs]``.
            Y_raw: Raw targets ``[N, n_outputs]``.

        Returns:
            ``(X_proc, Y_proc)`` after the pipelines, with this module's
            train/eval flag restored to its prior value on exit.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                X_proc = self.input_pipeline(X_raw)
                Y_proc = self.output_pipeline(Y_raw)
        finally:
            if was_training:
                self.train()
        return X_proc, Y_proc

    def _make_optimizer(self) -> torch.optim.Optimizer:
        """Construct the optimizer specified by ``fit_config.optimizer``.

        Returns:
            A ``torch.optim.Optimizer`` over the model's trainable params.

        Raises:
            ValueError: if ``fit_config.optimizer`` is not one of
                ``"adam"`` | ``"sgd"`` | ``"lbfgs"``.
        """
        fc = self.fit_config
        params = [p for p in self.parameters() if p.requires_grad]
        if fc.optimizer == "adam":
            return torch.optim.Adam(params, lr=fc.lr)
        if fc.optimizer == "sgd":
            return torch.optim.SGD(params, lr=fc.lr)
        if fc.optimizer == "lbfgs":
            return torch.optim.LBFGS(params, lr=fc.lr, max_iter=20)
        raise ValueError(f"Unknown optimizer: {fc.optimizer!r}")

    def _batches(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        batch_size: int | None,
        W: torch.Tensor | None = None,
    ):
        """Yield shuffled minibatches of ``(X, Y)``, plus ``W`` slice if given.

        Args:
            X:          Inputs ``[N, ...]``.
            Y:          Targets ``[N, ...]``.
            batch_size: Minibatch size; ``None`` means full-batch.
            W:          Optional weights ``[N]`` or ``[N, ...]`` to slice
                        alongside. When provided, each yielded tuple gains
                        a third element ``W[sl]`` (axis-0 slice, so 2-D
                        per-target weights work too).

        Yields:
            ``(batch_X, batch_Y)`` (W=None) or ``(batch_X, batch_Y, batch_W)``
            (W given) slices using a fresh permutation each call.
        """
        n = X.shape[0]
        bs = batch_size or n
        idx = torch.randperm(n, device=X.device)
        for start in range(0, n, bs):
            sl = idx[start : start + bs]
            if W is None:
                yield X[sl], Y[sl]
            else:
                yield X[sl], Y[sl], W[sl]

    def _train(
        self,
        X_raw: torch.Tensor,
        Y_raw: torch.Tensor,
        l1: float,
        l2: float,
        sample_weight: torch.Tensor | None = None,
    ) -> None:
        """Run one full training phase with the given L1/L2 coefficients.

        Three preprocessing strategies coexist here:

        * **Deterministic SGD / Adam path** — when neither pipeline is
          stochastic in train mode (e.g. ``KDEQuantile``,
          ``RandomizedQuantileGPD(randomize_ties=False)``, all parametric
          transforms), the entire training set is preprocessed *once* in eval
          mode and the cached tensor is sliced per batch. Heavy transforms
          like KDE are no longer paid per step.
        * **Stochastic SGD / Adam path** — when any pipeline is stochastic
          (``RandomizedQuantileGPD(randomize_ties=True)``), each minibatch
          step sees a fresh rank draw on tied rows, so the optimizer
          integrates over the rank ambiguity at the standard
          data-augmentation cadence.
        * **LBFGS path** — the entire training set is preprocessed *once*
          up front in eval mode regardless of pipeline type. LBFGS performs
          many line-search evaluations per outer step and requires a stable,
          deterministic objective, which is incompatible with stochastic
          PIT or per-batch noise injection.

        Args:
            X_raw:         Raw inputs ``[N, n_inputs]``.
            Y_raw:         Raw targets ``[N, n_outputs]``.
            l1:            L1 weight-penalty coefficient (skipped if ``0``).
            l2:            L2 weight-penalty coefficient (skipped if ``0``).
            sample_weight: Optional weights ``[N]`` or ``[N, n_outputs]``,
                           already normalised by ``fit`` (1-D: global mean
                           1.0; 2-D: each column to mean 1.0).
        """
        fc = self.fit_config
        nc = self.noise_config
        rc = self.reg_config
        base_loss = _resolve_loss(self.loss_spec)
        optimizer = self._make_optimizer()

        # Optional validation split: split on raw rows, then preprocess the
        # val set once in eval mode so the val objective is deterministic
        # (early stopping needs a stable signal).
        (
            X_train_raw, Y_train_raw, X_val_raw, Y_val_raw,
            sw_train, sw_val,
        ) = self._maybe_split(X_raw, Y_raw, fc.val_split, fc.seed, sample_weight)
        do_es = fc.early_stopping_patience is not None and X_val_raw is not None
        if do_es:
            X_val, Y_val = self._preprocess_eval(X_val_raw, Y_val_raw)
        best_val = math.inf
        patience = 0

        is_lbfgs = fc.optimizer == "lbfgs"
        # Proximal-Adam / FISTA L1 path (only meaningful when l1>0 and the
        # optimizer is gradient-based — LBFGS does line search and is
        # incompatible with prox steps inside its inner objective).
        use_proximal = bool(rc.l1_proximal) and l1 > 0 and not is_lbfgs
        prox_state = _ProxFistaState([self.W]) if use_proximal else None
        # If both pipelines are deterministic in train mode, the per-batch
        # pipeline call is wasted work — output is identical every epoch.
        # Preprocess once and reuse the cached tensor every step.
        deterministic_pipelines = (
            not self.input_pipeline.is_stochastic_in_train()
            and not self.output_pipeline.is_stochastic_in_train()
        )
        cache_preprocessed = is_lbfgs or deterministic_pipelines

        if cache_preprocessed:
            X_train_cached, Y_train_cached = self._preprocess_eval(
                X_train_raw, Y_train_raw
            )

        for epoch in range(fc.epochs):
            self.train()
            for _aug_pass in range(fc.n_aug):
                if is_lbfgs:
                    for batch_tup in self._batches(
                        X_train_cached, Y_train_cached, fc.batch_size,
                        W=sw_train,
                    ):
                        if sw_train is None:
                            batch_X, batch_Y = batch_tup
                            batch_W = None
                        else:
                            batch_X, batch_Y, batch_W = batch_tup
                        # Closure is rebuilt per batch (it captures
                        # batch_X / batch_Y); LBFGS calls it many times
                        # internally per step. No pipeline call inside —
                        # the data is already preprocessed.
                        def closure():
                            optimizer.zero_grad()
                            feats = self._features(batch_X) * self.active_mask
                            pred = feats @ self.W + self.b
                            loss = _weighted_loss_value(
                                self.loss_spec, base_loss, pred, batch_Y, batch_W,
                            )
                            if l1 > 0:
                                loss = loss + l1 * self.W.abs().sum()
                            if l2 > 0:
                                loss = loss + l2 * (self.W ** 2).sum()
                            loss.backward()
                            return loss

                        optimizer.step(closure)
                    continue

                # SGD / Adam path: either slice the cached preprocessed tensor
                # (deterministic pipelines) or preprocess each batch fresh in
                # train mode (stochastic PIT fires once per step).
                if cache_preprocessed:
                    batch_iter = self._batches(
                        X_train_cached, Y_train_cached, fc.batch_size, W=sw_train,
                    )
                else:
                    batch_iter = self._batches(
                        X_train_raw, Y_train_raw, fc.batch_size, W=sw_train,
                    )
                for batch_tup in batch_iter:
                    if sw_train is None:
                        b_X, b_Y = batch_tup
                        b_W = None
                    else:
                        b_X, b_Y, b_W = batch_tup
                    if cache_preprocessed:
                        batch_X, batch_Y = b_X, b_Y
                    else:
                        with torch.no_grad():
                            batch_X = self.input_pipeline(b_X)
                            batch_Y = self.output_pipeline(b_Y)

                    x = self._inject(batch_X, nc.input_additive_std, nc.input_multiplicative_std)
                    y = self._inject(batch_Y, nc.output_additive_std, nc.output_multiplicative_std)

                    feats = self._features(x) * self.active_mask
                    pred = feats @ self.W + self.b
                    loss = _weighted_loss_value(
                        self.loss_spec, base_loss, pred, y, b_W,
                    )
                    # In proximal mode L1 leaves the loss entirely; we apply
                    # it via soft-threshold after the Adam step.
                    if l1 > 0 and not use_proximal:
                        loss = loss + l1 * self.W.abs().sum()
                    if l2 > 0:
                        loss = loss + l2 * (self.W ** 2).sum()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if prox_state is not None:
                        prox_state.step(fc.lr, l1)

            # Validation / early stopping. Eval mode disables randomized PIT
            # so the val loss is comparable across epochs.
            if do_es:
                self.eval()
                with torch.no_grad():
                    feats_v = self._features(X_val) * self.active_mask
                    pred_v = feats_v @ self.W + self.b
                    val_loss = float(_weighted_loss_value(
                        self.loss_spec, base_loss, pred_v, Y_val, sw_val,
                    ).item())
                if val_loss < best_val - 1e-9:
                    best_val = val_loss
                    patience = 0
                else:
                    patience += 1
                    if patience >= fc.early_stopping_patience:
                        if fc.verbose:
                            print(f"  early stop at epoch {epoch + 1} (val={val_loss:.4e})")
                        break

            # Verbose path: emit a stable train-loss readout periodically.
            # Use an eval-mode preprocess so the printed number is not
            # contaminated by the per-step PIT noise that drives training.
            if fc.verbose and (epoch + 1) % 50 == 0:
                X_train_eval, Y_train_eval = self._preprocess_eval(X_train_raw, Y_train_raw)
                with torch.no_grad():
                    feats_t = self._features(X_train_eval) * self.active_mask
                    pred_t = feats_t @ self.W + self.b
                    train_loss = float(_weighted_loss_value(
                        self.loss_spec, base_loss, pred_t, Y_train_eval, sw_train,
                    ).item())
                print(f"  epoch {epoch + 1}: train_loss={train_loss:.4e}")

        # Replace the FISTA look-ahead Z_K with the prox-clean W_K so
        # downstream predict / save use the actually-sparse iterate.
        if prox_state is not None:
            prox_state.finalize()

    @staticmethod
    def _maybe_split(
        X: torch.Tensor,
        Y: torch.Tensor,
        val_split: float,
        seed: int | None,
        W: torch.Tensor | None = None,
    ) -> tuple:
        """Random train/validation split with a seeded generator.

        A fixed seed produces a reproducible split, so early-stopping
        decisions are deterministic across runs. Out-of-range
        ``val_split`` values short-circuit to "no split".

        Args:
            X:         Inputs ``[N, ...]``.
            Y:         Targets ``[N, ...]``.
            val_split: Fraction of rows to allocate to the validation set.
                       Values ``<= 0`` or ``>= 1`` mean no split.
            seed:      Optional generator seed; ``None`` uses a fresh draw.
            W:         Optional weights ``[N]`` or ``[N, ...]`` to split
                       alongside (axis-0 slice, so 2-D per-target weights
                       work too).

        Returns:
            ``(X_train, Y_train, X_val, Y_val, W_train, W_val)``. ``X_val``
            / ``Y_val`` are ``None`` when no split was performed; ``W_train``
            / ``W_val`` mirror ``W`` (both ``None`` if ``W is None``).
        """
        if val_split <= 0 or val_split >= 1:
            return X, Y, None, None, W, None
        n = X.shape[0]
        n_val = max(1, int(n * val_split))
        g = torch.Generator(device=X.device)
        if seed is not None:
            g.manual_seed(seed)
        idx = torch.randperm(n, generator=g, device=X.device)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        W_train = W[train_idx] if W is not None else None
        W_val = W[val_idx] if W is not None else None
        return (
            X[train_idx], Y[train_idx], X[val_idx], Y[val_idx], W_train, W_val,
        )
