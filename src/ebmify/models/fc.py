"""Fully-connected neural network regressor.

Mirrors the API of ``LinearModel``/``GAM`` (``fit`` / ``predict`` / ``save``
/ ``load_state``) but uses a multilayer perceptron forward pass and applies
L1 / L2 to every linear layer's weight (not just a readout). The primary
regularizer is per-batch input/output Gaussian noise injection
(``NoiseConfig``); weight penalties are an additional knob on top.

Usage:

    from ebmify.models import FCNet, FitConfig, NoiseConfig

    model = FCNet(
        n_inputs=10, n_outputs=2,
        hidden_dims=(128, 64), activation="gelu",
        fit_config=FitConfig(epochs=500, lr=1e-2, seed=0),
        noise_config=NoiseConfig(input_additive_std=0.05,
                                 input_multiplicative_std=0.02),
    )
    model.fit(X, Y)
    Y_pred = model.predict(X_new)  # in original scale

Reuses the preprocessing stack and the train/eval split from the rest of
the package, so the input/output pipelines (``quantile_gpd``, ``robust``,
``yeo_johnson``, ...) and the LBFGS / early-stopping / n_aug behaviors
behave the same as for ``LinearModel``/``GAM``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

from ._base import (
    _ProxFistaState,
    _normalise_sample_weight,
    _resolve_loss,
    _to_tensor,
    _weighted_loss_value,
    feature_leverage,
)
from ._config import FitConfig, NoiseConfig, PreprocessConfig, RegConfig
from ._scaler import TransformPipeline, make_pipeline


class OddPiecewiseReLU(nn.Module):
    """Odd piecewise-linear activation ``g(x) = sign(x) * h(|x|)`` with ReLU ramps.

    On ``t = |x|``, ``h`` is linear between knots ``t_k`` with slopes ``m_k`` on
    ``[0, t_1), [t_1, t_2), ...``. Implemented as ``K`` ReLUs on ``|x|`` (not
    ``2K``). Default knots/slopes match ``example/relu_odd_piecewise_demo.py``.
    """

    def __init__(
        self,
        knots: Sequence[float] | None = None,
        slopes: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        k = (0.25, 0.5, 1.0, 2.0) if knots is None else tuple(float(v) for v in knots)
        s = (1.0, 0.5, 0.25, 0.125, 0.0625) if slopes is None else tuple(float(v) for v in slopes)
        if len(s) != len(k) + 1:
            raise ValueError("slopes must have length len(knots)+1 (one slope per interval).")
        self.register_buffer("knots", torch.tensor(k, dtype=torch.float32))
        self.register_buffer("piecewise_slopes", torch.tensor(s, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        knots = self.knots.to(device=x.device, dtype=x.dtype)
        slopes = self.piecewise_slopes.to(device=x.device, dtype=x.dtype)
        m0 = slopes[0]
        delta = slopes[1:] - slopes[:-1]
        t = torch.abs(x)
        h = m0 * t + (delta * torch.relu(t.unsqueeze(-1) - knots)).sum(dim=-1)
        return torch.sign(x) * h


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "elu": nn.ELU,
    "odd_piecewise": OddPiecewiseReLU,
}


def _resolve_activation(spec: str | type[nn.Module] | Callable[[], nn.Module]) -> type[nn.Module] | Callable[[], nn.Module]:
    if isinstance(spec, str):
        if spec not in _ACTIVATIONS:
            valid = ", ".join(sorted(_ACTIVATIONS))
            raise ValueError(f"Unknown activation {spec!r}. Valid: {valid} or a callable.")
        return _ACTIVATIONS[spec]
    return spec


def _rff_dim_factor(in_dim: int) -> float:
    """``sqrt(d)`` dimensionality factor for RFF length-scale parametrization.

    In d-dim, an isotropic Gaussian frequency draw has ``||omega|| ~ sqrt(d)/ell``,
    so the effective RBF kernel bandwidth scales with input dim if you don't
    correct for it. We absorb that by parametrizing with an effective length
    scale ``ell_eff = length_scale * sqrt(d)`` everywhere, which makes the
    user-facing ``length_scale`` dimensionless: the same numeric value gives
    comparable kernel smoothness whether ``d=2`` or ``d=200``. The median
    heuristic divides the raw median pairwise distance by ``sqrt(d)`` before
    storing, so ``"median"`` mode is unchanged in effect — it still draws
    ``omega ~ N(0, median^{-2} I)`` under the hood.
    """
    return math.sqrt(int(in_dim))


def _normalize_length_scale_spec(
    length_scale: str | float | Sequence[float],
) -> str | tuple[float, ...]:
    """Canonicalize a ``length_scale`` argument to ``"median"`` or a tuple of floats.

    Accepts a positive float, a sequence of positive floats (multi-scale),
    or the string ``"median"``. Returns a hashable canonical form so it
    round-trips through ``save`` / ``load_state`` regardless of whether
    the caller used a list or tuple.
    """
    if isinstance(length_scale, str):
        if length_scale != "median":
            raise ValueError(
                f"length_scale string must be 'median', got {length_scale!r}"
            )
        return "median"
    if isinstance(length_scale, (int, float)) and not isinstance(length_scale, bool):
        v = float(length_scale)
        if v <= 0:
            raise ValueError(f"length_scale must be > 0, got {length_scale}")
        return (v,)
    try:
        vals = tuple(float(v) for v in length_scale)
    except TypeError as exc:
        raise ValueError(
            "length_scale must be a positive float, sequence of positive "
            f"floats, or 'median', got {length_scale!r}"
        ) from exc
    if len(vals) == 0:
        raise ValueError("length_scale sequence must not be empty.")
    if any(v <= 0 for v in vals):
        raise ValueError(f"all length_scale values must be > 0, got {vals}")
    return vals


class RFFLayer(nn.Module):
    """Random Fourier Features layer ``phi(x) = sqrt(2/M) * cos(x @ omega + b)``.

    Frozen ``(omega, b)`` drawn from ``N(0, (ell * sqrt(d))^{-2} I)`` x
    ``U[0, 2*pi]``. By Bochner, this approximates the RBF kernel
    ``k(x, x') = exp(-||x - x'||^2 / (2 * (ell*sqrt(d))^2))`` on the layer's
    input space — whatever space that is (raw inputs, normalized block
    inputs, or trunk activations). The ``sqrt(d)`` dimensionality factor
    (see :func:`_rff_dim_factor`) makes ``length_scale`` comparable across
    input dims.

    The bandwidth ``ell`` is set by:

    * a positive float passed at construction (resolved immediately);
    * a sequence of K positive floats — multi-scale RFF: ``n_features``
      columns are partitioned into K contiguous groups of sizes
      ``n_features // K`` (with the first ``n_features % K`` groups getting
      one extra column), and group ``k`` draws ``omega`` columns from
      ``N(0, (ell_k * sqrt(d))^{-2} I)``. The resulting kernel is the
      uniform mixture ``(1/K) * sum_k k_{ell_k}`` — useful when no single
      bandwidth captures the relevant scales of the target;
    * ``"median"`` plus a calibration sample via :meth:`init_bandwidth`
      (resolved from the median pairwise distance of the calibration data,
      then divided by ``sqrt(d)`` to give the dim-normalized value).
      Single-scale only.

    ``omega``, ``phase``, and ``length_scale`` are buffers — frozen, no grad,
    persisted in ``state_dict`` so the projection round-trips through
    ``save`` / ``load_state``. ``length_scale`` is shape ``[K]``;
    ``feature_scale_idx`` (non-persistent, derived from ``n_features`` and
    ``K``) records which scale each output column belongs to.
    """

    def __init__(
        self,
        in_dim: int,
        n_features: int,
        length_scale: str | float | Sequence[float] = "median",
        rff_seed: int = 0,
        median_pairs: int = 1000,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be > 0, got {in_dim}")
        if n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}")
        spec = _normalize_length_scale_spec(length_scale)
        n_scales = 1 if spec == "median" else len(spec)
        if n_features < n_scales:
            raise ValueError(
                f"n_features ({n_features}) must be >= number of length "
                f"scales ({n_scales})"
            )
        self.in_dim = int(in_dim)
        self.n_features = int(n_features)
        self.length_scale_spec = spec
        self.rff_seed = int(rff_seed)
        self.median_pairs = int(median_pairs)
        self.register_buffer(
            "feature_scale_idx",
            self._partition(self.n_features, n_scales),
            persistent=False,
        )
        self.register_buffer("omega", torch.zeros(self.in_dim, self.n_features))
        self.register_buffer("phase", torch.zeros(self.n_features))
        # All-zero length_scale means uninitialized; all-positive means resolved.
        self.register_buffer("length_scale", torch.zeros(n_scales))
        if spec != "median":
            self._sample(spec)

    @staticmethod
    def _partition(n_features: int, n_scales: int) -> torch.Tensor:
        """Map each output column to its length-scale index (contiguous groups)."""
        base = n_features // n_scales
        rem = n_features % n_scales
        idx = torch.empty(n_features, dtype=torch.long)
        offset = 0
        for k in range(n_scales):
            size = base + (1 if k < rem else 0)
            idx[offset : offset + size] = k
            offset += size
        return idx

    def is_initialized(self) -> bool:
        return bool(torch.all(self.length_scale > 0).item())

    def _sample(self, ells: Sequence[float]) -> None:
        ells_t = torch.as_tensor(
            [float(e) for e in ells],
            dtype=self.length_scale.dtype,
            device=self.length_scale.device,
        )
        if ells_t.numel() != self.length_scale.numel():
            raise ValueError(
                f"expected {self.length_scale.numel()} length scales, got "
                f"{ells_t.numel()}"
            )
        if (ells_t <= 0).any():
            raise ValueError(f"all length_scale values must be > 0, got {tuple(ells)}")
        device = self.omega.device
        dtype = self.omega.dtype
        # Effective bandwidth = ell * sqrt(d). See _rff_dim_factor.
        ell_eff = (ells_t * _rff_dim_factor(self.in_dim)).to(device=device, dtype=dtype)
        per_feature_ell_eff = ell_eff[self.feature_scale_idx.to(device)]
        g = torch.Generator(device=device)
        g.manual_seed(self.rff_seed)
        omega = torch.randn(
            self.in_dim, self.n_features,
            generator=g, device=device, dtype=dtype,
        ) / per_feature_ell_eff[None, :]
        phase = torch.rand(
            self.n_features,
            generator=g, device=device, dtype=dtype,
        ) * (2.0 * math.pi)
        self.omega.copy_(omega)
        self.phase.copy_(phase)
        self.length_scale.copy_(ells_t)

    @torch.no_grad()
    def init_bandwidth(self, X: torch.Tensor) -> None:
        """Resolve ``length_scale`` from a calibration sample if ``"median"``.

        No-op when the layer is already initialized (numeric ``length_scale``
        passed at construction, or a previous ``init_bandwidth`` call).
        ``"median"`` is single-scale only — pass an explicit sequence of
        floats to use a multi-scale RFF.
        """
        if self.is_initialized():
            return
        if self.length_scale_spec != "median":
            raise RuntimeError(
                "RFFLayer not initialized but length_scale_spec is not 'median' "
                f"(got {self.length_scale_spec!r})."
            )
        n = int(X.shape[0])
        if n < 2:
            ell = 1.0
        else:
            g = torch.Generator(device=X.device)
            g.manual_seed(self.rff_seed)
            i = torch.randint(0, n, (self.median_pairs,), generator=g, device=X.device)
            j = torch.randint(0, n, (self.median_pairs,), generator=g, device=X.device)
            mask = i != j
            if int(mask.sum().item()) == 0:
                ell = 1.0
            else:
                d = torch.linalg.norm(X[i[mask]] - X[j[mask]], dim=1)
                med = float(d.median().item())
                # Divide raw median by sqrt(d): the stored value is the
                # dimension-normalized length_scale; _sample re-applies sqrt(d).
                # Net effect: omega ~ N(0, median^{-2} I), same as classic
                # median-heuristic RFF.
                ell = (med / _rff_dim_factor(self.in_dim)) if med > 0 else 1.0
        self._sample((ell,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_initialized():
            raise RuntimeError(
                "RFFLayer not initialized. Pass a numeric length_scale at "
                "construction, or call init_bandwidth(X) before use."
            )
        proj = x @ self.omega + self.phase
        scale = math.sqrt(2.0 / float(self.n_features))
        return scale * torch.cos(proj)


class _ResidualMLPBlock(nn.Module):
    """Pre-norm residual block that can change width.
    RMSNorm is required to deal with x magnitudes possibly growing
    bigger after each layer.

    f(x) = Linear2( act( Linear1( LN(x) ) ) )
    y    = skip(x) + f(x)

    If ``in_dim != out_dim``, the skip path uses a learned projection.
    """

    def __init__(self, in_dim: int, out_dim: int, act_factory: Callable[[], nn.Module]) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        # self.ln = nn.LayerNorm(self.in_dim)
        self.ln = nn.RMSNorm(self.in_dim)
        # self.dp = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.in_dim, self.out_dim)
        self.act = act_factory()
        # self.act = OddPiecewiseReLU()
        self.fc2 = nn.Linear(self.out_dim, self.out_dim)
        self.skip = nn.Identity() if self.in_dim == self.out_dim else nn.Linear(self.in_dim, self.out_dim)
        # self.skip = nn.Linear(self.in_dim, self.out_dim)
        # self.beta = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = self.fc2(self.act(self.dp(self.fc1(self.ln(x)))))
        y = self.fc2(self.act(self.fc1(self.ln(x))))
        # y = self.fc2(self.act(self.fc1(x)))
        # beta = torch.sigmoid(self.beta)
        # beta = self.beta
        # y = self.fc2(self.act(self.fc1(beta*self.act(x))))
        return self.skip(x) + y
        # return self.ln(self.skip(x) + y)


class _ResidualRFFBlock(nn.Module):
    """Pre-norm residual block with an RFF inner nonlinearity.

    ``f(x) = W2 @ rff(LN(x)) + b2``, ``y = skip(x) + f(x)`` where
    ``rff(z) = sqrt(2/M) * cos(Omega @ z + b)`` with frozen ``(Omega, b)``.
    By Bochner this is kernel ridge on the LN-normalized block input,
    with ``W2`` as the learnable readout. If ``in_dim != out_dim``, the
    skip path uses a learned linear projection.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_features: int,
        length_scale: str | float | Sequence[float] = "median",
        rff_seed: int = 0,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.ln = nn.RMSNorm(self.in_dim)
        self.rff = RFFLayer(
            in_dim=self.in_dim, n_features=int(n_features),
            length_scale=length_scale, rff_seed=int(rff_seed),
        )
        self.fc2 = nn.Linear(int(n_features), self.out_dim)
        self.skip = (
            nn.Identity() if self.in_dim == self.out_dim
            else nn.Linear(self.in_dim, self.out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc2(self.rff(self.ln(x)))
        return self.skip(x) + y


class _ResidualMLP(nn.Module):
    """Residual MLP trunk + linear readout, with optional RFF input lift,
    output pre-readout RFF, and RFF residual blocks.

    Input/output RFFs are **concatenated** with the surrounding raw
    features — they augment, they don't replace:

    * input lift: ``in_proj`` sees ``[x ; cos(Omega_in @ x + b_in)]``,
      width ``n_inputs + M_in``;
    * output RFF: ``readout`` sees ``[h_trunk ; cos(Omega_out @ h_trunk + b_out)]``,
      width ``hidden_dims[-1] + M_out``.

    Concatenation preserves the original linear identity path through both
    layers, so if the RFF features don't help on a region the model can
    fall back to the raw representation. Last-layer leverage is then
    computed on the concatenated feature space, which inherits boundedness
    on the RFF half and unboundedness on the linear half — typically the
    dominant signal in the leverage diagonal comes from the bounded RFF
    components, which is exactly the OOD-tameness property we want.

    Forward: ``x -> [x; rff_in(x)] -> in_proj -> blocks -> [h; rff_out(h)] -> readout``
    (with the RFF concatenations skipped if the corresponding ``RFFLayer``
    is ``None``). The trunk (everything before ``readout``) is what
    :meth:`trunk` returns, which is the feature space last-layer Bayesian
    uncertainty operates on.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        hidden_dims: Sequence[int],
        act_factory: Callable[[], nn.Module],
        input_rff: "RFFLayer | None" = None,
        output_rff: "RFFLayer | None" = None,
        block_type: str = "linear",
        block_rff_features: int | None = None,
        block_rff_length_scale: str | float | Sequence[float] = "median",
        block_rff_seed_base: int = 2,
    ) -> None:
        super().__init__()
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one width.")
        if block_type not in ("linear", "rff"):
            raise ValueError(f"block_type must be 'linear' or 'rff', got {block_type!r}")
        if block_type == "rff" and block_rff_features is None:
            raise ValueError("block_type='rff' requires block_rff_features.")
        self.n_inputs = int(n_inputs)
        self.n_outputs = int(n_outputs)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)

        self.input_rff = input_rff
        in_proj_in = (
            self.n_inputs + input_rff.n_features
            if input_rff is not None
            else self.n_inputs
        )
        self.in_proj = nn.Linear(in_proj_in, self.hidden_dims[0])

        if block_type == "linear":
            self.blocks = nn.ModuleList(
                [
                    _ResidualMLPBlock(self.hidden_dims[i], self.hidden_dims[i + 1], act_factory)
                    for i in range(len(self.hidden_dims) - 1)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    _ResidualRFFBlock(
                        in_dim=self.hidden_dims[i],
                        out_dim=self.hidden_dims[i + 1],
                        n_features=int(block_rff_features),
                        length_scale=block_rff_length_scale,
                        rff_seed=block_rff_seed_base + i,
                    )
                    for i in range(len(self.hidden_dims) - 1)
                ]
            )

        self.output_rff = output_rff
        readout_in = (
            self.hidden_dims[-1] + output_rff.n_features
            if output_rff is not None
            else self.hidden_dims[-1]
        )
        self.readout = nn.Linear(readout_in, self.n_outputs)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_rff is not None:
            h = torch.cat([x, self.input_rff(x)], dim=-1)
        else:
            h = x
        h = self.in_proj(h)
        for b in self.blocks:
            h = b(h)
        if self.output_rff is not None:
            h = torch.cat([h, self.output_rff(h)], dim=-1)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.trunk(x))


class FCNet(nn.Module):
    """Multilayer perceptron regressor with the ebmify training stack.

    Architecture: residual MLP with pre-norm LayerNorm blocks. Residual
    connections are better for uncertainty estimation because activations
    might zero out predictions or features, which would lead to confident
    estimates on OOD data.

    Highly recommended to use L1 and L2 (default 1e-3 for both) together
    for good boundaries and OOD detection.

    Concretely: ``in_proj`` maps inputs to ``hidden_dims[0]``, then we apply
    residual blocks that can change width (a learned skip projection is used
    when widths differ), then a final linear readout to ``n_outputs``.

    Args:
        n_inputs:        Number of input features.
        n_outputs:       Number of output targets.
        hidden_dims:     Hidden layer widths (default ``(128, 64)``).
        activation:      Activation key in ``_ACTIVATIONS`` or a zero-arg
                         callable returning an ``nn.Module`` (default
                         ``"odd_piecewise"``).
        fit_config:      Training-loop settings.
        reg_config:      L1 / L2 weight penalties (applied to every Linear
                         layer's weight, not biases).
        noise_config:    Per-batch input/output noise injection (the
                         primary regularizer).
        preprocess:      Input/output preprocessing pipelines.
        loss:            ``"mse"`` | ``"mae"`` | ``"huber"`` or a callable.
        device:          Torch device.
        input_rff:       If set, augment the input with a frozen RFF lift
                         of this many features. ``in_proj`` then sees
                         ``[x ; cos(Omega_in @ x + b_in)]`` (width
                         ``n_inputs + input_rff``), so the linear identity
                         path through ``x`` is preserved alongside the
                         spectral lift (``None`` disables).
        input_rff_length_scale: Bandwidth for the input lift. ``"median"``
                         (default) calibrates from preprocessed training
                         inputs at fit time; a positive float resolves
                         immediately at construction. May also be a
                         sequence of K positive floats for a multi-scale
                         RFF — the ``input_rff`` columns are partitioned
                         evenly across the K scales (with the first
                         ``input_rff % K`` groups getting one extra
                         column), and each group's omega is sampled at
                         its own bandwidth. ``"median"`` is single-scale
                         only.
        output_rff:      If set, augment the trunk activations with a
                         frozen RFF map of this many features. ``readout``
                         then sees ``[h_trunk ; cos(Omega_out @ h_trunk + b_out)]``
                         (width ``hidden_dims[-1] + output_rff``), and
                         ``model.features(X)`` returns this concatenation.
                         Last-layer leverage on those features inherits
                         the bounded RFF half's OOD-tameness while keeping
                         the unbounded linear half available where it
                         helps (``None`` disables).
        output_rff_length_scale: Bandwidth for the output RFF, calibrated
                         on the pre-concat trunk activations at fit time
                         when ``"median"``. Also accepts a sequence of
                         positive floats for a multi-scale RFF (see
                         ``input_rff_length_scale``).
        block_type:      ``"linear"`` (default) for the standard
                         ``W2 @ act(W1 @ LN(x) + b1) + b2`` residual block,
                         or ``"rff"`` to replace the inner nonlinearity
                         with a frozen RFF map (``W2 @ rff(LN(x)) + b2``).
        block_rff_features: Required when ``block_type="rff"`` — RFF width
                         used inside each residual block.
        block_rff_length_scale: Bandwidth for in-block RFF maps,
                         calibrated per block on its LN-normalized input
                         when ``"median"``. Also accepts a sequence of
                         positive floats for a multi-scale RFF (see
                         ``input_rff_length_scale``).
        rff_seed:        Base RNG seed for RFF projections. The input lift
                         uses ``rff_seed``, the output RFF uses
                         ``rff_seed+1``, and block-i RFF uses
                         ``rff_seed+2+i`` so they're independent.

    The ``sqrt(d)`` dimensionality factor (see :func:`_rff_dim_factor`)
    makes ``length_scale`` dimension-normalized: the same numeric value
    gives comparable kernel smoothness whether ``d=2`` or ``d=200``.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        hidden_dims: Sequence[int] = (128, 64),
        activation: str | Callable[[], nn.Module] = "odd_piecewise",
        fit_config: FitConfig | None = None,
        reg_config: RegConfig | None = None,
        noise_config: NoiseConfig | None = None,
        preprocess: PreprocessConfig | None = None,
        loss: str | Callable = "mse",
        device: str | torch.device = "cpu",
        *,
        input_rff: int | None = None,
        input_rff_length_scale: str | float | Sequence[float] = "median",
        output_rff: int | None = None,
        output_rff_length_scale: str | float | Sequence[float] = "median",
        block_type: str = "linear",
        block_rff_features: int | None = None,
        block_rff_length_scale: str | float | Sequence[float] = "median",
        rff_seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_dims: tuple[int, ...] = tuple(int(h) for h in hidden_dims)
        self.fit_config = fit_config or FitConfig()
        self.reg_config = reg_config or RegConfig()
        self.noise_config = noise_config or NoiseConfig()
        self.preprocess = preprocess or PreprocessConfig()
        self.loss_spec = loss
        self.activation_spec = activation
        self.device = torch.device(device) if isinstance(device, str) else device

        # Stash RFF config so save() can dump it and load_state() can verify.
        # Length-scale specs are canonicalized so list/tuple inputs compare equal.
        self._rff_config: dict = {
            "input_rff": None if input_rff is None else int(input_rff),
            "input_rff_length_scale": _normalize_length_scale_spec(
                input_rff_length_scale
            ),
            "output_rff": None if output_rff is None else int(output_rff),
            "output_rff_length_scale": _normalize_length_scale_spec(
                output_rff_length_scale
            ),
            "block_type": block_type,
            "block_rff_features": (
                None if block_rff_features is None else int(block_rff_features)
            ),
            "block_rff_length_scale": _normalize_length_scale_spec(
                block_rff_length_scale
            ),
            "rff_seed": int(rff_seed),
        }

        act_factory = _resolve_activation(activation)
        if callable(act_factory):
            act_ctor: Callable[[], nn.Module] = act_factory  # type: ignore[assignment]
        else:
            act_ctor = act_factory  # type: ignore[assignment]

        input_rff_layer: RFFLayer | None = None
        if input_rff is not None:
            input_rff_layer = RFFLayer(
                in_dim=n_inputs, n_features=int(input_rff),
                length_scale=input_rff_length_scale, rff_seed=int(rff_seed),
            )
        output_rff_layer: RFFLayer | None = None
        if output_rff is not None:
            output_rff_layer = RFFLayer(
                in_dim=self.hidden_dims[-1], n_features=int(output_rff),
                length_scale=output_rff_length_scale, rff_seed=int(rff_seed) + 1,
            )

        self.net = _ResidualMLP(
            n_inputs=n_inputs, n_outputs=n_outputs,
            hidden_dims=self.hidden_dims, act_factory=act_ctor,
            input_rff=input_rff_layer, output_rff=output_rff_layer,
            block_type=block_type,
            block_rff_features=block_rff_features,
            block_rff_length_scale=block_rff_length_scale,
            block_rff_seed_base=int(rff_seed) + 2,
        )

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

        self._fitted = False
        self.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, Y, sample_weight=None) -> "FCNet":
        """Fit preprocessing pipelines, then train the network end-to-end.

        Args:
            X: Input matrix ``[N, n_inputs]`` (numpy or torch).
            Y: Target matrix ``[N, n_outputs]`` (numpy or torch).
            sample_weight: Optional loss weights. Pass ``[N]`` (or ``[N, 1]``)
                for per-sample weights shared across all output columns; pass
                ``[N, n_outputs]`` for per-target weights (e.g.
                ``SampleSet.to_tensors`` / ``to_matrix`` returns this shape,
                with one weight per (sample, target column) reflecting that
                target's label-overlap concurrency). 1-D inputs are
                normalised to global mean 1.0; 2-D inputs are normalised
                per-column to mean 1.0 — different target columns have
                different concurrency scales and must not be collapsed
                together. Mean-1 normalisation keeps the loss magnitude
                comparable to the unweighted case so previously-tuned L1/L2
                coefficients transfer without re-tuning. ``None`` keeps the
                unweighted loss.

        Returns:
            ``self`` (chainable).
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

        with torch.no_grad():
            self.input_pipeline.fit(X_t)
            self.output_pipeline.fit(Y_t)
            self._init_rff_bandwidths(X_t)

        self._train(
            X_t, Y_t, l1=self.reg_config.l1, l2=self.reg_config.l2,
            sample_weight=sw_t,
        )
        self._fitted = True
        return self

    @torch.no_grad()
    def _init_rff_bandwidths(self, X_raw: torch.Tensor) -> None:
        """Calibrate any ``"median"`` RFF bandwidths from training data.

        Numeric ``length_scale`` was already resolved at construction; this
        only fires for layers that asked for ``"median"``. Layers are
        calibrated *in order* — input lift on preprocessed inputs, each
        block's RFF on its own LN-normalized input, output RFF on the
        post-block trunk activations (the pre-concat trunk output, so the
        bandwidth reflects the space the RFF projection actually sees) —
        so each layer's median is taken on the actual space it will
        operate on at training time.
        """
        was_training = self.training
        self.eval()
        try:
            X_proc = self.input_pipeline(X_raw)
            net = self.net
            if net.input_rff is not None:
                net.input_rff.init_bandwidth(X_proc)
            # Mirror trunk(): concat raw + input-rff features before in_proj.
            if net.input_rff is not None:
                h = torch.cat([X_proc, net.input_rff(X_proc)], dim=-1)
            else:
                h = X_proc
            h = net.in_proj(h)
            for blk in net.blocks:
                if isinstance(blk, _ResidualRFFBlock):
                    blk.rff.init_bandwidth(blk.ln(h))
                h = blk(h)
            if net.output_rff is not None:
                # Calibrate on the pre-concat trunk activations: that's the
                # input space the RFF projection itself sees in trunk().
                net.output_rff.init_bandwidth(h)
        finally:
            if was_training:
                self.train()

    def predict(self, X) -> torch.Tensor:
        """Predict targets in the original (un-transformed) output scale."""
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call .fit(X, Y) first.")
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                X_t = _to_tensor(X, self.device)
                X_proc = self.input_pipeline(X_t)
                Y_proc = self.net(X_proc)
                Y = self.output_pipeline.inverse(Y_proc)
        finally:
            if was_training:
                self.train()
        return Y

    def features(self, X) -> torch.Tensor:
        """Return the penultimate-layer activations -- the features that
        feed the final linear readout.

        Treating the readout as a Bayesian linear regression on these
        features is the standard "deep GP via last-layer linearization"
        recipe. Useful for leverage-based uncertainty:
        ``h(x*) = phi(x*)^T (Phi^T Phi + lam I)^{-1} phi(x*)``. NOTE:
        the readout has a bias by default, so when computing leverage
        externally you must augment ``Phi`` with an all-ones column to
        cover the ``+ b`` term — or just call :meth:`feature_leverage`
        (or the module-level :func:`feature_leverage`) which handle
        that for you.

        Returns:
            ``[N, last_hidden_dim]`` features in the trained feature space.

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
                phi = self.net.trunk(X_proc)
        finally:
            if was_training:
                self.train()
        return phi

    def feature_leverage(
        self,
        X_train,
        X_query,
        ridge: float = 1e-3,
        *,
        bias: bool = True,
    ) -> torch.Tensor:
        """Last-layer leverage / Bayesian-linear posterior variance scaling.

        Mirrors :meth:`_RegularizedRegressor.feature_leverage` so the same
        recipe works uniformly across model classes. Returns the diagonal
        of ``phi(x*)^T (Phi_train^T Phi_train + r I)^{-1} phi(x*)`` over
        the trunk features ``phi(x) = self.features(x)``.

        With ``bias=True`` (the default) the formula is augmented with an
        all-ones column so it covers the readout's bias term as well as
        its weight matrix. The readout is constructed as
        ``nn.Linear(..., bias=True)``, so the bias-aware variant is what
        you want unless you have a specific reason to ignore it.

        Args:
            X_train: Training inputs ``[N, n_inputs]``.
            X_query: Query inputs ``[M, n_inputs]``.
            ridge:   Diagonal load on ``Phi^T Phi``.
            bias:    Augment with an all-ones column for the readout's
                     bias term. Defaults to ``True``.

        Returns:
            ``[M]`` non-negative tensor.

        Raises:
            RuntimeError: if called before ``fit``.
        """
        Phi_train = self.features(X_train)
        Phi_query = self.features(X_query)
        return feature_leverage(Phi_train, Phi_query, ridge, bias=bias)

    def save(self, path: str | Path) -> None:
        """Serialize architecture + state to disk."""
        torch.save(
            {
                "n_inputs": self.n_inputs,
                "n_outputs": self.n_outputs,
                "hidden_dims": list(self.hidden_dims),
                "rff_config": dict(self._rff_config),
                "fitted": self._fitted,
                "state_dict": self.state_dict(),
            },
            path,
        )

    def load_state(self, path: str | Path) -> "FCNet":
        """Restore weights from a file produced by ``save``.

        The receiver must already be constructed with the same
        ``n_inputs`` / ``n_outputs`` / ``hidden_dims`` / activation, as
        well as matching RFF kwargs (``input_rff``, ``output_rff``,
        ``block_type``, etc.). RFF buffers (``omega``, ``phase``,
        ``length_scale``) round-trip through ``state_dict``, so the
        calibrated projections are exactly preserved.
        """
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            # weights_only=True can reject benign primitive dicts on some
            # torch versions. Fall back to the unsafe loader for files we
            # produced ourselves.
            state = torch.load(path, map_location=self.device, weights_only=False)
        saved_rff = state.get("rff_config")
        if saved_rff is not None:
            saved_norm = dict(saved_rff)
            for key in (
                "input_rff_length_scale",
                "output_rff_length_scale",
                "block_rff_length_scale",
            ):
                if key in saved_norm:
                    saved_norm[key] = _normalize_length_scale_spec(saved_norm[key])
            if saved_norm != self._rff_config:
                raise ValueError(
                    "RFF config mismatch between saved file and receiver:\n"
                    f"  saved   = {saved_norm}\n"
                    f"  receiver= {self._rff_config}\n"
                    "Reconstruct FCNet with the same RFF kwargs before load_state()."
                )
        self.load_state_dict(state["state_dict"])
        self._fitted = state.get("fitted", False)
        return self

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _linear_weights(self) -> list[torch.Tensor]:
        """All Linear-layer ``weight`` tensors (used by L1/L2 penalties)."""
        return [m.weight for m in self.net.modules() if isinstance(m, nn.Linear)]

    @staticmethod
    def _inject(x: torch.Tensor, add_std: float, mul_std: float) -> torch.Tensor:
        """Inject additive + multiplicative Gaussian noise (no-op if both 0)."""
        if add_std > 0:
            x = x + add_std * torch.randn_like(x)
        if mul_std > 0:
            x = x * (1.0 + mul_std * torch.randn_like(x))
        return x

    def _preprocess_eval(
        self, X_raw: torch.Tensor, Y_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run input + output pipelines in eval mode (deterministic) under no_grad."""
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
        n = X.shape[0]
        bs = batch_size or n
        idx = torch.randperm(n, device=X.device)
        for start in range(0, n, bs):
            sl = idx[start : start + bs]
            if W is None:
                yield X[sl], Y[sl]
            else:
                yield X[sl], Y[sl], W[sl]

    def _reg_term(
        self, l1: float, l2: float, *, include_l1: bool = True,
    ) -> torch.Tensor | float:
        """Sum of L1/L2 penalties over every Linear-layer weight.

        Set ``include_l1=False`` to skip the L1 term (proximal-Adam mode
        moves L1 out of the loss and into a post-step soft-threshold).
        """
        l1_eff = l1 if include_l1 else 0.0
        if l1_eff <= 0 and l2 <= 0:
            return 0.0
        weights = self._linear_weights()
        total: torch.Tensor | float = 0.0
        for w in weights:
            if l1_eff > 0:
                total = total + l1_eff * w.abs().sum()
            if l2 > 0:
                total = total + l2 * (w ** 2).sum()
        return total

    def _train(
        self,
        X_raw: torch.Tensor,
        Y_raw: torch.Tensor,
        l1: float,
        l2: float,
        sample_weight: torch.Tensor | None = None,
    ) -> None:
        """One full training phase (Adam/SGD per-batch, or LBFGS full-batch).

        Three preprocessing strategies, same as the linear-in-features base:

        * **Deterministic SGD / Adam path** -- when neither pipeline is
          stochastic in train mode (e.g. ``KDEQuantile``,
          ``RandomizedQuantileGPD(randomize_ties=False)``, all parametric
          transforms), the entire training set is preprocessed *once* in eval
          mode and the cached tensor is sliced per batch. KDE is no longer
          paid per step.
        * **Stochastic SGD / Adam path** -- when any pipeline is stochastic
          (``RandomizedQuantileGPD(randomize_ties=True)``), each minibatch
          step preprocesses fresh in train mode so the optimizer integrates
          over the rank ambiguity at the standard data-augmentation cadence.
        * **LBFGS path** -- preprocesses once up front in eval mode (line
          search requires a stable objective) and skips noise injection.
        """
        fc = self.fit_config
        nc = self.noise_config
        rc = self.reg_config
        base_loss = _resolve_loss(self.loss_spec)
        optimizer = self._make_optimizer()

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
        # Proximal-Adam / FISTA L1 over every Linear layer's weight (biases
        # are not penalized). Tracks one shared FISTA momentum scalar
        # across all weight tensors.
        use_proximal = bool(rc.l1_proximal) and l1 > 0 and not is_lbfgs
        prox_state = (
            _ProxFistaState(self._linear_weights()) if use_proximal else None
        )
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
                        def closure():
                            optimizer.zero_grad()
                            pred = self.net(batch_X)
                            loss = _weighted_loss_value(
                                self.loss_spec, base_loss, pred, batch_Y, batch_W,
                            ) + self._reg_term(l1, l2)
                            loss.backward()
                            return loss

                        optimizer.step(closure)
                    continue

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

                    pred = self.net(x)
                    loss = _weighted_loss_value(
                        self.loss_spec, base_loss, pred, y, b_W,
                    ) + self._reg_term(l1, l2, include_l1=not use_proximal)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if prox_state is not None:
                        prox_state.step(fc.lr, l1)

            if do_es:
                self.eval()
                with torch.no_grad():
                    pred_v = self.net(X_val)
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

            if fc.verbose and (epoch + 1) % 50 == 0:
                X_train_eval, Y_train_eval = self._preprocess_eval(X_train_raw, Y_train_raw)
                with torch.no_grad():
                    pred_t = self.net(X_train_eval)
                    train_loss = float(_weighted_loss_value(
                        self.loss_spec, base_loss, pred_t, Y_train_eval, sw_train,
                    ).item())
                print(f"  epoch {epoch + 1}: train_loss={train_loss:.4e}")

        # Replace each weight's FISTA look-ahead with the prox-clean iterate.
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
