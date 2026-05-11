"""Configuration dataclasses for the regularized regressors.

These configs are passed to ``LinearModel`` and ``GAM`` to control fitting,
regularization, noise injection / data augmentation, and the input/output
preprocessing pipelines.

Defaults are tuned for "robust without thinking": Adam at ``lr=1e-2`` for 500
epochs, no regularization, no noise, and:

* inputs preprocessed with ``["quantile_gpd"]`` (RQT-GPD: empirical-CDF body
  + GPD tails, output ~ N(0, 1) marginals; handles atoms, heavy tails, and
  outliers without hyperparameter tuning),
* outputs preprocessed with ``["robust", "yeo_johnson"]`` (smooth monotone
  transform; preferred over RQT-GPD for outputs because the inverse is
  smooth/parametric, avoiding GPD-extrapolation noise on the predict path).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._scaler import TransformPipeline


@dataclass
class FitConfig:
    """Optimizer / training-loop settings.

    Args:
        epochs:                     Number of full passes over the training data.
        batch_size:                 Mini-batch size; ``None`` means full-batch.
        lr:                         Learning rate.
        optimizer:                  ``"adam"``, ``"sgd"``, or ``"lbfgs"``.
        val_split:                  Fraction of (X, Y) held out for early stopping.
                                    ``0`` disables validation.
        early_stopping_patience:    Stop after this many epochs without val improvement.
                                    ``None`` disables early stopping.
        n_aug:                      Number of noisy augmentation copies of the
                                    training set per epoch (``>= 1``). ``1`` disables.
        seed:                       RNG seed for reproducibility (torch + numpy).
        verbose:                    Print loss every ~50 epochs if True.
    """

    epochs: int = 500
    batch_size: int | None = None
    lr: float = 1e-2
    optimizer: str = "adam"
    val_split: float = 0.0
    early_stopping_patience: int | None = None
    n_aug: int = 1
    seed: int | None = None
    verbose: bool = False


@dataclass
class RegConfig:
    """Regularization settings.

    Both ``l1`` and ``l2`` are penalty coefficients on the weight matrix
    (excluding the bias). Their effect:

    * ``l1 > 0, l2 == 0``                    -> Lasso.
    * ``l1 == 0, l2 > 0``                    -> Ridge.
    * ``l1 > 0, l2 > 0, l1_then_l2=False``   -> Elastic net.
    * ``l1 > 0, l2 > 0, l1_then_l2=True``    -> Two-phase: fit with L1, prune
                                                weights with ``|w| < l1_select_threshold``,
                                                then re-fit surviving weights with L2 only.

    Args:
        l1:                  L1 (lasso) penalty coefficient.
        l2:                  L2 (ridge) penalty coefficient.
        l1_then_l2:          If True, use the two-phase L1-select-then-L2 procedure.
        l1_select_threshold: Threshold on ``|w|`` below which weights are pruned
                             between phases (only used when ``l1_then_l2`` is True).
        l1_proximal:         If True, apply L1 via proximal gradient (ISTA
                             with Adam as the smooth-step descent) instead
                             of adding ``l1 * |W|.sum()`` to the loss. The
                             "L1 in the loss" path with Adam never produces
                             exact zeros — Adam's running second-moment
                             smooths the subgradient kink at 0, so any
                             ``l1`` large enough to drive ~zero is also
                             large enough to bias the surviving weights
                             heavily. Proximal mode pulls ``l1`` out of the
                             loss, lets Adam descend on the smooth (data +
                             L2) part, then applies soft-thresholding
                             ``W ← sign(W) * relu(|W| - lr·l1)`` after every
                             optimizer step.
                             Note on scale: because Adam's per-step
                             displacement is bounded by ``lr`` (not by the
                             gradient magnitude), the effective threshold
                             is ``lr * l1`` regardless of feature scale —
                             so ``l1`` typically needs to be 10-100× larger
                             than the equivalent "L1 in loss" coefficient
                             to actually drive coordinates to zero. FISTA /
                             Nesterov momentum is intentionally NOT layered
                             on top because the Adam descent step isn't
                             Lipschitz-bounded by ``1/lr`` and the iterate
                             can run away (the ``_ProxFistaState`` helper
                             still supports it for diagnostics, but the
                             default in :meth:`step` is ``fista=False``).
                             Recommended whenever you actually want
                             sparsity (``l1_then_l2`` selection, or pure
                             Lasso fits where exact zeros matter).
    """

    l1: float = 0.0
    l2: float = 0.0
    l1_then_l2: bool = False
    l1_select_threshold: float = 1e-4
    l1_proximal: bool = False


@dataclass
class NoiseConfig:
    """Per-batch noise injection settings.

    All noise is applied in *transformed space* (after the preprocessing
    pipeline), where the data is approximately zero-mean unit-scale, so the
    ``std`` arguments are interpretable as fractions of a standard deviation.

    Args:
        input_additive_std:        ``x -> x + N(0, sigma)`` on transformed inputs.
        input_multiplicative_std:  ``x -> x * (1 + N(0, sigma))`` on transformed inputs.
        output_additive_std:       ``y -> y + N(0, sigma)`` on transformed targets.
        output_multiplicative_std: ``y -> y * (1 + N(0, sigma))`` on transformed targets.
    """

    input_additive_std: float = 0.0
    input_multiplicative_std: float = 0.0
    output_additive_std: float = 0.0
    output_multiplicative_std: float = 0.0


@dataclass
class PreprocessConfig:
    """Input / output preprocessing pipeline.

    ``input_transforms`` and ``output_transforms`` may each be:

    * a list of transform names (composed left -> right; inverse applied
      in reverse order); valid names are
      ``"standard"``, ``"robust"``, ``"minmax"``, ``"yeo_johnson"``, ``"identity"``.
    * a fully constructed ``TransformPipeline`` instance (advanced).

    Defaults:

    * ``input_transforms = ["quantile_gpd"]`` — RQT-GPD maps each input to
      exactly N(0, 1) marginals via empirical CDF body + GPD tails. No
      hyperparameter (auto threshold via Hall 1990 scaling), handles atomic
      mass / spike-and-tail / arbitrarily heavy tails. For shape-sensitive
      models (linear, kernel, distance-based) this is a strict improvement
      over a parametric monotone transform like Yeo-Johnson; for
      shape-invariant adaptive models (GAM, trees) it is statistically
      indistinguishable from ``["robust", "yeo_johnson"]`` (paired t-test
      across 30 seeds: t=1.33, p≈0.19), so this default works for both.

    * ``output_transforms = ["robust", "yeo_johnson"]`` — kept smooth and
      parametric. The inverse runs on every ``predict`` call; YJ's inverse
      is a closed-form analytic function, while RQT-GPD's inverse uses
      empirical interpolation + GPD extrapolation which can amplify noise
      in extrapolated tail regions when the model's predicted z value lands
      outside the body. ``RobustScale`` first puts the target on a stable
      numeric range so the YJ lambda fit isn't warped by output outliers.

    Args:
        input_transforms:  Pipeline applied to inputs.
        output_transforms: Pipeline applied to targets (inverted on prediction).
        minmax_range:      Output range for any ``"minmax"`` step in either pipeline.
        yeo_johnson_criterion: ``"w2"`` (default; minimize Wasserstein-2 distance to
                               N(0,1)) or ``"mle"`` (classical scipy MLE estimator).
        kde_bandwidth_factor: Multiplier on Silverman's bandwidth for any
                              ``"kde_quantile"`` step. ``> 1`` preserves geometric
                              gaps; ``< 1`` pulls the marginal closer to N(0, 1).
                              Default ``1.0`` is plain Silverman.
        n_bins:               If set to ``K >= 2``, switches any ``"quantile_gpd"``
                              step into binned mode: the body PIT output ``u`` is
                              snapped to one of K equiprobable bin midpoints inside
                              ``[q_lo, q_hi]`` (tail values fold into the boundary
                              bins). Only applies to ``"quantile_gpd"``; ``None``
                              (default) leaves the transform continuous.
    """

    input_transforms: list[str] | "TransformPipeline" = field(
        default_factory=lambda: ["quantile_gpd"]
    )
    output_transforms: list[str] | "TransformPipeline" = field(
        default_factory=lambda: ["robust"]
    )
    minmax_range: tuple[float, float] = (-1.0, 1.0)
    yeo_johnson_criterion: str = "w2"
    kde_bandwidth_factor: float = 1.0
    n_bins: int | None = None
