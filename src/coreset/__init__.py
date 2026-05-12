"""Coreset selection from feature matrices.

The library is feature-agnostic: callers stream a ``(N, D)`` feature
matrix ``Phi`` through a :class:`StandardizedView`, then run one or more
selection algorithms. Feature extraction itself lives outside this
package (see ``example/cifar/coreset/`` for the CIFAR pipelines).

Algorithms
----------
- :func:`greedy_max_variance` -- maintains ``A_inv`` via Sherman-Morrison
  and picks the largest-leverage sample each iteration.
- :func:`ridge_leverage_sample` -- importance samples by ridge leverage
  (mixed uniform tail or Bernoulli-without-replacement).
- :func:`spectral_rank_coverage` -- stratified per-bucket coverage using
  rank-in-bucket targets.

Public surface
--------------
::

    from coreset import (
        StandardizedView, fit_view,
        compute_eig,
        greedy_max_variance,
        ridge_leverage_sample,
        spectral_rank_coverage,
        compute_aux_targets,
    )

The CLI orchestrator (``python -m coreset.cli``) wires preprocess -> eig
-> algorithms -> aux targets and persists artifacts to disk.
"""

from coreset.preprocess import StandardizedView, fit_view
from coreset.eig import compute_eig
from coreset.greedy import greedy_max_variance
from coreset.leverage import ridge_leverage_sample
from coreset.spectral_rank import spectral_rank_coverage
from coreset.aux_targets import compute_aux_targets

__all__ = [
    "StandardizedView",
    "fit_view",
    "compute_eig",
    "greedy_max_variance",
    "ridge_leverage_sample",
    "spectral_rank_coverage",
    "compute_aux_targets",
]
