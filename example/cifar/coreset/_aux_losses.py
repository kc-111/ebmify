"""Shared auxiliary-loss helper for coreset training scripts.

Wraps the aux-target tensors produced by ``coreset.cli`` (one ``aux_*.pt``
per algorithm under ``artifacts/<algo>/``) into:

- ``AuxBundle``: per-algorithm container with the on-disk target tensors,
  metadata (``target_dim``, ``loss_kind``, ``per_coord_weights``), and a
  ready-to-use ``nn.ModuleDict`` of linear heads on top of an embedding.
- ``aux_loss_terms(...)``: given the current batch's embedding, the
  heads, the *batch-aligned* target tensors, and the per-aux lambdas,
  returns the weighted total auxiliary loss plus a per-aux dict for
  logging.

Both the supervised (``cifar_train_from_artifacts.py``) and SSL
(``cifar_ssl_train_from_artifacts.py``) scripts pull batch-aligned aux
target rows by *coreset position* ``[0, k)``: in the supervised loop
the position is the permutation index ``b``; in the SSL loop it is
``batch["sample_idx"]`` (added by ``spt.data.FromTorchDataset``).

Aux target families and their loss kinds:

- ``spectral_coords``  -> per-coord weighted MSE regression onto
  ``(n_top_eigvecs,)`` ridge-weighted projections.
- ``bucket_ranks``     -> MSE regression onto ``(n_buckets,)`` uniform
  rank values in ``[0, 1]``.
- ``leverage_score``   -> MSE regression onto a scalar.
- ``home_bucket``      -> ``CrossEntropyLoss`` classification over
  ``n_buckets`` classes.
- ``feature_distill``  -> MSE regression onto ``(D,)`` standardized
  backbone features (target file absent on artifacts built before this
  was added; silently skipped in that case).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


AUX_NAMES = (
    "spectral_coords",
    "bucket_ranks",
    "leverage_score",
    "home_bucket",
    "feature_distill",
)


@dataclass
class AuxSpec:
    """How a single auxiliary target is consumed by the trainer.

    Attributes:
        name: Aux target name (matches the on-disk ``aux_<name>.pt`` stem).
        target_dim: Output dimension of the linear head.
        loss_kind: One of ``"weighted_mse"``, ``"mse"``, ``"ce"``.
        per_coord_weights: Optional ``(target_dim,)`` float tensor of
            non-negative per-coordinate weights (used only for
            ``weighted_mse``).
    """

    name: str
    target_dim: int
    loss_kind: str
    per_coord_weights: torch.Tensor | None = None


@dataclass
class AuxBundle:
    """All aux targets + heads for a single algorithm.

    Attributes:
        targets: Maps aux name -> ``(k, target_dim)`` (or ``(k,)``) tensor
            of per-coreset-sample aux targets, indexed by coreset
            position ``[0, k)``.
        specs: Maps aux name -> :class:`AuxSpec`.
        heads: ``nn.ModuleDict`` with one ``nn.Linear(emb_dim, target_dim)``
            per aux name. Caller is responsible for ``.to(device)``,
            ``.train()`` / ``.eval()``, and adding to the optimizer.
    """

    targets: dict[str, torch.Tensor]
    specs: dict[str, AuxSpec]
    heads: nn.ModuleDict


def discover_aux_targets(algo_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, AuxSpec]]:
    """Load whichever ``aux_*.pt`` files exist under ``algo_dir``.

    Missing files are silently skipped so artifacts built before a new
    target was added still work. ``aux_spectral_coords.pt`` is paired
    with its companion ``aux_spectral_weights.pt`` (used as
    per-coordinate weights in ``weighted_mse``); if the weights file is
    absent the loss falls back to plain MSE.

    Args:
        algo_dir: Path to ``<artifacts>/<algo>/``.

    Returns:
        Tuple of (targets, specs) dicts; both keyed by aux name. May be
        empty if no aux files are present.
    """
    targets: dict[str, torch.Tensor] = {}
    specs: dict[str, AuxSpec] = {}

    sc_path = algo_dir / "aux_spectral_coords.pt"
    if sc_path.exists():
        coords = torch.load(sc_path, weights_only=False).to(torch.float32)
        weights_path = algo_dir / "aux_spectral_weights.pt"
        per_coord = (torch.load(weights_path, weights_only=False).to(torch.float32)
                     if weights_path.exists() else None)
        targets["spectral_coords"] = coords
        specs["spectral_coords"] = AuxSpec(
            name="spectral_coords",
            target_dim=int(coords.shape[1]),
            loss_kind="weighted_mse" if per_coord is not None else "mse",
            per_coord_weights=per_coord,
        )

    br_path = algo_dir / "aux_bucket_ranks.pt"
    if br_path.exists():
        br = torch.load(br_path, weights_only=False).to(torch.float32)
        targets["bucket_ranks"] = br
        specs["bucket_ranks"] = AuxSpec(
            name="bucket_ranks",
            target_dim=int(br.shape[1]),
            loss_kind="mse",
        )

    lv_path = algo_dir / "aux_leverage_score.pt"
    if lv_path.exists():
        lv = torch.load(lv_path, weights_only=False).to(torch.float32).unsqueeze(-1)
        targets["leverage_score"] = lv
        specs["leverage_score"] = AuxSpec(
            name="leverage_score",
            target_dim=1,
            loss_kind="mse",
        )

    hb_path = algo_dir / "aux_home_bucket.pt"
    if hb_path.exists():
        hb = torch.load(hb_path, weights_only=False).to(torch.long)
        n_classes = int(hb.max().item()) + 1
        targets["home_bucket"] = hb
        specs["home_bucket"] = AuxSpec(
            name="home_bucket",
            target_dim=n_classes,
            loss_kind="ce",
        )

    fd_path = algo_dir / "aux_feature_distill.pt"
    if fd_path.exists():
        fd = torch.load(fd_path, weights_only=False).to(torch.float32)
        targets["feature_distill"] = fd
        specs["feature_distill"] = AuxSpec(
            name="feature_distill",
            target_dim=int(fd.shape[1]),
            loss_kind="mse",
        )

    return targets, specs


def build_aux_heads(emb_dim: int, specs: dict[str, AuxSpec]) -> nn.ModuleDict:
    """One ``nn.Linear(emb_dim, target_dim)`` per aux spec, gathered in a ModuleDict."""
    heads = nn.ModuleDict({
        name: nn.Linear(emb_dim, spec.target_dim)
        for name, spec in specs.items()
    })
    return heads


def aux_loss_terms(
    emb: torch.Tensor,
    heads: nn.ModuleDict,
    targets_batch: dict[str, torch.Tensor],
    specs: dict[str, AuxSpec],
    lambdas: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sum-of-weighted aux losses on a batch.

    Args:
        emb: ``(B, emb_dim)`` features.
        heads: ``ModuleDict`` of linear heads keyed by aux name.
        targets_batch: Per-aux ``(B, ...)`` target tensor for *this* batch
            (caller indexes the full ``(k, ...)`` aux tensor by the
            batch's coreset positions).
        specs: Per-aux :class:`AuxSpec`.
        lambdas: Per-aux loss weight. Missing keys or zeros disable that
            aux head (no forward pass through it).

    Returns:
        Tuple of (total_loss, per_aux_log) where ``per_aux_log`` maps the
        aux name to its *unweighted* scalar loss (useful for printing).
        ``total_loss`` is a zero-d tensor on the same device/dtype as
        ``emb`` even if no aux is active.
    """
    total = emb.new_zeros(())
    logs: dict[str, float] = {}
    for name, spec in specs.items():
        lam = float(lambdas.get(name, 0.0))
        if lam <= 0.0 or name not in heads or name not in targets_batch:
            continue
        head = heads[name]
        target = targets_batch[name].to(emb.device, non_blocking=True)
        pred = head(emb)
        if spec.loss_kind == "ce":
            # ``target`` is either a hard label tensor ``(B,)`` of class
            # ids (canonical case) or a soft float tensor ``(B, K)`` of
            # mixed one-hots (Mixup/CutMix turned the hard label into a
            # convex combination of two). Both routes minimize the same
            # cross-entropy objective.
            if target.ndim >= 2 and target.dtype != torch.long:
                loss = -(target.to(pred.dtype) * F.log_softmax(pred, dim=-1)
                         ).sum(dim=-1).mean()
            else:
                loss = F.cross_entropy(pred, target.to(torch.long))
        elif spec.loss_kind == "weighted_mse":
            w = spec.per_coord_weights.to(emb.device)
            loss = (w * (pred - target.to(pred.dtype)).square()).mean()
        else:  # "mse"
            loss = F.mse_loss(pred, target.to(pred.dtype))
        total = total + lam * loss
        logs[name] = float(loss.detach())
    return total, logs


def add_aux_lambda_args(ap: argparse.ArgumentParser) -> None:
    """Adds one ``--aux-<name>`` float flag (default 0.0 = off) per aux target.

    Call after creating the group you want them in. Helps in --help output:
    every aux name shows up with its default in one consistent block.
    """
    helps = {
        "spectral_coords": ("regress top eigvec coords (per-coord ridge-weighted MSE)"),
        "bucket_ranks": "regress per-bucket uniform ranks in [0,1] (MSE)",
        "leverage_score": "regress scalar ridge leverage h_i (MSE)",
        "home_bucket": "classify each sample's home bucket id (cross-entropy)",
        "feature_distill": "regress standardized backbone features phi_i (MSE)",
    }
    for name in AUX_NAMES:
        ap.add_argument(
            f"--aux-{name.replace('_', '-')}",
            type=float, default=0.0,
            dest=f"aux_{name}",
            help=f"loss weight for {name}: {helps[name]} (0=disable head)",
        )


def collect_aux_lambdas(args: argparse.Namespace) -> dict[str, float]:
    """Pulls the per-aux lambdas out of an argparse Namespace as a dict."""
    return {name: float(getattr(args, f"aux_{name}", 0.0)) for name in AUX_NAMES}


def index_targets(
    full_targets: dict[str, torch.Tensor], pos: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """``{name: full[pos]}`` -- ``pos`` indexes coreset position ``[0, k)``."""
    pos_cpu = pos.to("cpu", dtype=torch.long)
    return {name: t.index_select(0, pos_cpu) for name, t in full_targets.items()}
