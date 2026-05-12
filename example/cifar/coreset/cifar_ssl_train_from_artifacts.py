"""SSL pretrain LeJEPA-recon on coresets produced by ``coreset.cli``.

Reads ``<artifacts_root>/<algo>/indices.pt`` for each algorithm and runs
one LeJEPA-recon SSL pretraining per algorithm, recording the best
validation linear-probe top-1 and kNN accuracy reported by
``spt.callbacks.OnlineProbe`` / ``OnlineKNN`` (same probes that
``ssl_pretrain_recon.py`` uses).

Outputs land in ``example/out/coreset/`` and include a JSON results file
plus one bar plot per metric.

Usage:
    python example/cifar/coreset/cifar_build_coreset_ssl.py
    python example/cifar/coreset/cifar_ssl_train_from_artifacts.py \\
        --artifacts example/cifar/cache/coreset/ssl_resnet18_recon \\
        --epochs 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Subset

import stable_pretraining as spt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import REPO_ROOT  # noqa: F401, E402

from ebmify.models.losses import make_regularizer  # noqa: E402
from ssl_pretrain_recon import (  # noqa: E402
    EMB_DIM,
    ConvDecoder,
    _make_projector,
    _make_train_tf,
    _make_val_tf,
    forward,
)

from _artifacts import (  # noqa: E402
    artifacts_help, default_artifacts, resolve_artifacts,
)
from _aux_losses import (  # noqa: E402
    aux_loss_terms, build_aux_heads, collect_aux_lambdas,
    discover_aux_targets, index_targets,
)

ALGO_CHOICES = ["greedy", "leverage", "spectral_rank"]
DEFAULT_TAG = "ssl_resnet18_recon"  # written by cifar_build_coreset_ssl.py


class _Fmt(argparse.ArgumentDefaultsHelpFormatter,
           argparse.RawDescriptionHelpFormatter):
    """Keeps the docstring intact + always shows '(default: X)' inline,
    even for flags without an explicit help string."""

    def _get_help_string(self, action):
        h = action.help or ""
        if (action.default is not argparse.SUPPRESS
                and action.default is not None
                and "%(default)" not in h
                and not action.required):
            h = (h + " " if h else "") + "(default: %(default)s)"
        return h


_ALGO_STYLE = {
    "greedy":        ("C3", "Greedy max-variance"),
    "leverage":      ("C0", "Ridge leverage sample"),
    "spectral_rank": ("C2", "Spectral-rank coverage"),
}

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


class CaptureBestMetrics(Callback):
    """Track running max of val linear-probe top-1 and kNN accuracy."""

    def __init__(self) -> None:
        super().__init__()
        self.best_top1 = 0.0
        self.best_knn = 0.0

    def on_validation_end(self, trainer, pl_module) -> None:
        m = trainer.callback_metrics
        top1 = m.get("eval/linear_probe_top1_epoch", None)
        knn = m.get("eval/knn_probe_accuracy", None)
        if top1 is not None:
            self.best_top1 = max(self.best_top1, float(top1))
        if knn is not None:
            self.best_knn = max(self.best_knn, float(knn))


def _make_subset_dataloaders(subset_idx: np.ndarray, *, batch_size: int,
                             num_workers: int):
    """Build train/val dataloaders over a CIFAR-10 subset for SSL pretraining."""
    cifar_train = torchvision.datasets.CIFAR10(
        root=str(REPO_ROOT), train=True, download=False,
    )
    cifar_val = torchvision.datasets.CIFAR10(
        root=str(REPO_ROOT), train=False, download=False,
    )
    train_subset = Subset(cifar_train, [int(i) for i in subset_idx])
    train_ds = spt.data.FromTorchDataset(
        train_subset, names=["image", "label"], transform=_make_train_tf(),
    )
    val_ds = spt.data.FromTorchDataset(
        cifar_val, names=["image", "label"], transform=_make_val_tf(),
    )
    train_dl = DataLoader(
        dataset=train_ds, batch_size=batch_size, num_workers=num_workers,
        drop_last=True, shuffle=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        pin_memory=True,
    )
    val_dl = DataLoader(
        dataset=val_ds, batch_size=batch_size,
        num_workers=max(1, num_workers // 2),
        persistent_workers=num_workers > 0,
    )
    return train_dl, val_dl


def _make_forward_with_aux():
    """Build a LeJEPA-recon ``forward`` that also adds aux-head losses.

    The returned function calls the unmodified ``ssl_pretrain_recon.forward``
    and, when the module has ``aux_*`` state attached and the batch
    carries a ``sample_idx`` field (auto-added by
    ``spt.data.FromTorchDataset``), looks up aux targets by coreset
    position, applies the per-aux linear heads on the first view's
    backbone embedding, and adds the weighted aux losses to
    ``out["loss"]``.

    Returns:
        A callable with signature ``forward(self, batch, stage) -> dict``
        suitable for ``spt.Module(forward=...)``. The returned dict has
        the same keys as the wrapped forward plus per-aux training/eval
        scalars logged via ``self.log``.
    """
    def forward_with_aux(self, batch, stage):
        out = forward(self, batch, stage)
        if "loss" not in out:
            return out
        if not getattr(self, "_aux_active", False):
            return out
        # FromTorchDataset adds sample_idx per item. After the two-view
        # collate, batch["sample_idx"] may be a list-of-tensors (one per
        # view) or a single tensor; the per-view sample_idx for view 0
        # gives the coreset position we need.
        sidx = batch.get("sample_idx", None)
        if sidx is None:
            return out
        if isinstance(sidx, (list, tuple)):
            sidx = sidx[0]
        # Take view-0 embeddings only: first B rows of the concatenated
        # (V*B, D) tensor packed inside ``out["embedding"]``.
        B = int(sidx.numel())
        emb_v0 = out["embedding"][:B]
        targets_batch = index_targets(self.aux_targets, sidx)
        aux_total, aux_logs = aux_loss_terms(
            emb_v0, self.aux_heads, targets_batch,
            self.aux_specs, self.aux_lambdas,
        )
        out["loss"] = out["loss"] + aux_total
        for k, v in aux_logs.items():
            self.log(f"{stage}/aux_{k}", v,
                     on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/aux_total", aux_total.detach(),
                 on_step=True, on_epoch=True, sync_dist=True)
        return out
    return forward_with_aux


def _train_one_coreset(args, subset_idx: np.ndarray, run_name: str,
                       art: Path, algo: str,
                       aux_lambdas: dict[str, float]) -> dict:
    """Run LeJEPA-recon SSL pretraining on ``subset_idx`` and return best probe metrics.

    Args:
        args: Parsed argparse namespace (epochs, lr, projector dims, etc.).
        subset_idx: Length-``k`` array of CIFAR-10 train-set indices that
            define this algorithm's coreset.
        run_name: Lightning run tag; controls log + checkpoint paths.
        art: ``<artifacts>/`` root; the per-algo aux files live at
            ``art/<algo>/aux_*.pt``.
        algo: Algorithm name (e.g. ``"greedy"``). Used to locate the
            per-algorithm aux targets.
        aux_lambdas: Map ``aux_name -> loss weight``; 0.0 disables that
            aux head entirely.

    Returns:
        Dict with keys:
            ``best_linear_probe_top1``: max OnlineProbe top-1 across epochs.
            ``best_knn_acc``: max OnlineKNN accuracy across epochs.
            ``aux_active``: list of aux names whose heads were trained
                (intersection of ``aux_lambdas > 0`` and on-disk targets).
    """
    pl.seed_everything(args.seed, workers=True)

    train_dl, val_dl = _make_subset_dataloaders(
        subset_idx, batch_size=args.batch_size, num_workers=args.num_workers,
    )

    backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)
    backbone.fc = nn.Identity()
    projector = _make_projector(EMB_DIM, args.proj_dim, args.proj_hidden)
    decoder = ConvDecoder(in_dim=EMB_DIM, base=args.dec_base)
    regularizer = make_regularizer(args.regularizer, num_proj=args.num_proj)

    # Discover which aux targets this algorithm has on disk. Heads are
    # only built for aux names that (a) are present in <algo>/ and (b)
    # have lambda > 0; anything else is skipped silently (logged once).
    aux_targets, aux_specs = discover_aux_targets(art / algo)
    active = [n for n in aux_specs if aux_lambdas.get(n, 0.0) > 0]
    missing = [n for n, w in aux_lambdas.items() if w > 0 and n not in aux_specs]
    if missing:
        print(f"  [aux] requested but missing on disk for {algo}: {missing}")
    aux_active = len(active) > 0
    aux_heads = build_aux_heads(EMB_DIM, aux_specs) if aux_active else None
    if aux_active:
        print(f"  [aux] active heads for {algo}: {active}  "
              f"lambdas={ {n: aux_lambdas[n] for n in active} }")

    chosen_forward = _make_forward_with_aux() if aux_active else forward
    module = spt.Module(
        backbone=backbone,
        projector=projector,
        decoder=decoder,
        forward=chosen_forward,
        regularizer=regularizer,
        lambd=args.lambd,
        lambd_recon=args.lambd_recon,
        inv_tol=args.inv_tol,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            "interval": "epoch",
        },
    )
    # Attach aux state to the Module so forward_with_aux can find it.
    # nn.Module submodules (aux_heads) get their params auto-included in
    # the optimizer; plain attrs (aux_targets / specs / lambdas) ride
    # along as ordinary Python state.
    if aux_active:
        module.aux_heads = aux_heads
        module.aux_targets = aux_targets
        module.aux_specs = aux_specs
        module.aux_lambdas = aux_lambdas
        module._aux_active = True
    else:
        module._aux_active = False

    linear_probe = spt.callbacks.OnlineProbe(
        module,
        name="linear_probe",
        input="embedding",
        target="label",
        probe=nn.Linear(EMB_DIM, 10),
        loss=nn.CrossEntropyLoss(),
        metrics={
            "top1": torchmetrics.classification.MulticlassAccuracy(10),
            "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
        },
    )
    knn_probe = spt.callbacks.OnlineKNN(
        name="knn_probe", input="embedding", target="label",
        queue_length=min(20000, len(subset_idx)),
        metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(10)},
        input_dim=EMB_DIM, k=10,
    )

    cap = CaptureBestMetrics()
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(LOG_DIR / run_name / "checkpoints"),
        save_last=True, save_top_k=0,
    )
    logger = CSVLogger(save_dir=str(LOG_DIR), name=run_name, version="")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        callbacks=[knn_probe, linear_probe, cap, ckpt_cb],
        precision=args.precision,
        logger=logger,
        default_root_dir=str(LOG_DIR / run_name),
    )
    trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)
    return {
        "best_linear_probe_top1": cap.best_top1,
        "best_knn_acc":           cap.best_knn,
        "aux_active":             active if aux_active else [],
    }


def _discover_algos(art: Path) -> list[str]:
    return sorted(
        d.name for d in art.iterdir()
        if d.is_dir() and (d / "indices.pt").exists()
    )


def _bar_plot(results: list[dict], full_acc: float | None,
              metric: str, out_path: Path, tag: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.arange(len(results))
    accs = [r[metric] * 100 for r in results]
    colors = [_ALGO_STYLE.get(r["algorithm"], ("C5", r["algorithm"]))[0]
              for r in results]
    ax.bar(xs, accs, color=colors)
    for x, a in zip(xs, accs):
        ax.text(x, a + 0.3, f"{a:.2f}", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{r['algorithm']}\nk={r['budget']}" for r in results],
        rotation=0, fontsize=9,
    )
    ax.set_ylabel(f"val {metric.replace('best_', '').replace('_', ' ')} (%)")
    ax.set_title(f"{tag}: LeJEPA-recon SSL pretrain on coreset (per algorithm)")
    if full_acc is not None:
        ax.axhline(full_acc * 100, color="gray", ls="--", lw=0.8,
                   label=f"full 50k = {full_acc * 100:.2f}%")
        ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=_Fmt)

    g = ap.add_argument_group("data")

    # Which coreset bundle to pretrain on. Accepts a bare <tag>
    # (resolved against example/cifar/cache/coreset/) or a full path.
    # resolve_artifacts() validates that at least one <algo>/indices.pt
    # exists. Defaults to DEFAULT_TAG if that tag is present on disk;
    # otherwise the flag becomes required with a helpful error listing
    # available tags from artifacts_help().
    _art_default, _art_required = default_artifacts(DEFAULT_TAG)
    g.add_argument("--artifacts", default=_art_default, required=_art_required,
                   type=resolve_artifacts, metavar="TAG_OR_PATH",
                   help=artifacts_help(preferred=DEFAULT_TAG))

    # Restrict which coreset algorithms to pretrain on. Defaults to
    # every <algo>/indices.pt under --artifacts. The library-supported
    # choices are:
    #   greedy        -- greedy max-variance pick (diverse-but-noisy)
    #   leverage      -- ridge leverage-score sampling (importance weighted)
    #   spectral_rank -- greedy max-coverage on rank strata (uniform-ish)
    g.add_argument("--algorithms", nargs="+", default=None,
                   choices=ALGO_CHOICES, metavar="ALGO",
                   help=("restrict to these algos (default: all under --artifacts). "
                         "Choices: "
                         "greedy=greedy max-variance, "
                         "leverage=ridge leverage sampling, "
                         "spectral_rank=greedy max-coverage on rank strata"))

    g = ap.add_argument_group("trainer")

    # SSL pretraining epochs. 200 is the LeJEPA-recon default that
    # matches ssl_pretrain_recon.py on the full 50k. On a coreset the
    # online probe usually plateaus earlier; drop to 100 for smoke tests.
    g.add_argument("--epochs",       type=int,   default=200)

    # Mini-batch size for SSL. 256 is the LeJEPA-recon recipe; the
    # sigreg/w1/w2 regularizer estimates uniformity *within* a batch,
    # so going much smaller (<128) noticeably degrades the uniformity
    # signal. Increase if you have GPU memory headroom.
    g.add_argument("--batch-size",   type=int,   default=256, dest="batch_size")

    # AdamW peak learning rate. 5e-4 is the LeJEPA-recon default;
    # paired with linear-warmup + cosine annealing inside spt.Module.
    # SSL is more lr-sensitive than supervised -- changing this can
    # easily move probe top-1 by 1-2%.
    g.add_argument("--lr",           type=float, default=5e-4, help="AdamW peak lr (warmup + cosine)")

    # AdamW L2 decoupled weight-decay. 5e-4 is conservative; LeJEPA
    # works fine across 1e-4..1e-3. Heavier decay can suppress the
    # projector and starve the uniformity regularizer.
    g.add_argument("--weight-decay", type=float, default=5e-4, dest="weight_decay")

    # DataLoader workers per dataloader. 8 is good for a single
    # multi-core box; reduce if you see DataLoader memory pressure
    # (each worker holds its own augmentation pipeline + transforms).
    # 0 = main-thread loading (useful for debugging).
    g.add_argument("--num-workers",  type=int,   default=8,    dest="num_workers")

    # Lightning trainer precision. Choices:
    #   16-mixed   -- fp16 mixed precision via AMP (fast on V100/T4/A100/3090,
    #                 default for speed; some loss-scale fiddling on edge cases)
    #   32-true    -- pure fp32 (slowest but most numerically stable;
    #                 use this if you see NaNs in the sigreg/w1/w2 terms)
    #   bf16-mixed -- bfloat16 mixed precision (Ampere+/H100 only;
    #                 fp32-like dynamic range, fastest and most stable
    #                 modern choice when hardware supports it)
    g.add_argument("--precision",                default="16-mixed",
                   choices=["16-mixed", "32-true", "bf16-mixed"],
                   help=("lightning precision. "
                         "16-mixed=fp16 AMP (fast, broad GPU support), "
                         "32-true=full fp32 (slow, most stable), "
                         "bf16-mixed=bfloat16 AMP (Ampere+ only, recommended on H100/A100)"))

    # RNG seed for SSL pretraining (model init, dataloader shuffle,
    # transform RNG). MUST match the --seed used by
    # cifar_ssl_eval_from_artifacts.py, because the eval script
    # locates the checkpoint via the run-name pattern
    # "ssl_coreset_<tag>_<algo>_s<seed>". Changing this here without
    # also changing it at eval time will produce "checkpoint missing".
    g.add_argument("--seed",         type=int,   default=0,
                   help="must match --seed at eval time (used in run-name lookup)")

    # Auxiliary loss weights. Each attaches a separate nn.Linear head on
    # the 512-D backbone embedding (view 0 only) and adds the head's
    # per-target loss to the LeJEPA-recon total loss with the given
    # lambda. 0.0 disables that head entirely (no forward pass, no
    # parameters trained). Targets are loaded from
    # <artifacts>/<algo>/aux_<name>.pt and indexed by ``sample_idx``
    # (the coreset position auto-added by ``spt.data.FromTorchDataset``).
    g_aux = ap.add_argument_group("auxiliary losses (lambda per aux target; 0=off)")

    # Regress projections onto the top eigenvectors of Phi^T Phi.
    # Loss is per-coordinate MSE weighted by w_j = 1/sqrt(sigma2[j] + lam),
    # so noisy trailing coords contribute less. Head: 512 -> n_top_eigvecs.
    g_aux.add_argument("--aux-spectral-coords", type=float, default=0.0,
                       dest="aux_spectral_coords",
                       help="weight for spectral_coords aux (per-coord ridge-weighted MSE; 0=off)")

    # Regress per-bucket uniform ranks in [0, 1] (one rank per bucket).
    # Encourages the embedding to know its rank-position within every
    # bucket of equal-mass eigenvectors. Head: 512 -> n_buckets.
    g_aux.add_argument("--aux-bucket-ranks", type=float, default=0.0,
                       dest="aux_bucket_ranks",
                       help="weight for bucket_ranks aux (per-bucket uniform-rank MSE; 0=off)")

    # Regress scalar ridge leverage h_i; uniformly upweights samples that
    # carry unique information about the Gram matrix. Head: 512 -> 1.
    g_aux.add_argument("--aux-leverage-score", type=float, default=0.0,
                       dest="aux_leverage_score",
                       help="weight for leverage_score aux (scalar leverage MSE; 0=off)")

    # Classify each sample's home-bucket id b*_i = argmax_b S_{i,b}.
    # Acts like a coarse cluster label derived from the spectrum.
    # Head: 512 -> n_buckets, cross-entropy.
    g_aux.add_argument("--aux-home-bucket", type=float, default=0.0,
                       dest="aux_home_bucket",
                       help="weight for home_bucket aux (cross-entropy over bucket id; 0=off)")

    # Regress standardized backbone features phi_i ("teacher distillation"
    # from the encoder that built the coreset). Head: 512 -> D.
    g_aux.add_argument("--aux-feature-distill", type=float, default=0.0,
                       dest="aux_feature_distill",
                       help="weight for feature_distill aux (frozen-teacher feature MSE; 0=off)")

    g = ap.add_argument_group("LeJEPA-recon hyperparameters (rarely changed)")

    # Uniformity regularizer used to push projector outputs toward
    # an isotropic Gaussian on random 1-D projections. Choices:
    #   sigreg -- per-projection sigma regularizer toward N(0,1) std
    #             (LeJEPA default; cheapest, most stable)
    #   w1     -- 1-Wasserstein distance to N(0,1) on each 1-D projection
    #             (sharper tails, slightly slower)
    #   w2     -- 2-Wasserstein distance to N(0,1) on each 1-D projection
    #             (more sensitive to outliers than w1)
    g.add_argument("--regularizer",  default="sigreg", choices=["sigreg", "w1", "w2"],
                   help=("uniformity regularizer. "
                         "sigreg=per-proj sigma toward N(0,1) (default), "
                         "w1=1-Wasserstein to N(0,1) per projection, "
                         "w2=2-Wasserstein to N(0,1) per projection"))

    # Weight on the uniformity regularizer term in the total loss.
    # 0.05 is the LeJEPA-recon default. Too high -> projector
    # collapses to perfect Gaussian but loses semantic content;
    # too low -> dimensional/representational collapse.
    g.add_argument("--lambd",        type=float, default=0.05, help="weight on uniformity regularizer")

    # Weight on the pixel-space reconstruction (decoder) term.
    # 0.1 keeps recon as an auxiliary anchor against collapse without
    # dominating the invariance + uniformity objective. Set to 0 to
    # disable recon entirely (turns this into vanilla LeJEPA).
    g.add_argument("--lambd-recon",  type=float, default=0.1,  dest="lambd_recon",
                   help="weight on pixel-space reconstruction term (0 = disable recon)")

    # Slack on the joint-embedding invariance MSE between two
    # augmented views. 0.0 = strict MSE (LeJEPA default). Small
    # positive values (e.g. 0.01) act like a hinge that ignores
    # tiny disagreements -- useful only if invariance dominates.
    g.add_argument("--inv-tol",      type=float, default=0.0,  dest="inv_tol",
                   help="invariance MSE slack for joint-embedding (0 = strict MSE)")

    # Projector output dimensionality. The uniformity regularizer
    # operates here, not on the 512-D backbone embedding. 64 is the
    # LeJEPA default and a good speed/quality tradeoff; raising it
    # beyond ~128 rarely helps and slows the regularizer.
    g.add_argument("--proj-dim",     type=int,   default=64,   dest="proj_dim")

    # Projector MLP hidden width (3-layer MLP: 512 -> H -> H -> proj_dim).
    # 2048 matches LeJEPA-recon. Reduce to 1024 if you're memory-bound.
    g.add_argument("--proj-hidden",  type=int,   default=2048, dest="proj_hidden")

    # Number of random 1-D projections used by sigreg/w1/w2 each step.
    # 1024 is the LeJEPA default; the regularizer's variance is ~ 1 / num_proj,
    # so dropping below 256 makes the signal noisy. Higher = smoother
    # gradient but linear time cost.
    g.add_argument("--num-proj",     type=int,   default=1024, dest="num_proj",
                   help="# random 1-D projections used by sigreg/w1/w2")

    # ConvDecoder base channel width. The decoder upsamples from
    # the 512-D embedding back to 32x32x3 via a small transposed-conv
    # stack with channel pattern (base*4, base*2, base). 256 matches
    # ssl_pretrain_recon.py; halve to 128 to shrink decoder memory.
    g.add_argument("--dec-base",     type=int,   default=256,  dest="dec_base",
                   help="conv decoder base channel width")

    g = ap.add_argument_group("plot")

    # Optional reference line on the bar plots (e.g. your full-50k
    # SSL linear-probe accuracy). Pass as a fraction in [0, 1], not
    # a percentage. Default None = no reference line drawn.
    g.add_argument("--full-acc", type=float, default=None, dest="full_acc",
                   help="50k-train reference accuracy, drawn as a dashed line")

    args = ap.parse_args()

    art = args.artifacts  # already a validated Path
    tag = art.name
    algos = args.algorithms or _discover_algos(art)
    if not algos:
        raise RuntimeError(f"no <algo>/indices.pt under {art}")

    out_dir = REPO_ROOT / "example" / "out" / "coreset"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{tag}_ssl_coreset_results.json"
    plot_path_lin = out_dir / f"{tag}_ssl_coreset_linprobe.png"
    plot_path_knn = out_dir / f"{tag}_ssl_coreset_knn.png"

    # Per-aux loss weights collected once for the whole run (same weights
    # applied to every algorithm). 0.0 means "head not attached".
    aux_lambdas = collect_aux_lambdas(args)

    results: list[dict] = []
    for algo in algos:
        idx = torch.load(art / algo / "indices.pt").cpu().numpy().astype(np.int64)
        run_name = f"ssl_coreset_{tag}_{algo}_s{args.seed}"
        print(f"\n=== {run_name}  ({len(idx)} samples) ===")
        metrics = _train_one_coreset(args, idx, run_name, art, algo, aux_lambdas)
        results.append({
            "algorithm": algo,
            "budget": int(len(idx)),
            "epochs": args.epochs,
            "seed": args.seed,
            "aux_lambdas": aux_lambdas,
            "aux_active": sorted(metrics.pop("aux_active", [])),
            **metrics,
        })

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults -> {results_path}")

    print("\n=== summary ===")
    print(f"  {'algorithm':<18} {'budget':>8} {'lin-top1':>10} {'knn-acc':>10}")
    for r in results:
        print(f"  {r['algorithm']:<18} {r['budget']:>8} "
              f"{r.get('best_linear_probe_top1', 0)*100:>9.2f}%  "
              f"{r.get('best_knn_acc', 0)*100:>9.2f}%")

    _bar_plot(results, args.full_acc, "best_linear_probe_top1",
              plot_path_lin, tag)
    _bar_plot(results, args.full_acc, "best_knn_acc",
              plot_path_knn, tag)
    print(f"plots -> {plot_path_lin}, {plot_path_knn}")


if __name__ == "__main__":
    main()
