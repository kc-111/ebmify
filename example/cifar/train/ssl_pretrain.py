"""SSL pretraining on CIFAR-10 with the ebmify W1 regularizer.

Loss::

    L = (1 - lambd) * inv  +  lambd * reg

- ``reg``: SlicedW1 from ``ebmify/models/losses.py`` — pushes the stack
  of projections toward N(0, I).
- ``inv``: margin-hinge invariance under an iid N(0, I) prior. Strict
  invariance corresponds to ``inv_tol = 0``; ``inv_tol > 0`` clamps
  pressure below the natural view dispersion floor
  ``E[||z_i - z_bar||^2] = D (V - 1)/V``.

Default ``lambd = 0.8`` so the W1 regularizer dominates with sim at 0.2x.

Uses ``stable_pretraining`` (cloned outside this repo) for the
LightningModule wrapper, backbone helper, transforms, and online probes.
Saves the trained backbone state dict to
``example/cifar/cache/cifar10_ssl_resnet18.pt`` for downstream OOD eval.

Usage:
    python example/cifar/train/ssl_pretrain.py --epochs 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.forward import _get_views_list

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import REPO_ROOT  # noqa: F401, E402
from ebmify.models.losses import make_regularizer  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

EMB_DIM = 512  # ResNet18 penultimate width.


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _make_train_tf() -> transforms.MultiViewTransform:
    """Two-view SSL augmentation per the user's recipe.

    scale=(0.2, 1.0) is a softer crop than the (0.08, 1.0) SimCLR/VICReg
    default, well-suited to 32x32 images where aggressive crops can drop
    all foreground content.
    """
    one = transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.2, hue=0.1, p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.PILGaussianBlur(p=0.5),
        transforms.RandomSolarize(p=0.2, threshold=0.5),
        transforms.ToImage(**spt.data.static.CIFAR10),
    )
    return transforms.MultiViewTransform([one, one])


def _make_val_tf() -> transforms.Compose:
    return transforms.Compose(
        transforms.RGB(),
        transforms.Resize((32, 32)),
        transforms.ToImage(**spt.data.static.CIFAR10),
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _make_data(batch_size: int, num_workers: int) -> spt.data.DataModule:
    # torchvision.datasets.CIFAR10 reads from <root>/cifar-10-batches-py/...
    # The repo already has that directory at REPO_ROOT (populated by
    # download_cifar.py / cifar_data.py), so pointing root=REPO_ROOT
    # reuses the existing pickle archives. download=False prevents any
    # redownload if the integrity check were to fail.
    cifar_train = torchvision.datasets.CIFAR10(
        root=str(REPO_ROOT), train=True, download=False,
    )
    cifar_val = torchvision.datasets.CIFAR10(
        root=str(REPO_ROOT), train=False, download=False,
    )
    train_ds = spt.data.FromTorchDataset(
        cifar_train, names=["image", "label"], transform=_make_train_tf(),
    )
    val_ds = spt.data.FromTorchDataset(
        cifar_val, names=["image", "label"], transform=_make_val_tf(),
    )

    train_dl = torch.utils.data.DataLoader(
        dataset=train_ds, batch_size=batch_size, num_workers=num_workers,
        drop_last=True, shuffle=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        pin_memory=True,
    )
    val_dl = torch.utils.data.DataLoader(
        dataset=val_ds, batch_size=batch_size,
        num_workers=max(1, num_workers // 2),
        persistent_workers=num_workers > 0,
    )
    return spt.data.DataModule(train=train_dl, val=val_dl)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _make_projector(in_dim: int, out_dim: int, hidden: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

def forward(self, batch, stage):
    """Two-view forward.

    SSL loss is computed only when ``views`` is a list; otherwise we just
    expose ``embedding`` for downstream (probe / eval) consumers. If
    ``self.ema_backbone`` is installed (by EMABackboneCallback with
    probe=True), the embedding handed to probe callbacks is computed
    from the EMA shadow under no_grad; SSL loss still uses raw
    embeddings so backbone training is unchanged.
    """
    out: dict = {}
    views = _get_views_list(batch)
    ema_bb = getattr(self, "ema_backbone", None)

    if views is None:
        emb = self.backbone(batch["image"])
        if ema_bb is not None:
            with torch.no_grad():
                out["embedding"] = ema_bb(batch["image"])
        else:
            out["embedding"] = emb
        out["projection"] = self.projector(emb)
        return out

    live_emb = [self.backbone(v["image"]) for v in views]
    live_z = [self.projector(e) for e in live_emb]

    z_stack = torch.stack(live_z, dim=0)        # (V, B, D)
    V, _, D = z_stack.shape
    reg = self.regularizer(z_stack)
    mean_z = z_stack.mean(dim=0, keepdim=True)

    # Margin-hinge invariance: free pressure below prior dispersion floor.
    per_sample_sq = (z_stack - mean_z).square().sum(dim=-1)
    prior_floor = D * (V - 1) / V
    margin = self.inv_tol * prior_floor
    inv = torch.clamp(per_sample_sq - margin, min=0.0).mean() / D

    loss = self.lambd * reg + (1.0 - self.lambd) * inv
    out["loss"] = loss

    # Stack views for probe callbacks; optionally route through EMA shadow.
    if ema_bb is not None:
        with torch.no_grad():
            probe_emb = [ema_bb(v["image"]) for v in views]
    else:
        probe_emb = live_emb
    out["embedding"] = torch.cat(probe_emb, dim=0)
    out["projection"] = torch.cat(live_z, dim=0)
    if "label" in views[0]:
        out["label"] = torch.cat([v["label"] for v in views], dim=0)

    self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
    self.log(f"{stage}/reg",  reg,  on_step=True, on_epoch=True, sync_dist=True)
    self.log(f"{stage}/inv",  inv,  on_step=True, on_epoch=True, sync_dist=True)

    return out


# ---------------------------------------------------------------------------
# Backbone-save callback
# ---------------------------------------------------------------------------

class SaveBackboneCallback(Callback):
    """Dump just the backbone state dict on train_end, ready for OOD eval.

    Avoids loading the full Lightning checkpoint downstream; the eval
    script only needs the encoder weights.
    """

    def __init__(self, save_path: Path):
        super().__init__()
        self.save_path = Path(save_path)

    def on_train_end(self, trainer, pl_module):
        self.save_path.parent.mkdir(exist_ok=True, parents=True)
        sd = pl_module.backbone.state_dict()
        config = {"arch": "resnet18-spt-low-res", "emb_dim": EMB_DIM,
                  "norm": "CIFAR10"}
        torch.save({"state_dict": sd, "config": config}, self.save_path)
        print(f"[save] backbone state dict -> {self.save_path}")


class EMABackboneCallback(Callback):
    """Maintain an EMA shadow of the backbone; save it at train_end.

    Parameters (not buffers) are EMA-averaged each train batch via
    ``torch.optim.swa_utils.AveragedModel``. At ``on_train_end`` we run
    one un-augmented pass over view 0 of the train loader to recompute
    BN running stats matched to the EMA parameters -- without this the
    saved checkpoint's BN buffers are stuck at init and embeddings will
    be garbage. Saved to ``save_path``; downstream probes can load it
    by pointing ``--ssl-tag`` at the suffixed file.
    """

    def __init__(self, decay: float, save_path: Path, datamodule,
                 probe: bool = False):
        super().__init__()
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema decay must be in (0, 1); got {decay}")
        self.decay = decay
        self.save_path = Path(save_path)
        self.datamodule = datamodule
        self.probe = probe
        self.ema: torch.optim.swa_utils.AveragedModel | None = None

    def on_fit_start(self, trainer, pl_module):
        d = self.decay

        def avg_fn(avg, p, _):
            return d * avg + (1.0 - d) * p

        self.ema = torch.optim.swa_utils.AveragedModel(
            pl_module.backbone, avg_fn=avg_fn,
        )
        if self.probe:
            # Register as submodule so Lightning manages train/eval mode +
            # device placement, and so forward() can find it on pl_module.
            # The forward pass through this module during training also
            # naturally updates its BN running stats -- so we can skip
            # the post-hoc BN refresh.
            pl_module.ema_backbone = self.ema.module
            print(f"[ema] online probe will read embeddings from EMA "
                  f"shadow (decay={d})")
        else:
            print(f"[ema] tracking backbone with decay={d}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.ema is not None:
            self.ema.update_parameters(pl_module.backbone)

    @torch.no_grad()
    def _refresh_bn(self, pl_module) -> None:
        bn = [m for m in self.ema.modules()
              if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        if not bn:
            return
        saved = [(m, m.momentum) for m in bn]
        for m in bn:
            m.reset_running_stats()
            m.momentum = None  # cumulative average
        self.ema.train()
        device = pl_module.device
        for batch in self.datamodule.train_dataloader():
            views = _get_views_list(batch)
            imgs = views[0]["image"] if views is not None else batch["image"]
            self.ema(imgs.to(device, non_blocking=True))
        for m, mom in saved:
            m.momentum = mom

    def on_train_end(self, trainer, pl_module):
        if self.ema is None:
            return
        if not self.probe:
            # EMA was never run during training -> its BN buffers are at
            # init. Refresh by running one cumulative-mean pass over view 0
            # of the train loader. With probe=True, BN tracked naturally
            # via the probe forward, so this pass would only churn stats.
            print("[ema] refreshing BN stats over view 0 of train loader ...")
            self._refresh_bn(pl_module)
        self.save_path.parent.mkdir(exist_ok=True, parents=True)
        sd = self.ema.module.state_dict()
        config = {"arch": "resnet18-spt-low-res", "emb_dim": EMB_DIM,
                  "norm": "CIFAR10", "ema_decay": self.decay}
        torch.save({"state_dict": sd, "config": config}, self.save_path)
        print(f"[save] EMA backbone (decay={self.decay}) -> {self.save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--regularizer", default="w2",
                   choices=["sigreg", "w1", "w2"])
    p.add_argument("--lambd", type=float, default=0.05,
                   help="Weight on reg; (1 - lambd) weights invariance.")
    p.add_argument("--inv-tol", type=float, default=0.0,
                   help="Margin epsilon in [0, 1] on the invariance hinge.")
    p.add_argument("--proj-dim", type=int, default=64)
    p.add_argument("--proj-hidden", type=int, default=2048)
    p.add_argument("--num-proj", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", default="16-mixed")
    p.add_argument("--tag", default="",
                   help="Tag appended to log/checkpoint paths.")
    p.add_argument("--ema-decay", type=float, default=0.0, dest="ema_decay",
                   help="If > 0, also track an EMA shadow of the backbone "
                        "and save it to <cache>/cifar10_ssl_resnet18[_tag]_ema.pt. "
                        "Typical: 0.999 (short runs), 0.9999 (long runs).")
    p.add_argument("--probe-on-ema", action="store_true", dest="probe_on_ema",
                   help="Route the online linear/kNN probes through the "
                        "EMA backbone. Requires --ema-decay > 0. Logged "
                        "probe metrics then reflect EMA-weight quality.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    pl.seed_everything(args.seed, workers=True)

    run_name = f"cifar10_w1{('_' + args.tag) if args.tag else ''}"
    print(f"[run] {run_name}  lambd={args.lambd}  epochs={args.epochs}")

    data = _make_data(args.batch_size, args.num_workers)

    backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)
    backbone.fc = nn.Identity()
    projector = _make_projector(EMB_DIM, args.proj_dim, args.proj_hidden)
    regularizer = make_regularizer(args.regularizer, num_proj=args.num_proj)

    module = spt.Module(
        backbone=backbone,
        projector=projector,
        forward=forward,
        regularizer=regularizer,
        lambd=args.lambd,
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
        name="knn_probe",
        input="embedding",
        target="label",
        queue_length=20000,
        metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(10)},
        input_dim=EMB_DIM,
        k=10,
    )

    tag_suffix = ('_' + args.tag) if args.tag else ''
    backbone_path = CACHE_DIR / f"cifar10_ssl_resnet18{tag_suffix}.pt"
    save_cb = SaveBackboneCallback(backbone_path)

    LOG_DIR.mkdir(exist_ok=True, parents=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(LOG_DIR / run_name / "checkpoints"),
        save_last=True, save_top_k=0,
    )
    logger = CSVLogger(save_dir=str(LOG_DIR), name=run_name, version="")

    if args.probe_on_ema and not args.ema_decay > 0.0:
        raise SystemExit("--probe-on-ema requires --ema-decay > 0")

    callbacks = [knn_probe, linear_probe, ckpt_cb, save_cb]
    if args.ema_decay > 0.0:
        ema_path = CACHE_DIR / f"cifar10_ssl_resnet18{tag_suffix}_ema.pt"
        callbacks.append(EMABackboneCallback(
            args.ema_decay, ema_path, data, probe=args.probe_on_ema,
        ))

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        precision=args.precision,
        logger=logger,
        default_root_dir=str(LOG_DIR / run_name),
    )

    data.setup("fit")
    trainer.fit(
        module,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )


if __name__ == "__main__":
    main()
