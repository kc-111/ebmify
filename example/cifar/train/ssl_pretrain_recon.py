"""SSL pretraining with an added reconstruction head.

Motivation: vanilla LeJEPA (``ssl_pretrain.py``) trains invariance to
strong pixel augmentations, which makes the backbone *pixel-statistics
blind*. Under leverage-OOD with ``centered+L2`` preprocessing, that
shows up as Gaussian noise looking *more* in-distribution than cifar100
(see ``LEVERAGE_FINDINGS.md`` Section 2.5).

Hypothesis: bolting a reconstruction head onto the LeJEPA backbone
preserves some pixel-statistics information in the embedding without
losing the semantic-invariants axis. If the trade-off is feature
coverage (Section 2.6), a jointly-trained "aug in -> z -> aug out"
encoder should recover both axes from a single backbone — i.e. do
in-network what ``cifar_concat_features_test.py`` does at inference.

Loss::

    L = (1 - lambd) * inv  +  lambd * reg  +  lambd_recon * recon

- ``inv``, ``reg``: same margin-hinge invariance + W1 regularizer as
  ``ssl_pretrain.py``.
- ``recon``: MSE between decoded image and the *augmented* input view
  in normalized-pixel space. We reconstruct the augmented view rather
  than a canonical view so the head genuinely demands per-view pixel
  information from the embedding.

The decoder is discarded at the end of training; only the backbone
state dict is saved, so all downstream OOD / probe scripts work
unchanged (just point ``--ssl-tag recon`` at them).

Usage:
    python example/cifar/train/ssl_pretrain_recon.py --epochs 1000
    python example/cifar/train/ssl_pretrain_recon.py --lambd-recon 0.3 --tag recon_03
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
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

EMB_DIM = 512


# ---------------------------------------------------------------------------
# Transforms (identical to ssl_pretrain.py)
# ---------------------------------------------------------------------------

def _make_train_tf() -> transforms.MultiViewTransform:
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
# Model: projector (SSL) + decoder (recon)
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


class ConvDecoder(nn.Module):
    """Symmetric to the ResNet18 low-resolution stem: 512 -> (256, 2, 2)
    via 1x1 conv on a tiled embedding, then four stride-2 upsamples to
    (3, 32, 32). Trained jointly; thrown away at the end."""

    def __init__(self, in_dim: int = EMB_DIM, base: int = 256) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.base = base
        self.proj = nn.Linear(in_dim, base * 2 * 2)
        self.up = nn.Sequential(
            self._block(base,     base // 2),   # 2 -> 4
            self._block(base // 2, base // 4),  # 4 -> 8
            self._block(base // 4, base // 8),  # 8 -> 16
            self._block(base // 8, base // 16), # 16 -> 32
        )
        self.head = nn.Conv2d(base // 16, 3, kernel_size=3, padding=1)

    @staticmethod
    def _block(c_in: int, c_out: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z).view(z.shape[0], self.base, 2, 2)
        h = self.up(h)
        return self.head(h)  # raw (no tanh); we predict normalized pixels


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

def forward(self, batch, stage):
    out: dict = {}
    views = _get_views_list(batch)

    if views is None:
        emb = self.backbone(batch["image"])
        out["embedding"] = emb
        out["projection"] = self.projector(emb)
        return out

    live_emb = [self.backbone(v["image"]) for v in views]
    live_z = [self.projector(e) for e in live_emb]

    z_stack = torch.stack(live_z, dim=0)        # (V, B, D_proj)
    V, _, D = z_stack.shape
    reg = self.regularizer(z_stack)
    mean_z = z_stack.mean(dim=0, keepdim=True)
    per_sample_sq = (z_stack - mean_z).square().sum(dim=-1)
    prior_floor = D * (V - 1) / V
    margin = self.inv_tol * prior_floor
    inv = torch.clamp(per_sample_sq - margin, min=0.0).mean() / D
    ssl_loss = self.lambd * reg + (1.0 - self.lambd) * inv

    # Per-view reconstruction: predict the augmented view from its own
    # backbone embedding. Loss is MSE in normalized-pixel space (the
    # space ToImage(**CIFAR10) hands us).
    recon_loss = z_stack.new_zeros(())
    if self.lambd_recon > 0.0:
        recons = [self.decoder(e) for e in live_emb]
        targets = [v["image"] for v in views]
        recon_loss = sum(F.mse_loss(r, t) for r, t in zip(recons, targets)) / V

    loss = ssl_loss + self.lambd_recon * recon_loss
    out["loss"] = loss

    out["embedding"] = torch.cat(live_emb, dim=0)
    out["projection"] = torch.cat(live_z, dim=0)
    if "label" in views[0]:
        out["label"] = torch.cat([v["label"] for v in views], dim=0)

    self.log(f"{stage}/loss",  loss,        on_step=True, on_epoch=True, sync_dist=True)
    self.log(f"{stage}/reg",   reg,         on_step=True, on_epoch=True, sync_dist=True)
    self.log(f"{stage}/inv",   inv,         on_step=True, on_epoch=True, sync_dist=True)
    self.log(f"{stage}/recon", recon_loss,  on_step=True, on_epoch=True, sync_dist=True)
    return out


# ---------------------------------------------------------------------------
# Backbone-save callback (identical to ssl_pretrain.py)
# ---------------------------------------------------------------------------

class SaveBackboneCallback(Callback):
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--regularizer", default="sigreg",
                   choices=["sigreg", "w1", "w2"])
    p.add_argument("--lambd", type=float, default=0.05,
                   help="Weight on reg; (1 - lambd) weights invariance.")
    p.add_argument("--lambd-recon", type=float, default=0.1, dest="lambd_recon",
                   help="Weight on pixel-MSE reconstruction term. "
                        "0 reproduces vanilla LeJEPA.")
    p.add_argument("--inv-tol", type=float, default=0.0)
    p.add_argument("--proj-dim", type=int, default=64)
    p.add_argument("--proj-hidden", type=int, default=2048)
    p.add_argument("--num-proj", type=int, default=1024)
    p.add_argument("--dec-base", type=int, default=256, dest="dec_base",
                   help="Decoder channel width at the 2x2 stem.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", default="16-mixed")
    p.add_argument("--tag", default="recon",
                   help="Tag appended to log/checkpoint paths. "
                        "Default 'recon' so it doesn't clobber the "
                        "vanilla LeJEPA checkpoint.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    pl.seed_everything(args.seed, workers=True)

    run_name = f"cifar10_w1{('_' + args.tag) if args.tag else ''}"
    print(f"[run] {run_name}  lambd={args.lambd}  "
          f"lambd_recon={args.lambd_recon}  epochs={args.epochs}")

    data = _make_data(args.batch_size, args.num_workers)

    backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)
    backbone.fc = nn.Identity()
    projector = _make_projector(EMB_DIM, args.proj_dim, args.proj_hidden)
    decoder = ConvDecoder(in_dim=EMB_DIM, base=args.dec_base)
    regularizer = make_regularizer(args.regularizer, num_proj=args.num_proj)

    module = spt.Module(
        backbone=backbone,
        projector=projector,
        decoder=decoder,
        forward=forward,
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

    tag = args.tag if args.tag else "recon"
    backbone_path = CACHE_DIR / f"cifar10_ssl_resnet18_{tag}.pt"
    save_cb = SaveBackboneCallback(backbone_path)

    LOG_DIR.mkdir(exist_ok=True, parents=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(LOG_DIR / run_name / "checkpoints"),
        save_last=True, save_top_k=0,
    )
    logger = CSVLogger(save_dir=str(LOG_DIR), name=run_name, version="")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        callbacks=[knn_probe, linear_probe, ckpt_cb, save_cb],
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
