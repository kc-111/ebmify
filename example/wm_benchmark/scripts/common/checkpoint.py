"""Local checkpoint callback — replaces upstream's SaveCkptCallback.

Writes a self-contained ``weights_epoch_<N>.pt`` per save into the run dir
under our own ``data/runs/<method>/<run_id>/`` tree, plus a ``last.pt``
symlink-style mirror for the most recent epoch. The payload uses the same
keys swm.wm.utils.load_pretrained expects: ``model_state_dict`` and
``cfg``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch

LOG = logging.getLogger("wm_benchmark")


class LocalSaveCkptCallback(pl.Callback):
    """Persist model weights every ``every_n_epochs`` and at train end.

    Stores ``{model_state_dict, cfg}`` so that downstream loaders that
    follow swm's convention (e.g. ``swm.wm.utils.load_pretrained``) can
    pick checkpoints back up.
    """

    def __init__(self, run_dir: Path, *, every_n_epochs: int = 1,
                 cfg_payload: Any | None = None, key: str = "world_model"):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.cfg_payload = cfg_payload
        self.key = key

    def _save(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *, suffix: str) -> None:
        target = self.run_dir / f"weights_{suffix}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": pl_module.state_dict(),
            "cfg": self.cfg_payload,
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
        }
        torch.save(payload, target)
        last = self.run_dir / "last.pt"
        try:
            if last.is_symlink() or last.exists():
                last.unlink()
            last.symlink_to(target.name)
        except OSError:
            torch.save(payload, last)
        LOG.info("saved checkpoint %s", target)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        ep = trainer.current_epoch
        if (ep + 1) % self.every_n_epochs != 0:
            return
        self._save(trainer, pl_module, suffix=f"epoch_{ep + 1}")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._save(trainer, pl_module, suffix=f"final_ep{trainer.current_epoch + 1}")


def resume_path(run_dir: Path) -> Path | None:
    """Return ``run_dir/last.pt`` if present (for PL ``ckpt_path`` resume)."""
    p = Path(run_dir) / "last.pt"
    return p if p.exists() else None


__all__ = ["LocalSaveCkptCallback", "resume_path"]
