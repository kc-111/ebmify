"""Local logging stack: CSVLogger (always on) + opt-in WandbLogger + a
VerboseProgressCallback that streams per-step diagnostics to stderr and to
``progress.log``.

Run-dir layout:

    data/runs/<method>/<run_id>/
      config.yaml      # snapshot of the resolved dataclass
      metrics.csv      # PL CSVLogger output
      progress.log     # streaming text log
      weights_epoch_<N>.pt
"""
from __future__ import annotations

import logging as _stdlog
import math
import sys
import time
from dataclasses import is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import CSVLogger

# Optional: WandB only imported lazily so unused trainers stay light.
try:
    from lightning.pytorch.loggers import WandbLogger
except Exception:  # pragma: no cover
    WandbLogger = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Run-dir helpers


def make_run_id(prefix: str | None = None) -> str:
    """Timestamp-based run id, ``YYYYmmdd-HHMMSS`` optionally prefixed."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_{stamp}" if prefix else stamp


def make_run_dir(method: str, run_id: str, root: Path) -> Path:
    """Return ``root / "runs" / method / run_id`` after mkdir-ing it."""
    p = root / "runs" / method / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Loggers


def make_loggers(run_dir: Path, *, wandb_enabled: bool = False,
                 wandb_project: str | None = None,
                 wandb_entity: str | None = None,
                 wandb_run_name: str | None = None,
                 wandb_id: str | None = None) -> list[Any]:
    """Build the loggers passed to ``pl.Trainer(logger=...)``.

    Always returns a CSVLogger. Appends a WandbLogger only when
    ``wandb_enabled=True`` and the package import succeeded.
    """
    csv = CSVLogger(save_dir=str(run_dir), name="", version="")
    if not wandb_enabled:
        return [csv]
    if WandbLogger is None:
        raise RuntimeError("wandb_enabled=True but lightning.pytorch.loggers.WandbLogger could not be imported")
    wb = WandbLogger(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_run_name,
        id=wandb_id,
        save_dir=str(run_dir),
        log_model=False,
        resume="allow",
    )
    return [csv, wb]


# ---------------------------------------------------------------------------
# Verbose progress callback


def _grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        # detach to avoid building autograd graph
        total += p.grad.detach().float().norm(2).item() ** 2
    return math.sqrt(total)


def setup_text_logger(run_dir: Path, name: str = "wm_benchmark") -> _stdlog.Logger:
    """Configure a stderr+file logger writing to ``run_dir/progress.log``."""
    logger = _stdlog.getLogger(name)
    logger.setLevel(_stdlog.INFO)
    logger.handlers.clear()
    fmt = _stdlog.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    fh = _stdlog.FileHandler(run_dir / "progress.log")
    fh.setFormatter(fmt)
    sh = _stdlog.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


class VerboseProgressCallback(pl.Callback):
    """Log per-N-step training diagnostics: loss, grad-norms, throughput, ETA.

    Numbers are computed cheaply (no extra forward passes) and printed to
    both stderr and ``progress.log`` so a `tail -f` works in any terminal
    without spinning up wandb.
    """

    def __init__(self, every_n_steps: int = 50, *, logger_name: str = "wm_benchmark"):
        super().__init__()
        self.every_n_steps = max(1, int(every_n_steps))
        self._logger_name = logger_name
        self._log = _stdlog.getLogger(logger_name)
        self._t_step_start: float | None = None
        self._samples_seen: int = 0
        self._t_train_start: float | None = None

    # ----- lifecycle

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._t_train_start = time.time()
        self._t_step_start = time.time()
        self._log.info("train start: max_epochs=%s steps_per_epoch=%s",
                       trainer.max_epochs, getattr(trainer, "num_training_batches", "?"))

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._log.info("epoch %d/%s start", trainer.current_epoch,
                       trainer.max_epochs if trainer.max_epochs is not None else "?")

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                            outputs, batch, batch_idx: int) -> None:
        # batch_size is best-effort; PL exposes it on the trainer datamodule
        if isinstance(batch, dict):
            bs_field = next((v for v in batch.values() if isinstance(v, torch.Tensor)), None)
            bs = bs_field.shape[0] if bs_field is not None else 1
        elif isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], torch.Tensor):
            bs = batch[0].shape[0]
        else:
            bs = 1
        self._samples_seen += bs

        step = trainer.global_step
        if step == 0 or step % self.every_n_steps != 0:
            return

        now = time.time()
        elapsed = now - (self._t_step_start or now)
        sps = self._samples_seen / elapsed if elapsed > 0 else float("nan")
        self._samples_seen = 0
        self._t_step_start = now

        # loss + grad norms
        loss = None
        if isinstance(outputs, dict) and "loss" in outputs:
            try:
                loss = float(outputs["loss"].detach())
            except Exception:
                loss = None
        elif torch.is_tensor(outputs):
            loss = float(outputs.detach())

        gnorm = _grad_norm(pl_module.parameters())

        # ETA from progress fraction over epochs
        if trainer.max_epochs and trainer.num_training_batches and self._t_train_start:
            total_steps = trainer.max_epochs * trainer.num_training_batches
            frac = step / max(1, total_steps)
            wall = now - self._t_train_start
            eta = (wall / frac - wall) if frac > 0 else float("nan")
        else:
            eta = float("nan")

        # lr (first param group only — extra groups end up in the CSV anyway)
        try:
            lr = trainer.optimizers[0].param_groups[0]["lr"]
        except Exception:
            lr = float("nan")

        msg = (f"step={step} ep={trainer.current_epoch} "
               f"loss={loss:.4f} " if loss is not None
               else f"step={step} ep={trainer.current_epoch} ")
        msg += f"|g|={gnorm:.3g} lr={lr:.3g} sps={sps:.1f} eta={eta/60:.1f}min"
        self._log.info(msg)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._t_train_start is None:
            return
        wall = time.time() - self._t_train_start
        self._log.info("train end: wall=%.1f min global_step=%d",
                       wall / 60, trainer.global_step)


__all__ = [
    "CSVLogger",
    "WandbLogger",
    "VerboseProgressCallback",
    "make_loggers",
    "make_run_dir",
    "make_run_id",
    "setup_text_logger",
]
