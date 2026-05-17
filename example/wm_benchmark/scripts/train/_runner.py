"""Shared run-skeleton for every local trainer.

Each ``scripts/train/<method>.py`` defines two glue functions:

    def build(omega_cfg) -> (data_module, spt.Module)
    METHOD = "lewm"  # used for run_dir naming

…and then calls ``run_trainer(cfg, build, METHOD)``. ``run_trainer``
handles all the boring bits: seeding, run-dir creation, snapshot of the
config, logger wiring, callback wiring, checkpoint resume, and the final
``spt.Manager()`` call.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import lightning.pytorch as pl
import stable_pretraining as spt

# sys.path bootstrap so 'scripts.common' / '_paths' resolve regardless of cwd.
_BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
if str(_BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_ROOT))

from _paths import DATA_DIR  # noqa: E402
from scripts.common import checkpoint as ckptmod  # noqa: E402
from scripts.common import config as cfgmod  # noqa: E402
from scripts.common import logging as logmod  # noqa: E402
from scripts.common import seeding  # noqa: E402


def run_trainer(cfg: Any, build: Callable[[Any], tuple[Any, spt.Module]],
                method: str) -> Path:
    """Drive the training loop for ``method`` using ``build(cfg)``.

    ``cfg`` is the local dataclass cfg (not yet OmegaConf). ``build``
    returns ``(data_module, spt.Module)`` after converting cfg → OmegaConf
    on the upstream side.

    Returns the run directory.
    """
    seeding.set_seed(cfg.seed)

    run_id = logmod.make_run_id(prefix=method)
    run_dir = logmod.make_run_dir(method, run_id, DATA_DIR)

    log = logmod.setup_text_logger(run_dir)
    log.info("run_dir=%s", run_dir)

    cfgmod.save_yaml(cfg, run_dir / "config.yaml")

    data, module = build(cfg)

    wandb = getattr(cfg, "wandb", None)
    loggers = logmod.make_loggers(
        run_dir,
        wandb_enabled=bool(getattr(wandb, "enabled", False)),
        wandb_project=getattr(wandb, "project", None),
        wandb_entity=getattr(wandb, "entity", None),
        wandb_run_name=getattr(wandb, "name", None) or method,
        wandb_id=getattr(wandb, "id", None),
    )

    callbacks = [
        logmod.VerboseProgressCallback(every_n_steps=int(getattr(cfg, "log_every_n_steps", 50))),
        ckptmod.LocalSaveCkptCallback(
            run_dir,
            every_n_epochs=int(getattr(cfg, "save_every_n_epochs", 1)),
            cfg_payload=cfgmod.to_dict(cfg),
        ),
    ]

    trainer_kwargs = cfg.trainer.asdict_for_pl()
    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=callbacks,
        logger=loggers,
        num_sanity_val_steps=1,
        enable_checkpointing=True,
        default_root_dir=str(run_dir),
    )

    ckpt_path = ckptmod.resume_path(run_dir)
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data,
        ckpt_path=str(ckpt_path) if ckpt_path else None,
    )
    manager()
    log.info("done. run_dir=%s", run_dir)
    return run_dir


def run_two_phase_trainer(
    cfg: Any,
    build_value: Callable[[Any], tuple[Any, spt.Module]],
    build_actor: Callable[[Any, spt.Module], tuple[Any, spt.Module]],
    method: str,
) -> Path:
    """Two-phase variant for GC-IQL / GC-IVL / HILP.

    Phase 1 trains the value/critic model under
    ``cfg.goal_probabilities``. Phase 2 freezes those weights and trains
    an actor under ``cfg.actor_goal_probabilities`` via AWR.

    Each phase writes to its own subdir under ``data/runs/<method>/<run_id>/``.
    """
    seeding.set_seed(cfg.seed)
    run_id = logmod.make_run_id(prefix=method)
    run_dir = logmod.make_run_dir(method, run_id, DATA_DIR)
    log = logmod.setup_text_logger(run_dir)
    log.info("run_dir=%s", run_dir)
    cfgmod.save_yaml(cfg, run_dir / "config.yaml")

    cfg_payload = cfgmod.to_dict(cfg)
    wandb = getattr(cfg, "wandb", None)
    wb_enabled = bool(getattr(wandb, "enabled", False))
    wb_kw = dict(
        wandb_enabled=wb_enabled,
        wandb_project=getattr(wandb, "project", None),
        wandb_entity=getattr(wandb, "entity", None),
    )
    trainer_kwargs = cfg.trainer.asdict_for_pl()

    # ---- Phase 1: value/critic
    if bool(getattr(cfg, "train_value", True)):
        log.info("=== phase 1: value/critic ===")
        phase_dir = run_dir / "value"
        phase_dir.mkdir(parents=True, exist_ok=True)
        data_v, mod_v = build_value(cfg)
        loggers = logmod.make_loggers(
            phase_dir, **wb_kw,
            wandb_run_name=f"{method}-value", wandb_id=None,
        )
        callbacks = [
            logmod.VerboseProgressCallback(every_n_steps=int(getattr(cfg, "log_every_n_steps", 50))),
            ckptmod.LocalSaveCkptCallback(phase_dir,
                                          every_n_epochs=int(getattr(cfg, "save_every_n_epochs", 1)),
                                          cfg_payload=cfg_payload, key="value"),
        ]
        trainer = pl.Trainer(**trainer_kwargs, callbacks=callbacks, logger=loggers,
                              num_sanity_val_steps=1, enable_checkpointing=True,
                              default_root_dir=str(phase_dir))
        spt.Manager(trainer=trainer, module=mod_v, data=data_v,
                    ckpt_path=str(ckptmod.resume_path(phase_dir)) if ckptmod.resume_path(phase_dir) else None)()
    else:
        log.info("phase 1 skipped (train_value=False)")
        # Caller is responsible for building a value model ready to pass to the actor.
        data_v, mod_v = build_value(cfg)

    # ---- Phase 2: actor
    log.info("=== phase 2: actor ===")
    phase_dir = run_dir / "actor"
    phase_dir.mkdir(parents=True, exist_ok=True)
    data_a, mod_a = build_actor(cfg, mod_v)
    loggers = logmod.make_loggers(
        phase_dir, **wb_kw,
        wandb_run_name=f"{method}-actor", wandb_id=None,
    )
    callbacks = [
        logmod.VerboseProgressCallback(every_n_steps=int(getattr(cfg, "log_every_n_steps", 50))),
        ckptmod.LocalSaveCkptCallback(phase_dir,
                                      every_n_epochs=int(getattr(cfg, "save_every_n_epochs", 1)),
                                      cfg_payload=cfg_payload, key="actor"),
    ]
    trainer = pl.Trainer(**trainer_kwargs, callbacks=callbacks, logger=loggers,
                          num_sanity_val_steps=1, enable_checkpointing=True,
                          default_root_dir=str(phase_dir))
    spt.Manager(trainer=trainer, module=mod_a, data=data_a,
                ckpt_path=str(ckptmod.resume_path(phase_dir)) if ckptmod.resume_path(phase_dir) else None)()
    log.info("done. run_dir=%s", run_dir)
    return run_dir
