"""Local dataclass-config for the PreJEPA trainer.

Mirrors upstream ``config/prejepa.yaml`` — frozen pretrained vision
backbone + causal predictor over video clips.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ._common import OptimizerCfg, PredictorCfg, TrainerCfg, WandbCfg


@dataclass
class BackboneCfg:
    name: str = "facebook/dinov2-small"
    type: str = "dinov2_small"
    is_video_encoder: bool = False


@dataclass
class WMEncoding:
    proprio: int = 10
    action: int = 10


@dataclass
class WMCfg:
    history_size: int = 3
    num_preds: int = 1
    encoding: WMEncoding = field(default_factory=WMEncoding)


@dataclass
class PreJEPAConfig:
    output_model_name: str = "prejepa"
    seed: int = 42

    dataset_name: str = "pusht_expert_train_video"
    cache_dir: str | None = None
    n_steps: int = 4  # = wm.num_preds + wm.history_size
    frameskip: int = 5

    batch_size: int = 32
    num_workers: int = 16
    train_split: float = 0.9

    image_size: int = 224
    patch_size: int = 14
    log_every_n_steps: int = 50
    save_every_n_epochs: int = 5

    backbone: BackboneCfg = field(default_factory=BackboneCfg)
    wm: WMCfg = field(default_factory=WMCfg)
    predictor: PredictorCfg = field(default_factory=lambda: PredictorCfg(size="small"))
    optimizer: OptimizerCfg = field(default_factory=lambda: OptimizerCfg(lr=5e-4, weight_decay=0.0))
    trainer: TrainerCfg = field(default_factory=lambda: TrainerCfg(
        max_epochs=10, precision="16-mixed", strategy="ddp",
    ))
    wandb: WandbCfg = field(default_factory=WandbCfg)
