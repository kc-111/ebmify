"""Local dataclass-config for the PLDM trainer.

Mirrors upstream ``config/pldm.yaml`` — ViT-tiny + predictor + 8-term
loss with per-term ``enabled``/``weight``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ._common import (
    DataCfg,
    LoaderCfg,
    OptimizerCfg,
    PredictorCfg,
    TrainerCfg,
    WandbCfg,
)


@dataclass
class WMCfg:
    type: str = "lewm"  # upstream pldm.yaml literally sets this to 'lewm'
    history_size: int = 3
    num_preds: int = 1
    embed_dim: int = 192
    use_proprio: bool = False


@dataclass
class LossTerm:
    enabled: bool = True
    weight: float = 0.0


@dataclass
class LossCfg:
    sigreg: LossTerm = field(default_factory=lambda: LossTerm(enabled=True, weight=0.0))
    temp_straight: LossTerm = field(default_factory=lambda: LossTerm(enabled=False, weight=0.1))
    std: LossTerm = field(default_factory=lambda: LossTerm(enabled=True, weight=18.0))
    std_t: LossTerm = field(default_factory=lambda: LossTerm(enabled=True, weight=0.7))
    cov: LossTerm = field(default_factory=lambda: LossTerm(enabled=True, weight=12.0))
    cov_t: LossTerm = field(default_factory=lambda: LossTerm(enabled=True, weight=0.0))
    temp_align: LossTerm = field(default_factory=lambda: LossTerm(enabled=True, weight=0.2))
    idm: LossTerm = field(default_factory=lambda: LossTerm(enabled=True, weight=0.0))


@dataclass
class PLDMConfig:
    output_model_name: str = "pldm"
    seed: int = 3072
    img_size: int = 224
    patch_size: int = 14
    encoder_scale: str = "tiny"
    encoder_resnet9: bool = False
    train_value_function: bool = False
    projector_loss_weight: float = 1e-3
    train_split: float = 0.9
    log_every_n_steps: int = 50
    save_every_n_epochs: int = 5

    trainer: TrainerCfg = field(default_factory=lambda: TrainerCfg(precision="bf16"))
    loader: LoaderCfg = field(default_factory=LoaderCfg)
    optimizer: OptimizerCfg = field(default_factory=lambda: OptimizerCfg(lr=5e-5, weight_decay=1e-3))
    data: DataCfg = field(default_factory=lambda: DataCfg(
        name="pusht_expert_train_video",
        num_steps=4,
        frameskip=5,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
    ))
    wm: WMCfg = field(default_factory=WMCfg)
    predictor: PredictorCfg = field(default_factory=PredictorCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    wandb: WandbCfg = field(default_factory=WandbCfg)
