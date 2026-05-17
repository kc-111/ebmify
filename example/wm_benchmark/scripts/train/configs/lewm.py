"""Local dataclass-config for the LeWM trainer.

Mirrors upstream ``config/lewm.yaml``: ViT-tiny encoder + transformer
predictor + SIGReg projector loss.
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
    type: str = "lewm"
    history_size: int = 3
    num_preds: int = 1
    embed_dim: int = 192
    use_proprio: bool = False


@dataclass
class SigregCfg:
    weight: float = 0.09
    kwargs: dict = field(default_factory=lambda: {"knots": 17, "num_proj": 1024})


@dataclass
class LossCfg:
    sigreg: SigregCfg = field(default_factory=SigregCfg)


@dataclass
class LeWMConfig:
    output_model_name: str = "lewm"
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
