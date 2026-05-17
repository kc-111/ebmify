"""Local dataclass-config for the GCBC trainer.

Mirrors upstream ``config/gcbc.yaml`` — DINO or ViT-tiny encoder +
``swm.wm.gcrl.Predictor``, behavioral cloning loss only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ._common import (
    DinoWMCfg,
    OptimizerCfg,
    PredictorCfg,
    TrainerCfg,
    WandbCfg,
)


@dataclass
class GCBCConfig:
    output_model_name: str = "dinogcbc"
    dataset_name: str = "pusht_expert_train.h5"
    seed: int = 42

    image_size: int = 224
    patch_size: int = 14
    n_steps: int = 3  # = dinowm.num_preds + dinowm.history_size - 1
    frameskip: int = 1

    batch_size: int = 128
    num_workers: int = 8
    train_split: float = 0.9
    train_subset_fraction: float = 1.0
    log_every_n_steps: int = 50
    save_every_n_epochs: int = 5

    encoder_type: str = "dino"  # 'dino' | 'vit_tiny'
    predictor_lr: float = 1e-4
    proprio_encoder_lr: float = 1e-4
    encoder_lr: float = 1e-4

    dinowm: DinoWMCfg = field(default_factory=DinoWMCfg)
    predictor: PredictorCfg = field(default_factory=lambda: PredictorCfg(dropout=0.0, emb_dropout=0.0))
    optimizer: OptimizerCfg = field(default_factory=lambda: OptimizerCfg(lr=1e-4, weight_decay=0.0))
    trainer: TrainerCfg = field(default_factory=lambda: TrainerCfg(
        max_epochs=100, devices=1, precision="bf16-mixed",
    ))
    wandb: WandbCfg = field(default_factory=WandbCfg)
