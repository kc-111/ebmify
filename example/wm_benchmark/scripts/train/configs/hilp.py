"""Local dataclass-config for the HILP trainer.

Mirrors upstream ``config/hilp.yaml`` — Hilbert foundation policy
(MetricValuePredictor over latent space) with goal_probabilities skewed
toward random/geometric_future.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ._common import (
    DinoWMCfg,
    GoalProbCfg,
    OptimizerCfg,
    PredictorCfg,
    TrainerCfg,
    WandbCfg,
)


@dataclass
class HILPConfig:
    output_model_name: str = "dinohilp"
    dataset_name: str = "pusht_expert_train.h5"
    seed: int = 42

    image_size: int = 224
    patch_size: int = 14
    n_steps: int = 4
    frameskip: int = 1

    batch_size: int = 128
    num_workers: int = 8
    train_split: float = 0.9
    log_every_n_steps: int = 50
    save_every_n_epochs: int = 5

    encoder_type: str = "dino"
    train_value: bool = True

    predictor_lr: float = 3e-4
    proprio_encoder_lr: float = 3e-4
    action_encoder_lr: float = 3e-4
    encoder_lr: float = 3e-4

    discount: float = 0.99
    expectile: float = 0.9
    awr_alpha: float = 3.0
    value_ema_tau: float = 0.995
    goal_gamma: float = 0.99

    dinowm: DinoWMCfg = field(default_factory=lambda: DinoWMCfg(td_offset=1))
    predictor: PredictorCfg = field(default_factory=lambda: PredictorCfg(dropout=0.0, emb_dropout=0.0))
    optimizer: OptimizerCfg = field(default_factory=lambda: OptimizerCfg(lr=3e-4, weight_decay=0.0))
    trainer: TrainerCfg = field(default_factory=lambda: TrainerCfg(
        max_epochs=100, devices=1, precision="bf16-mixed", gradient_clip_val=1.0,
    ))
    goal_probabilities: GoalProbCfg = field(default_factory=lambda: GoalProbCfg(
        random=0.375, geometric_future=0.625, uniform_future=0.0, current=0.0,
    ))
    actor_goal_probabilities: GoalProbCfg = field(default_factory=lambda: GoalProbCfg(
        random=0.5, geometric_future=0.0, uniform_future=0.5, current=0.0,
    ))
    wandb: WandbCfg = field(default_factory=WandbCfg)
