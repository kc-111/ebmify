"""Shared dataclasses reused across training-method configs.

These mirror upstream's nested yaml structure field-for-field so the
local trainers can call into the same swm/spt code paths without
reshaping data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Trainer / loader / optimizer / data


@dataclass
class TrainerCfg:
    max_epochs: int = 100
    devices: Any = "auto"
    accelerator: str = "gpu"
    precision: str = "bf16-mixed"
    strategy: str = "auto"
    gradient_clip_val: float = 1.0

    def asdict_for_pl(self) -> dict[str, Any]:
        """Subset of fields safe to splat into ``pl.Trainer(**...)``."""
        out = {
            "max_epochs": self.max_epochs,
            "devices": self.devices,
            "accelerator": self.accelerator,
            "precision": self.precision,
            "gradient_clip_val": self.gradient_clip_val,
        }
        if self.strategy and self.strategy != "auto":
            out["strategy"] = self.strategy
        return out


@dataclass
class LoaderCfg:
    batch_size: int = 128
    num_workers: int = 6
    drop_last: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 3
    pin_memory: bool = True
    shuffle: bool = True


@dataclass
class OptimizerCfg:
    type: str = "AdamW"
    lr: float = 5e-5
    weight_decay: float = 1e-3


@dataclass
class DataCfg:
    """Mirrors ``config/data/<x>.yaml``."""
    name: str = "pusht_expert_train_video"
    num_steps: int = 4  # = wm.num_preds + wm.history_size
    frameskip: int = 5
    keys_to_load: list[str] = field(
        default_factory=lambda: ["pixels", "action", "observation"]
    )
    keys_to_cache: list[str] = field(default_factory=lambda: ["action", "observation"])
    keys_to_merge: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# WandB / logging


@dataclass
class WandbCfg:
    enabled: bool = False
    project: str = "stable-wm"
    entity: str = "stable-wm"
    name: str | None = None
    id: str | None = None


# ---------------------------------------------------------------------------
# Predictor (transformer-based) — same shape across all methods


@dataclass
class PredictorCfg:
    depth: int = 6
    heads: int = 16
    mlp_dim: int = 2048
    dim_head: int = 64
    dropout: float = 0.0
    emb_dropout: float = 0.0
    # prejepa uses ``size`` as a label only
    size: str = "small"


# ---------------------------------------------------------------------------
# Goal-conditioned RL: shared goal-sampling block


@dataclass
class GoalProbCfg:
    random: float = 0.3
    geometric_future: float = 0.5
    uniform_future: float = 0.0
    current: float = 0.2

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.random, self.geometric_future, self.uniform_future, self.current)


# ---------------------------------------------------------------------------
# DinoWM block (shared across gcbc/gciql/gcivl/hilp)


@dataclass
class DinoWMCfg:
    history_size: int = 3
    num_preds: int = 1
    td_offset: int = 1
    use_proprio_encoder: bool = False
    proprio_dim: int = 4
    proprio_embed_dim: int = 10
    action_dim: int = 2
    action_embed_dim: int = 10
