"""OpenWebText token-stream loader.

Loads ``train.bin`` / ``val.bin`` produced by ``prepare.py`` as uint16
numpy memmaps and serves random fixed-length windows for next-token
prediction. Re-opening the memmap inside ``get_batch`` is intentional:
it avoids file-handle leaks across long runs and survives any future
worker forks.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch

OWT_ROOT = Path(__file__).resolve().parent
META_PATH = OWT_ROOT / "meta.json"


def _bin_path(split: Literal["train", "val"]) -> Path:
    return OWT_ROOT / f"{split}.bin"


def load_meta() -> dict:
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"{META_PATH} not found; run example/openwebtext_lm/prepare.py first."
        )
    return json.loads(META_PATH.read_text())


def get_batch(
    split: Literal["train", "val"],
    batch_size: int,
    seq_len: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    path = _bin_path(split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run example/openwebtext_lm/prepare.py first."
        )
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    ix = np.random.randint(0, len(arr) - seq_len - 1, size=batch_size)
    x = torch.from_numpy(
        np.stack([arr[i : i + seq_len].astype(np.int64) for i in ix])
    )
    y = torch.from_numpy(
        np.stack([arr[i + 1 : i + 1 + seq_len].astype(np.int64) for i in ix])
    )
    pin = torch.cuda.is_available() and str(device) != "cpu"
    if pin:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def lm_ckpt_path(model_name: str, arch: str) -> Path:
    cache_dir = OWT_ROOT / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"owt_lm_{arch}_{model_name}.pt"
