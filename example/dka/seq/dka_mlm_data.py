"""BERT-style masked-token batches built on top of openwebtext_lm's bins.

Reuses ``train.bin`` / ``val.bin`` produced by
``example/openwebtext_lm/prepare.py``. For each sampled window:

  - 15% of positions are picked.
  - Of those, 80% replaced with the mask sentinel (GPT-2 EOT = 50256),
    10% replaced with a uniform random token, 10% left unchanged
    (standard BERT recipe).
  - Returns ``(x_input, x_orig, mask)`` so the loss can be computed only
    at masked positions.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import EXAMPLE_ROOT  # noqa: F401, E402

OWT_DIR = EXAMPLE_ROOT / "openwebtext_lm"
MASK_ID = 50256  # GPT-2 EOT, reused as [MASK] for the demo
VOCAB_SIZE = 50257


def _bin_path(split: Literal["train", "val"]) -> Path:
    return OWT_DIR / f"{split}.bin"


def assert_bins_present() -> None:
    miss = [p for p in (_bin_path("train"), _bin_path("val")) if not p.exists()]
    if miss:
        raise FileNotFoundError(
            "openwebtext_lm bins missing: "
            + ", ".join(str(p) for p in miss)
            + "\nRun:  python example/openwebtext_lm/prepare.py"
        )


def get_masked_batch(
    split: Literal["train", "val"],
    batch_size: int,
    seq_len: int,
    device: torch.device | str,
    *,
    mask_frac: float = 0.15,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(x_input, x_orig, mask)``.

    ``x_input`` is the corrupted sequence the model sees; ``x_orig`` is
    the ground-truth tokens; ``mask`` (bool) marks the positions where
    the loss should be evaluated.
    """
    assert_bins_present()
    arr = np.memmap(_bin_path(split), dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(arr) - seq_len - 1, size=batch_size)
    orig = np.stack([arr[i : i + seq_len].astype(np.int64) for i in ix])
    orig_t = torch.from_numpy(orig)

    # Sample mask positions
    probs = torch.rand(batch_size, seq_len)
    mask = probs < mask_frac                                # (B, L) bool

    # Of the masked positions, decide replacement strategy
    rand_role = torch.rand(batch_size, seq_len)
    do_mask = mask & (rand_role < 0.8)
    do_rand = mask & (rand_role >= 0.8) & (rand_role < 0.9)
    # remaining 10% of masked positions keep the original token

    x_input = orig_t.clone()
    x_input[do_mask] = MASK_ID
    if do_rand.any():
        rand_tokens = torch.randint(0, VOCAB_SIZE, (int(do_rand.sum().item()),),
                                    dtype=torch.long)
        x_input[do_rand] = rand_tokens

    pin = torch.cuda.is_available() and str(device) != "cpu"
    if pin:
        x_input = x_input.pin_memory().to(device, non_blocking=True)
        orig_t = orig_t.pin_memory().to(device, non_blocking=True)
        mask = mask.pin_memory().to(device, non_blocking=True)
    else:
        x_input = x_input.to(device)
        orig_t = orig_t.to(device)
        mask = mask.to(device)
    return x_input, orig_t, mask
