"""Evaluate val loss / perplexity / bits-per-token for a trained checkpoint."""
from __future__ import annotations

import argparse
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import OWT_ROOT  # noqa: F401, E402

from model import GPT, GPTConfig  # noqa: E402
from owt_data import get_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--n-batches", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=None,
                    help="override the checkpoint's seq_len")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def load_model(ckpt_path: Path, device: torch.device, seq_len_override: int | None) -> GPT:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = blob["config"]
    seq_len = seq_len_override or cfg["seq_len"]
    model = GPT(
        GPTConfig(
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            d_model=cfg["d_model"],
            seq_len=seq_len,
            vocab_size=cfg.get("vocab_size", 50257),
            arch=cfg["arch"],
        )
    ).to(device)
    model.load_state_dict(blob["state_dict"], strict=False)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model = load_model(args.ckpt, device, args.seq_len)
    seq_len = model.cfg.seq_len

    use_bf16 = args.dtype == "bf16" and device.type == "cuda"
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )

    losses = torch.zeros(args.n_batches)
    with torch.no_grad():
        for i in range(args.n_batches):
            x, y = get_batch("val", args.batch, seq_len, device)
            with autocast_ctx:
                _, loss = model(x, y)
            losses[i] = loss.item()

    mean_loss = losses.mean().item()
    print(f"ckpt:           {args.ckpt}")
    print(f"arch:           {model.cfg.arch}")
    print(f"seq_len:        {seq_len}")
    print(f"batches:        {args.n_batches} x {args.batch}")
    print(f"mean_loss:      {mean_loss:.4f}")
    print(f"perplexity:     {math.exp(mean_loss):.3f}")
    print(f"bits_per_token: {mean_loss / math.log(2):.4f}")


if __name__ == "__main__":
    main()
