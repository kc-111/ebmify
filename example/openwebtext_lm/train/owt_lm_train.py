"""Train a small GPT (llama-style or gpt2-style) on OpenWebText.

Expects ``train.bin`` / ``val.bin`` / ``meta.json`` produced by
``prepare.py``. Single-GPU, bf16 by default, AdamW + cosine LR with
linear warmup, grad-clip, periodic eval + checkpoint.

Example:

    python example/openwebtext_lm/train/owt_lm_train.py \\
        --arch llama --model tiny --max-iters 20000 \\
        --batch 32 --grad-accum 4 --eval-interval 500
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import OWT_ROOT  # noqa: F401, E402  (registers subdir paths)

from model import PRESETS, GPT  # noqa: E402
from owt_data import get_batch, lm_ckpt_path, load_meta  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", choices=["llama", "gpt2"], default="llama")
    ap.add_argument("--model", choices=list(PRESETS), default="tiny")
    ap.add_argument("--seq-len", type=int, default=None,
                    help="override preset seq_len (default: preset)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--warmup-iters", type=int, default=200)
    ap.add_argument("--max-iters", type=int, default=20000)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--eval-iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model (PT 2.x)")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def build_optimizer(model: torch.nn.Module, lr: float, wd: float, betas):
    """AdamW with weight decay on 2D params only (matrices + embeddings)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = torch.cuda.is_available()
    try:
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
    except TypeError:
        return torch.optim.AdamW(groups, lr=lr, betas=betas)


def lr_at(step: int, *, lr_max: float, lr_min: float, warmup: int, total: int) -> float:
    if step < warmup:
        return lr_max * (step + 1) / max(1, warmup)
    if step >= total:
        return lr_min
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, *, batch_size, seq_len, device, eval_iters, autocast_ctx):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, batch_size, seq_len, device)
            with autocast_ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    meta = load_meta()
    assert meta["vocab_size"] == 50257, f"unexpected vocab in meta.json: {meta}"

    preset = PRESETS[args.model]
    seq_len = args.seq_len or preset["seq_len"]

    device = torch.device(args.device)
    model = GPT.from_preset(
        args.model, arch=args.arch, vocab_size=meta["vocab_size"], seq_len=seq_len
    ).to(device)

    use_bf16 = args.dtype == "bf16" and device.type == "cuda"
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )

    if args.compile:
        model = torch.compile(model)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"model: arch={args.arch} preset={args.model} d={preset['d_model']} "
        f"layers={preset['n_layer']} heads={preset['n_head']} seq_len={seq_len}",
        flush=True,
    )
    print(f"params: total={n_total:,} trainable={n_train:,}", flush=True)
    print(f"tokens/step = batch*grad_accum*seq_len = "
          f"{args.batch * args.grad_accum * seq_len:,}", flush=True)

    opt = build_optimizer(
        model, lr=args.lr, wd=args.weight_decay, betas=(args.beta1, args.beta2)
    )

    log_dir = OWT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"owt_lm_{args.arch}_{args.model}.jsonl"
    ckpt_path = lm_ckpt_path(args.model, args.arch)

    cfg_for_ckpt = {
        **vars(args),
        **preset,
        "seq_len": seq_len,
        "arch": args.arch,
        "vocab_size": meta["vocab_size"],
    }

    def save_ckpt(step: int) -> None:
        sd = model.state_dict()
        # If compiled, strip the "_orig_mod." prefix for portability.
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
        torch.save({"state_dict": sd, "config": cfg_for_ckpt, "step": step}, ckpt_path)

    t_run = time.time()
    t_iter = time.time()
    log_path.write_text("")  # truncate previous run's log

    for step in range(args.max_iters + 1):
        lr = lr_at(
            step,
            lr_max=args.lr,
            lr_min=args.min_lr,
            warmup=args.warmup_iters,
            total=args.max_iters,
        )
        for g in opt.param_groups:
            g["lr"] = lr

        if step > 0 and step % args.eval_interval == 0:
            losses = estimate_loss(
                model,
                batch_size=args.batch,
                seq_len=seq_len,
                device=device,
                eval_iters=args.eval_iters,
                autocast_ctx=autocast_ctx,
            )
            elapsed = time.time() - t_run
            print(
                f"step {step:6d} | lr {lr:.2e} | "
                f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
                f"ppl_val {math.exp(losses['val']):.2f} | "
                f"elapsed {elapsed:.0f}s",
                flush=True,
            )
            with log_path.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "step": step,
                            "train_loss": losses["train"],
                            "val_loss": losses["val"],
                            "val_ppl": math.exp(losses["val"]),
                            "lr": lr,
                            "elapsed_s": elapsed,
                        }
                    )
                    + "\n"
                )
            save_ckpt(step)

        if step == args.max_iters:
            break

        # gradient accumulation
        opt.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            x, y = get_batch("train", args.batch, seq_len, device)
            with autocast_ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % 50 == 0:
            dt = time.time() - t_iter
            t_iter = time.time()
            print(
                f"step {step:6d} | lr {lr:.2e} | "
                f"train_loss {(loss.item() * args.grad_accum):.4f} | "
                f"dt {dt * 1000 / max(1, 50):.0f} ms/iter",
                flush=True,
            )

    save_ckpt(args.max_iters)
    print(f"done. checkpoint: {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
