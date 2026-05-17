"""BERT-style MLM training with a DKA stack on OpenWebText tokens.

A short demo, not a real BERT run. Defaults keep the run under a few
minutes on a single GPU; use the §C.1 ablation knobs to compare
rank rules, output forms, and recurrent depths.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import DKA_ROOT  # noqa: F401, E402

from dka_mlm_data import VOCAB_SIZE, get_masked_batch  # noqa: E402
from ebmify.models.dka import DKABlock, RMSNorm, SoftmaxAttentionBlock  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", choices=["dka", "softmax"], default="dka",
                    help="block type: 'dka' = DKABlock; 'softmax' = standard MHA+FFN baseline")
    ap.add_argument("--max-iters", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    # DKA-only
    ap.add_argument("--d-feat", type=int, default=64)
    ap.add_argument("--phi-hidden", type=int, default=-1,
                    help="hidden dim of φ's 2-layer MLP; -1 = 4*d_feat, 0 = single Linear")
    ap.add_argument("--rank-rule", choices=["full", "fixed_r", "adaptive"], default="adaptive")
    ap.add_argument("--fixed-r", type=int, default=16)
    ap.add_argument("--rho", type=float, default=1e-2)
    ap.add_argument("--r-max", type=int, default=64)
    ap.add_argument("--output-form", choices=["kalman", "none", "gated"], default="kalman")
    ap.add_argument("--recurrent-depth", type=int, default=1)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--use-ffn", action="store_true",
                    help="add a SwiGLU FFN after each DKA layer (off by default)")
    ap.add_argument("--phi-type", choices=["swiglu", "mlp"], default="swiglu",
                    help="DKA: SwiGLU (default) or GELU 2-layer MLP for φ")
    # Shared
    ap.add_argument("--n-heads", type=int, default=4,
                    help="# of attention heads (d_model must divide; applies to both arches)")
    ap.add_argument("--causal", action="store_true",
                    help="causal variant. NB: BERT-style MLM is bidirectional by design — "
                         "use this only if you really want causal masking.")
    # Softmax-only
    ap.add_argument("--ffn-hidden", type=int, default=-1)
    # Training
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-iters", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--mask-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="default")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def _build_block(args: argparse.Namespace) -> nn.Module:
    if args.arch == "dka":
        phi_hidden = None if args.phi_hidden < 0 else args.phi_hidden
        return DKABlock(
            d_model=args.d_model,
            d_feat=args.d_feat,
            n_heads=args.n_heads,
            causal=args.causal,
            phi_type=args.phi_type,
            recurrent_depth=args.recurrent_depth,
            phi_hidden=phi_hidden,
            rank_rule=args.rank_rule,
            fixed_r=args.fixed_r,
            rho=args.rho,
            r_max=args.r_max,
            output_form=args.output_form,
            lam=args.lam,
            target="x",
            use_ffn=args.use_ffn,
        )
    if args.arch == "softmax":
        ffn_hidden = None if args.ffn_hidden < 0 else args.ffn_hidden
        return SoftmaxAttentionBlock(
            d_model=args.d_model,
            n_heads=args.n_heads,
            causal=args.causal,
            ffn_hidden=ffn_hidden,
        )
    raise ValueError(f"unknown arch: {args.arch!r}")


class MLMModel(nn.Module):
    def __init__(self, args: argparse.Namespace, vocab_size: int) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, args.d_model)
        self.pos_emb = nn.Embedding(args.seq_len, args.d_model)
        self.blocks = nn.ModuleList([_build_block(args) for _ in range(args.n_layers)])
        self.out_norm = RMSNorm(args.d_model)
        self.head = nn.Linear(args.d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0)
        h = self.tok_emb(ids) + self.pos_emb(pos)
        for blk in self.blocks:
            h, _ = blk(h)
        return self.head(self.out_norm(h))


def lr_at(step: int, *, lr_max: float, warmup: int, total: int) -> float:
    if step < warmup:
        return lr_max * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return lr_max * max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate(model, *, args, device, autocast_ctx):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, orig, mask = get_masked_batch(
                split, args.batch, args.seq_len, device,
                mask_frac=args.mask_frac, seed=args.seed + 10_000 + k,
            )
            with autocast_ctx:
                logits = model(x)
                tgt = orig[mask]
                logit = logits[mask]
                loss = F.cross_entropy(logit, tgt)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model = MLMModel(args, VOCAB_SIZE).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"arch={args.arch}  params: trainable={n_params:,}", flush=True)

    use_bf16 = args.dtype == "bf16" and device.type == "cuda"
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )

    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    log_dir = DKA_ROOT / "logs"
    cache_dir = DKA_ROOT / "cache"
    log_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    stem = f"{args.arch}_mlm_{args.tag}"
    log_path = log_dir / f"{stem}.jsonl"
    ckpt_path = cache_dir / f"{stem}.pt"
    log_path.write_text("")

    t_run = time.time()
    for step in range(args.max_iters + 1):
        lr = lr_at(step, lr_max=args.lr, warmup=args.warmup_iters, total=args.max_iters)
        for g in opt.param_groups:
            g["lr"] = lr

        if step > 0 and step % args.eval_interval == 0:
            losses = estimate(model, args=args, device=device, autocast_ctx=autocast_ctx)
            elapsed = time.time() - t_run
            uniform_ce = math.log(VOCAB_SIZE)
            print(
                f"step {step:5d} | lr {lr:.2e} | "
                f"train {losses['train']:.4f} | val {losses['val']:.4f} "
                f"(uniform={uniform_ce:.2f}) | elapsed {elapsed:.0f}s",
                flush=True,
            )
            with log_path.open("a") as f:
                f.write(json.dumps({
                    "step": step,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "uniform_ce": uniform_ce,
                    "lr": lr,
                    "elapsed_s": elapsed,
                }) + "\n")

        if step == args.max_iters:
            break

        x, orig, mask = get_masked_batch(
            "train", args.batch, args.seq_len, device,
            mask_frac=args.mask_frac, seed=args.seed + step,
        )
        with autocast_ctx:
            logits = model(x)
            tgt = orig[mask]
            logit = logits[mask]
            loss = F.cross_entropy(logit, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % 50 == 0:
            print(
                f"step {step:5d} | lr {lr:.2e} | "
                f"train_loss {loss.item():.4f}",
                flush=True,
            )

    torch.save({"state_dict": model.state_dict(), "config": vars(args)}, ckpt_path)
    print(f"done. checkpoint: {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
