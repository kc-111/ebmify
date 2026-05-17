"""Synthetic associative recall training: DKA vs softmax.

Task format per sample (length = 2*N + 2):
    [k1, v1, k2, v2, ..., kN, vN, q_key, <ANS>]
Target:
    value paired with q_key (predict at final <ANS> position).

This is a simple synthetic benchmark to compare retrieval behavior and
throughput/optimization between DKA and softmax blocks.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import DKA_ROOT  # noqa: F401, E402

from ebmify.models.dka import DKABlock, RMSNorm, SoftmaxAttentionBlock  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", choices=["dka", "softmax"], default="dka")
    ap.add_argument("--max-iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n-pairs", type=int, default=256)
    ap.add_argument("--key-vocab", type=int, default=128)
    ap.add_argument("--value-vocab", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    # DKA-only knobs
    ap.add_argument("--d-feat", type=int, default=64)
    ap.add_argument("--phi-hidden", type=int, default=-1)
    ap.add_argument("--phi-type", choices=["swiglu", "mlp"], default="swiglu")
    ap.add_argument("--d-value", type=int, default=-1)
    ap.add_argument("--value-type", choices=["linear", "swiglu", "mlp"], default="swiglu")
    ap.add_argument("--value-hidden", type=int, default=-1)
    ap.add_argument("--output-form", choices=["kalman", "none", "gated"], default="kalman")
    ap.add_argument("--recurrent-depth", type=int, default=1)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--use-ffn", action="store_true")
    # Shared/softmax
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--ffn-hidden", type=int, default=-1)
    # Optimization
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="default")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def _build_block(args: argparse.Namespace) -> nn.Module:
    if args.arch == "dka":
        phi_hidden = None if args.phi_hidden < 0 else args.phi_hidden
        value_hidden = None if args.value_hidden < 0 else args.value_hidden
        d_value = None if args.d_value < 0 else args.d_value
        return DKABlock(
            d_model=args.d_model,
            d_feat=args.d_feat,
            n_heads=args.n_heads,
            causal=args.causal,
            phi_type=args.phi_type,
            phi_hidden=phi_hidden,
            target="value",
            d_value=d_value,
            value_type=args.value_type,
            value_hidden=value_hidden,
            output_form=args.output_form,
            recurrent_depth=args.recurrent_depth,
            lam=args.lam,
            use_ffn=args.use_ffn,
        )
    ffn_hidden = None if args.ffn_hidden < 0 else args.ffn_hidden
    return SoftmaxAttentionBlock(
        d_model=args.d_model,
        n_heads=args.n_heads,
        causal=args.causal,
        ffn_hidden=ffn_hidden,
    )


def _sample_assoc_batch(
    *,
    batch: int,
    n_pairs: int,
    key_vocab: int,
    value_vocab: int,
    ans_token: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one synthetic associative-recall batch."""
    keys = torch.randint(0, key_vocab, (batch, n_pairs), generator=generator)
    vals = torch.randint(0, value_vocab, (batch, n_pairs), generator=generator)
    q_idx = torch.randint(0, n_pairs, (batch,), generator=generator)
    b_idx = torch.arange(batch)
    q_keys = keys[b_idx, q_idx]
    target_vals = vals[b_idx, q_idx] + key_vocab

    seq_len = 2 * n_pairs + 2
    x = torch.empty(batch, seq_len, dtype=torch.long)
    x[:, 0 : 2 * n_pairs : 2] = keys
    x[:, 1 : 2 * n_pairs : 2] = vals + key_vocab
    x[:, -2] = q_keys
    x[:, -1] = ans_token
    return x.to(device), target_vals.to(device)


class AssocRecallModel(nn.Module):
    def __init__(self, args: argparse.Namespace, vocab_size: int, seq_len: int) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, args.d_model)
        self.pos_emb = nn.Embedding(seq_len, args.d_model)
        self.blocks = nn.ModuleList([_build_block(args) for _ in range(args.n_layers)])
        self.out_norm = RMSNorm(args.d_model)
        self.head = nn.Linear(args.d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        h = self.tok_emb(ids) + self.pos_emb(pos)
        for blk in self.blocks:
            h, _ = blk(h, return_info=False)
        return self.head(self.out_norm(h))


@torch.no_grad()
def estimate(
    model: nn.Module,
    *,
    args: argparse.Namespace,
    device: torch.device,
    autocast_ctx,
    seed_offset: int,
    ans_token: int,
) -> dict[str, float]:
    model.eval()
    losses = torch.zeros(args.eval_iters)
    accs = torch.zeros(args.eval_iters)
    gen = torch.Generator().manual_seed(args.seed + seed_offset)
    for i in range(args.eval_iters):
        x, tgt = _sample_assoc_batch(
            batch=args.batch,
            n_pairs=args.n_pairs,
            key_vocab=args.key_vocab,
            value_vocab=args.value_vocab,
            ans_token=ans_token,
            device=device,
            generator=gen,
        )
        with autocast_ctx:
            logits = model(x)
            pred = logits[:, -1, :]
            loss = F.cross_entropy(pred, tgt)
        acc = (pred.argmax(dim=-1) == tgt).float().mean()
        losses[i] = loss.item()
        accs[i] = acc.item()
    model.train()
    return {"loss": losses.mean().item(), "acc": accs.mean().item()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    seq_len = 2 * args.n_pairs + 2
    ans_token = args.key_vocab + args.value_vocab + 1
    vocab_size = args.key_vocab + args.value_vocab + 2

    model = AssocRecallModel(args, vocab_size=vocab_size, seq_len=seq_len).to(device)
    if args.compile:
        model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"arch={args.arch} params={n_params:,} seq_len={seq_len} "
        f"compile={args.compile} causal={args.causal}",
        flush=True,
    )

    use_bf16 = args.dtype == "bf16" and device.type == "cuda"
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    )

    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    log_dir = DKA_ROOT / "logs"
    cache_dir = DKA_ROOT / "cache"
    log_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    stem = f"{args.arch}_assoc_recall_{args.tag}"
    log_path = log_dir / f"{stem}.jsonl"
    ckpt_path = cache_dir / f"{stem}.pt"
    log_path.write_text("")

    train_gen = torch.Generator().manual_seed(args.seed + 1234)
    t_run = time.time()
    for step in range(args.max_iters + 1):
        if step > 0 and step % args.eval_interval == 0:
            tr = estimate(
                model,
                args=args,
                device=device,
                autocast_ctx=autocast_ctx,
                seed_offset=10_000,
                ans_token=ans_token,
            )
            te = estimate(
                model,
                args=args,
                device=device,
                autocast_ctx=autocast_ctx,
                seed_offset=20_000,
                ans_token=ans_token,
            )
            elapsed = time.time() - t_run
            print(
                f"step {step:5d} | train loss {tr['loss']:.4f} acc {tr['acc']:.3f} | "
                f"test loss {te['loss']:.4f} acc {te['acc']:.3f} | elapsed {elapsed:.0f}s",
                flush=True,
            )
            with log_path.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "step": step,
                            "train_loss": tr["loss"],
                            "train_acc": tr["acc"],
                            "test_loss": te["loss"],
                            "test_acc": te["acc"],
                            "elapsed_s": elapsed,
                        }
                    )
                    + "\n"
                )

        if step == args.max_iters:
            break

        x, tgt = _sample_assoc_batch(
            batch=args.batch,
            n_pairs=args.n_pairs,
            key_vocab=args.key_vocab,
            value_vocab=args.value_vocab,
            ans_token=ans_token,
            device=device,
            generator=train_gen,
        )
        with autocast_ctx:
            logits = model(x)
            pred = logits[:, -1, :]
            loss = F.cross_entropy(pred, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % 50 == 0:
            acc = (pred.argmax(dim=-1) == tgt).float().mean().item()
            print(
                f"step {step:5d} | train_loss {loss.item():.4f} train_acc {acc:.3f}",
                flush=True,
            )

    torch.save({"state_dict": model.state_dict(), "config": vars(args)}, ckpt_path)
    print(f"done. checkpoint: {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
