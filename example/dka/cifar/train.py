"""Train a DKA/softmax patch-token classifier on CIFAR."""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import DKA_ROOT  # noqa: F401, E402

from dka_cifar_data import get_arrays, patchify  # noqa: E402
from ebmify.models.dka import DKABlock, RMSNorm, SoftmaxAttentionBlock  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", choices=["dka", "softmax"], default="dka",
                    help="block type: 'dka' = DKABlock; 'softmax' = standard MHA+FFN baseline")
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    ap.add_argument("--patch-size", type=int, choices=[2, 4, 8], default=4)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layers", type=int, default=4)
    # DKA-only knobs
    ap.add_argument("--d-feat", type=int, default=64)
    ap.add_argument("--phi-hidden", type=int, default=-1,
                    help="hidden dim of φ's 2-layer MLP; -1 = 4*d_feat, 0 = single Linear")
    ap.add_argument("--d-value", type=int, default=-1,
                    help="per-head value dim for learned value projection; -1 uses d_model/n_heads")
    ap.add_argument("--value-type", choices=["linear", "swiglu", "mlp"], default="swiglu",
                    help="value projection type")
    ap.add_argument("--value-hidden", type=int, default=-1,
                    help="value FFN hidden dim when --value-type is swiglu/mlp; -1 uses default")
    ap.add_argument("--kernel-strides", type=int, nargs="+", default=[1, 2, 4],
                    help="multi-kernel stride factors (default: 1 2 4)")
    ap.add_argument("--output-form", choices=["kalman", "none", "gated"], default="kalman")
    ap.add_argument("--recurrent-depth", type=int, default=1)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--use-ffn", action="store_true",
                    help="add a SwiGLU FFN after each DKA layer (off by default)")
    ap.add_argument("--phi-type", choices=["swiglu", "mlp"], default="swiglu",
                    help="DKA: SwiGLU (default) or GELU 2-layer MLP for φ")
    # Shared
    ap.add_argument("--n-heads", type=int, default=4,
                    help="# of attention heads (d_model must divide; applies to both DKA and softmax)")
    ap.add_argument("--causal", action="store_true",
                    help="use the causal variant of the chosen arch")
    # Softmax-only knobs
    ap.add_argument("--ffn-hidden", type=int, default=-1,
                    help="softmax FFN hidden; -1 = SwiGLU default (~8/3·d_model rounded up)")
    # Training
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--iters-per-epoch", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--compile", action="store_true",
                    help="wrap model with torch.compile for faster steady-state training")
    ap.add_argument("--amp", choices=["none", "bf16", "fp16"], default="bf16",
                    help="mixed precision mode on CUDA")
    ap.add_argument("--tf32", action="store_true",
                    help="enable TF32 matmul/cuDNN on CUDA")
    ap.add_argument("--val-iters", type=int, default=-1,
                    help="validation iterations per epoch; -1 = full val sweep")
    ap.add_argument("--log-interval", type=int, default=20,
                    help="print train progress every N train iterations")
    ap.add_argument("--debug-gate-every", type=int, default=0,
                    help="if >0, log variance-gate stats every N train steps")
    ap.add_argument("--noise-token-std", type=float, default=0.05,
                    help="std of Gaussian perturbation added to the learned noise token embedding")
    ap.add_argument("--noise-token-eval", action="store_true",
                    help="also sample Gaussian noise token perturbation at eval time")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="default")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def _build_block(args: argparse.Namespace, d_model: int) -> nn.Module:
    if args.arch == "dka":
        phi_hidden = None if args.phi_hidden < 0 else args.phi_hidden
        value_hidden = None if args.value_hidden < 0 else args.value_hidden
        d_value = None if args.d_value < 0 else args.d_value
        kernel_strides = tuple(sorted({max(1, int(s)) for s in args.kernel_strides}))
        return DKABlock(
            d_model=d_model,
            d_feat=args.d_feat,
            n_heads=args.n_heads,
            causal=args.causal,
            phi_type=args.phi_type,
            recurrent_depth=args.recurrent_depth,
            phi_hidden=phi_hidden,
            output_form=args.output_form,
            lam=args.lam,
            target="value",
            d_value=d_value,
            value_type=args.value_type,
            value_hidden=value_hidden,
            kernel_strides=kernel_strides,
            use_ffn=args.use_ffn,
        )
    if args.arch == "softmax":
        ffn_hidden = None if args.ffn_hidden < 0 else args.ffn_hidden
        return SoftmaxAttentionBlock(
            d_model=d_model,
            n_heads=args.n_heads,
            causal=args.causal,
            ffn_hidden=ffn_hidden,
        )
    raise ValueError(f"unknown arch: {args.arch!r}")


class PatchClassifier(nn.Module):
    """Linear patch embed -> stack of blocks -> CLS-token classifier head.

    Same wrapper for either arch; the block type is selected by ``args.arch``.
    """

    def __init__(self, patch_dim: int, n_classes: int, args: argparse.Namespace) -> None:
        super().__init__()
        self.in_proj = nn.Linear(patch_dim, args.d_model)
        self.noise_token_std = float(args.noise_token_std)
        self.noise_token_eval = bool(args.noise_token_eval)
        self.noise_token = nn.Parameter(torch.zeros(1, 1, args.d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.d_model))
        self.blocks = nn.ModuleList(
            [_build_block(args, args.d_model) for _ in range(args.n_layers)]
        )
        self.out_norm = RMSNorm(args.d_model)
        self.cls_head = nn.Linear(args.d_model, n_classes)
        nn.init.normal_(self.noise_token, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, *, return_info: bool = False) -> tuple[torch.Tensor, list[dict]]:
        h = self.in_proj(x)
        noise_tok = self.noise_token.expand(h.shape[0], 1, h.shape[-1])
        if self.noise_token_std > 0.0 and (self.training or self.noise_token_eval):
            noise_tok = noise_tok + self.noise_token_std * torch.randn_like(noise_tok)
        cls = self.cls_token.expand(h.shape[0], 1, h.shape[-1])
        # Keep CLS last so causal variants can read both patch and noise tokens.
        h = torch.cat([h, noise_tok, cls], dim=1)
        all_infos: list[dict] = [] if return_info else []
        for blk in self.blocks:
            h, infos = blk(h, return_info=return_info)
            if return_info:
                all_infos.extend(infos)
        h = self.out_norm(h)
        logits = self.cls_head(h[:, -1, :])
        return logits, all_infos


def _sample_classification_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    patch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = images.shape[0]
    if generator is None:
        idx = torch.randint(0, n, (batch_size,))
    else:
        idx = torch.randint(0, n, (batch_size,), generator=generator)
    batch_imgs = images[idx].to(device)
    batch_labels = labels[idx].to(device=device, dtype=torch.long)
    tokens = patchify(batch_imgs, patch_size)
    return tokens, batch_labels


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    use_cuda = device.type == "cuda"
    if args.tf32 and use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    train_images, train_labels = get_arrays(args.dataset, "train")
    val_images, val_labels = get_arrays(args.dataset, "test")
    patch_dim = 3 * args.patch_size * args.patch_size
    L = (32 // args.patch_size) ** 2
    n_classes = 10 if args.dataset == "cifar10" else 100
    print(f"data: {args.dataset} train={tuple(train_images.shape)} "
          f"val={tuple(val_images.shape)} L={L} patch_dim={patch_dim}", flush=True)

    model = PatchClassifier(patch_dim=patch_dim, n_classes=n_classes, args=args).to(device)
    if args.compile:
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"arch={args.arch}  params: trainable={n_params:,}  "
        f"compile={args.compile}  amp={args.amp}  tf32={args.tf32}",
        flush=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))
    use_bf16_amp = use_cuda and args.amp == "bf16"
    use_fp16_amp = use_cuda and args.amp == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_amp)

    def _autocast_ctx():
        if use_bf16_amp:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_fp16_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    log_dir = DKA_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    cache_dir = DKA_ROOT / "cache"
    cache_dir.mkdir(exist_ok=True)
    tag = args.tag
    stem = f"{args.arch}_cifar_cls_{args.dataset}_{tag}"
    log_path = log_dir / f"{stem}.jsonl"
    ckpt_path = cache_dir / f"{stem}.pt"
    log_path.write_text("")

    rng_train = torch.Generator().manual_seed(args.seed)
    rng_val = torch.Generator().manual_seed(args.seed + 1)

    t_run = time.time()
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_seen = 0
        for step in range(1, args.iters_per_epoch + 1):
            tokens, labels = _sample_classification_batch(
                train_images, train_labels, args.batch, args.patch_size,
                device, generator=rng_train,
            )
            debug_gate = args.debug_gate_every > 0 and (step % args.debug_gate_every == 0)
            with _autocast_ctx():
                logits, infos = model(tokens, return_info=debug_gate)
                loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            if use_fp16_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            running_loss += loss.item() * labels.numel()
            preds = logits.argmax(dim=-1)
            running_correct += (preds == labels).sum().item()
            running_seen += labels.numel()
            if debug_gate and infos:
                # Report from the first block/recurrent step as a quick health signal.
                gate_info = infos[0]
                gate = gate_info.get("gate", None)
                v = gate_info.get("v", None)
                v_raw = gate_info.get("v_raw", None)
                clamp_frac = gate_info.get("clamp_frac", 0.0)
                if gate is not None and v is not None and v_raw is not None:
                    gate_det = gate.detach()
                    v_det = v.detach()
                    v_raw_det = v_raw.detach()
                    print(
                        f"gate_debug step={step} "
                        f"K[min/mean/max]={gate_det.min().item():.4f}/{gate_det.mean().item():.4f}/{gate_det.max().item():.4f} "
                        f"v[min/mean/max]={v_det.min().item():.4f}/{v_det.mean().item():.4f}/{v_det.max().item():.4f} "
                        f"v_raw[min/mean/max]={v_raw_det.min().item():.4f}/{v_raw_det.mean().item():.4f}/{v_raw_det.max().item():.4f} "
                        f"clamp_frac={clamp_frac:.4f}",
                        flush=True,
                    )
            if args.log_interval > 0 and (step % args.log_interval == 0 or step == args.iters_per_epoch):
                elapsed_train = time.time() - t_epoch
                it_s = step / max(elapsed_train, 1e-9)
                eta_s = (args.iters_per_epoch - step) / max(it_s, 1e-9)
                avg_loss = running_loss / max(running_seen, 1)
                acc = running_correct / max(running_seen, 1)
                print(
                    f"epoch {epoch:3d} | train {step:4d}/{args.iters_per_epoch} | "
                    f"loss {loss.item():.5f} | avg {avg_loss:.5f} | acc {acc:.3f} | "
                    f"{it_s:.2f} it/s | eta {eta_s:.0f}s",
                    flush=True,
                )
        train_loss = running_loss / max(running_seen, 1)
        train_acc = running_correct / max(running_seen, 1)
        t_train = time.time() - t_epoch

        # val
        t_val_start = time.time()
        model.eval()
        n_val_iters = max(1, val_images.shape[0] // args.batch)
        if args.val_iters > 0:
            n_val_iters = min(n_val_iters, args.val_iters)
        v_loss_sum = 0.0
        v_correct = 0
        v_seen = 0
        with torch.no_grad():
            for step in range(1, n_val_iters + 1):
                tokens, labels = _sample_classification_batch(
                    val_images, val_labels, args.batch, args.patch_size,
                    device, generator=rng_val,
                )
                with _autocast_ctx():
                    logits, _ = model(tokens)
                    v_loss_sum += F.cross_entropy(logits, labels).item() * labels.numel()
                preds = logits.argmax(dim=-1)
                v_correct += (preds == labels).sum().item()
                v_seen += labels.numel()
                if args.log_interval > 0 and (step % args.log_interval == 0 or step == n_val_iters):
                    print(
                        f"epoch {epoch:3d} | val   {step:4d}/{n_val_iters}",
                        flush=True,
                    )
        val_loss = v_loss_sum / max(v_seen, 1)
        val_acc = v_correct / max(v_seen, 1)
        t_val = time.time() - t_val_start

        elapsed = time.time() - t_run
        epoch_elapsed = time.time() - t_epoch
        print(
            f"epoch {epoch:3d} | train_loss {train_loss:.5f} | train_acc {train_acc:.3f} | "
            f"val_loss {val_loss:.5f} | val_acc {val_acc:.3f} | "
            f"train_s {t_train:.0f} | val_s {t_val:.0f} | epoch_s {epoch_elapsed:.0f} | elapsed {elapsed:.0f}s",
            flush=True,
        )
        with log_path.open("a") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "elapsed_s": elapsed,
            }) + "\n")

    torch.save({
        "state_dict": model.state_dict(),
        "config": vars(args),
        "patch_dim": patch_dim,
        "num_classes": n_classes,
    }, ckpt_path)
    print(f"done. checkpoint: {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
