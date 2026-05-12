"""Train a CIFAR-adapted ResNet18 on CIFAR-10 to ~95.5-96% test top-1.

Single-pass SGD recipe, roughly the same wall-clock as the original 100-epoch
baseline (AMP + channels_last speed up each epoch enough to afford 200 epochs).
The augmentation stack is the centerpiece -- everything else is lightweight
recipe hygiene.

Augmentation (all per-sample, vectorized on GPU):
- per-sample reflect-pad random crop + per-sample hflip
- per-sample random affine: rotation +/-15deg, translation +/-5%
  (single batched ``affine_grid`` + ``grid_sample`` with reflection padding)
- per-sample color jitter: brightness, contrast, saturation each +/-0.2
- Cutout 16x16
- Mixup(alpha=0.2) <-> CutMix(alpha=1.0) alternation, mix prob 0.8
- label smoothing 0.1, soft cross-entropy

Recipe hygiene:
- CIFAR stem (3x3 s1, no maxpool), zero-init residual BN gamma
- SGD + Nesterov, decoupled weight decay (5e-4 on conv/linear, 0 on BN/bias)
- 5-epoch linear LR warmup -> cosine over 200 epochs
- AMP bfloat16 (no GradScaler needed) + channels_last memory format
- EMA of weights (decay 0.999); EMA weights are saved under ``state_dict``
  so the downstream OOD eval picks them up automatically. Raw weights are
  kept under ``raw_state_dict``.

Usage:
    python example/cifar/train/cifar_resnet18_train.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import REPO_ROOT  # noqa: F401, E402
from cifar_data import load_cifar_test, load_cifar_train  # noqa: E402

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
# Grayscale weights used by saturation jitter (Rec. 601 luma).
_LUMA = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Architecture: ResNet-18 CIFAR + zero-init residual.
# ---------------------------------------------------------------------------

def make_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """torchvision ResNet18 retuned for 32x32 inputs (CIFAR adaptation).

    Replaces the stem 7x7/s2 conv + maxpool with a 3x3/s1 conv and an
    Identity maxpool. Zero-inits the last BN gamma of each BasicBlock so
    blocks start as identity (He et al., "Bag of tricks").
    """
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    for m in model.modules():
        if isinstance(m, models.resnet.BasicBlock):
            nn.init.zeros_(m.bn2.weight)
    return model


# ---------------------------------------------------------------------------
# Augmentation: all per-sample, all vectorized.
# ---------------------------------------------------------------------------

def random_crop_flip(x: torch.Tensor, pad: int = 4, size: int = 32) -> torch.Tensor:
    """Per-sample hflip + per-sample reflect-pad random crop, vectorized.

    Uses ``unfold`` to enumerate all (2*pad+1)**2 crop offsets, then picks
    one per image with advanced indexing.
    """
    B = x.shape[0]
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    patches = padded.unfold(2, size, 1).unfold(3, size, 1)  # (B,C,P,P,32,32)
    h = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    w = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    idx = torch.arange(B, device=x.device)
    return patches[idx, :, h, w].contiguous()


def random_affine(x: torch.Tensor, max_rot_deg: float = 15.0,
                  max_translate: float = 0.05) -> torch.Tensor:
    """Per-sample affine: random rotation + translation, reflection-padded.

    ``affine_grid`` works in normalized [-1, 1] coordinates, so a translation
    of ``max_translate`` (fraction of image size) is 2x that in grid units.
    """
    B, _, _, _ = x.shape
    device = x.device
    angle = (torch.rand(B, device=device) * 2 - 1) * (max_rot_deg * math.pi / 180.0)
    tx = (torch.rand(B, device=device) * 2 - 1) * (2.0 * max_translate)
    ty = (torch.rand(B, device=device) * 2 - 1) * (2.0 * max_translate)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    theta = torch.stack([
        torch.stack([cos_a, -sin_a, tx], dim=-1),
        torch.stack([sin_a,  cos_a, ty], dim=-1),
    ], dim=-2)  # (B, 2, 3)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear",
                         padding_mode="reflection", align_corners=False)


def color_jitter(x: torch.Tensor, brightness: float = 0.2,
                 contrast: float = 0.2, saturation: float = 0.2
                 ) -> torch.Tensor:
    """Per-sample brightness/contrast/saturation scalars in [0,1] pixel space."""
    B = x.shape[0]
    device = x.device
    b = 1.0 + (torch.rand(B, device=device) * 2 - 1) * brightness
    c = 1.0 + (torch.rand(B, device=device) * 2 - 1) * contrast
    s = 1.0 + (torch.rand(B, device=device) * 2 - 1) * saturation
    b = b[:, None, None, None]
    c = c[:, None, None, None]
    s = s[:, None, None, None]
    x = x * b
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * c + mean
    gray = (x * _LUMA.to(device)).sum(dim=1, keepdim=True)
    x = gray + s * (x - gray)
    return x.clamp(0.0, 1.0)


def cutout(x: torch.Tensor, size: int = 16) -> torch.Tensor:
    """Zero a random ``size x size`` square per image (centers allowed off-edge)."""
    B, _, H, W = x.shape
    cy = torch.randint(0, H, (B,), device=x.device)
    cx = torch.randint(0, W, (B,), device=x.device)
    ys = torch.arange(H, device=x.device)[None, :]
    xs = torch.arange(W, device=x.device)[None, :]
    half = size // 2
    my = (ys >= (cy[:, None] - half)) & (ys < (cy[:, None] + half))  # (B, H)
    mx = (xs >= (cx[:, None] - half)) & (xs < (cx[:, None] + half))  # (B, W)
    mask = (my[:, :, None] & mx[:, None, :])[:, None, :, :]          # (B,1,H,W)
    return x.masked_fill(mask, 0.0)


def mixup_or_cutmix(x: torch.Tensor, y_oh: torch.Tensor, *,
                    mixup_alpha: float, cutmix_alpha: float, mix_prob: float,
                    extras: dict[str, torch.Tensor] | None = None,
                    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """With prob ``mix_prob``, apply Mixup XOR CutMix (50/50) to the batch.

    Args:
        x: ``(B, C, H, W)`` already-normalized batch.
        y_oh: ``(B, K)`` one-hot labels.
        mixup_alpha, cutmix_alpha: Beta-distribution shape params for
            the Mixup and CutMix coefficients.
        mix_prob: Probability of applying any mix this step (else
            pass-through). When mix fires, Mixup vs CutMix is 50/50.
        extras: Optional ``{name -> tensor}`` of additional per-sample
            tensors that should be linearly blended by the same ``lam``
            and the same ``perm`` as ``y_oh``. Used to keep aux-loss
            targets aligned with the mixed image (one-hot home_bucket,
            spectral_coords, feature_distill, etc.).

    Returns:
        Tuple ``(x_mixed, y_mixed, extras_mixed)``. When ``extras`` is
        ``None``, ``extras_mixed`` is an empty dict; otherwise its keys
        match ``extras`` and each tensor is blended identically to
        ``y_oh``.
    """
    extras_mixed: dict[str, torch.Tensor] = (
        dict(extras) if extras else {}
    )
    if torch.rand(1).item() >= mix_prob:
        return x, y_oh, extras_mixed
    use_cutmix = torch.rand(1).item() < 0.5
    B, _, H, W = x.shape
    perm = torch.randperm(B, device=x.device)
    if use_cutmix:
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        cut_w = int(W * math.sqrt(1.0 - lam))
        cut_h = int(H * math.sqrt(1.0 - lam))
        cy = int(torch.randint(0, H, (1,)).item())
        cx = int(torch.randint(0, W, (1,)).item())
        y1 = max(0, cy - cut_h // 2)
        y2 = min(H, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(W, cx + cut_w // 2)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
        lam = 1.0 - ((y2 - y1) * (x2 - x1)) / (H * W)
    else:
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        lam = max(lam, 1.0 - lam)
        x = lam * x + (1.0 - lam) * x[perm]
    y_mix = lam * y_oh + (1.0 - lam) * y_oh[perm]
    if extras:
        # Same lam + same perm as the label mix: keeps every extra target
        # aligned to the same convex combination as the mixed image, so
        # the aux head trained on the mixed embedding regresses against
        # the matching convex combination of targets.
        extras_mixed = {k: lam * v + (1.0 - lam) * v[perm] for k, v in extras.items()}
    return x, y_mix, extras_mixed


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                       smoothing: float) -> torch.Tensor:
    K = logits.shape[-1]
    smoothed = (1.0 - smoothing) * targets + smoothing / K
    return -(smoothed * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def normalize(x: torch.Tensor, device: str) -> torch.Tensor:
    return (x - CIFAR_MEAN.to(device)) / CIFAR_STD.to(device)


# ---------------------------------------------------------------------------
# Optimizer / scheduler / eval helpers.
# ---------------------------------------------------------------------------

def _split_decay_params(model: nn.Module
                        ) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Conv/linear weights get weight decay; BN scales/biases do not."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def _make_warmup_cosine(opt: torch.optim.Optimizer, total_steps: int,
                        warmup_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, *,
             batch_size: int, device: str) -> float:
    model.eval()
    correct = 0
    for i in range(0, X.shape[0], batch_size):
        xb = normalize(X[i:i + batch_size], device)
        xb = xb.contiguous(memory_format=torch.channels_last)
        logits = model(xb)
        correct += int((logits.argmax(-1) == y[i:i + batch_size]).sum())
    return correct / X.shape[0]


@torch.no_grad()
def _update_bn_stats(model: nn.Module, X: torch.Tensor, device: str,
                     batch_size: int) -> None:
    """Refresh BN running stats with a single un-augmented cumulative pass.

    EMA weights drift away from the per-step BN statistics, so we recompute
    ``running_mean`` / ``running_var`` (momentum=None => cumulative mean)
    before eval. Mirror this pre-eval at save time.
    """
    bn = [m for m in model.modules()
          if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bn:
        return
    saved = [m.momentum for m in bn]
    for m in bn:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for i in range(0, X.shape[0], batch_size):
        xb = normalize(X[i:i + batch_size], device)
        xb = xb.contiguous(memory_format=torch.channels_last)
        model(xb)
    for m, mom in zip(bn, saved):
        m.momentum = mom


# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------

def train(
    model: nn.Module, X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: torch.Tensor, y_te: torch.Tensor, device: str, *,
    epochs: int, warmup_epochs: int, batch_size: int, lr: float,
    momentum: float, weight_decay: float, label_smoothing: float,
    mixup_alpha: float, cutmix_alpha: float, mix_prob: float,
    cutout_size: int, affine_rot_deg: float, affine_translate: float,
    color_brightness: float, color_contrast: float, color_saturation: float,
    ema_decay: float,
    # Optional aux-head extension (coreset pipeline). When provided,
    # adds a per-aux linear head on the 512-D pre-fc feature (captured
    # via a forward hook on ``model.avgpool``) and adds the head's
    # weighted loss to the main soft-CE. Aux targets are mixed by the
    # same lam/perm as the labels (see ``mixup_or_cutmix(extras=...)``).
    aux_heads: nn.ModuleDict | None = None,
    aux_targets: dict[str, torch.Tensor] | None = None,
    aux_specs: dict | None = None,
    aux_lambdas: dict[str, float] | None = None,
) -> dict:
    X_t = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_tr, dtype=torch.long, device=device)
    n = X_t.shape[0]
    num_classes = int(y_t.max().item()) + 1

    use_aux = (
        aux_heads is not None and aux_targets and aux_specs
        and aux_lambdas and any(v > 0 for v in aux_lambdas.values())
    )
    feat_cache: dict[str, torch.Tensor] = {}
    hook_handle = None
    if use_aux:
        # Local import keeps this script standalone-runnable; the aux
        # helper lives next to the coreset trainer that consumes it.
        from _aux_losses import aux_loss_terms, index_targets  # type: ignore

        def _capture(_module, _inp, out):
            feat_cache["emb"] = out.flatten(1)
        hook_handle = model.avgpool.register_forward_hook(_capture)
        aux_heads.to(device)
        # Pre-build per-aux device-resident target tensors keyed by
        # coreset position. For CE aux ("home_bucket"), pre-expand to
        # one-hot float so it can be linearly mixed by Mixup/CutMix.
        aux_full: dict[str, torch.Tensor] = {}
        for name, t in aux_targets.items():
            t_d = t.to(device, non_blocking=True)
            spec = aux_specs[name]
            if spec.loss_kind == "ce":
                t_d = F.one_hot(t_d.long(), spec.target_dim).float()
            else:
                t_d = t_d.to(torch.float32)
            aux_full[name] = t_d

    decay, no_decay = _split_decay_params(model)
    if use_aux:
        # Aux head linear weights (ndim=2) -> decay group;
        # aux head biases (ndim=1) -> no-decay group. Same rule as the
        # main model. Adding to the existing groups keeps SGD's
        # momentum state contiguous.
        ah_decay, ah_no_decay = _split_decay_params(aux_heads)
        decay = decay + ah_decay
        no_decay = no_decay + ah_no_decay
    opt = torch.optim.SGD(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, momentum=momentum, nesterov=True,
    )

    steps_per_epoch = math.ceil(n / batch_size)
    sched = _make_warmup_cosine(opt, epochs * steps_per_epoch,
                                warmup_epochs * steps_per_epoch)

    amp_enabled = device == "cuda" and torch.cuda.is_bf16_supported()

    def _ema_avg(avg, p, _):
        return ema_decay * avg + (1.0 - ema_decay) * p
    ema = torch.optim.swa_utils.AveragedModel(model, avg_fn=_ema_avg)
    ema.to(memory_format=torch.channels_last)

    rng = np.random.default_rng(0)
    best_acc = 0.0
    best_ema_acc = 0.0
    history: list[dict] = []
    for ep in range(epochs):
        model.train()
        if use_aux:
            aux_heads.train()
        idx = rng.permutation(n)
        loss_sum = 0.0
        ce_sum = 0.0
        aux_sum_by_name: dict[str, float] = {}
        correct = 0
        nb = 0
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            xb = X_t[b]
            yb = y_t[b]

            # Geometry + photometry in pixel space [0,1].
            xb = random_crop_flip(xb)
            xb = random_affine(xb, max_rot_deg=affine_rot_deg,
                               max_translate=affine_translate)
            xb = color_jitter(xb, brightness=color_brightness,
                              contrast=color_contrast,
                              saturation=color_saturation)
            xb = normalize(xb, device)
            # Cutout after normalize -> the masked region becomes the
            # dataset mean in pixel space (standard practice).
            if cutout_size > 0:
                xb = cutout(xb, size=cutout_size)
            y_oh = F.one_hot(yb, num_classes).float()
            # Coreset position `b` indexes the on-disk aux tensors
            # (rows aligned to positions [0, n) by construction).
            aux_batch: dict[str, torch.Tensor] = {}
            if use_aux:
                pos = torch.as_tensor(b, device=device, dtype=torch.long)
                for name, t_full in aux_full.items():
                    aux_batch[name] = t_full.index_select(0, pos)
            # Mixup/CutMix blends image, label, and every aux target by
            # the same lam + perm so the captured embedding aligns with
            # the convex combination of aux targets.
            xb, y_target, aux_batch = mixup_or_cutmix(
                xb, y_oh, mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha, mix_prob=mix_prob,
                extras=aux_batch if use_aux else None,
            )
            xb = xb.contiguous(memory_format=torch.channels_last)

            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=amp_enabled):
                logits = model(xb)
                ce = soft_cross_entropy(logits, y_target, label_smoothing)
                loss = ce
                aux_logs: dict[str, float] = {}
                if use_aux:
                    aux_total, aux_logs = aux_loss_terms(
                        feat_cache["emb"], aux_heads, aux_batch,
                        aux_specs, aux_lambdas,
                    )
                    loss = loss + aux_total

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            ema.update_parameters(model)

            loss_sum += float(loss.detach())
            ce_sum += float(ce.detach())
            for k, v in aux_logs.items():
                aux_sum_by_name[k] = aux_sum_by_name.get(k, 0.0) + v
            # Mixup/CutMix blurs labels; train-acc vs hard yb is a
            # divergence signal, not a quality metric.
            correct += int((logits.argmax(-1) == yb).sum())
            nb += 1

        aux_log_running = {k: v / nb for k, v in aux_sum_by_name.items()}
        train_acc = correct / n
        test_acc = evaluate(model, X_te, y_te,
                            batch_size=batch_size, device=device)
        best_acc = max(best_acc, test_acc)
        lr_now = opt.param_groups[0]["lr"]
        aux_str = ""
        if aux_log_running:
            aux_str = "  aux={" + " ".join(
                f"{k}:{v:.3f}" for k, v in aux_log_running.items()) + "}"
        line = (f"  epoch {ep+1:3d}/{epochs}  lr={lr_now:.4f}  "
                f"loss={loss_sum/nb:.3f}  ce={ce_sum/nb:.3f}{aux_str}  "
                f"train_acc={train_acc*100:.2f}  "
                f"test_acc={test_acc*100:.2f}  (best={best_acc*100:.2f})")

        # EMA BN refresh is a full extra train-set pass; every 10 epochs.
        ema_acc: float | None = None
        if ep + 1 == epochs or (ep + 1) % 10 == 0:
            _update_bn_stats(ema, X_t, device, batch_size)
            ema_acc = evaluate(ema, X_te, y_te,
                               batch_size=batch_size, device=device)
            best_ema_acc = max(best_ema_acc, ema_acc)
            line += f"  ema_acc={ema_acc*100:.2f}  (best={best_ema_acc*100:.2f})"
        print(line)

        history.append({"epoch": ep + 1, "loss": loss_sum / nb,
                        "ce": ce_sum / nb, "aux": dict(aux_log_running),
                        "train_acc": train_acc, "test_acc": test_acc,
                        "ema_acc": ema_acc})

    _update_bn_stats(ema, X_t, device, batch_size)
    final_ema_acc = evaluate(ema, X_te, y_te,
                             batch_size=batch_size, device=device)
    best_ema_acc = max(best_ema_acc, final_ema_acc)
    if hook_handle is not None:
        hook_handle.remove()
    return {"history": history,
            "best_acc": best_acc,
            "best_ema_acc": best_ema_acc,
            "final_ema_acc": final_ema_acc,
            "ema_state": ema.module.state_dict()}


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def resnet18_ckpt_path(tag: str = "") -> Path:
    cache = Path(__file__).resolve().parent.parent / "cache"
    cache.mkdir(exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    return cache / f"cifar10_resnet18{suffix}.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--warmup-epochs", type=int, default=5,
                    dest="warmup_epochs")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4,
                    dest="weight_decay")
    ap.add_argument("--label-smoothing", type=float, default=0.1,
                    dest="label_smoothing")
    ap.add_argument("--mixup-alpha", type=float, default=0.2,
                    dest="mixup_alpha")
    ap.add_argument("--cutmix-alpha", type=float, default=1.0,
                    dest="cutmix_alpha")
    ap.add_argument("--mix-prob", type=float, default=0.8, dest="mix_prob")
    ap.add_argument("--cutout-size", type=int, default=16, dest="cutout_size")
    ap.add_argument("--affine-rot-deg", type=float, default=15.0,
                    dest="affine_rot_deg")
    ap.add_argument("--affine-translate", type=float, default=0.05,
                    dest="affine_translate")
    ap.add_argument("--color-brightness", type=float, default=0.2,
                    dest="color_brightness")
    ap.add_argument("--color-contrast", type=float, default=0.2,
                    dest="color_contrast")
    ap.add_argument("--color-saturation", type=float, default=0.2,
                    dest="color_saturation")
    ap.add_argument("--ema-decay", type=float, default=0.999, dest="ema_decay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    X_tr, y_tr = load_cifar_train("cifar10")
    X_te, y_te = load_cifar_test("cifar10")
    X_te_t = torch.as_tensor(X_te, dtype=torch.float32, device=device)
    y_te_t = torch.as_tensor(y_te, dtype=torch.long, device=device)
    print(f"  cifar10 train: {X_tr.shape}, test: {X_te.shape}")

    model = make_resnet18_cifar(num_classes=10).to(device)
    model = model.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters: {n_params:,}")

    print(f"\nTraining ResNet18-CIFAR (epochs={args.epochs}, "
          f"batch={args.batch}, lr={args.lr}, wd={args.weight_decay}) ...")
    print(f"  aug: crop+flip, affine(rot={args.affine_rot_deg}deg, "
          f"t={args.affine_translate}), "
          f"color(b={args.color_brightness}, c={args.color_contrast}, "
          f"s={args.color_saturation}), cutout={args.cutout_size}, "
          f"mixup={args.mixup_alpha}, cutmix={args.cutmix_alpha} "
          f"(mix_prob={args.mix_prob}), ls={args.label_smoothing}, "
          f"ema={args.ema_decay}")
    out = train(
        model, X_tr, y_tr, X_te_t, y_te_t, device,
        epochs=args.epochs, warmup_epochs=args.warmup_epochs,
        batch_size=args.batch, lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha,
        mix_prob=args.mix_prob, cutout_size=args.cutout_size,
        affine_rot_deg=args.affine_rot_deg,
        affine_translate=args.affine_translate,
        color_brightness=args.color_brightness,
        color_contrast=args.color_contrast,
        color_saturation=args.color_saturation,
        ema_decay=args.ema_decay,
    )

    config = {"arch": "resnet18-cifar", "num_classes": 10}
    ckpt = resnet18_ckpt_path(args.tag)
    # state_dict = EMA weights (better generalization); used by downstream
    # OOD eval. raw_state_dict kept as a fallback for ablations.
    torch.save({"state_dict": out["ema_state"],
                "raw_state_dict": model.state_dict(),
                "config": config,
                "best_acc": out["best_acc"],
                "best_ema_acc": out["best_ema_acc"],
                "final_ema_acc": out["final_ema_acc"]}, ckpt)
    print(f"\nsaved {ckpt}")
    print(f"  config: {json.dumps(config)}")
    print(f"  best raw test_acc = {out['best_acc']*100:.2f}%")
    print(f"  best EMA test_acc = {out['best_ema_acc']*100:.2f}%  "
          f"(final={out['final_ema_acc']*100:.2f}%)")


if __name__ == "__main__":
    main()
