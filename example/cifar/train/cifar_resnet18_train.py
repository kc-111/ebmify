"""Train a CIFAR-adapted ResNet18 on CIFAR-10 (supervised, from scratch).

Standard CIFAR recipe: first conv is 3x3 stride=1 (not 7x7 stride=2) and
the initial maxpool is dropped, so the 32x32 input is preserved through
the stages. ~100 epochs of SGD + cosine LR + random crop / hflip gets
~93-94% test accuracy.

The trained network is the supervised analogue of the unsupervised VAE
encoder: the 512-d pre-fc feature serves as `z = f(x)` for the OOD
eval script (``cifar_resnet18_ood_threshold.py``).

Usage:
    python example/cifar/train/cifar_resnet18_train.py --epochs 100
"""

from __future__ import annotations

import argparse
import json
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

# Per-channel CIFAR-10 normalization stats (computed on the train split).
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def make_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """torchvision ResNet18 retuned for 32x32 inputs.

    Replaces the stem 7x7 stride-2 conv + maxpool with a 3x3 stride-1
    conv and an Identity maxpool, which is the standard CIFAR adaptation.
    Leaves the four residual stages and final avgpool intact.
    """
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


def augment(x: torch.Tensor) -> torch.Tensor:
    """Per-sample hflip + per-batch random 32x32 crop from a 4-px reflect pad.

    Per-batch crop (one offset for the whole batch) is a small concession
    to keep this fully vectorized; combined with per-sample hflip and 100
    epochs of cosine LR it still hits ~93%+ accuracy.
    """
    B = x.shape[0]
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    pad = F.pad(x, (4, 4, 4, 4), mode="reflect")
    h = int(torch.randint(0, 9, (1,)).item())
    w = int(torch.randint(0, 9, (1,)).item())
    return pad[:, :, h:h + 32, w:w + 32]


def normalize(x: torch.Tensor, device: str) -> torch.Tensor:
    return (x - CIFAR_MEAN.to(device)) / CIFAR_STD.to(device)


def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, *,
             batch_size: int, device: str) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = normalize(X[i:i + batch_size], device)
            logits = model(xb)
            correct += int((logits.argmax(-1) == y[i:i + batch_size]).sum())
    return correct / X.shape[0]


def train(
    model: nn.Module, X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: torch.Tensor, y_te: torch.Tensor,
    device: str, *, epochs: int, batch_size: int, lr: float,
    momentum: float, weight_decay: float,
) -> dict:
    X_t = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_tr, dtype=torch.long, device=device)
    n = X_t.shape[0]

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    rng = np.random.default_rng(0)
    best_acc = 0.0
    history: list[dict] = []
    for ep in range(epochs):
        model.train()
        idx = rng.permutation(n)
        loss_sum = 0.0
        correct = 0
        nb = 0
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            xb = X_t[b]
            yb = y_t[b]
            xb = augment(xb)
            xb = normalize(xb, device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach())
            correct += int((logits.argmax(-1) == yb).sum())
            nb += 1
        sched.step()
        train_acc = correct / n
        test_acc = evaluate(model, X_te, y_te,
                            batch_size=batch_size, device=device)
        best_acc = max(best_acc, test_acc)
        lr_now = opt.param_groups[0]["lr"]
        print(f"  epoch {ep+1:3d}/{epochs}  lr={lr_now:.4f}  "
              f"loss={loss_sum/nb:.3f}  train_acc={train_acc*100:.2f}  "
              f"test_acc={test_acc*100:.2f}  (best={best_acc*100:.2f})")
        history.append({"epoch": ep + 1, "loss": loss_sum / nb,
                        "train_acc": train_acc, "test_acc": test_acc})
    return {"history": history, "best_acc": best_acc}


def resnet18_ckpt_path(tag: str = "") -> Path:
    cache = Path(__file__).resolve().parent.parent / "cache"
    cache.mkdir(exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    return cache / f"cifar10_resnet18{suffix}.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4,
                    dest="weight_decay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    X_tr, y_tr = load_cifar_train("cifar10")
    X_te, y_te = load_cifar_test("cifar10")
    X_te_t = torch.as_tensor(X_te, dtype=torch.float32, device=device)
    y_te_t = torch.as_tensor(y_te, dtype=torch.long, device=device)
    print(f"  cifar10 train: {X_tr.shape}, test: {X_te.shape}")

    model = make_resnet18_cifar(num_classes=10).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters: {n_params:,}")

    print(f"\nTraining ResNet18-CIFAR (epochs={args.epochs}, batch={args.batch}, "
          f"lr={args.lr}, wd={args.weight_decay}) ...")
    out = train(
        model, X_tr, y_tr, X_te_t, y_te_t, device,
        epochs=args.epochs, batch_size=args.batch, lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay,
    )

    config = {"arch": "resnet18-cifar", "num_classes": 10}
    ckpt = resnet18_ckpt_path(args.tag)
    torch.save({"state_dict": model.state_dict(), "config": config,
                "best_acc": out["best_acc"]}, ckpt)
    print(f"\nsaved {ckpt}")
    print(f"  config: {json.dumps(config)}")
    print(f"  best test_acc = {out['best_acc']*100:.2f}%")


if __name__ == "__main__":
    main()
