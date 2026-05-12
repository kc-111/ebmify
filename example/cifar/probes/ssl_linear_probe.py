"""Linear probe on top of the frozen SSL ResNet18 backbone.

Diagnostic for the leverage-OOD result where cifar100 looked "more in
distribution" than cifar10 test. Two competing explanations:

(a) The SSL features are cifar10-specific: cifar100 features collapse
    near origin, which gives them small ||z||^2 and therefore small
    h(z). Linear probe accuracy on cifar100 would be near random (~1-15%).

(b) The SSL features are generic visual features that transfer fine to
    cifar100, and the OOD failure is purely a leverage methodology
    artifact (Mahalanobis from origin doesn't capture semantic OOD).
    Linear probe accuracy on cifar100 would be 30-50%+.

Procedure: load the frozen SSL backbone, encode train + test of the
chosen dataset once, fit a single nn.Linear classifier on the train
features (frozen backbone), report top-1 / top-5 test accuracy.

Usage:
    python example/cifar/probes/ssl_linear_probe.py --dataset cifar100
    python example/cifar/probes/ssl_linear_probe.py --dataset cifar10  # sanity check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import stable_pretraining as spt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import REPO_ROOT  # noqa: F401, E402

from cifar_data import load_cifar_test, load_cifar_train  # noqa: E402
from cifar_resnet18_train import CIFAR_MEAN, CIFAR_STD  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def load_ssl_backbone(ckpt_path: Path, device: str) -> tuple[nn.Module, int]:
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = raw["state_dict"]
    backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)
    backbone.fc = nn.Identity()
    backbone.load_state_dict(sd)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone.to(device), 512


def encode(model: nn.Module, x: torch.Tensor, *,
           batch_size: int, device: str) -> torch.Tensor:
    mean = CIFAR_MEAN.to(device)
    std = CIFAR_STD.to(device)
    feats = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            chunk = (x[i:i + batch_size] - mean) / std
            feats.append(model(chunk))
    return torch.cat(feats, dim=0)


def topk_acc(logits: torch.Tensor, y: torch.Tensor, k: int) -> float:
    _, topk = logits.topk(k, dim=-1)
    return float((topk == y.unsqueeze(-1)).any(dim=-1).float().mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100")
    ap.add_argument("--ckpt", type=str, default="example/cifar/cache/cifar10_ssl_resnet18_recon.pt",
                    help="Path to SSL backbone checkpoint.")
    ap.add_argument("--tag", type=str, default="",
                    help="Tag used at SSL training time.")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256, dest="batch_size")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        suffix = f"_{args.tag}" if args.tag else ""
        ckpt_path = CACHE_DIR / f"cifar10_ssl_resnet18{suffix}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Run ssl_pretrain.py first."
        )
    print(f"loading SSL ResNet18 backbone from {ckpt_path} ...")
    model, z_dim = load_ssl_backbone(ckpt_path, device)

    print(f"loading {args.dataset} train/test ...")
    X_tr, y_tr = load_cifar_train(args.dataset)
    X_te, y_te = load_cifar_test(args.dataset)
    X_tr_t = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
    X_te_t = torch.as_tensor(X_te, dtype=torch.float32, device=device)
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.long, device=device)
    y_te_t = torch.as_tensor(y_te, dtype=torch.long, device=device)
    n_classes = int(max(int(y_tr.max()), int(y_te.max()))) + 1
    print(f"  classes: {n_classes}, train: {tuple(X_tr.shape)}, "
          f"test: {tuple(X_te.shape)}")

    print("encoding once through the frozen backbone ...")
    Z_tr = encode(model, X_tr_t, batch_size=args.batch_size, device=device)
    Z_te = encode(model, X_te_t, batch_size=args.batch_size, device=device)
    print(f"  Z_tr: {tuple(Z_tr.shape)}  ||z|| median="
          f"{Z_tr.norm(dim=1).median().item():.3f}")
    print(f"  Z_te: {tuple(Z_te.shape)}  ||z|| median="
          f"{Z_te.norm(dim=1).median().item():.3f}")

    # Standardize per-feature on train stats. Standard practice for linear
    # probes; makes optimization well-conditioned without changing the
    # underlying separability of the features.
    mu = Z_tr.mean(dim=0, keepdim=True)
    sigma = Z_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
    Z_tr_n = (Z_tr - mu) / sigma
    Z_te_n = (Z_te - mu) / sigma

    probe = nn.Linear(z_dim, n_classes).to(device)
    opt = torch.optim.AdamW(
        probe.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    n = Z_tr_n.shape[0]
    rng = np.random.default_rng(args.seed)
    print(f"\ntraining linear probe ({z_dim} -> {n_classes}) "
          f"for {args.epochs} epochs ...")
    best_top1 = 0.0
    best_top5 = 0.0
    top1 = 0.0
    top5 = 0.0
    for ep in range(args.epochs):
        probe.train()
        idx = rng.permutation(n)
        loss_sum = 0.0
        correct = 0
        nb = 0
        for s in range(0, n, args.batch_size):
            b = idx[s:s + args.batch_size]
            logits = probe(Z_tr_n[b])
            yb = y_tr_t[b]
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach())
            correct += int((logits.argmax(-1) == yb).sum())
            nb += 1
        sched.step()

        probe.eval()
        with torch.no_grad():
            logits_te = probe(Z_te_n)
            top1 = topk_acc(logits_te, y_te_t, 1)
            top5 = topk_acc(logits_te, y_te_t, 5)
        best_top1 = max(best_top1, top1)
        best_top5 = max(best_top5, top5)
        if (ep + 1) % max(1, args.epochs // 20) == 0 or ep == args.epochs - 1:
            train_acc = correct / n
            lr_now = opt.param_groups[0]["lr"]
            print(f"  ep {ep+1:3d}/{args.epochs}  lr={lr_now:.4f}  "
                  f"loss={loss_sum/nb:.3f}  train_top1={train_acc*100:.2f}  "
                  f"test_top1={top1*100:.2f}  test_top5={top5*100:.2f}")

    print(f"\nfinal:    top-1 = {top1*100:.2f}%   top-5 = {top5*100:.2f}%")
    print(f"best:     top-1 = {best_top1*100:.2f}%   "
          f"top-5 = {best_top5*100:.2f}%")
    print(f"chance:   top-1 = {100.0/n_classes:.2f}%   "
          f"top-5 = {100.0*5/n_classes:.2f}%")


if __name__ == "__main__":
    main()
