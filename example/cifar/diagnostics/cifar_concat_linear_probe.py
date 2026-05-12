"""Linear-probe accuracy on CIFAR10 (test) and CIFAR100 (transfer)
for VAE, LeJEPA, and their concatenation.

Companion to ``cifar_concat_features_test.py``: that script showed the
concat representation gives both pixel-stat (Gaussian noise) AND
semantic (cifar100) OOD detection. This one tests whether the same
concat also helps for the classification probe -- which is the
canonical "feature quality" measure.

Per-piece per-feature standardization on train stats (the usual probe
preprocessing) before concatenation, so each piece contributes on a
comparable scale and the linear classifier can re-weight as needed.

Reports top-1 / top-5 test accuracy for six (representation, dataset)
combinations.

Usage:
    python example/cifar/diagnostics/cifar_concat_linear_probe.py
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import REPO_ROOT  # noqa: F401, E402

from cifar_data import (  # noqa: E402
    cifar_ckpt_path, load_cifar_test, load_cifar_train,
)
from cifar_vae_train import load_vae  # noqa: E402
from ssl_linear_probe import (  # noqa: E402
    encode as encode_lejepa_normed,
    load_ssl_backbone, topk_acc,
)

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def encode_vae(vae, x: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            mu, _ = vae.encode(x[i:i + batch_size])
            outs.append(mu)
    return torch.cat(outs, dim=0)


def standardize(Z_tr: torch.Tensor, Z_te: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
    mu = Z_tr.mean(dim=0, keepdim=True)
    sigma = Z_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (Z_tr - mu) / sigma, (Z_te - mu) / sigma


def fit_probe(Z_tr: torch.Tensor, y_tr: torch.Tensor,
              Z_te: torch.Tensor, y_te: torch.Tensor, *,
              n_classes: int, epochs: int, lr: float,
              weight_decay: float, batch_size: int, device: str,
              seed: int, tag: str) -> dict:
    torch.manual_seed(seed)
    z_dim = Z_tr.shape[1]
    probe = nn.Linear(z_dim, n_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr,
                            weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Z_tr.shape[0]
    rng = np.random.default_rng(seed)
    best_top1 = best_top5 = 0.0
    top1 = top5 = 0.0
    print(f"  [{tag}] training linear probe ({z_dim} -> {n_classes}) "
          f"for {epochs} epochs ...")
    for ep in range(epochs):
        probe.train()
        idx = rng.permutation(n)
        loss_sum = 0.0
        correct = 0
        nb = 0
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            logits = probe(Z_tr[b])
            yb = y_tr[b]
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
            logits_te = probe(Z_te)
            top1 = topk_acc(logits_te, y_te, 1)
            top5 = topk_acc(logits_te, y_te, 5)
        best_top1 = max(best_top1, top1)
        best_top5 = max(best_top5, top5)
        if (ep + 1) % max(1, epochs // 10) == 0 or ep == epochs - 1:
            print(f"    ep {ep+1:4d}/{epochs}  loss={loss_sum/nb:.3f}  "
                  f"train_top1={correct/n*100:5.2f}  "
                  f"test_top1={top1*100:5.2f}  test_top5={top5*100:5.2f}")
    return dict(
        top1=top1, top5=top5,
        best_top1=best_top1, best_top5=best_top5,
        z_dim=z_dim,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssl-tag", type=str, default="")
    ap.add_argument("--vae-z", type=int, default=256)
    ap.add_argument("--vae-beta", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=256, dest="batch_size")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    # --- Load backbones -----------------------------------------------
    suffix = f"_{args.ssl_tag}" if args.ssl_tag else ""
    ssl_ckpt = CACHE_DIR / f"cifar10_ssl_resnet18{suffix}.pt"
    if not ssl_ckpt.exists():
        raise FileNotFoundError(f"No SSL ckpt at {ssl_ckpt}")
    print(f"loading LeJEPA from {ssl_ckpt} ...")
    ssl_model, ssl_dim = load_ssl_backbone(ssl_ckpt, device)

    vae_ckpt = cifar_ckpt_path("cifar10", args.vae_z, args.vae_beta)
    if not vae_ckpt.exists():
        raise FileNotFoundError(f"No VAE ckpt at {vae_ckpt}")
    print(f"loading VAE from {vae_ckpt} ...")
    vae = load_vae(vae_ckpt, device)
    vae_dim = args.vae_z

    # --- Encode and probe each dataset --------------------------------
    results: dict[str, dict[str, dict]] = {}
    for dataset in ["cifar10", "cifar100"]:
        print(f"\n=== {dataset} ===")
        X_tr, y_tr = load_cifar_train(dataset)
        X_te, y_te = load_cifar_test(dataset)
        X_tr_t = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
        X_te_t = torch.as_tensor(X_te, dtype=torch.float32, device=device)
        y_tr_t = torch.as_tensor(y_tr, dtype=torch.long, device=device)
        y_te_t = torch.as_tensor(y_te, dtype=torch.long, device=device)
        n_classes = int(max(int(y_tr.max()), int(y_te.max()))) + 1
        print(f"  classes: {n_classes}, train: {tuple(X_tr.shape)}, "
              f"test: {tuple(X_te.shape)}")

        print("  encoding through LeJEPA ...")
        Z_ssl_tr = encode_lejepa_normed(ssl_model, X_tr_t,
                                        batch_size=args.batch_size,
                                        device=device)
        Z_ssl_te = encode_lejepa_normed(ssl_model, X_te_t,
                                        batch_size=args.batch_size,
                                        device=device)
        print("  encoding through VAE encoder ...")
        Z_vae_tr = encode_vae(vae, X_tr_t, batch_size=args.batch_size)
        Z_vae_te = encode_vae(vae, X_te_t, batch_size=args.batch_size)
        print(f"  shapes: VAE {tuple(Z_vae_tr.shape)} "
              f"LeJEPA {tuple(Z_ssl_tr.shape)}")

        # Per-block standardization on train stats.
        Z_vae_tr_n, Z_vae_te_n = standardize(Z_vae_tr, Z_vae_te)
        Z_ssl_tr_n, Z_ssl_te_n = standardize(Z_ssl_tr, Z_ssl_te)
        Z_cat_tr = torch.cat([Z_vae_tr_n, Z_ssl_tr_n], dim=-1)
        Z_cat_te = torch.cat([Z_vae_te_n, Z_ssl_te_n], dim=-1)

        results[dataset] = {}
        for rep_name, Ztr, Zte in [
            ("vae",        Z_vae_tr_n, Z_vae_te_n),
            ("lejepa",     Z_ssl_tr_n, Z_ssl_te_n),
            ("vae+lejepa", Z_cat_tr,   Z_cat_te),
        ]:
            tag = f"{rep_name} on {dataset}"
            res = fit_probe(
                Ztr, y_tr_t, Zte, y_te_t,
                n_classes=n_classes, epochs=args.epochs, lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size, device=device,
                seed=args.seed, tag=tag,
            )
            res["chance"] = 100.0 / n_classes
            res["chance5"] = 500.0 / n_classes
            results[dataset][rep_name] = res
            print(f"    --> top-1={res['top1']*100:5.2f}  "
                  f"top-5={res['top5']*100:5.2f}  "
                  f"(best top-1={res['best_top1']*100:5.2f})")

        del Z_vae_tr, Z_vae_te, Z_ssl_tr, Z_ssl_te
        del Z_vae_tr_n, Z_vae_te_n, Z_ssl_tr_n, Z_ssl_te_n
        del Z_cat_tr, Z_cat_te
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Summary ------------------------------------------------------
    print("\n=== summary: linear-probe test accuracy (%) ===")
    hdr = f"  {'representation':<14} " + " ".join(
        f"{ds + ' top1':>14} {ds + ' top5':>14}" for ds in ["cifar10", "cifar100"]
    )
    print(hdr)
    for rep_name in ["vae", "lejepa", "vae+lejepa"]:
        row = f"  {rep_name:<14}"
        for ds in ["cifar10", "cifar100"]:
            r = results[ds][rep_name]
            row += f" {r['top1']*100:>14.2f} {r['top5']*100:>14.2f}"
        print(row)
    print(f"  {'chance':<14}", end="")
    for ds in ["cifar10", "cifar100"]:
        r = results[ds]["vae"]  # any rep -- chance is the same
        print(f" {r['chance']:>14.2f} {r['chance5']:>14.2f}", end="")
    print()


if __name__ == "__main__":
    main()
