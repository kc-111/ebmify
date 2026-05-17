"""Eval a trained DKA denoiser at multiple noise levels.

Reports val MSE + PSNR in image space (after depatchify and clamp to
[0, 1]), and compares against the identity baseline (do-nothing PSNR of
the noisy input).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import DKA_ROOT  # noqa: F401, E402

import argparse as _argparse  # noqa: E402

from dka_cifar_data import depatchify, get_arrays, get_noisy_clean_batch  # noqa: E402
from train import PatchDenoiser  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n-batches", type=int, default=40)
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.05, 0.1, 0.2, 0.4])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def load(ckpt_path: Path, device: torch.device) -> tuple[PatchDenoiser, dict]:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = blob["config"]
    patch_dim = blob["patch_dim"]
    args = _argparse.Namespace(**cfg)
    model = PatchDenoiser(patch_dim=patch_dim, args=args).to(device)
    model.load_state_dict(blob["state_dict"], strict=True)
    model.eval()
    return model, cfg


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, cfg = load(args.ckpt, device)
    val_images, _ = get_arrays(cfg["dataset"], "test")
    patch_size = cfg["patch_size"]

    print(f"ckpt:     {args.ckpt}")
    print(f"dataset:  {cfg['dataset']}  patch_size={patch_size}  "
          f"trained_sigma={cfg['noise_sigma']}")
    print(f"{'sigma':>6}  {'mse_img':>10}  {'psnr_db':>10}  "
          f"{'id_psnr_db':>12}  {'gain_db':>9}")
    rng = torch.Generator().manual_seed(args.seed)
    with torch.no_grad():
        for sigma in args.sigmas:
            mse_running = 0.0
            id_mse_running = 0.0
            for _ in range(args.n_batches):
                noisy, clean = get_noisy_clean_batch(
                    val_images, args.batch, patch_size, sigma, device, generator=rng,
                )
                pred, _ = model(noisy)
                pred_img = depatchify(pred, patch_size).clamp(0.0, 1.0)
                clean_img = depatchify(clean, patch_size).clamp(0.0, 1.0)
                noisy_img = depatchify(noisy, patch_size).clamp(0.0, 1.0)
                mse_running += F.mse_loss(pred_img, clean_img).item()
                id_mse_running += F.mse_loss(noisy_img, clean_img).item()
            mse = mse_running / args.n_batches
            id_mse = id_mse_running / args.n_batches
            psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
            id_psnr = 10.0 * math.log10(1.0 / max(id_mse, 1e-12))
            print(
                f"{sigma:>6.3f}  {mse:>10.5f}  {psnr:>10.2f}  "
                f"{id_psnr:>12.2f}  {psnr - id_psnr:>+9.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
