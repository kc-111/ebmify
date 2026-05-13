"""SSL-backbone CIFAR-10 coreset builder.

This is the entry point that *runs the three coreset algorithms* on
LeJEPA-recon SSL ResNet-18 features. It loads the SSL checkpoint
produced by ``example/cifar/train/ssl_pretrain_recon.py`` and reuses
the same extract -> ``coreset.cli`` -> diagnostics path as the
supervised builder. Class balance is intentionally *not* applied here
since SSL training is label-agnostic.

Outputs:
    example/cifar/cache/phi_ssl_resnet18_<tag>.pt
    example/cifar/cache/coreset/ssl_resnet18_<tag>/
    example/out/coreset/ssl_resnet18_<tag>_coreset_selection_diagnostics.png
    example/out/coreset/ssl_resnet18_<tag>_coreset_spectrum.png

Usage:
    python example/cifar/coreset/cifar_build_coreset_ssl.py
    python example/cifar/coreset/cifar_build_coreset_ssl.py --ssl-tag recon --budget-k 10000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import REPO_ROOT  # noqa: F401, E402

from _extract_common import extract_and_select  # noqa: E402


def _ssl_ckpt_path(tag: str) -> Path:
    return REPO_ROOT / "example" / "cifar" / "cache" / f"cifar10_ssl_resnet18_{tag}.pt"


def _load_ssl_backbone(ckpt_path: Path, device: str) -> nn.Module:
    import stable_pretraining as spt  # heavy dep, local import
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)
    backbone.fc = nn.Identity()
    backbone.load_state_dict(raw["state_dict"])
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone.to(device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--ssl-tag", default="recon_ema", dest="ssl_tag",
        help=("SSL checkpoint tag. Loads "
              "example/cifar/cache/cifar10_ssl_resnet18_<tag>.pt (produced "
              "by example/cifar/train/ssl_pretrain_recon.py)."),
    )
    ap.add_argument(
        "--budget-k", type=int, default=5000, dest="budget_k",
        help=("Coreset size k -- number of samples each algorithm selects "
              "from the 50k CIFAR-10 train set."),
    )
    ap.add_argument(
        "--ridge", type=float, default=1e-3,
        help=("Ridge lambda used by 'leverage' and 'spectral_rank' when "
              "regularizing the spectrum (Phi^T Phi + lambda*I)."),
    )
    ap.add_argument(
        "--n-buckets", type=int, default=64, dest="n_buckets",
        help=("Number of equal-mass eigenvector buckets used by "
              "spectral_rank_coverage (B). Bigger B = finer per-band coverage."),
    )
    ap.add_argument(
        "--n-top-eigvecs", type=int, default=64, dest="n_top_eigvecs",
        help=("Width of the spectral_coords aux target. Does NOT affect "
              "which samples get selected -- only downstream supervision."),
    )
    ap.add_argument(
        "--batch", type=int, default=256,
        help="Mini-batch size for the feature-extraction forward pass.",
    )
    ap.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for leverage sampling and spectral_rank tie-breaking.",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    ckpt = _ssl_ckpt_path(args.ssl_tag)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No SSL checkpoint at {ckpt}. Run "
            f"`python example/cifar/train/ssl_pretrain_recon.py` first."
        )
    print(f"loading SSL backbone from {ckpt}")
    model = _load_ssl_backbone(ckpt, device)
    tag = f"ssl_resnet18_{args.ssl_tag}"

    extract_and_select(
        tag=tag,
        model=model,
        norm="cifar",
        resize=None,
        batch_size=args.batch,
        device=device,
        budget_k=args.budget_k,
        ridge_lambda=args.ridge,
        n_buckets=args.n_buckets,
        n_top_eigvecs=args.n_top_eigvecs,
        standardize="center_l2",
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
