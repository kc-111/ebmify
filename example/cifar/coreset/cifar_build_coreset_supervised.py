"""Supervised-backbone CIFAR-10 coreset builder.

This is the entry point that *runs the three coreset algorithms* (greedy
max-variance, ridge leverage, spectral-rank coverage) on supervised
ResNet-18 features. It:

1. Loads either the locally-trained CIFAR ResNet-18 (default) or the
   ImageNet-pretrained ResNet-18 (``--backbone imagenet``).
2. Extracts features on the full 50k CIFAR-10 train set.
3. Invokes ``coreset.cli`` to run all three selection algorithms and
   emit aux targets.
4. Renders the diagnostic plots.

Outputs:
    example/cifar/cache/phi_<tag>.pt                  feature matrix + labels
    example/cifar/cache/coreset/<tag>/                coreset artifacts
    example/out/coreset/<tag>_coreset_selection_diagnostics.png
    example/out/coreset/<tag>_coreset_spectrum.png

Usage:
    python example/cifar/coreset/cifar_build_coreset_supervised.py
    python example/cifar/coreset/cifar_build_coreset_supervised.py --backbone imagenet
    python example/cifar/coreset/cifar_build_coreset_supervised.py --budget-k 10000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tvm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import REPO_ROOT  # noqa: F401, E402

from cifar_resnet18_train import make_resnet18_cifar  # noqa: E402

from _extract_common import extract_and_select  # noqa: E402


def _load_supervised_ckpt(device: str) -> nn.Module:
    ckpt_path = REPO_ROOT / "example" / "cifar" / "cache" / "cifar10_resnet18.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No supervised checkpoint at {ckpt_path}. "
            f"Run `python example/cifar/train/cifar_resnet18_train.py` first."
        )
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = raw.get("config", {})
    model = make_resnet18_cifar(num_classes=int(cfg.get("num_classes", 10)))
    model.load_state_dict(raw["state_dict"])
    model.fc = nn.Identity()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def _load_imagenet_backbone(device: str) -> nn.Module:
    weights = tvm.ResNet18_Weights.IMAGENET1K_V1
    model = tvm.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--backbone", choices=["supervised", "imagenet"], default="supervised",
        help=("Feature extractor. 'supervised' uses the locally-trained CIFAR "
              "ResNet-18 ckpt (example/cifar/cache/cifar10_resnet18.pt); "
              "'imagenet' uses torchvision IMAGENET1K_V1 weights."),
    )
    ap.add_argument(
        "--budget-k", type=int, default=500, dest="budget_k",
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
              "spectral_rank_coverage (B). Bigger B = finer per-band coverage; "
              "see src/coreset/spectral_rank.py for the full algorithm."),
    )
    ap.add_argument(
        "--n-top-eigvecs", type=int, default=64, dest="n_top_eigvecs",
        help=("Width of the spectral_coords aux target (aux_spectral_coords.pt "
              "is (k, N) where N = n_top_eigvecs). Does NOT affect which "
              "samples get selected -- only downstream supervision."),
    )
    ap.add_argument(
        "--batch", type=int, default=256,
        help="Mini-batch size for the feature-extraction forward pass.",
    )
    ap.add_argument(
        "--resize", type=int, default=None,
        help=("Square resize before feature extraction. Only meaningful with "
              "--backbone imagenet (defaults to 224); ignored otherwise."),
    )
    ap.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for leverage sampling and spectral_rank tie-breaking.",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    if args.backbone == "supervised":
        model = _load_supervised_ckpt(device)
        norm = "cifar"
        resize = None
        tag = "supervised_resnet18"
    else:
        model = _load_imagenet_backbone(device)
        norm = "imagenet"
        resize = args.resize if args.resize is not None else 224
        tag = "imagenet_resnet18"

    extract_and_select(
        tag=tag,
        model=model,
        norm=norm,
        resize=resize,
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
