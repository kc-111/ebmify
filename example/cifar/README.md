# CIFAR experiments

CIFAR-10 / CIFAR-100 pipelines for leverage-as-EBM,
leverage-as-posterior-variance, and leverage-as-OOD. See
`LEVERAGE_FINDINGS.md` for the consolidated write-up of what works,
what doesn't, and why.

## Layout

```
cifar/
├── LEVERAGE_FINDINGS.md       this directory's writeup
├── README.md                  (you are here)
├── _paths.py                  shared sys.path bootstrap
├── cifar_data.py              CIFAR-10/100 numpy loaders + ckpt paths
│
├── train/                     training entry points
│   ├── cifar_resnet18_train.py
│   ├── cifar_vae_train.py
│   └── ssl_pretrain.py
│
├── ood/                       per-backbone OOD threshold
│   ├── cifar_resnet18_ood_threshold.py
│   ├── cifar_ssl_ood_threshold.py
│   ├── cifar_dinov2_ood_threshold.py
│   └── cifar_vae_ood_threshold.py
│
├── ebm/                       leverage-as-EBM / Langevin sampling
│   └── cifar_vae_langevin.py
│
├── diagnostics/               bandwidth/centering/concat scans
│   ├── cifar_resnet18_bandwidth_scan.py
│   ├── cifar_memorization_scan.py
│   ├── cifar_centering_comparison.py
│   ├── cifar_concat_features_test.py
│   └── cifar_concat_linear_probe.py
│
├── probes/                    linear probes & sanity checks
│   ├── ssl_linear_probe.py
│   └── cifar_vae_sanity.py
│
├── archive/                   superseded scripts (do not import)
├── cache/                     trained checkpoints
└── logs/                      ssl_pretrain logs (Lightning CSV)
```

Any script in any subdir can import from any other (cross-subdir
imports are handled by `_paths.py`, which every subdir script imports
once near the top).

## Backbones

| Tag        | Where                                  | What                                 |
|------------|----------------------------------------|--------------------------------------|
| `resnet18` | `train/cifar_resnet18_train.py`        | Supervised ResNet18 on CIFAR-10      |
| `ssl`      | `train/ssl_pretrain.py`                | LeJEPA ResNet18 (joint-embed inv.)   |
| `dinov2`   | torch.hub.load (no training script)    | DINOv2 ViT-B/14, frozen Meta release |
| `vae`      | `train/cifar_vae_train.py`             | β-VAE encoder μ                      |

## Suggested reading order

1. `LEVERAGE_FINDINGS.md` — framework and findings
2. `probes/cifar_vae_sanity.py` + `train/cifar_vae_train.py` — VAE basics
3. `ebm/cifar_vae_langevin.py` — leverage as EBM in action
4. `ood/cifar_*_ood_threshold.py` — leverage as OOD per backbone
5. `diagnostics/cifar_centering_comparison.py` and
   `diagnostics/cifar_memorization_scan.py` — the diagnostics that
   informed the writeup
6. `diagnostics/cifar_concat_features_test.py` and
   `diagnostics/cifar_concat_linear_probe.py` — stacking backbones,
   OOD vs probe trade-off

## Archive

`archive/` holds three superseded scripts kept for historical
reference:

- `cifar_resnet_ood_threshold.py` — ResNet50 ImageNet-pretrained,
  superseded by `ood/cifar_resnet18_ood_threshold.py` plus DINOv2.
- `cifar_vae_ood_ell_sweep.py` — older ell sweep, superseded by
  `diagnostics/cifar_memorization_scan.py`.
- `cifar_vae_ood_eval.py` — older x→z OOD eval, superseded by
  `ood/cifar_vae_ood_threshold.py`.

These do *not* go through `_paths.py` and may have stale imports
relative to the current layout.

## Outputs

Figures land in `../out/`; checkpoints in `cache/`; ssl_pretrain
logs in `logs/`.
