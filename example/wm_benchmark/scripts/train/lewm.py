"""Local LeWM trainer — fork of ``upstream_scripts/train/lewm.py``.

Reuses upstream's ``lejepa_forward`` + model builder verbatim; this script
owns argparse/dataclass config, local CSVLogger + verbose progress logging,
and writes checkpoints under ``data/runs/lewm/<run_id>/``.

Usage:

    python scripts/train/lewm.py
    python scripts/train/lewm.py trainer.max_epochs=10 loader.batch_size=64
    python scripts/train/lewm.py --config-file my_overrides.yaml
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

_BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
if str(_BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_ROOT))

import torch  # noqa: E402

import stable_pretraining as spt  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.data import column_normalizer as get_column_normalizer  # noqa: E402
from stable_worldmodel.wm.lewm import LeWM  # noqa: E402
from stable_worldmodel.wm.lewm.module import MLP, Embedder, Predictor  # noqa: E402
from stable_worldmodel.wm.loss import SIGReg  # noqa: E402

from scripts.common import config as cfgmod  # noqa: E402
from scripts.train._runner import run_trainer  # noqa: E402
from scripts.train._upstream import load_upstream, to_omegaconf  # noqa: E402
from scripts.train.configs.lewm import LeWMConfig  # noqa: E402

METHOD = "lewm"


def _build(cfg: LeWMConfig):
    """Construct (data_module, spt.Module) from a local dataclass cfg."""
    upstream = load_upstream("lewm")
    oc = to_omegaconf(cfg)
    # upstream's `lejepa_forward` expects cfg attributes; pass `oc` through.

    # ---- dataset
    dataset_cfg = dict(oc.data)
    dataset_name = dataset_cfg.pop("name")
    dataset = swm.data.load_dataset(dataset_name, transform=None, **dataset_cfg)

    transforms = [upstream.get_img_preprocessor("pixels", "pixels", img_size=cfg.img_size)]
    from omegaconf import open_dict
    with open_dict(oc):
        for col in oc.data.keys_to_load:
            if col.startswith("pixels"):
                continue
            transforms.append(get_column_normalizer(dataset, col, col))
            setattr(oc.wm, f"{col}_dim", dataset.get_dim(col))

    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, **{k: getattr(cfg.loader, k) for k in
                     ("batch_size", "num_workers", "drop_last", "persistent_workers",
                      "prefetch_factor", "pin_memory", "shuffle")},
        generator=rnd_gen,
    )
    val_kwargs = {k: getattr(cfg.loader, k) for k in
                  ("batch_size", "num_workers", "persistent_workers",
                   "prefetch_factor", "pin_memory")}
    val_kwargs.update(shuffle=False, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_set, **val_kwargs)

    # ---- model
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.embed_dim or hidden_dim
    action_dim = int(oc.wm.get("action_dim", 0)) or dataset.get_dim("action")
    effective_act_dim = cfg.data.frameskip * action_dim

    predictor = Predictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        depth=cfg.predictor.depth,
        heads=cfg.predictor.heads,
        mlp_dim=cfg.predictor.mlp_dim,
        dim_head=cfg.predictor.dim_head,
        dropout=cfg.predictor.dropout,
        emb_dropout=cfg.predictor.emb_dropout,
    )
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    world_model = LeWM(encoder=encoder, predictor=predictor,
                       action_encoder=action_encoder,
                       projector=projector, pred_proj=pred_proj)

    total_steps = cfg.trainer.max_epochs * len(train_loader)
    optim = {
        "model_opt": {
            "modules": "model",
            "optimizer": {"type": cfg.optimizer.type,
                          "lr": cfg.optimizer.lr,
                          "weight_decay": cfg.optimizer.weight_decay},
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR",
                          "warmup_steps": max(1, int(0.01 * total_steps)),
                          "max_steps": total_steps},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train_loader, val=val_loader)
    module = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(upstream.lejepa_forward, cfg=oc),
        optim=optim,
    )
    return data_module, module


def main(argv: list[str] | None = None) -> int:
    cfg = cfgmod.from_argv(LeWMConfig, argv, description=__doc__)
    run_trainer(cfg, _build, METHOD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
