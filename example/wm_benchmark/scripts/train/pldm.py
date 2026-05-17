"""Local PLDM trainer — fork of ``upstream_scripts/train/pldm.py``.

Reuses upstream's ``pldm_forward`` + PLDM model + IDM + losses verbatim;
local scaffolding handles config/logger/checkpoint.

Usage:

    python scripts/train/pldm.py
    python scripts/train/pldm.py trainer.max_epochs=10 loss.sigreg.weight=0.05
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

_BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
if str(_BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import stable_pretraining as spt  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.data import column_normalizer as get_column_normalizer  # noqa: E402
from stable_worldmodel.wm.loss import PLDMLoss, TemporalStraighteningLoss  # noqa: E402
from stable_worldmodel.wm.pldm import PLDM  # noqa: E402
from stable_worldmodel.wm.pldm.module import MLP, Embedder, Predictor  # noqa: E402

from scripts.common import config as cfgmod  # noqa: E402
from scripts.train._runner import run_trainer  # noqa: E402
from scripts.train._upstream import load_upstream, to_omegaconf  # noqa: E402
from scripts.train.configs.pldm import PLDMConfig  # noqa: E402

METHOD = "pldm"


def _build(cfg: PLDMConfig):
    upstream = load_upstream("pldm")
    oc = to_omegaconf(cfg)

    # ---- dataset
    dataset_cfg = dict(oc.data)
    dataset_name = dataset_cfg.pop("name")
    dataset = swm.data.load_dataset(dataset_name, transform=None, **dataset_cfg)
    img_proc = upstream.get_img_preprocessor("pixels", "pixels", cfg.img_size)

    extra = []
    for col in oc.data.keys_to_load:
        if col == "pixels":
            continue
        extra.append(get_column_normalizer(dataset, col, col))
    for col in oc.data.get("keys_to_merge", {}):
        extra.append(get_column_normalizer(dataset, col, col))

    from omegaconf import open_dict
    with open_dict(oc):
        for col in oc.data.keys_to_load:
            if col == "pixels":
                continue
            setattr(oc.wm, f"{col}_dim", dataset.get_dim(col))

    dataset.transform = spt.data.transforms.Compose(img_proc, *extra)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen,
    )

    loader_kw = {k: getattr(cfg.loader, k) for k in
                 ("batch_size", "num_workers", "drop_last", "persistent_workers",
                  "prefetch_factor", "pin_memory", "shuffle")}
    train_loader = DataLoader(train_set, **loader_kw, generator=rnd_gen)
    val_kw = dict(loader_kw); val_kw["shuffle"] = False; val_kw["drop_last"] = False
    val_loader = DataLoader(val_set, **val_kw)

    # ---- model
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale, patch_size=cfg.patch_size, image_size=cfg.img_size,
        pretrained=False, use_mask_token=False,
    )
    hidden = encoder.config.hidden_size
    embed_dim = cfg.wm.embed_dim or hidden
    action_dim = int(oc.wm.get("action_dim", 0)) or dataset.get_dim("action")
    effective_act_dim = cfg.data.frameskip * action_dim

    predictor = Predictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim, hidden_dim=hidden, output_dim=hidden,
        depth=cfg.predictor.depth, heads=cfg.predictor.heads,
        mlp_dim=cfg.predictor.mlp_dim, dim_head=cfg.predictor.dim_head,
        dropout=cfg.predictor.dropout, emb_dropout=cfg.predictor.emb_dropout,
    )
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(input_dim=hidden, output_dim=embed_dim, hidden_dim=2048, norm_fn=nn.BatchNorm1d)
    pred_proj = MLP(input_dim=hidden, output_dim=embed_dim, hidden_dim=2048, norm_fn=nn.BatchNorm1d)
    idm = MLP(input_dim=2 * embed_dim, hidden_dim=512, output_dim=effective_act_dim)

    world_model = PLDM(encoder=encoder, predictor=predictor,
                       action_encoder=action_encoder,
                       projector=projector, pred_proj=pred_proj)
    models = {"model": world_model, "idm": idm}
    losses = {"pldm": PLDMLoss(), "path_straight": TemporalStraighteningLoss()}

    total_steps = cfg.trainer.max_epochs * len(train_loader)
    optim = {}
    for name in models:
        optim[f"{name}_opt"] = {
            "modules": name,
            "optimizer": {"type": cfg.optimizer.type,
                          "lr": cfg.optimizer.lr,
                          "weight_decay": cfg.optimizer.weight_decay},
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR",
                          "warmup_steps": max(1, int(0.01 * total_steps)),
                          "max_steps": total_steps},
            "interval": "epoch",
        }

    data_module = spt.data.DataModule(train=train_loader, val=val_loader)
    module = spt.Module(**models, **losses,
                        forward=partial(upstream.pldm_forward, cfg=oc),
                        optim=optim)
    return data_module, module


def main(argv: list[str] | None = None) -> int:
    cfg = cfgmod.from_argv(PLDMConfig, argv, description=__doc__)
    run_trainer(cfg, _build, METHOD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
