"""Local PreJEPA trainer — fork of ``upstream_scripts/train/prejepa.py``.

Reuses upstream's ``get_encoder`` / ``get_img_preprocessor`` /
``dinowm_forward`` verbatim and re-implements the dataset+model+optim
wiring (upstream's `run` body is tightly coupled to Hydra context).
"""
from __future__ import annotations

import sys
from collections import OrderedDict
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

from scripts.common import config as cfgmod  # noqa: E402
from scripts.train._runner import run_trainer  # noqa: E402
from scripts.train._upstream import load_upstream, to_omegaconf  # noqa: E402
from scripts.train.configs.prejepa import PreJEPAConfig  # noqa: E402

METHOD = "prejepa"


def _build(cfg: PreJEPAConfig):
    upstream = load_upstream("prejepa")
    oc = to_omegaconf(cfg)
    from omegaconf import open_dict

    encoding_keys = list(oc.wm.get("encoding", {}).keys())
    keys_to_load = ["pixels"] + encoding_keys

    dataset = swm.data.load_dataset(
        oc.dataset_name,
        num_steps=oc.n_steps,
        frameskip=oc.frameskip,
        transform=None,
        cache_dir=oc.get("cache_dir", None),
        keys_to_load=keys_to_load,
        keys_to_cache=encoding_keys,
    )

    normalizers = [get_column_normalizer(dataset, c, c)
                   for c in oc.wm.get("encoding", {})]
    if oc.backbone.get("is_video_encoder", False):
        from transformers import AutoVideoProcessor
        from stable_worldmodel.data.transforms import VideoPipeline  # type: ignore[attr-defined]
        processor = AutoVideoProcessor.from_pretrained(oc.backbone.name)
        transform = spt.data.transforms.Compose(
            VideoPipeline(processor, source="pixels", target="pixels"),
            spt.data.transforms.Resize(oc.image_size, source="pixels", target="pixels"),
            *normalizers,
        )
    else:
        transform = spt.data.transforms.Compose(
            upstream.get_img_preprocessor("pixels", "pixels", oc.image_size),
            *normalizers,
        )
    dataset.transform = transform

    with open_dict(oc) as oc:
        oc.extra_dims = {}
        for key in oc.wm.get("encoding", {}):
            if key not in dataset.column_names:
                raise ValueError(f"encoding key '{key}' not in dataset columns")
            dim = dataset.get_dim(key)
            oc.extra_dims[key] = dim if key != "action" else dim * oc.frameskip

    rnd_gen = torch.Generator().manual_seed(oc.seed)
    train_set, val_set = spt.data.random_split(
        dataset, [oc.train_split, 1 - oc.train_split], generator=rnd_gen,
    )

    train_loader = DataLoader(
        train_set, batch_size=oc.batch_size, num_workers=oc.num_workers,
        drop_last=True, persistent_workers=True, pin_memory=True, shuffle=True,
        generator=rnd_gen,
    )
    val_loader = DataLoader(
        val_set, batch_size=oc.batch_size, num_workers=oc.num_workers, pin_memory=True,
    )

    encoder, embed_dim, num_patches, interp_pos_enc = upstream.get_encoder(oc)
    embed_dim += sum(oc.wm.get("encoding", {}).values())
    if oc.backbone.get("is_video_encoder", False):
        num_patches += num_patches * (oc.n_steps // 4)

    predictor_kwargs = {k: v for k, v in oc.predictor.items() if k != "size"}
    predictor = swm.wm.prejepa.CausalPredictor(
        num_patches=num_patches, num_frames=oc.wm.history_size,
        dim=embed_dim, **predictor_kwargs,
    )

    extra_encoders = nn.ModuleDict(OrderedDict(
        (key, swm.wm.prejepa.Embedder(in_chans=oc.extra_dims[key], emb_dim=ed))
        for key, ed in oc.wm.get("encoding", {}).items()
    ))

    world_model = swm.wm.PreJEPA(
        encoder=spt.backbone.EvalOnly(encoder),
        predictor=predictor,
        extra_encoders=extra_encoders,
        history_size=oc.wm.history_size,
        num_pred=oc.wm.num_preds,
        interpolate_pos_encoding=interp_pos_enc,
    )

    module = spt.Module(
        model=world_model,
        forward=partial(upstream.dinowm_forward, cfg=oc),
        optim={"model_opt": {"modules": "model", "optimizer": dict(oc.optimizer)}},
    )
    data = spt.data.DataModule(train=train_loader, val=val_loader)
    return data, module


def main(argv: list[str] | None = None) -> int:
    cfg = cfgmod.from_argv(PreJEPAConfig, argv, description=__doc__)
    run_trainer(cfg, _build, METHOD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
