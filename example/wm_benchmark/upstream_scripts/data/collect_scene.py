# Copied verbatim from stable-worldmodel @ 463ab63517b043ca6c3753b01e34ea6f145497c6
# Source: https://github.com/galilai-group/stable-worldmodel/blob/463ab63517b043ca6c3753b01e34ea6f145497c6/scripts/data/collect_scene.py
# License: MIT (see example/wm_benchmark/ATTRIBUTION.md)
import os
from pathlib import Path

os.environ['MUJOCO_GL'] = 'egl'

import hydra
import numpy as np
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm
from stable_worldmodel.envs.ogbench import ExpertPolicy


@hydra.main(version_base=None, config_path='./config', config_name='ogb')
def run(cfg: DictConfig):
    """Run parallel data collection script"""

    world = swm.World(
        'swm/OGBScene-v0',
        **cfg.world,
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=False,
        mode='data_collection',
    )

    options = cfg.get('options')
    options = OmegaConf.to_object(options) if options is not None else None

    rng = np.random.default_rng(cfg.seed)
    world.set_policy(ExpertPolicy())

    world.collect(
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / 'ogbench/scene_single_expert.lance',
        episodes=cfg.num_traj,
        seed=rng.integers(0, 1_000_000).item(),
        options=options,
    )

    logging.success(
        '🎉🎉🎉 Completed data collection for ogbench scene  🎉🎉🎉'
    )


if __name__ == '__main__':
    run()
