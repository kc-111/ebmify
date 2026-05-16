# Copied verbatim from stable-worldmodel @ 463ab63517b043ca6c3753b01e34ea6f145497c6
# Source: https://github.com/galilai-group/stable-worldmodel/blob/463ab63517b043ca6c3753b01e34ea6f145497c6/scripts/data/collect_pointmaze.py
# License: MIT (see example/wm_benchmark/ATTRIBUTION.md)
from pathlib import Path

import stable_worldmodel as swm
from stable_worldmodel.policy import RandomPolicy

world = swm.World(
    'swm/OGBMaze-v0',
    num_envs=4,
    image_shape=(224, 224),
    loco_env_type='point',
    maze_env_type='maze',
    maze_type='teleport',
    ob_type='pixels',
    max_episode_steps=21,
)

world.set_policy(RandomPolicy())

world.collect(
    path=Path(swm.data.utils.get_cache_dir())
    / 'datasets'
    / 'pointmaze-teleport-navigate-v0.lance',
    episodes=2,
    seed=0,
)
