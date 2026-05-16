# Copied verbatim from stable-worldmodel @ 463ab63517b043ca6c3753b01e34ea6f145497c6
# Source: https://github.com/galilai-group/stable-worldmodel/blob/463ab63517b043ca6c3753b01e34ea6f145497c6/scripts/examples/breakout.py
# License: MIT (see example/wm_benchmark/ATTRIBUTION.md)
from pathlib import Path

import stable_worldmodel as swm
from stable_worldmodel.policy import RandomPolicy


VIDEO_DIR = Path(__file__).parent / 'videos' / 'breakout'

world = swm.World(
    'ALE/Breakout-v5',
    num_envs=1,
    image_shape=(224, 160),
    max_episode_steps=1000,
    goal_conditioned=False,
    render_mode='rgb_array',
)
world.set_policy(RandomPolicy(seed=0))

world.evaluate(episodes=1, seed=0, video=VIDEO_DIR)
