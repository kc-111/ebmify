# World-Model Benchmark (stable-worldmodel wrapper)

Thin wrapper around **stable-worldmodel** (galilai-group, MIT) — a unified
offline world-modeling framework with a `swm.World(...)` API, Lance-format
data, a `world.evaluate(...)` protocol, and a baseline zoo (DINO-WM, PLDM,
LeWM, GCBC, IQL / IVL / HILP).

This directory does not vendor a benchmark. It provides:

- **Sibling install** of stable-worldmodel (cloned to `~/Desktop/...`, pip
  installed editable into ebmify's venv).
- A **verbatim copy of upstream's `scripts/` tree** under
  `upstream_scripts/`, pinned to commit
  [`463ab63`](https://github.com/galilai-group/stable-worldmodel/tree/463ab63517b043ca6c3753b01e34ea6f145497c6/scripts).
  Every file carries a 3-line header pointing back at its source. See
  [`ATTRIBUTION.md`](ATTRIBUTION.md).
- A few thin entrypoints in `scripts/` (env setup, bootstrap health check,
  collect dispatcher, verify smoke test) that defer to the upstream code.
- A gitignored `data/` dir that becomes `$STABLEWM_HOME` (datasets +
  checkpoints land here, not in `~/.stable_worldmodel/`).

## Connection to ebmify

The leverage-as-EBM machinery in this repo defines an energy
$h(x) = \phi(x)^\top (\Phi^\top \Phi + \lambda I)^{-1} \phi(x)$ on the
last-layer features of any backbone. On these benchmarks the natural follow-up
is a *transition* energy — e.g. $h([x_t, x_{t+1}])$ or
$h(x_{t+1} \mid x_t, a_t)$ — scored against stable-worldmodel's planners
(`CEMSolver`, `iCEMSolver`, `MPPI`, `WorldModelPolicy`) via
`world.evaluate(...)`. This example sets up the data + envs + baselines needed
to build that bridge; no EBM training code lives here yet.

## Install

Three one-time steps. See `requirements.txt` for verbatim commands.

```bash
# 1. Clone stable-worldmodel as a sibling and pip-install editable with [env]
cd ~/Desktop
git clone https://github.com/galilai-group/stable-worldmodel.git
cd stable-worldmodel
uv pip install -e '.[env]'              # [env] brings ogbench, dm_control via gymnasium[all], etc.
uv pip install 'pymunk>=7'              # swm/PushT-v1 needs pymunk 7's on_collision API

# 2. Our thin extras
cd /home/kevin-cheung/Desktop/ebmify
uv pip install -r example/wm_benchmark/requirements.txt

# 3. Point swm's cache at this example's data/  (once per shell)
source example/wm_benchmark/scripts/env.sh
```

The sibling-clone pattern mirrors how `stable_pretraining` is consumed across
`example/cifar/` (`example/cifar/train/ssl_pretrain.py`).

## Verify

```bash
python example/wm_benchmark/scripts/bootstrap.py    # confirms swm imports + STABLEWM_HOME
python example/wm_benchmark/scripts/verify.py       # builds 1 World per benchmark env, step + render
python example/wm_benchmark/scripts/verify.py --env pointmaze   # subset
```

`verify.py` exercises the same `swm.World(...)` calls used by
`upstream_scripts/data/collect_*.py`, so a green verify means the collect
scripts will instantiate cleanly.

## Collect data

```bash
# Dispatcher into the upstream collect scripts (hydra-based). Trailing args
# pass through to the upstream argv:
python example/wm_benchmark/scripts/collect.py pointmaze
python example/wm_benchmark/scripts/collect.py pusht_fov  num_traj=10
python example/wm_benchmark/scripts/collect.py tworooms
python example/wm_benchmark/scripts/collect.py reacher
python example/wm_benchmark/scripts/collect.py cube

# Or invoke the upstream script directly (same effect):
python example/wm_benchmark/upstream_scripts/data/collect_pointmaze.py
```

Lance files land under `$STABLEWM_HOME/datasets/` — i.e.
`example/wm_benchmark/data/datasets/`.

## Train baselines

Upstream ships 7 baseline trainers under `upstream_scripts/train/`
(hydra-configured): `pldm.py`, `prejepa.py`, `lewm.py`, `gcbc.py`,
`gciql.py`, `gcivl.py`, `hilp.py`. Run them directly:

```bash
python example/wm_benchmark/upstream_scripts/train/pldm.py
python example/wm_benchmark/upstream_scripts/train/gcbc.py
# etc. — each script has its own config under upstream_scripts/train/config/
```

## Evaluate

```bash
python example/wm_benchmark/upstream_scripts/plan/eval_wm.py   # world-model + planner
python example/wm_benchmark/upstream_scripts/plan/eval_ff.py   # feed-forward baseline
```

Per-task config under `upstream_scripts/plan/config/{pusht,tworoom,reacher,cube}.yaml`.

## Visualize

```bash
python example/wm_benchmark/upstream_scripts/visualization/visualize_dataset.py
python example/wm_benchmark/upstream_scripts/visualization/visualize_trajectories.py
python example/wm_benchmark/upstream_scripts/visualization/visualize_env.py
python example/wm_benchmark/upstream_scripts/visualization/visualize_value_function.py
```

Configs under `upstream_scripts/visualization/configs/`.

## Per-env summary

Five benchmark envs (kwargs taken from each `upstream_scripts/data/collect_*.py`
at the pinned SHA):

| Tag         | `env_id`            | Obs        | Notable kwargs                                                              | Upstream collect script |
|-------------|---------------------|------------|------------------------------------------------------------------------------|--------------------------|
| pointmaze   | `swm/OGBMaze-v0`    | 224×224 RGB | `loco_env_type='point', maze_env_type='maze', maze_type='teleport'`         | `collect_pointmaze.py`   |
| pusht_fov   | `swm/PushT-v1`      | 224×224 RGB | sweeps every non-default variation                                          | `collect_pusht_fov.py`   |
| tworooms    | `swm/TwoRoom-v1`    | 224×224 RGB | `ExpertPolicy(action_noise=2.0, action_repeat_prob=0.05)`                   | `collect_tworooms.py`    |
| reacher     | (cfg-driven)        | DMC visual | `MUJOCO_GL=glfw`, `RandomPolicy`                                            | `collect_reacher.py`     |
| cube        | `swm/OGBCube-v0`    | 224×224 RGB | `env_type='single', multiview=True, mode='data_collection'`                 | `collect_cube.py`        |

The full `upstream_scripts/data/` tree also includes `collect_antmaze.py`,
`collect_dmc.py`, `collect_scene.py`, `collect_pusht_toy.py`,
`collect_tworooms_single_var.py`, `collect_weak_pusht.py`,
`collect_weak_discrete_pusht.py`. They're available but not part of the
default verify trip.

## Layout

```
example/wm_benchmark/
├── README.md                       # this file
├── ATTRIBUTION.md                  # upstream SHA + provenance details
├── requirements.txt                # thin extras on top of stable-worldmodel
├── _paths.py                       # sys.path bootstrap (mirrors example/cifar)
├── scripts/                        # our thin wrappers
│   ├── env.sh                      # source me to set STABLEWM_HOME
│   ├── bootstrap.py                # one-shot health check
│   ├── collect.py                  # dispatcher into upstream_scripts/data/collect_*.py
│   └── verify.py                   # smoke test: build 1 World per benchmark env
├── upstream_scripts/               # verbatim copy of upstream's scripts/ tree
│   ├── benchmark/                  # compare_h5_lance.py, convert.py, configs/
│   ├── data/                       # 13 collect_*.py + convert.py + config/
│   ├── examples/                   # breakout.py, dmc.py
│   ├── expert/                     # train_fetch_policy.py, train_policies.py
│   ├── plan/                       # eval_ff.py, eval_wm.py, config/
│   ├── train/                      # 7 baseline trainers + config/
│   ├── visualization/              # 4 visualize_*.py + utils.py + configs/
│   └── minerl.sh
└── data/                           # gitignored; $STABLEWM_HOME target
```

## Citations

If you use this scaffold, please cite the underlying papers and frameworks.

```bibtex
@misc{stableworldmodel2025,
  title        = {stable-worldmodel: A unified framework for offline world-model benchmarking},
  author       = {Galilai Group},
  year         = {2025},
  howpublished = {\url{https://github.com/galilai-group/stable-worldmodel}}
}

@misc{zhou2024dinowm,
  title         = {DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning},
  author        = {Gaoyue Zhou and Hengkai Pan and Yann LeCun and Lerrel Pinto},
  year          = {2024},
  eprint        = {2411.04983},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2411.04983}
}

@misc{terver2025jepawms,
  title         = {What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?},
  author        = {Basile Terver and Tsung-Yen Yang and Jean Ponce and Adrien Bardes and Yann LeCun},
  year          = {2025},
  eprint        = {2512.24497},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2512.24497}
}

@inproceedings{chi2023diffusionpolicy,
  title     = {Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author    = {Cheng Chi and Siyuan Feng and Yilun Du and Zhenjia Xu and Eric Cousineau and Benjamin Burchfiel and Shuran Song},
  booktitle = {Proceedings of Robotics: Science and Systems (RSS)},
  year      = {2023},
  url       = {https://arxiv.org/abs/2303.04137}
}

@inproceedings{florence2022implicitbc,
  title     = {Implicit Behavioral Cloning},
  author    = {Florence, Pete and Lynch, Corey and Zeng, Andy and Ramirez, Oscar A. and Wahid, Ayzaan and Downs, Laura and Wong, Adrian and Lee, Johnny and Mordatch, Igor and Tompson, Jonathan},
  booktitle = {Proceedings of the 5th Conference on Robot Learning (CoRL)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {164},
  pages     = {158--168},
  year      = {2022},
  publisher = {PMLR},
  url       = {https://arxiv.org/abs/2109.00137}
}

@misc{fu2020d4rl,
  title         = {D4RL: Datasets for Deep Data-Driven Reinforcement Learning},
  author        = {Justin Fu and Aviral Kumar and Ofir Nachum and George Tucker and Sergey Levine},
  year          = {2020},
  eprint        = {2004.07219},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2004.07219}
}
```
