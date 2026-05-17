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
  [`ATTRIBUTION.md`](ATTRIBUTION.md). Kept around as the reference snapshot
  to diff against — not the primary surface.
- **Forked-and-localized scripts** under `scripts/{data,train,eval,common}/`:
  Hydra/WandB replaced with dataclass-config + CSVLogger (WandB opt-in),
  verbose progress callback (grad norms, throughput, ETA), uniform argparse
  collectors, and a `RolloutRecorder` that dumps frames/actions/timings to
  `data/eval_runs/`. Upstream model code (`stable_worldmodel.wm.*`) is
  imported as-is, so trained checkpoints stay binary-compatible with
  upstream eval.
- A gitignored `data/` dir that becomes `$STABLEWM_HOME` (datasets +
  checkpoints + run dirs land here, not in `~/.stable_worldmodel/`).

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
# Dispatcher into the local collectors under scripts/data/. Trailing args
# pass through to the collector's argparse:
python example/wm_benchmark/scripts/collect.py pointmaze
python example/wm_benchmark/scripts/collect.py pusht_fov --num-traj 10
python example/wm_benchmark/scripts/collect.py tworooms
python example/wm_benchmark/scripts/collect.py reacher
python example/wm_benchmark/scripts/collect.py cube
python example/wm_benchmark/scripts/collect.py antmaze_minari --variant umaze --max-episodes 100

# Or invoke a collector directly:
python example/wm_benchmark/scripts/data/collect_pointmaze.py --num-traj 10
```

Lance files land under `$STABLEWM_HOME/datasets/` — i.e.
`example/wm_benchmark/data/datasets/`.

### AntMaze via minari

`scripts/data/collect_antmaze_minari.py` pulls D4RL antmaze trajectories
through [minari](https://github.com/Farama-Foundation/Minari) and replays
each episode through `gymnasium-robotics`' `AntMaze_*-v5` in
`render_mode='rgb_array'` to recover pixels. The output schema mirrors
the columns produced by `swm/OGBMaze-v0` (ant) — `pixels`,
`goal_pixels`, `action`, `proprio`, `reward`, `terminated`, `truncated`
— so the dataset drops straight into the same training pipeline.

Replay rendering is slow (~5–20 ms/frame); the default `--max-episodes
100` keeps a single run minute-scale. Pass `--prefer state` (default) to
re-set qpos/qvel per step for an exact pixel match, or `--prefer action`
to step the recorded actions through the env (cheaper but drifts).

## Train baselines

Local trainers live under `scripts/train/`. Each is a fork of the matching
`upstream_scripts/train/<method>.py`: the model wiring is imported
verbatim from upstream's module, and the per-method `scripts/train/configs/<method>.py`
mirrors upstream's `config/<method>.yaml` as a frozen `@dataclass`. Hydra
is replaced by `--config-file foo.yaml --override key=value`; WandB is
opt-in via `--wandb`. A `VerboseProgressCallback` streams per-step
grad-norms / throughput / ETA to stderr and `data/runs/<method>/<run_id>/progress.log`.

```bash
python example/wm_benchmark/scripts/train/lewm.py    --dataset tworoom_expert --max-epochs 1
python example/wm_benchmark/scripts/train/pldm.py    --dataset tworoom_expert
python example/wm_benchmark/scripts/train/prejepa.py --dataset tworoom_expert
python example/wm_benchmark/scripts/train/gcbc.py    --dataset tworoom_expert
python example/wm_benchmark/scripts/train/gciql.py   --dataset tworoom_expert   # two-phase
python example/wm_benchmark/scripts/train/gcivl.py   --dataset tworoom_expert   # two-phase
python example/wm_benchmark/scripts/train/hilp.py    --dataset tworoom_expert   # two-phase
```

Outputs: `data/runs/<method>/<run_id>/{config.yaml, metrics.csv,
progress.log, weights_epoch_*.pt, last.pt}`. Two-phase trainers
(gciql/gcivl/hilp) split into `value/` and `actor/` subdirs.

## Evaluate

```bash
# WM + planner: optimizes actions to reach a goal under the learned dynamics.
python example/wm_benchmark/scripts/eval/eval_wm.py \
    --checkpoint data/runs/lewm/<run_id>/weights_epoch_1.pt \
    --env tworooms --planner cem --episodes 8 --eval-budget 50

# Direct GC-policy rollout (no action optimization).
python example/wm_benchmark/scripts/eval/eval_policy.py \
    --checkpoint data/runs/gcbc/<run_id>/weights_epoch_1.pt \
    --env tworooms --episodes 8
```

`--planner` accepts `cem | icem | mppi | ebmify`; the last is a stub
that raises `NotImplementedError` and is the future hook for a Langevin /
SAM-Adams planner over actions using `src/ebmify/sampler/samadams.py`.

Outputs land under `data/eval_runs/<run_id>/`:
`frames_ep<NN>.mp4`, `actions_ep<NN>.npy`, `rollouts.csv`, `timing.csv`,
`summary.json` (success rate, mean reward, mean steps, per-step wallclock).

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

The local `scripts/data/` tree additionally covers `collect_antmaze.py`,
`collect_dmc.py`, `collect_scene.py`, `collect_pusht_toy.py`,
`collect_tworooms_single_var.py`, `collect_weak_pusht.py`,
`collect_weak_discrete_pusht.py`, plus the new
`collect_antmaze_minari.py`. They're available but not part of the
default verify trip.

## Layout

```
example/wm_benchmark/
├── README.md                       # this file
├── ATTRIBUTION.md                  # upstream SHA + provenance details
├── requirements.txt                # thin extras on top of stable-worldmodel
├── _paths.py                       # sys.path bootstrap (mirrors example/cifar)
├── scripts/                        # primary surface
│   ├── env.sh                      # source me to set STABLEWM_HOME
│   ├── bootstrap.py                # one-shot health check
│   ├── verify.py                   # smoke test: build 1 World per benchmark env
│   ├── collect.py                  # dispatcher into scripts/data/collect_*.py
│   ├── common/                     # shared: config, logging, seeding, checkpoint, lance_io
│   ├── data/                       # forked collectors (13 collect_*.py + convert.py)
│   │                               # incl. collect_antmaze_minari.py (D4RL via minari)
│   ├── train/                      # forked trainers (7 methods) + configs/
│   └── eval/                       # planner adapters + RolloutRecorder + eval_wm / eval_policy
├── upstream_scripts/               # verbatim reference snapshot (diff target)
│   ├── benchmark/                  # compare_h5_lance.py, convert.py, configs/
│   ├── data/                       # 13 collect_*.py + convert.py + config/
│   ├── examples/                   # breakout.py, dmc.py
│   ├── expert/                     # train_fetch_policy.py, train_policies.py
│   ├── plan/                       # eval_ff.py, eval_wm.py, config/
│   ├── train/                      # 7 baseline trainers + config/
│   ├── visualization/              # 4 visualize_*.py + utils.py + configs/
│   └── minerl.sh
└── data/                           # gitignored; $STABLEWM_HOME target
    ├── datasets/                   # .lance files
    ├── runs/<method>/<run_id>/     # training: weights_epoch_*.pt, config.yaml, metrics.csv, progress.log
    └── eval_runs/<run_id>/         # eval: frames_ep<NN>.mp4, actions_ep<NN>.npy, rollouts.csv, summary.json
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
