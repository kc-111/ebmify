"""Local fork-and-localized trainers for the wm_benchmark.

Each ``<method>.py`` in this folder owns scaffolding (argparse/dataclass
config, local logging, checkpointing into ``data/runs/...``) and reuses
the model-builder + forward functions from the SHA-pinned
``upstream_scripts/train/<method>.py`` snapshot via
``scripts.train._upstream``.
"""
