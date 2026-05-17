"""Local fork of upstream's data-collection scripts.

Each ``collect_<env>.py`` is a self-contained argparse-driven script
that drops the Hydra dependency. Output Lance files all land under
``$STABLEWM_HOME/datasets/`` (or ``data/datasets/`` once
``env.sh`` is sourced).
"""
