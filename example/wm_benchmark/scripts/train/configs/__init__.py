"""Dataclass-config modules — one per training method.

Each module exports a top-level ``<Method>Config`` dataclass whose
field tree mirrors upstream's ``config/<method>.yaml``. The shared
``DataCfg``, ``TrainerCfg``, etc. dataclasses live in ``_common``.
"""
