"""Dataclass-config + argparse loader, replacing upstream's Hydra surface.

Each ``scripts/train/configs/<method>.py`` defines a frozen-ish nested
dataclass with the same field names as upstream's `config/<method>.yaml`.
``from_argv`` parses:

    python scripts/train/lewm.py
    python scripts/train/lewm.py --config-file path/to/override.yaml
    python scripts/train/lewm.py override key.subkey=value other.flag=true

Trailing ``key=value`` pairs use a tiny dotted-path overlay so we keep the
Hydra-flavored UX without taking the Hydra dependency. Values are parsed
with `yaml.safe_load` so booleans, ints, floats, lists, and null work.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any, Type, TypeVar

import yaml

T = TypeVar("T")


# ---------------------------------------------------------------------------
# YAML round-trip


def to_dict(cfg: Any) -> dict[str, Any]:
    """Recursively turn a dataclass tree into plain dicts/lists/scalars."""
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return {f.name: to_dict(getattr(cfg, f.name)) for f in dataclasses.fields(cfg)}
    if isinstance(cfg, (list, tuple)):
        return [to_dict(v) for v in cfg]
    if isinstance(cfg, dict):
        return {k: to_dict(v) for k, v in cfg.items()}
    return cfg


def save_yaml(cfg: Any, path: Path) -> None:
    """Write a dataclass cfg to ``path`` as YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)


# ---------------------------------------------------------------------------
# Overlay


def _coerce_scalar(value: str) -> Any:
    """Use YAML to coerce ``"3"`` → int, ``"true"`` → bool, etc."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _apply_dotted(cfg: Any, dotted: str, value: Any) -> None:
    """Set ``cfg.<a>.<b>.<c>`` from a dotted path ``"a.b.c"``."""
    parts = dotted.split(".")
    obj = cfg
    for p in parts[:-1]:
        if dataclasses.is_dataclass(obj):
            obj = getattr(obj, p)
        elif isinstance(obj, dict):
            obj = obj[p]
        else:
            raise TypeError(f"cannot descend into {type(obj).__name__} at '{p}'")
    leaf = parts[-1]
    if dataclasses.is_dataclass(obj):
        if not any(f.name == leaf for f in dataclasses.fields(obj)):
            raise KeyError(f"unknown field '{dotted}' on {type(obj).__name__}")
        # nested-dataclass support: if leaf field is a dataclass and the
        # incoming value is a dict, merge field-by-field.
        current = getattr(obj, leaf)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            for k, v in value.items():
                _apply_dotted(current, k, v)
            return
        setattr(obj, leaf, value)
    elif isinstance(obj, dict):
        obj[leaf] = value
    else:
        raise TypeError(f"cannot set '{dotted}' on {type(obj).__name__}")


def _merge_dict(cfg: Any, data: dict[str, Any], prefix: str = "") -> None:
    """Apply a (possibly nested) dict of overrides onto ``cfg``."""
    for k, v in data.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            current = _get(cfg, path)
            if dataclasses.is_dataclass(current):
                _merge_dict(cfg, v, prefix=path)
                continue
        _apply_dotted(cfg, path, v)


def _get(cfg: Any, dotted: str) -> Any:
    obj = cfg
    for p in dotted.split("."):
        if dataclasses.is_dataclass(obj):
            obj = getattr(obj, p)
        elif isinstance(obj, dict):
            obj = obj[p]
        else:
            return None
    return obj


# ---------------------------------------------------------------------------
# Argparse entrypoint


def from_argv(cls: Type[T], argv: list[str] | None = None,
              *, description: str | None = None) -> T:
    """Build a config of type ``cls`` from CLI args.

    ``--config-file PATH`` loads a yaml file and overlays its contents on
    the default config; remaining positional args of the form ``a.b=value``
    overlay scalar (or YAML-parsed) values on top.
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-file", type=Path, default=None,
                        help="YAML file with overrides applied over the dataclass defaults")
    parser.add_argument("overrides", nargs="*",
                        help="dotted-path overrides, e.g. trainer.max_epochs=10 wm.embed_dim=384")
    ns = parser.parse_args(argv)

    cfg = cls()  # type: ignore[call-arg]  # defaults must be construct-without-args

    if ns.config_file is not None:
        with ns.config_file.open() as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"config file must be a mapping, got {type(data).__name__}")
        _merge_dict(cfg, data)

    for ov in ns.overrides:
        if "=" not in ov:
            raise ValueError(f"override missing '=': {ov!r}")
        key, _, raw = ov.partition("=")
        _apply_dotted(cfg, key.strip(), _coerce_scalar(raw.strip()))

    return cfg
