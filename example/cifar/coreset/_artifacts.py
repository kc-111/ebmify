"""Shared argparse helper: resolve --artifacts as either a tag or a full path.

The build scripts write artifacts to::

    example/cifar/cache/coreset/<tag>/<algo>/indices.pt

so in practice there's a small, on-disk-enumerable set of valid <tag>
values per machine. ``--artifacts`` therefore accepts:

- a bare tag name (e.g. ``supervised_resnet18``), resolved to
  ``example/cifar/cache/coreset/<tag>/``, or
- an arbitrary directory path (override).

The list of valid tags is computed at argparse-construction time by
scanning the cache; the discovered names are surfaced in the
``--help`` text so the user can see what's actually selectable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import REPO_ROOT  # noqa: E402

CORESET_CACHE = REPO_ROOT / "example" / "cifar" / "cache" / "coreset"


def discover_tags() -> list[str]:
    """Sorted list of tag names that contain at least one ``<algo>/indices.pt``."""
    if not CORESET_CACHE.is_dir():
        return []
    out: list[str] = []
    for d in sorted(CORESET_CACHE.iterdir()):
        if not d.is_dir():
            continue
        if any((sub / "indices.pt").exists()
               for sub in d.iterdir() if sub.is_dir()):
            out.append(d.name)
    return out


def resolve_artifacts(s: str) -> Path:
    """argparse ``type=`` for ``--artifacts``: tag-name OR directory path."""
    p = Path(s)
    if p.is_dir():
        return p.resolve()
    cand = CORESET_CACHE / s
    if cand.is_dir():
        return cand.resolve()
    tags = discover_tags()
    msg = (f"no coreset dir at '{p}' or '{cand}'. "
           f"available tags: {tags or '(none -- run a build script first)'}")
    raise argparse.ArgumentTypeError(msg)


def artifacts_help(preferred: str | None = None) -> str:
    """One-line help string listing the currently-discovered tags.

    If ``preferred`` is given, surface it as the recommended default so
    the user sees it in ``--help``.
    """
    tags = discover_tags()
    head = "tag name or path to coreset/<tag>/."
    if preferred:
        head = f"{head} canonical default for this script: {preferred}."
    if tags:
        return f"{head} available tags: {', '.join(tags)}"
    return f"{head} no tags discovered yet -- run a build script first"


def default_artifacts(preferred: str) -> tuple[str | None, bool]:
    """Resolve a script's default-tag against the on-disk cache.

    Returns ``(default, required)`` ready to pass to ``add_argument``:

    - if ``preferred`` exists in ``CORESET_CACHE``: ``(preferred, False)``
      -- argparse will route the string through ``type=resolve_artifacts``
      and resolve it to an absolute Path.
    - else: ``(None, True)`` -- script becomes required, with a helpful
      error pointing to available (or absent) tags.
    """
    if preferred in discover_tags():
        return preferred, False
    return None, True
