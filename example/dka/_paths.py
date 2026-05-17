"""Path bootstrap shared by all example/dka/ scripts.

Importing this module registers dka/ root, its subdirs, example/cifar,
example/openwebtext_lm, and the repo src/ on sys.path so any dka script
can import from any sibling. Mirrors example/cifar/_paths.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

DKA_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DKA_ROOT.parent.parent
EXAMPLE_ROOT = DKA_ROOT.parent


def _add(p: Path) -> None:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


_add(DKA_ROOT)
for _sub in ("cifar", "seq", "ablations", "benchmarks"):
    _add(DKA_ROOT / _sub)
_add(EXAMPLE_ROOT / "cifar")
_add(EXAMPLE_ROOT / "openwebtext_lm")
_add(REPO_ROOT / "src")
