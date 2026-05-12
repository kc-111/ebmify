"""Path bootstrap shared by all cifar/ scripts.

Importing this module registers cifar/ root, every cifar/ subdir, the
example/mnist and example/hetero dirs, and the repo src/ on sys.path,
so any cifar script can import from any other cifar script regardless
of which subdir it lives in. Also exposes ``REPO_ROOT`` for output
paths.

Usage from a script inside a subdir (e.g., ``ood/cifar_*.py``):

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _paths import REPO_ROOT  # noqa: F401  (registers subdir paths)
"""
from __future__ import annotations

import sys
from pathlib import Path

CIFAR_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CIFAR_ROOT.parent.parent
EXAMPLE_ROOT = CIFAR_ROOT.parent


def _add(p: Path) -> None:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


_add(CIFAR_ROOT)
for _sub in ("train", "ood", "ebm", "diagnostics", "probes"):
    _add(CIFAR_ROOT / _sub)
_add(EXAMPLE_ROOT / "mnist")
_add(EXAMPLE_ROOT / "hetero")
_add(REPO_ROOT / "src")
