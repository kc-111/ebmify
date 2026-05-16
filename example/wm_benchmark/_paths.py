"""Path bootstrap shared by all dino_wm_benchmark/ scripts.

Importing this module registers the benchmark root, ``env/`` and
``scripts/`` on sys.path, alongside the other example dirs and the repo
``src/``. This means a script anywhere inside ``example/dino_wm_benchmark``
can ``import env`` (so registered gym envs resolve their entry points) and
cross-import from sibling examples without ceremony.

Usage from a script:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _paths import REPO_ROOT, BENCHMARK_ROOT, DATA_DIR  # noqa: F401
"""
from __future__ import annotations

import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parent
EXAMPLE_ROOT = BENCHMARK_ROOT.parent
REPO_ROOT = EXAMPLE_ROOT.parent
DATA_DIR = BENCHMARK_ROOT / "data"


def _add(p: Path) -> None:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


_add(BENCHMARK_ROOT)
for _sub in ("env", "scripts"):
    _add(BENCHMARK_ROOT / _sub)
for _sibling in ("mnist", "cifar", "hetero"):
    _add(EXAMPLE_ROOT / _sibling)
_add(REPO_ROOT / "src")
