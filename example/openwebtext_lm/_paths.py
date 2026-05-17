"""Path bootstrap shared by all openwebtext_lm/ scripts.

Importing this module registers the openwebtext_lm/ root, its subdirs
(train, eval), and the repo src/ on sys.path so any script in any
subdir can import the local modules (model, owt_data) by name.

Usage from a script inside a subdir (e.g., ``train/owt_lm_train.py``):

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _paths import REPO_ROOT  # noqa: F401  (registers subdir paths)
"""
from __future__ import annotations

import sys
from pathlib import Path

OWT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = OWT_ROOT.parent.parent
EXAMPLE_ROOT = OWT_ROOT.parent


def _add(p: Path) -> None:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


_add(OWT_ROOT)
for _sub in ("train", "eval"):
    _add(OWT_ROOT / _sub)
_add(REPO_ROOT / "src")
