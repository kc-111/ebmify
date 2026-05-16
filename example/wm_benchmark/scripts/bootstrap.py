"""One-shot health check for the wm_benchmark scaffold.

Verifies that:
  1. ``stable_worldmodel`` imports (i.e. you've cloned and pip-installed it).
  2. ``STABLEWM_HOME`` points at this example's ``data/`` dir.
  3. The cache dir is writable.

Run after ``source scripts/env.sh``:

    python scripts/bootstrap.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import DATA_DIR  # noqa: E402


CLONE_HINT = """\
stable_worldmodel is not installed. Clone it as a sibling of ebmify and
install editable into the same venv:

    cd ~/Desktop
    git clone https://github.com/galilai-group/stable-worldmodel.git
    cd stable-worldmodel
    uv pip install -e .          # or: <venv>/bin/pip install -e .
"""

ENV_HINT = f"""\
STABLEWM_HOME is not set (or set to the wrong path).
Expected: {DATA_DIR}
Got: {os.environ.get('STABLEWM_HOME', '<unset>')}

Source the env script once per shell:

    source example/wm_benchmark/scripts/env.sh
"""


def main() -> int:
    try:
        import stable_worldmodel as swm
    except ImportError:
        print(CLONE_HINT, file=sys.stderr)
        return 1

    expected = str(DATA_DIR.resolve())
    got = os.environ.get("STABLEWM_HOME", "")
    if Path(got).resolve() != Path(expected):
        print(ENV_HINT, file=sys.stderr)
        return 1

    cache_dir = Path(swm.data.utils.get_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    probe = cache_dir / ".bootstrap_probe"
    probe.write_text("ok")
    probe.unlink()

    version = getattr(swm, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as pkg_version
            version = pkg_version("stable_worldmodel")
        except Exception:
            version = "unknown"

    print("stable_worldmodel:", version)
    print("STABLEWM_HOME:    ", expected)
    print("cache_dir:        ", cache_dir)
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
