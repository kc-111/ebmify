"""Thin dispatcher into upstream ``upstream_scripts/data/collect_<env>.py``.

Usage:

    python scripts/collect.py pointmaze
    python scripts/collect.py pusht_fov  num_traj=10
    python scripts/collect.py tworooms
    python scripts/collect.py reacher
    python scripts/collect.py cube

Trailing args are forwarded verbatim to the upstream script's argv, which is
the right shape for the hydra-based collect scripts (e.g.
``num_traj=10 world.image_shape=[112,112]``). Output lance files land under
``$STABLEWM_HOME/datasets/`` — `source scripts/env.sh` first.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import BENCHMARK_ROOT  # noqa: E402

UPSTREAM = BENCHMARK_ROOT / "upstream_scripts" / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("env", help="env tag, e.g. pointmaze / pusht_fov / tworooms / reacher / cube")
    parser.add_argument("rest", nargs=argparse.REMAINDER,
                        help="forwarded verbatim to upstream collect_<env>.py")
    ns = parser.parse_args()

    script = UPSTREAM / f"collect_{ns.env}.py"
    if not script.is_file():
        available = sorted(p.stem.removeprefix("collect_") for p in UPSTREAM.glob("collect_*.py"))
        print(f"unknown env: {ns.env!r}\navailable: {available}", file=sys.stderr)
        return 2

    # Hydra reads sys.argv; rewrite it so the upstream script sees its own name
    # at argv[0] and only the user-supplied overrides afterwards.
    sys.argv = [str(script)] + ns.rest
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
