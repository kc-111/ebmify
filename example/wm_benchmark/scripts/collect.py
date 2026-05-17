"""Thin dispatcher into local ``scripts/data/collect_<env>.py``.

Usage:

    python scripts/collect.py pointmaze
    python scripts/collect.py pusht_fov --num-traj 10
    python scripts/collect.py tworooms
    python scripts/collect.py reacher
    python scripts/collect.py cube
    python scripts/collect.py antmaze_minari --variant umaze

Trailing args are forwarded verbatim to the local collector's ``main(argv)``.
Output lance files land under ``$STABLEWM_HOME/datasets/`` — first run
``source scripts/env.sh`` to point STABLEWM_HOME at this benchmark's
``data/`` dir.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import BENCHMARK_ROOT  # noqa: E402

LOCAL = BENCHMARK_ROOT / "scripts" / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("env", help="env tag, e.g. pointmaze / pusht_fov / tworooms / reacher / cube / antmaze_minari")
    parser.add_argument("rest", nargs=argparse.REMAINDER,
                        help="forwarded verbatim to scripts/data/collect_<env>.py")
    ns = parser.parse_args()

    script = LOCAL / f"collect_{ns.env}.py"
    if not script.is_file():
        available = sorted(p.stem.removeprefix("collect_") for p in LOCAL.glob("collect_*.py"))
        print(f"unknown env: {ns.env!r}\navailable: {available}", file=sys.stderr)
        return 2

    sys.argv = [str(script)] + ns.rest
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
