"""Shared helpers used by every local ``collect_<env>.py``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
if str(_BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_ROOT))

import stable_worldmodel as swm  # noqa: E402


def datasets_root(cache_dir: str | None = None) -> Path:
    """Resolve the datasets/ root the same way upstream does."""
    base = Path(cache_dir) if cache_dir else Path(swm.data.utils.get_cache_dir())
    out = base / "datasets"
    out.mkdir(parents=True, exist_ok=True)
    return out


def base_argparser(description: str, *, default_num_traj: int = 100) -> argparse.ArgumentParser:
    """Build the argparse skeleton common to every collector.

    Subclasses (the per-env collectors) add their own env-specific flags
    after the call.
    """
    p = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-traj", type=int, default=default_num_traj,
                   help="number of episodes to collect")
    p.add_argument("--num-envs", type=int, default=4,
                   help="how many envs to run in parallel inside swm.World")
    p.add_argument("--max-episode-steps", type=int, default=100,
                   help="episode length budget passed to swm.World")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", type=str, default=None,
                   help="override $STABLEWM_HOME; defaults to whatever swm resolves")
    p.add_argument("--out-name", type=str, default=None,
                   help="lance output stem; defaults to the env's canonical name")
    return p
