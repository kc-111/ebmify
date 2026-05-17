"""Forked from ``upstream_scripts/data/convert.py`` — convert datasets
between registered formats (HDF5 ↔ Lance ↔ folder, plus HF repo ids and
``lerobot://`` URIs handled by ``stable_worldmodel.data.convert``).

Examples::

    python scripts/data/convert.py --source data.h5 --dest data.lance
    python scripts/data/convert.py --source quentinll/lewm-pusht \
        --dest /scratch/lewm_pusht.lance
    python scripts/data/convert.py --source data.lance --dest data/ \
        --dest-format folder
"""
from __future__ import annotations

import argparse

from stable_worldmodel.data import convert


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True,
                   help="Path, HF repo id, or scheme-prefixed identifier "
                        "(e.g. lerobot://lerobot/pusht).")
    p.add_argument("--dest", required=True,
                   help="Output path for the destination writer.")
    p.add_argument("--source-format", default=None,
                   help="Force the source format (skips autodetect).")
    p.add_argument("--dest-format", default="lance",
                   help="Registered writer name (default: lance).")
    p.add_argument("--cache-dir", default=None,
                   help="Override the dataset cache root.")
    p.add_argument("--mode", choices=("append", "overwrite", "error"),
                   default="append",
                   help="Destination writer mode (default: append).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    convert(
        args.source, args.dest,
        source_format=args.source_format,
        dest_format=args.dest_format,
        cache_dir=args.cache_dir,
        mode=args.mode,
    )
    print(f"[convert] {args.source} -> {args.dest} ({args.dest_format})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
