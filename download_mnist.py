"""Fetch the 4 MNIST IDX files into ./MNIST/raw/.

Skips files that already exist. Uses urllib + gzip from the standard library.

Source mirror: Google Cloud Vision dataset mirror.
"""

from __future__ import annotations

import gzip
import shutil
import sys
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "MNIST" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in FILES:
        idx_name = fname[:-3]  # strip .gz
        idx_path = out_dir / idx_name
        if idx_path.exists():
            print(f"  exists: {idx_path}")
            continue
        url = BASE + fname
        gz_path = out_dir / fname
        print(f"  downloading {url}")
        with urllib.request.urlopen(url) as resp, gz_path.open("wb") as out:
            shutil.copyfileobj(resp, out)
        print(f"  decompressing {gz_path}")
        with gzip.open(gz_path, "rb") as gz, idx_path.open("wb") as out:
            shutil.copyfileobj(gz, out)
        gz_path.unlink()
    print("MNIST ready at", out_dir)


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        print(f"network error: {e}", file=sys.stderr)
        sys.exit(1)
