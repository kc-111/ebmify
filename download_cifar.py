"""Fetch CIFAR-10 and CIFAR-100 python pickles into the repo root.

Creates ./cifar-10-batches-py/ and ./cifar-100-python/ at the same level
as this script. Skips fetching when the unpacked directory already exists.
"""

from __future__ import annotations

import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

ARCHIVES = [
    ("cifar-10-python.tar.gz",  "cifar-10-batches-py", "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"),
    ("cifar-100-python.tar.gz", "cifar-100-python",    "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"),
]


def main() -> None:
    root = Path(__file__).resolve().parent
    for tar_name, dir_name, url in ARCHIVES:
        out_dir = root / dir_name
        if out_dir.exists():
            print(f"  exists: {out_dir}")
            continue
        tar_path = root / tar_name
        print(f"  downloading {url}")
        with urllib.request.urlopen(url) as resp, tar_path.open("wb") as out:
            shutil.copyfileobj(resp, out)
        print(f"  extracting {tar_path}")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(root)
        tar_path.unlink()
    print("CIFAR ready under", root)


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        print(f"network error: {e}", file=sys.stderr)
        sys.exit(1)
