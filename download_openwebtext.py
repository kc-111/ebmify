"""Fetch Skylion007/openwebtext into the repo root for language modelling.

Creates ./openwebtext/ at the same level as this script. Skips fetching
when the directory already exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

REPO_ID = "Skylion007/openwebtext"
DIR_NAME = "openwebtext"


def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / DIR_NAME
    if out_dir.exists():
        print(f"  exists: {out_dir}")
        return
    print(f"  downloading {REPO_ID} -> {out_dir}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(out_dir),
    )
    print("OpenWebText ready under", out_dir)


if __name__ == "__main__":
    try:
        main()
    except HfHubHTTPError as e:
        print(f"hub error: {e}", file=sys.stderr)
        sys.exit(1)
