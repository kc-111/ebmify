# Attribution

This example wraps **stable-worldmodel** (galilai-group, MIT).

- Upstream: https://github.com/galilai-group/stable-worldmodel
- Pinned commit: `463ab63517b043ca6c3753b01e34ea6f145497c6`
- License: MIT (copyright 2025 galilai-group; see upstream `LICENSE`)

## Scripts copied verbatim

Every file under `upstream_scripts/` is a verbatim copy of the corresponding
file under `scripts/` in the upstream repo at the pinned commit above. Each
carries a 3-line header documenting the exact source path:

```python
# Copied verbatim from stable-worldmodel @ <SHA>
# Source: https://github.com/galilai-group/stable-worldmodel/blob/<SHA>/scripts/<path>
# License: MIT (see example/wm_benchmark/ATTRIBUTION.md)
```

If a file is modified locally, its header gets an additional `# MODIFIED: ...`
line documenting what changed and why. (None so far.)

## Resyncing from upstream

```bash
cd ~/Desktop/stable-worldmodel && git pull
rm -rf example/wm_benchmark/upstream_scripts
cp -r ~/Desktop/stable-worldmodel/scripts example/wm_benchmark/upstream_scripts
# Then re-run the header-stamping snippet (see git log of this dir for the
# Python one-liner) with the new SHA, and update the SHA above.
```

## Why a sibling clone, not vendored as a package

`stable_worldmodel` is a Python package, not a thin script bundle. Installing
it via `uv pip install -e ~/Desktop/stable-worldmodel` lets imports resolve
naturally and lets you `git pull` upstream updates without touching this
repo. This mirrors how `stable_pretraining` is consumed across
`example/cifar/` (see `example/cifar/train/ssl_pretrain.py`).
