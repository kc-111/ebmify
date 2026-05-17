"""Thin wrappers around swm's Lance I/O for our local collectors.

The new minari antmaze collector needs to write a Lance file with the
same column schema swm's ``OGBMaze`` antmaze emits. swm has internal
writers used by ``World.collect``, but they're not exposed as public
helpers; this module wraps them with a single ``write_lance`` entrypoint
keyed to the (numpy-everywhere) shape our minari renderer produces.

Reads route through ``swm.data.load_dataset`` and stay verbatim.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

# swm is required for both read and write paths.
import stable_worldmodel as swm  # noqa: F401  (kept for the read side)


def datasets_dir() -> Path:
    """Resolve ``$STABLEWM_HOME/datasets`` via swm so envs stay aligned."""
    from stable_worldmodel.data import utils as swm_utils

    root = swm_utils.get_cache_dir(sub_folder="datasets")
    return Path(root)


def write_lance(path: Path, episodes: Iterable[Mapping[str, Any]], *,
                overwrite: bool = False) -> Path:
    """Persist a sequence of episode dicts to a Lance file at ``path``.

    Each ``episode`` is a mapping where every value is either a ndarray
    of shape ``(T, ...)`` or a scalar broadcast across the T timesteps.
    The function flattens episodes into one PyArrow table keyed by
    ``episode_idx`` + ``step_idx`` and writes via ``pyarrow.dataset``.
    """
    import pyarrow as pa
    import pyarrow.dataset as pads

    path = Path(path)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists (pass overwrite=True to clobber)")
        # Remove dir-form Lance datasets cleanly.
        if path.is_dir():
            import shutil
            shutil.rmtree(path)
        else:
            path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize each episode to a per-column list of arrays.
    rows: dict[str, list[Any]] = {}
    for ep_idx, ep in enumerate(episodes):
        # determine T from the first array-valued column
        T = None
        for v in ep.values():
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                T = v.shape[0]
                break
        if T is None:
            raise ValueError(f"episode {ep_idx}: no array-valued column to infer T from")
        for k, v in ep.items():
            arr = v if isinstance(v, np.ndarray) else np.asarray([v] * T)
            if arr.shape[0] != T:
                raise ValueError(f"episode {ep_idx}, column {k!r}: shape[0]={arr.shape[0]} != T={T}")
            rows.setdefault(k, []).append(arr)
        rows.setdefault("episode_idx", []).append(np.full((T,), ep_idx, dtype=np.int64))
        rows.setdefault("step_idx", []).append(np.arange(T, dtype=np.int64))

    columns: dict[str, pa.Array] = {}
    for k, parts in rows.items():
        flat = np.concatenate(parts, axis=0)
        if flat.ndim == 1:
            columns[k] = pa.array(flat)
        else:
            # store higher-rank arrays as a fixed-shape tensor list per row
            inner_shape = list(flat.shape[1:])
            n = flat.shape[0]
            ext = pa.FixedShapeTensorType(pa.from_numpy_dtype(flat.dtype), inner_shape)
            storage = pa.array(flat.reshape(n, -1).tolist(),
                                pa.list_(pa.from_numpy_dtype(flat.dtype),
                                          int(np.prod(inner_shape))))
            columns[k] = pa.ExtensionArray.from_storage(ext, storage)

    table = pa.Table.from_pydict(columns)
    pads.write_dataset(table, str(path), format="lance")
    return path


__all__ = ["datasets_dir", "write_lance"]
