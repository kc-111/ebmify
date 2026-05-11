"""CIFAR-10 / CIFAR-100 loaders that read the official pickle archives.

Looks for ``cifar-10-batches-py/`` and ``cifar-100-python/`` at the repo
root (placed there by ``download_cifar.py``). Returns float32 images in
[0, 1] with shape ``(N, 3, 32, 32)`` and int64 labels.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def _flat_to_chw(flat: np.ndarray) -> np.ndarray:
    return flat.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0


def load_cifar10_train():
    root = _repo_root() / "cifar-10-batches-py"
    Xs, ys = [], []
    for i in range(1, 6):
        d = _load_pickle(root / f"data_batch_{i}")
        Xs.append(_flat_to_chw(d[b"data"]))
        ys.append(np.asarray(d[b"labels"], dtype=np.int64))
    return np.concatenate(Xs), np.concatenate(ys)


def load_cifar10_test():
    root = _repo_root() / "cifar-10-batches-py"
    d = _load_pickle(root / "test_batch")
    return _flat_to_chw(d[b"data"]), np.asarray(d[b"labels"], dtype=np.int64)


def load_cifar100_train():
    root = _repo_root() / "cifar-100-python"
    d = _load_pickle(root / "train")
    return _flat_to_chw(d[b"data"]), np.asarray(d[b"fine_labels"], dtype=np.int64)


def load_cifar100_test():
    root = _repo_root() / "cifar-100-python"
    d = _load_pickle(root / "test")
    return _flat_to_chw(d[b"data"]), np.asarray(d[b"fine_labels"], dtype=np.int64)


def load_cifar_train(dataset: str):
    if dataset == "cifar10":
        return load_cifar10_train()
    if dataset == "cifar100":
        return load_cifar100_train()
    raise ValueError(f"unknown dataset {dataset!r}; expected cifar10 or cifar100")


def load_cifar_test(dataset: str):
    if dataset == "cifar10":
        return load_cifar10_test()
    if dataset == "cifar100":
        return load_cifar100_test()
    raise ValueError(f"unknown dataset {dataset!r}; expected cifar10 or cifar100")


def cifar_ckpt_path(dataset: str, z_dim: int, beta: float) -> Path:
    cache_dir = Path(__file__).resolve().parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{dataset}_vae_z{z_dim}_beta{beta}.pt"
