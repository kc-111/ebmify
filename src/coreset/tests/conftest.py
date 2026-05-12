"""Shared synthetic fixtures.

We plant a feature matrix Phi = U S V^T + noise with known spectrum so that
eig recovery, leverage profile, and bucket structure are predictable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def synth_phi() -> dict:
    """Phi (1000, 32) with planted singular values [10, 8, 6, ..., small]."""
    rng = np.random.default_rng(0)
    N, D = 1000, 32
    # Diagonal singular values: first 8 are large, rest small.
    s = np.concatenate([np.linspace(10.0, 4.0, 8), 0.1 * np.ones(D - 8)])
    U = rng.standard_normal((N, D)).astype(np.float32)
    U, _ = np.linalg.qr(U)
    Vt = rng.standard_normal((D, D)).astype(np.float32)
    Vt, _ = np.linalg.qr(Vt)
    Phi = (U * s[None, :]) @ Vt
    Phi += 0.01 * rng.standard_normal(Phi.shape).astype(np.float32)
    return {
        "Phi": torch.from_numpy(Phi.astype(np.float32)),
        "N": N,
        "D": D,
        "singular_values": s.astype(np.float32),
        "Vt_true": Vt.astype(np.float32),
    }
