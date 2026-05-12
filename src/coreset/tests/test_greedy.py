"""Greedy: |S| == k, no duplicates, max_h trajectory non-increasing on average."""

from __future__ import annotations

import torch

from coreset.preprocess import fit_view
from coreset.greedy import greedy_max_variance


def test_greedy_basic(synth_phi, device):
    Phi = synth_phi["Phi"]
    view = fit_view(Phi, mode="center_l2", chunk_size=128, device=device, out_path=None)
    idx, stats = greedy_max_variance(view, lam=1e-2, k=50, seed_size=8, seed=0)
    assert idx.numel() == 50
    assert idx.unique().numel() == 50
    traj = stats["max_h_trajectory"]
    # Long-run trajectory should be downward; allow local bumps from refactors.
    head = sum(traj[:5]) / 5.0
    tail = sum(traj[-5:]) / 5.0
    assert tail <= head + 1e-3
