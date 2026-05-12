"""Leverage: replacement path returns k, no-replacement returns k, weights finite."""

from __future__ import annotations

import torch

from coreset.preprocess import fit_view
from coreset.eig import compute_eig
from coreset.leverage import ridge_leverage_sample, compute_leverage


def test_leverage_with_and_without_replacement_return_k(synth_phi, device):
    Phi = synth_phi["Phi"]
    view = fit_view(Phi, mode="center_l2", chunk_size=128, device=device, out_path=None)
    sigma2, V = compute_eig(view, lam=1e-3, low_rank=False)
    for replace in (True, False):
        idx, w, stats = ridge_leverage_sample(
            view, sigma2, V, lam=1e-3, k=100, alpha=0.2, replace=replace, seed=0
        )
        assert idx.numel() == 100
        assert torch.isfinite(w).all()
        assert stats["effective_dim"] > 0


def test_empirical_p_matches_target(synth_phi, device):
    Phi = synth_phi["Phi"]
    view = fit_view(Phi, mode="center_l2", chunk_size=128, device=device, out_path=None)
    sigma2, V = compute_eig(view, lam=1e-3, low_rank=False)
    h = compute_leverage(view, sigma2, V, lam=1e-3)
    p = (1 - 0.2) * (h / h.sum()) + 0.2 / view.N
    counts = torch.zeros(view.N, device=device)
    n_trials = 200
    k = 200
    g = torch.Generator(device=device).manual_seed(0)
    for _ in range(n_trials):
        idx = torch.multinomial(p, k, replacement=True, generator=g)
        counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
    emp = counts / (n_trials * k)
    rel_err = (emp - p).abs() / p.clamp(min=1e-6)
    # large-h indices should converge; allow up to 1.5 rel error for tails.
    top = torch.topk(p, 20).indices
    assert float(rel_err[top].mean()) < 0.5
