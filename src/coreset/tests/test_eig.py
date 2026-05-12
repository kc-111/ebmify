"""Eig recovers the planted spectrum within tolerance, low-rank path matches top-r."""

from __future__ import annotations

import torch

from coreset.preprocess import fit_view
from coreset.eig import compute_eig


def test_full_eig_recovers_planted_spectrum(synth_phi, device):
    Phi = synth_phi["Phi"]
    view = fit_view(Phi, mode="none", chunk_size=128, device=device, out_path=None)
    sigma2, V = compute_eig(view, lam=1e-3, low_rank=False)
    # singular values planted -> sigma2 = s^2 sorted desc
    true = torch.from_numpy(synth_phi["singular_values"]).pow(2).sort(descending=True).values
    # Up to noise floor and ridge-shift back.
    rel = (sigma2.cpu() - true).abs() / (true + 1e-6)
    assert float(rel[:8].max()) < 0.05


def test_low_rank_top_r_close_to_full(synth_phi, device):
    Phi = synth_phi["Phi"]
    view = fit_view(Phi, mode="none", chunk_size=128, device=device, out_path=None)
    s_full, V_full = compute_eig(view, lam=1e-3, low_rank=False)
    s_lr, V_lr = compute_eig(view, lam=1e-3, low_rank=True, r=8, n_iter=6, seed=0)
    # top 8 eigenvalues should match the full-eig top 8 closely
    rel = (s_lr.cpu() - s_full[:8].cpu()).abs() / (s_full[:8].cpu() + 1e-6)
    assert float(rel.max()) < 0.10
