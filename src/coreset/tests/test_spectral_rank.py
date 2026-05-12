"""Spectral-rank: Gonzalez selection produces uniform per-bucket marginals."""

from __future__ import annotations

import math

import torch

from coreset.preprocess import fit_view
from coreset.eig import compute_eig
from coreset.spectral_rank import spectral_rank_coverage


def test_spectral_rank_marginals_are_uniform(synth_phi, device):
    Phi = synth_phi["Phi"]
    view = fit_view(Phi, mode="center_l2", chunk_size=128, device=device, out_path=None)
    sigma2, V = compute_eig(view, lam=1e-3, low_rank=False)
    idx, ranks_at_sel, stats = spectral_rank_coverage(
        view, V, sigma2, lam=1e-3, k=160, n_buckets=8, seed=0
    )
    assert idx.numel() == 160
    assert idx.unique().numel() == 160  # no duplicates

    # Gonzalez farthest-first in rank space should yield marginals that
    # are approximately Uniform[0, 1] in every bucket: mean ~ 0.5,
    # std ~ 1/sqrt(12) ~= 0.2887.
    mean_rank = ranks_at_sel.mean(dim=0)
    std_rank = ranks_at_sel.std(dim=0)
    assert float((mean_rank - 0.5).abs().max()) < 0.05
    assert float((std_rank - 1.0 / math.sqrt(12.0)).abs().max()) < 0.05

    # Stats should expose the marginals so downstream diagnostics can verify.
    assert "mean_rank_per_bucket" in stats
    assert "std_rank_per_bucket" in stats
    assert len(stats["mean_rank_per_bucket"]) == 8
