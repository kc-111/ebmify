"""Aux targets: shape and dtype, spectral_coords reconstruction matches truncation."""

from __future__ import annotations

import torch

from coreset.preprocess import fit_view
from coreset.eig import compute_eig
from coreset.aux_targets import compute_aux_targets


def test_aux_shapes_and_dtypes(tmp_path, synth_phi, device):
    Phi = synth_phi["Phi"]
    view = fit_view(Phi, mode="center_l2", chunk_size=128, device=device, out_path=None)
    sigma2, V = compute_eig(view, lam=1e-3, low_rank=False)
    sel = torch.arange(0, 100, dtype=torch.long)
    saved = compute_aux_targets(
        view, sigma2, V, lam=1e-3,
        selected_idx=sel,
        targets=["spectral_coords", "bucket_ranks", "leverage_score", "home_bucket"],
        n_buckets=8, n_top_eigvecs=8,
        out_dir=tmp_path,
    )
    coords = torch.load(saved["spectral_coords"])
    wts = torch.load(saved["spectral_weights"])
    ranks = torch.load(saved["bucket_ranks"])
    lev = torch.load(saved["leverage_score"])
    hb = torch.load(saved["home_bucket"])
    assert coords.shape == (100, 8) and coords.dtype == torch.float32
    assert wts.shape == (8,) and wts.dtype == torch.float32
    assert ranks.shape == (100, 8) and ranks.dtype == torch.float32
    assert lev.shape == (100,) and lev.dtype == torch.float32
    assert hb.shape == (100,) and hb.dtype == torch.int64
