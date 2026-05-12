"""Preprocess: mean, L2 row norms, streaming idempotent."""

from __future__ import annotations

import torch

from coreset.preprocess import fit_view


def test_center_l2_row_norms_unit(synth_phi, device, tmp_path):
    Phi = synth_phi["Phi"]
    view = fit_view(Phi, mode="center_l2", chunk_size=128, device=device,
                    out_path=tmp_path / "preprocessing.pt")
    for _, chunk in view.stream():
        norms = chunk.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_center_zero_column_mean(synth_phi, device, tmp_path):
    Phi = synth_phi["Phi"]
    view = fit_view(Phi, mode="center", chunk_size=128, device=device,
                    out_path=tmp_path / "preprocessing.pt")
    acc = torch.zeros(synth_phi["D"], device=device, dtype=torch.float64)
    n = 0
    for _, chunk in view.stream():
        acc += chunk.to(torch.float64).sum(dim=0)
        n += chunk.shape[0]
    mean = (acc / n).abs().max().item()
    assert mean < 1e-4


def test_none_mode_passthrough(synth_phi, device):
    Phi = synth_phi["Phi"]
    view = fit_view(Phi, mode="none", chunk_size=128, device=device, out_path=None)
    idx = torch.tensor([0, 1, 2])
    out = view(idx).cpu()
    assert torch.allclose(out, Phi[idx], atol=0.0)
