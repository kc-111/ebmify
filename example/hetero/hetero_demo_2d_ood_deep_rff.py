"""Deep RFF version of ``hetero_demo_2d_ood``: an FCNet with an RFF input
lift, a residual MLP trunk, and an RFF pre-readout layer applied to the
moons + ring + spiral topology, with last-layer leverage as the
uncertainty signal.

Why the deep RFF stack on this stress-test:
  * The training set has *internal* holes (the empty disc inside the
    ring, the slot between the moons). The output-RFF features are
    bounded in ``[-sqrt(2/M), sqrt(2/M)]``, so the leverage signal
    that flags those holes can't be drowned out by unbounded blow-ups
    far OOD — a known failure mode of leverage on raw FCNet
    penultimate activations.
  * The MLP trunk in the middle gives the model capacity to resolve
    the curved manifolds (moons, spiral) that a single-layer RBF
    kernel struggles with. By the "MLP-features-as-data" reframe, the
    output-RFF layer is just RFF on whatever space the trunk has
    carved out, and Bochner gives a bona-fide kernel approximation in
    that space.
  * The input-RFF lift fixes the spectral bias of the bare MLP on the
    raw input plane, so the trunk can fit the high-frequency moon /
    spiral geometry from the start.

The leverage head is identical to the FCNet/GAM/RFF versions:

    h(x*)        = phi_out(x*)^T (Phi^T Phi + ridge*I)^-1 phi_out(x*)
    sigma_epi    = lam * sqrt(h(x*)),  lam = sigma_noise / sqrt(h_p95)
    sigma_epi    = sigma_max * tanh(sigma_epi_raw / sigma_max)

with ``phi_out = net.features(x)`` returning the bounded output-RFF
activations (or trunk activations if ``output_rff=None``).

Bandwidths are calibrated by the ``"median"`` heuristic at fit time:
``Omega_in`` from preprocessed inputs, ``Omega_out`` from trunk
activations after the residual blocks. The ``sqrt(d)`` dimensionality
factor in :class:`RFFLayer` makes those bandwidths dimension-normalized.

Run from the repo root:

    python example/hetero/hetero_demo_2d_ood_deep_rff.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ebmify.models import (
    FCNet,
    FitConfig,
    NoiseConfig,
    PreprocessConfig,
    RegConfig,
    feature_leverage,
)

from hetero_demo_2d_ood import (
    make_2d_complex_data, gauss_mix_landscape,
    classify_regions, _draw_shapes, _heat,
)


def _fmt_ell(t: torch.Tensor) -> str:
    """Render an RFFLayer ``length_scale`` buffer (shape ``[K]``)."""
    vals = t.detach().cpu().tolist()
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return "[" + ", ".join(f"{v:.2f}" for v in vals) + "]"


def main() -> None:
    X_tr, y_tr, geom = make_2d_complex_data(n=1000, seed=1)

    pc = PreprocessConfig(
        input_transforms=["standard"],
        # input_transforms=["kde_quantile"],
        # kde_bandwidth_factor=10.0,
        output_transforms=["standard"],
        # output_transforms=["kde_quantile"],
    )
    fc = FitConfig(epochs=1000, lr=5e-4, batch_size=128, seed=0, verbose=True)
    nc = NoiseConfig(
        input_additive_std=0.0, input_multiplicative_std=0.0,
        output_additive_std=0.0,
    )

    M_in = None
    M_out = None
    print(
        "Fitting deep RFF FCNet "
        f"(input_rff M={M_in} median, residual MLP, output_rff M={M_out} median) "
        "on complex (moons + ring + spiral) topology ..."
    )
    net = FCNet(
        n_inputs=2, n_outputs=1,
        hidden_dims=(32, 32, 32, 32),
        activation="odd_piecewise",
        fit_config=fc, reg_config=RegConfig(l2=1e-5),
        noise_config=nc, preprocess=pc,
        input_rff=M_in, input_rff_length_scale=[0.25, 0.5, 1.0],
        output_rff=M_out, output_rff_length_scale=[0.1],
        # block_type="rff",
        # block_rff_features=64,
        # block_rff_length_scale="median",
        rff_seed=0,
    )
    net.fit(X_tr, y_tr)
    # ell_in = _fmt_ell(net.net.input_rff.length_scale)
    # ell_out = _fmt_ell(net.net.output_rff.length_scale)
    # print(f"  resolved input_rff length_scale = {ell_in}")
    # print(f"  resolved output_rff length_scale = {ell_out}")

    G = 140
    X1_LO, X1_HI = -5.5, 6.0
    X2_LO, X2_HI = -4.0, 4.0
    grid_x1 = np.linspace(X1_LO, X1_HI, G).astype(np.float32)
    grid_x2 = np.linspace(X2_LO, X2_HI, G).astype(np.float32)
    x1g, x2g = np.meshgrid(grid_x1, grid_x2)
    X_grid = np.stack([x1g.ravel(), x2g.ravel()], axis=1)
    extent = (X1_LO, X1_HI, X2_LO, X2_HI)

    with torch.no_grad():
        Phi_train = net.features(X_tr)
        Phi_grid = net.features(X_grid)
        mu_train = net.predict(X_tr).cpu().numpy()
        mu = net.predict(X_grid).cpu().numpy()[:, 0].reshape(G, G)

    print("Computing last-layer leverage on output-RFF features ...")
    ridge = 1e-3
    h_train = feature_leverage(Phi_train, Phi_train, ridge=ridge, bias=True).cpu().numpy()
    h_grid = feature_leverage(Phi_train, Phi_grid, ridge=ridge, bias=True).cpu().numpy().reshape(G, G)
    h_char = float(np.percentile(h_train, 95))
    sigma_noise = float(np.std(y_tr - mu_train))
    lam = sigma_noise / np.sqrt(max(h_char, 1e-12))
    sigma_epi_raw = lam * np.sqrt(np.clip(h_grid, 0.0, None))
    sigma_max = 5.0 * sigma_noise
    sigma_epi = sigma_max * np.tanh(sigma_epi_raw / sigma_max)

    truth_mean = gauss_mix_landscape(x1g, x2g).astype(np.float32)
    abserr = np.abs(mu - truth_mean)

    masks = classify_regions(x1g, x2g, geom)
    in_data = masks["in_data"]; ring_hole = masks["ring_hole"]; ood = masks["ood"]

    def _stat(name: str, arr: np.ndarray) -> None:
        print(f"  {name:18s}  in_data={float(arr[in_data].mean()):.4f}  "
              f"ring_hole={float(arr[ring_hole].mean()):.4f}  "
              f"ood={float(arr[ood].mean()):.4f}")

    print(f"\nGrid cells per region: in_data={int(in_data.sum())}, "
          f"ring_hole={int(ring_hole.sum())}, ood={int(ood.sum())}")
    print("\nMean over region:")
    _stat("sigma_epistemic", sigma_epi)
    _stat("|mean - truth|", abserr)
    print("  Notes:")
    print("   * Output-RFF features are bounded, so OOD leverage stays tame")
    print("     even before the tanh saturation kicks in.")
    print("   * The MLP trunk between the input and output RFF layers gives")
    print("     enough capacity to resolve the moons/spiral geometry.")

    cx_r, cy_r = geom["ring"]["center"]
    r_outer = geom["ring"]["r_outer"]
    pad_r = 0.6
    n_slice = 400
    t_slice = np.linspace(-1.0, 1.0, n_slice).astype(np.float32)
    slice_len = r_outer + pad_r
    slice_x1 = (cx_r + t_slice * slice_len).astype(np.float32)
    slice_x2 = np.full_like(slice_x1, cy_r, dtype=np.float32)
    slice_X = np.stack([slice_x1, slice_x2], axis=1)
    with torch.no_grad():
        Phi_slice = net.features(slice_X)
    h_slice = feature_leverage(Phi_train, Phi_slice, ridge=ridge, bias=True).cpu().numpy()
    sigma_epi_slice_raw = lam * np.sqrt(np.clip(h_slice, 0.0, None))
    sigma_epi_slice = sigma_max * np.tanh(sigma_epi_slice_raw / sigma_max)
    abs_radius_slice = np.abs(t_slice * slice_len)
    in_hole_slice = abs_radius_slice < geom["ring"]["r_inner"]
    on_ring_slice = (
        (abs_radius_slice >= geom["ring"]["r_inner"])
        & (abs_radius_slice <= r_outer)
    )
    ood_slice = abs_radius_slice > r_outer

    print("\n1-D slice across the ring (x2 = ring center, x1 sweeps diameter):")
    print(f"  sigma_epi mean on ring     : "
          f"{float(sigma_epi_slice[on_ring_slice].mean()):.4f}")
    if in_hole_slice.any():
        print(f"  sigma_epi mean in hole     : "
              f"{float(sigma_epi_slice[in_hole_slice].mean()):.4f}")
        print(f"  sigma_epi peak in hole     : "
              f"{float(sigma_epi_slice[in_hole_slice].max()):.4f}")
    if ood_slice.any():
        print(f"  sigma_epi mean ood (slice) : "
              f"{float(sigma_epi_slice[ood_slice].mean()):.4f}")

    out_dir = Path(__file__).resolve().parent.parent / "out" / "hetero"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.7])
    axes = np.array([
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])],
    ])
    ax_slice = fig.add_subplot(gs[2, :])

    ax = axes[0, 0]
    ax.scatter(X_tr[:, 0], X_tr[:, 1], s=3, alpha=0.4, color="tab:gray", label="train")
    _draw_shapes(ax, geom)
    ax.set_xlim(X1_LO, X1_HI); ax.set_ylim(X2_LO, X2_HI)
    ax.set_aspect("equal")
    ax.set_title("Training data\n(2 moons + ring + spiral)", fontsize=9)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.legend(loc="upper left", fontsize=7)

    mu_vmin = float(truth_mean.min())
    mu_vmax = float(truth_mean.max())
    _heat(axes[0, 1], truth_mean, extent,
          "Truth mean (gauss-mix landscape)",
          geom, cmap="RdBu_r", vmin=mu_vmin, vmax=mu_vmax)
    _heat(axes[0, 2], mu, extent,
        #   f"Mean prediction (deep RFF)\nin M={M_in} ell_in={ell_in} | out M={M_out} ell_out={ell_out}",
        f"Mean prediction (deep RFF)\nout M={M_out}",
          geom, cmap="RdBu_r", vmin=mu_vmin, vmax=mu_vmax)

    _heat(axes[1, 0], abserr, extent,
          "|mean - truth|",
          geom, cmap="magma")

    sigma_vmax = float(sigma_epi.max())
    _heat(axes[1, 1], sigma_epi, extent,
          "sigma_epistemic\n(last-layer leverage on output-RFF features)",
          geom, cmap="magma", vmin=0.0, vmax=sigma_vmax)

    zr = 1.4 * geom["ring"]["r_outer"]
    z_extent = (cx_r - zr, cx_r + zr, cy_r - zr, cy_r + zr)
    z_mask = ((x1g >= z_extent[0]) & (x1g <= z_extent[1])
              & (x2g >= z_extent[2]) & (x2g <= z_extent[3]))
    rows = np.where(z_mask.any(axis=1))[0]
    cols = np.where(z_mask.any(axis=0))[0]
    sigma_zoom = sigma_epi[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    zoom_vmax = float(sigma_zoom.max())
    _heat(axes[1, 2], sigma_zoom, z_extent,
          "Zoom: ring + hole\n(sigma_epi must rise inside the empty disc)",
          geom, cmap="magma", vmin=0.0, vmax=zoom_vmax)

    ax_slice.plot(t_slice * slice_len, sigma_epi_slice, color="tab:purple",
                  lw=2.0, label="sigma_epi across ring diameter")
    ax_slice.fill_between(t_slice * slice_len, 0.0, sigma_epi_slice,
                          where=in_hole_slice, color="tab:orange", alpha=0.20,
                          label="ring hole (no training data)")
    ax_slice.fill_between(t_slice * slice_len, 0.0, sigma_epi_slice,
                          where=on_ring_slice, color="tab:blue", alpha=0.10,
                          label="on ring (data present)")
    ax_slice.fill_between(t_slice * slice_len, 0.0, sigma_epi_slice,
                          where=ood_slice, color="tab:red", alpha=0.10,
                          label="outside ring (far OOD)")
    ax_slice.axhline(sigma_noise, color="tab:gray", ls=":", lw=0.8,
                     label=f"sigma_noise ({sigma_noise:.2f})")
    ratio = (
        float(sigma_epi_slice[in_hole_slice].mean())
        / max(float(sigma_epi_slice[on_ring_slice].mean()), 1e-9)
    )
    ax_slice.set_xlim(-slice_len, slice_len)
    ax_slice.set_ylim(0.0, float(sigma_epi_slice.max()) * 1.20)
    ax_slice.set_xlabel("offset from ring center along x1 (units)")
    ax_slice.set_ylabel("sigma_epi")
    ax_slice.set_title(
        f"Deep RFF ring-diameter slice. Hole sigma_epi is {ratio:.2f}x the on-ring mean: "
        "leverage on bounded output-RFF features detects the topological hole.",
        fontsize=10,
    )
    ax_slice.legend(loc="upper center", fontsize=8, ncol=2)

    fig.suptitle(
        # f"2D deep RFF (input M={M_in} ell_in={ell_in}, MLP trunk, output M={M_out} ell_out={ell_out})\n"
        f"2D deep RFF (MLP trunk, output M={M_out})\n"
        f"+ closed-form last-layer epistemic on a complex training topology\n"
        "Train: 2 interlocking moons + thin annulus + spiral arm. "
        f"Predict on [{X1_LO:+.1f}, {X1_HI:+.1f}] x [{X2_LO:+.1f}, {X2_HI:+.1f}]. "
        "Row 0: data, truth, prediction. Row 1: |error|, sigma_epi (full), sigma_epi (ring zoom).",
        y=0.995, fontsize=10,
    )
    fig.tight_layout()
    out_path = out_dir / "hetero_demo_2d_ood_deep_rff.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
