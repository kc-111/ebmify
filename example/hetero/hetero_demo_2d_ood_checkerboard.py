"""Deep RFF + last-layer leverage on a checkerboard topology.

We sample 2D points uniformly inside the *white* cells of an
``n_cells x n_cells`` checkerboard (cell ``(i, j)`` is white iff
``(i + j)`` is even). The black cells are interior topological holes:
fully surrounded by data on all four sides, but the model never sees a
single training point inside them. The leverage head should light up
sharply on the black cells -- a much harder test than annular or slot
holes because the holes form a *periodic lattice*, so the model has
multiple repeating absent regions to flag.

Same FCNet/RFF/leverage stack as ``hetero_demo_2d_ood_deep_rff.py``;
only the data topology and the truth landscape change.

Run from the repo root:

    python example/hetero/hetero_demo_2d_ood_checkerboard.py
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


# ----------------------------------------------------------------------
# Truth landscape: a smooth surface defined everywhere (so the heatmap
# in black cells has a sensible target). cos(pi*x1) * cos(pi*x2) gives a
# checkerboard-shaped ground-truth signal that aligns with the cell
# boundaries.
# ----------------------------------------------------------------------

def checkerboard_landscape(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return (np.cos(np.pi * x1) * np.cos(np.pi * x2)).astype(x1.dtype)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def make_checkerboard_data(
    n: int = 1000, seed: int = 1, n_cells: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sample uniformly inside the white cells of an n_cells x n_cells board.

    Cell ``(i, j)`` occupies ``[i, i+1] x [j, j+1]`` for
    ``i, j in [0, n_cells)``; it is white iff ``(i + j)`` is even.
    """
    rng = np.random.default_rng(seed)
    white_cells = [
        (i, j) for i in range(n_cells) for j in range(n_cells)
        if (i + j) % 2 == 0
    ]
    n_white = len(white_cells)
    n_per = n // n_white
    extra = n - n_per * n_white

    pts = []
    for k, (i, j) in enumerate(white_cells):
        m = n_per + (1 if k < extra else 0)
        local = rng.uniform(0.0, 1.0, size=(m, 2)).astype(np.float32)
        local[:, 0] += i; local[:, 1] += j
        pts.append(local)
    X = np.concatenate(pts, axis=0)
    rng.shuffle(X)

    geom = {"n_cells": n_cells, "white_cells": white_cells}
    y = checkerboard_landscape(X[:, :1], X[:, 1:2]).astype(np.float32)
    return X, y, geom


# ----------------------------------------------------------------------
# Region masks: each grid pixel is on a white cell (in_data), on a black
# cell (interior hole), or outside the board (far OOD).
# ----------------------------------------------------------------------

def classify_regions(x1: np.ndarray, x2: np.ndarray, geom: dict) -> dict:
    n_cells = int(geom["n_cells"])
    in_bbox = (x1 >= 0.0) & (x1 < n_cells) & (x2 >= 0.0) & (x2 < n_cells)
    # Clip to a valid integer index even outside the board so the parity
    # array has no surprises; we mask back in_bbox at the end.
    i = np.floor(np.clip(x1, 0.0, n_cells - 1e-6)).astype(np.int64)
    j = np.floor(np.clip(x2, 0.0, n_cells - 1e-6)).astype(np.int64)
    parity = (i + j) % 2
    is_white = (parity == 0) & in_bbox
    is_black = (parity == 1) & in_bbox
    return {
        "in_data": is_white,
        "interior_hole": is_black,
        "ood": ~in_bbox,
    }


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------

def _draw_shapes(ax, geom: dict, color: str = "orange") -> None:
    n_cells = int(geom["n_cells"])
    for i in range(n_cells):
        for j in range(n_cells):
            if (i + j) % 2 == 0:
                ax.add_patch(plt.Rectangle(
                    (i, j), 1, 1, fill=False,
                    edgecolor=color, lw=0.8, alpha=0.8,
                ))


def _heat(ax, Z, extent, title, geom, *, cmap="viridis",
          vmin=None, vmax=None) -> None:
    im = ax.imshow(Z, extent=extent, origin="lower",
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="equal", interpolation="nearest")
    _draw_shapes(ax, geom)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _fmt_ell(t: torch.Tensor) -> str:
    vals = t.detach().cpu().tolist()
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return "[" + ", ".join(f"{v:.3f}" for v in vals) + "]"


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    n_cells = 4
    X_tr, y_tr, geom = make_checkerboard_data(n=1600, seed=1, n_cells=n_cells)

    pc = PreprocessConfig(
        input_transforms=["standard"],
        output_transforms=["standard"],
    )
    fc = FitConfig(epochs=1500, lr=5e-4, batch_size=128, seed=0, verbose=True)
    nc = NoiseConfig(
        input_additive_std=0.0, input_multiplicative_std=0.0,
        output_additive_std=0.0,
    )

    M_in = 32
    M_out = 64
    print(
        "Fitting deep RFF FCNet "
        f"(input_rff M={M_in}, residual MLP, output_rff M={M_out}) "
        f"on checkerboard ({n_cells}x{n_cells}, white-only) ..."
    )
    net = FCNet(
        n_inputs=2, n_outputs=1,
        hidden_dims=(64, 64),
        activation="odd_piecewise",
        fit_config=fc, reg_config=RegConfig(l2=1e-5),
        noise_config=nc, preprocess=pc,
        input_rff=M_in, input_rff_length_scale=[0.15, 0.3, 0.6],
        output_rff=M_out, output_rff_length_scale=[0.15, 0.3, 0.6],
        rff_seed=0,
    )
    net.fit(X_tr, y_tr)
    ell_in = _fmt_ell(net.net.input_rff.length_scale)
    ell_out = _fmt_ell(net.net.output_rff.length_scale)
    print(f"  resolved input_rff length_scale  = {ell_in}")
    print(f"  resolved output_rff length_scale = {ell_out}")

    G = 160
    PAD = 0.8
    X1_LO, X1_HI = -PAD, n_cells + PAD
    X2_LO, X2_HI = -PAD, n_cells + PAD
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
    sigma_max = 5.0 * sigma_noise
    sigma_epi = sigma_max * np.tanh(
        lam * np.sqrt(np.clip(h_grid, 0.0, None)) / sigma_max
    )

    truth_mean = checkerboard_landscape(x1g, x2g).astype(np.float32)
    abserr = np.abs(mu - truth_mean)

    masks = classify_regions(x1g, x2g, geom)
    in_data = masks["in_data"]
    interior_hole = masks["interior_hole"]
    ood = masks["ood"]

    def _stat(name: str, arr: np.ndarray) -> None:
        print(f"  {name:18s}  in_data={float(arr[in_data].mean()):.4f}  "
              f"interior_hole={float(arr[interior_hole].mean()):.4f}  "
              f"ood={float(arr[ood].mean()):.4f}")

    print(f"\nGrid cells per region: in_data={int(in_data.sum())}, "
          f"interior_hole={int(interior_hole.sum())}, ood={int(ood.sum())}")
    print("\nMean over region:")
    _stat("sigma_epistemic", sigma_epi)
    _stat("|mean - truth|", abserr)
    print("  Notes:")
    print("   * Black cells are surrounded by white cells on all four sides")
    print("     yet contain zero training data -- a periodic-lattice variant")
    print("     of the ring-hole test.")

    # 1-D slice across the diagonal of the board: (0, 0) -> (n_cells, n_cells).
    # Walks alternately white -> black -> white -> ... on the cell parity.
    n_slice = 400
    t_slice = np.linspace(-PAD, n_cells + PAD, n_slice).astype(np.float32)
    slice_x1 = t_slice
    slice_x2 = t_slice
    slice_X = np.stack([slice_x1, slice_x2], axis=1)
    with torch.no_grad():
        Phi_slice = net.features(slice_X)
    h_slice = feature_leverage(Phi_train, Phi_slice, ridge=ridge, bias=True).cpu().numpy()
    sigma_epi_slice = sigma_max * np.tanh(
        lam * np.sqrt(np.clip(h_slice, 0.0, None)) / sigma_max
    )
    in_bbox_slice = (slice_x1 >= 0.0) & (slice_x1 < n_cells)
    # Diagonal: (i, j) = (floor(x1), floor(x2)) and j == i along x1 == x2,
    # so parity = 2*i % 2 = 0 always. The diagonal stays on white cells!
    # Use an off-diagonal slice (x2 = x1 + 0.5) so we cross both colours.
    slice_x2_off = (slice_x1 + 0.5).astype(np.float32)
    slice_X_off = np.stack([slice_x1, slice_x2_off], axis=1)
    with torch.no_grad():
        Phi_slice_off = net.features(slice_X_off)
    h_slice_off = feature_leverage(Phi_train, Phi_slice_off, ridge=ridge, bias=True).cpu().numpy()
    sigma_epi_slice_off = sigma_max * np.tanh(
        lam * np.sqrt(np.clip(h_slice_off, 0.0, None)) / sigma_max
    )
    in_bbox_off = ((slice_x1 >= 0.0) & (slice_x1 < n_cells)
                   & (slice_x2_off >= 0.0) & (slice_x2_off < n_cells))
    i_off = np.floor(np.clip(slice_x1, 0.0, n_cells - 1e-6)).astype(np.int64)
    j_off = np.floor(np.clip(slice_x2_off, 0.0, n_cells - 1e-6)).astype(np.int64)
    parity_off = (i_off + j_off) % 2
    on_white = (parity_off == 0) & in_bbox_off
    in_black = (parity_off == 1) & in_bbox_off
    ood_off = ~in_bbox_off

    print("\n1-D slice along x2 = x1 + 0.5 (crosses alternating cells):")
    if on_white.any():
        print(f"  sigma_epi mean on white cells : "
              f"{float(sigma_epi_slice_off[on_white].mean()):.4f}")
    if in_black.any():
        print(f"  sigma_epi mean in black cells : "
              f"{float(sigma_epi_slice_off[in_black].mean()):.4f}")
    if ood_off.any():
        print(f"  sigma_epi mean ood (slice)    : "
              f"{float(sigma_epi_slice_off[ood_off].mean()):.4f}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    out_dir = Path(__file__).resolve().parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
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
    ax.set_title(f"Training data\n({n_cells}x{n_cells} checkerboard, white only)",
                 fontsize=9)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.legend(loc="upper left", fontsize=7)

    mu_vmin = float(truth_mean.min()); mu_vmax = float(truth_mean.max())
    _heat(axes[0, 1], truth_mean, extent,
          "Truth mean (cos(pi x1) cos(pi x2))",
          geom, cmap="RdBu_r", vmin=mu_vmin, vmax=mu_vmax)
    _heat(axes[0, 2], mu, extent,
          f"Mean prediction (deep RFF)\nin M={M_in} ell_in={ell_in} | out M={M_out} ell_out={ell_out}",
          geom, cmap="RdBu_r", vmin=mu_vmin, vmax=mu_vmax)

    _heat(axes[1, 0], abserr, extent, "|mean - truth|", geom, cmap="magma")
    sigma_vmax = float(sigma_epi.max())
    _heat(axes[1, 1], sigma_epi, extent,
          "sigma_epistemic\n(last-layer leverage on output-RFF features)",
          geom, cmap="magma", vmin=0.0, vmax=sigma_vmax)

    # Zoom to the central 2x2 cells.
    z_lo = (n_cells - 2) / 2.0
    z_hi = z_lo + 2.0
    z_extent = (z_lo, z_hi, z_lo, z_hi)
    z_mask = ((x1g >= z_extent[0]) & (x1g <= z_extent[1])
              & (x2g >= z_extent[2]) & (x2g <= z_extent[3]))
    rows = np.where(z_mask.any(axis=1))[0]
    cols = np.where(z_mask.any(axis=0))[0]
    sigma_zoom = sigma_epi[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    zoom_vmax = float(sigma_zoom.max())
    _heat(axes[1, 2], sigma_zoom, z_extent,
          "Zoom: central 2x2\n(sigma_epi must rise inside black cells)",
          geom, cmap="magma", vmin=0.0, vmax=zoom_vmax)

    ax_slice.plot(slice_x1, sigma_epi_slice_off, color="tab:purple",
                  lw=2.0, label="sigma_epi along x2 = x1 + 0.5")
    ax_slice.fill_between(slice_x1, 0.0, sigma_epi_slice_off,
                          where=in_black, color="tab:orange", alpha=0.20,
                          label="black cells (no data)")
    ax_slice.fill_between(slice_x1, 0.0, sigma_epi_slice_off,
                          where=on_white, color="tab:blue", alpha=0.10,
                          label="white cells (data present)")
    ax_slice.fill_between(slice_x1, 0.0, sigma_epi_slice_off,
                          where=ood_off, color="tab:red", alpha=0.10,
                          label="off the board (far OOD)")
    ax_slice.axhline(sigma_noise, color="tab:gray", ls=":", lw=0.8,
                     label=f"sigma_noise ({sigma_noise:.2f})")
    on_mean = (
        float(sigma_epi_slice_off[on_white].mean()) if on_white.any() else 1e-9
    )
    on_mean = max(on_mean, 1e-9)
    black_mean = (
        float(sigma_epi_slice_off[in_black].mean()) if in_black.any() else 0.0
    )
    ratio = black_mean / on_mean
    ax_slice.set_xlim(X1_LO, X1_HI)
    ax_slice.set_ylim(0.0, float(sigma_epi_slice_off.max()) * 1.20)
    ax_slice.set_xlabel("x1 (slice along x2 = x1 + 0.5)")
    ax_slice.set_ylabel("sigma_epi")
    ax_slice.set_title(
        f"Diagonal slice. Black-cell sigma_epi is {ratio:.2f}x the white-cell mean: "
        "leverage flags every black cell on the lattice.",
        fontsize=10,
    )
    ax_slice.legend(loc="upper center", fontsize=8, ncol=2)

    fig.suptitle(
        f"2D deep RFF on checkerboard topology "
        f"(input M={M_in} ell_in={ell_in}, MLP trunk, output M={M_out} ell_out={ell_out})\n"
        f"Train: white cells only of a {n_cells}x{n_cells} board. "
        f"Predict on [{X1_LO:+.1f}, {X1_HI:+.1f}] x [{X2_LO:+.1f}, {X2_HI:+.1f}]. "
        "Row 0: data, truth, prediction. Row 1: |error|, sigma_epi (full), sigma_epi (zoom).",
        y=0.995, fontsize=10,
    )
    fig.tight_layout()
    out_path = out_dir / "hetero_demo_2d_ood_checkerboard.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
