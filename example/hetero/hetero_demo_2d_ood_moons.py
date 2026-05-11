"""Deep RFF + last-layer leverage on sklearn ``make_moons``.

Two interleaved half-circles (top moon at the origin, bottom moon
shifted to ``(1, -0.5)``). The slot between the two moons and the
regions outside the bbox are unobserved; the leverage head should flag
both. Same FCNet/RFF/leverage stack as
``hetero_demo_2d_ood_deep_rff.py``.

Run from the repo root:

    python example/hetero/hetero_demo_2d_ood_moons.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons

from ebmify.models import (
    FCNet,
    FitConfig,
    NoiseConfig,
    PreprocessConfig,
    RegConfig,
    feature_leverage,
)


# ----------------------------------------------------------------------
# Truth landscape
# ----------------------------------------------------------------------

# Bumps placed on the actual moon arms, alternating signs for structure.
# sklearn make_moons:
#   * top moon    = upper half-circle of unit radius at the origin
#                   -> x in [-1, 1], y in [0, 1]
#   * bottom moon = lower half-circle of unit radius at (1, +0.5)
#                   -> x in [0, 2], y in [-0.5, 0.5]
_MOONS_BUMPS: tuple[tuple[float, float, float, float], ...] = (
    (-1.0,  0.0,  1.2, 0.3),   # peak: top moon left tip
    ( 0.0,  1.0, -1.0, 0.3),   # trough: top moon apex
    ( 2.0,  0.5,  1.2, 0.3),   # peak: bottom moon right tip
    ( 1.0, -0.5, -1.0, 0.3),   # trough: bottom moon bottom
)


def moons_landscape(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x1, dtype=np.float64)
    for cx, cy, a, s in _MOONS_BUMPS:
        out += a * np.exp(-((x1 - cx) ** 2 + (x2 - cy) ** 2) / (2.0 * s ** 2))
    return out.astype(x1.dtype)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def make_moons_data(
    n: int = 1000, seed: int = 1, noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """sklearn ``make_moons`` + smooth gauss-mix label.

    sklearn's top moon is the upper half-circle of unit radius at the
    origin; the bottom moon is the lower half-circle of unit radius
    centered at ``(1, +0.5)``.
    """
    X, _ = make_moons(n_samples=n, noise=noise, random_state=seed)
    X = X.astype(np.float32)
    geom = {
        "moon_top":    {"center": (0.0, 0.0), "radius": 1.0,
                         "angle_lo": 0.0, "angle_hi": np.pi,
                         "thickness": 3.0 * noise + 0.04},
        "moon_bottom": {"center": (1.0, 0.5), "radius": 1.0,
                         "angle_lo": np.pi, "angle_hi": 2.0 * np.pi,
                         "thickness": 3.0 * noise + 0.04},
    }
    y = moons_landscape(X[:, :1], X[:, 1:2]).astype(np.float32)
    return X, y, geom


# ----------------------------------------------------------------------
# Region masks
# ----------------------------------------------------------------------

def _moon_mask(
    x1: np.ndarray, x2: np.ndarray, m: dict, pad: float = 0.0,
) -> np.ndarray:
    cx, cy = m["center"]
    r = np.sqrt((x1 - cx) ** 2 + (x2 - cy) ** 2)
    theta = np.arctan2(x2 - cy, x1 - cx)
    th = np.where(theta < m["angle_lo"], theta + 2 * np.pi, theta)
    in_angle = (th >= m["angle_lo"]) & (th <= m["angle_hi"])
    in_radial = np.abs(r - m["radius"]) <= (m["thickness"] + pad)
    return in_angle & in_radial


def classify_regions(
    x1: np.ndarray, x2: np.ndarray, geom: dict,
    *, x1_lo: float, x1_hi: float, x2_lo: float, x2_hi: float,
) -> dict:
    mt = _moon_mask(x1, x2, geom["moon_top"])
    mb = _moon_mask(x1, x2, geom["moon_bottom"])
    in_data = mt | mb
    in_bbox = ((x1 >= x1_lo) & (x1 <= x1_hi)
               & (x2 >= x2_lo) & (x2 <= x2_hi))
    slot = in_bbox & (~in_data)
    return {
        "moon_top": mt, "moon_bottom": mb,
        "in_data": in_data, "slot": slot, "ood": ~in_bbox,
    }


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------

def _draw_shapes(ax, geom: dict, color: str = "orange") -> None:
    for key in ("moon_top", "moon_bottom"):
        m = geom[key]
        cx, cy = m["center"]; rr = m["radius"]; tk = m["thickness"]
        th = np.linspace(m["angle_lo"], m["angle_hi"], 200)
        for r in (rr - tk, rr + tk):
            ax.plot(cx + r * np.cos(th), cy + r * np.sin(th),
                    color=color, lw=1.0, alpha=0.85)


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
    X_tr, y_tr, geom = make_moons_data(n=1000, seed=1)

    pc = PreprocessConfig(
        input_transforms=["standard"],
        output_transforms=["standard"],
    )
    fc = FitConfig(epochs=1000, lr=5e-4, batch_size=128, seed=0, verbose=True)
    nc = NoiseConfig(
        input_additive_std=0.0, input_multiplicative_std=0.0,
        output_additive_std=0.0,
    )

    M_in = 32
    M_out = 64
    print(
        "Fitting deep RFF FCNet "
        f"(input_rff M={M_in}, residual MLP, output_rff M={M_out}) "
        "on sklearn make_moons ..."
    )
    net = FCNet(
        n_inputs=2, n_outputs=1,
        hidden_dims=(32, 32),
        activation="odd_piecewise",
        fit_config=fc, reg_config=RegConfig(l2=1e-5),
        noise_config=nc, preprocess=pc,
        input_rff=M_in, input_rff_length_scale=[0.1, 0.25, 0.5],
        output_rff=M_out, output_rff_length_scale=[0.1, 0.25, 0.5],
        rff_seed=0,
    )
    net.fit(X_tr, y_tr)
    ell_in = _fmt_ell(net.net.input_rff.length_scale)
    ell_out = _fmt_ell(net.net.output_rff.length_scale)
    print(f"  resolved input_rff length_scale  = {ell_in}")
    print(f"  resolved output_rff length_scale = {ell_out}")

    G = 140
    X1_LO, X1_HI = -2.0, 3.0
    X2_LO, X2_HI = -1.7, 1.7
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

    truth_mean = moons_landscape(x1g, x2g).astype(np.float32)
    abserr = np.abs(mu - truth_mean)

    # bbox for "in_bbox" slightly tighter than the full grid: cells outside
    # the data envelope are far OOD even if inside the plot extent.
    masks = classify_regions(
        x1g, x2g, geom,
        x1_lo=-1.5, x1_hi=2.5, x2_lo=-1.3, x2_hi=1.3,
    )
    in_data = masks["in_data"]; slot = masks["slot"]; ood = masks["ood"]

    def _stat(name: str, arr: np.ndarray) -> None:
        print(f"  {name:18s}  in_data={float(arr[in_data].mean()):.4f}  "
              f"slot={float(arr[slot].mean()):.4f}  "
              f"ood={float(arr[ood].mean()):.4f}")

    print(f"\nGrid cells per region: in_data={int(in_data.sum())}, "
          f"slot={int(slot.sum())}, ood={int(ood.sum())}")
    print("\nMean over region:")
    _stat("sigma_epistemic", sigma_epi)
    _stat("|mean - truth|", abserr)
    print("  Notes:")
    print("   * The 'slot' is the empty interior region between the two")
    print("     moons -- a topological hole the leverage head should flag.")

    # 1-D slice across the slot. At x1 = 0.5, x2 sweeps from below the
    # bottom moon, up through the slot, into the top moon, and out.
    n_slice = 400
    t_slice = np.linspace(-1.6, 1.6, n_slice).astype(np.float32)
    slice_x1 = np.full_like(t_slice, 0.5, dtype=np.float32)
    slice_x2 = t_slice
    slice_X = np.stack([slice_x1, slice_x2], axis=1)
    with torch.no_grad():
        Phi_slice = net.features(slice_X)
    h_slice = feature_leverage(Phi_train, Phi_slice, ridge=ridge, bias=True).cpu().numpy()
    sigma_epi_slice = sigma_max * np.tanh(
        lam * np.sqrt(np.clip(h_slice, 0.0, None)) / sigma_max
    )
    on_top = _moon_mask(slice_x1, slice_x2, geom["moon_top"])
    on_bot = _moon_mask(slice_x1, slice_x2, geom["moon_bottom"])
    on_moon = on_top | on_bot
    # At x1 = 0.5 the bottom moon edge sits at x2 ~= -0.37 and the top
    # moon edge at x2 ~= +0.87, so the interior slot is roughly
    # x2 in (-0.37, +0.87). `~on_moon` excludes the band itself.
    in_slot = (~on_moon) & (t_slice > -0.3) & (t_slice < 0.8)
    ood_slice = (~on_moon) & ~in_slot

    print("\n1-D slice through the slot (x1 = 0.5, x2 sweep):")
    if on_moon.any():
        print(f"  sigma_epi mean on moons : "
              f"{float(sigma_epi_slice[on_moon].mean()):.4f}")
    if in_slot.any():
        print(f"  sigma_epi mean in slot  : "
              f"{float(sigma_epi_slice[in_slot].mean()):.4f}")

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
    ax.set_title("Training data\n(sklearn make_moons)", fontsize=9)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.legend(loc="upper left", fontsize=7)

    mu_vmin = float(truth_mean.min()); mu_vmax = float(truth_mean.max())
    _heat(axes[0, 1], truth_mean, extent,
          "Truth mean (gauss-mix landscape)",
          geom, cmap="RdBu_r", vmin=mu_vmin, vmax=mu_vmax)
    _heat(axes[0, 2], mu, extent,
          f"Mean prediction (deep RFF)\nin M={M_in} ell_in={ell_in} | out M={M_out} ell_out={ell_out}",
          geom, cmap="RdBu_r", vmin=mu_vmin, vmax=mu_vmax)

    _heat(axes[1, 0], abserr, extent, "|mean - truth|", geom, cmap="magma")
    sigma_vmax = float(sigma_epi.max())
    _heat(axes[1, 1], sigma_epi, extent,
          "sigma_epistemic\n(last-layer leverage on output-RFF features)",
          geom, cmap="magma", vmin=0.0, vmax=sigma_vmax)

    z_extent = (-1.5, 2.5, -1.5, 1.5)
    z_mask = ((x1g >= z_extent[0]) & (x1g <= z_extent[1])
              & (x2g >= z_extent[2]) & (x2g <= z_extent[3]))
    rows = np.where(z_mask.any(axis=1))[0]
    cols = np.where(z_mask.any(axis=0))[0]
    sigma_zoom = sigma_epi[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    zoom_vmax = float(sigma_zoom.max())
    _heat(axes[1, 2], sigma_zoom, z_extent,
          "Zoom: moons + slot\n(sigma_epi must rise inside the slot)",
          geom, cmap="magma", vmin=0.0, vmax=zoom_vmax)

    ax_slice.plot(t_slice, sigma_epi_slice, color="tab:purple",
                  lw=2.0, label="sigma_epi at x1 = 0.5")
    ax_slice.fill_between(t_slice, 0.0, sigma_epi_slice,
                          where=in_slot, color="tab:orange", alpha=0.20,
                          label="slot (no data)")
    ax_slice.fill_between(t_slice, 0.0, sigma_epi_slice,
                          where=on_moon, color="tab:blue", alpha=0.10,
                          label="on moon (data present)")
    ax_slice.fill_between(t_slice, 0.0, sigma_epi_slice,
                          where=ood_slice, color="tab:red", alpha=0.10,
                          label="far OOD")
    ax_slice.axhline(sigma_noise, color="tab:gray", ls=":", lw=0.8,
                     label=f"sigma_noise ({sigma_noise:.2f})")
    on_mean = float(sigma_epi_slice[on_moon].mean()) if on_moon.any() else 1e-9
    on_mean = max(on_mean, 1e-9)
    slot_mean = float(sigma_epi_slice[in_slot].mean()) if in_slot.any() else 0.0
    ratio = slot_mean / on_mean
    ax_slice.set_xlim(-1.6, 1.6)
    ax_slice.set_ylim(0.0, float(sigma_epi_slice.max()) * 1.20)
    ax_slice.set_xlabel("x2 (offset, x1 = 0.5)")
    ax_slice.set_ylabel("sigma_epi")
    ax_slice.set_title(
        f"x2 slice through the slot. Slot sigma_epi is {ratio:.2f}x the on-moon mean: "
        "leverage flags the interior gap between the two moons.",
        fontsize=10,
    )
    ax_slice.legend(loc="upper center", fontsize=8, ncol=2)

    fig.suptitle(
        f"2D deep RFF on sklearn make_moons "
        f"(input M={M_in} ell_in={ell_in}, MLP trunk, output M={M_out} ell_out={ell_out})\n"
        "Train: two interleaved half-circle moons. "
        f"Predict on [{X1_LO:+.1f}, {X1_HI:+.1f}] x [{X2_LO:+.1f}, {X2_HI:+.1f}]. "
        "Row 0: data, truth, prediction. Row 1: |error|, sigma_epi (full), sigma_epi (zoom).",
        y=0.995, fontsize=10,
    )
    fig.tight_layout()
    out_path = out_dir / "hetero_demo_2d_ood_moons.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
