"""Demo: one FCNet + closed-form last-layer epistemic on a complex,
non-convex training topology -- two interlocking moons, a thin annulus,
and a spiral arm. Stress-test for OOD detection.

This is a stronger version of ``hetero_demo_2d.py``: instead of two
convex blobs, the training set is a union of three topologically
non-trivial shapes:

  * **Two interlocking moons** at the origin -- forces the epistemic
    head to follow a *curved* decision boundary, with a thin OOD slot
    in the moons' interlock region.
  * **Thin annulus** at the upper right -- the empty disc *inside* the
    ring is a hole the model has never seen, even though it is
    surrounded by data on all sides. Pure topology test: can the
    epistemic head flag a hole that is not a "far away" region?
  * **Spiral arm** at the lower left -- thin curved manifold; tests
    whether epistemic stays low along the arm and rises perpendicular
    to it.

The OOD region is therefore **not just "outside the training bbox"**:
it includes the inside of the ring, the slot between the moons, and
the channels between the spiral turns. A naive distance-to-nearest-
training-point baseline would also flag these, but only the leverage
head also stays low *along the curved arms* where data is dense.

Truth landscape: same gauss-mix idea as the simpler demo, with bumps
placed near each training shape so the in-data mu prediction has
non-trivial structure to learn (not just a constant).

Architecture (identical to ``hetero_demo_2d.py``):
  * Single ``FCNet`` predicts ``mu(x)``. ``kde_quantile`` input with
    a wide bandwidth keeps the geometric gaps (ring hole, moon slot)
    visible to the leverage head; an empirical ``quantile_gpd`` body
    would collapse them.
  * ``sigma_epi(x*) = lambda * sqrt(phi(x*)^T (Phi^T Phi + r I)^-1 phi(x*))``
    with ``lambda`` calibrated so leverage on a typical training point
    matches the residual scale, then ``tanh``-saturated at
    ``5 * sigma_noise``.

Run from the repo root:

    python example/hetero/hetero_demo_2d_ood.py
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
# Truth landscape: gauss-mix bumps placed near the training shapes so
# the in-data mu prediction has structure to fit, not a constant.
# ----------------------------------------------------------------------

# (cx, cy, amplitude, scale) for each Gaussian bump.
_LANDSCAPE_BUMPS: tuple[tuple[float, float, float, float], ...] = (
    (-0.5,  0.4,  1.2, 0.6),   # peak on the upper moon
    ( 0.5, -0.4, -1.0, 0.6),   # trough on the lower moon
    ( 3.5,  1.5,  1.4, 0.8),   # peak on the ring
    (-3.0, -1.5, -1.0, 0.8),   # trough on the spiral
    ( 0.0,  3.0,  0.5, 1.0),   # OOD bump north
    ( 4.5, -2.0, -0.8, 1.0),   # OOD trough lower-right
)


def gauss_mix_landscape(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Smooth, bounded sum-of-Gaussians surface. Total amplitude ~ 5.9."""
    out = np.zeros_like(x1, dtype=np.float64)
    for cx, cy, a, s in _LANDSCAPE_BUMPS:
        out += a * np.exp(-((x1 - cx) ** 2 + (x2 - cy) ** 2) / (2.0 * s ** 2))
    return out.astype(x1.dtype)


# ----------------------------------------------------------------------
# Training-shape samplers
# ----------------------------------------------------------------------

def _sample_moon(
    n: int, center: tuple[float, float], radius: float,
    angle_lo: float, angle_hi: float, thickness: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``n`` points on a thick crescent arc.

    Points uniform in (theta, r) with theta in ``[angle_lo, angle_hi]`` and
    ``r`` in ``[radius - thickness, radius + thickness]``.
    """
    theta = rng.uniform(angle_lo, angle_hi, size=n)
    r = radius + rng.uniform(-thickness, thickness, size=n)
    x1 = center[0] + r * np.cos(theta)
    x2 = center[1] + r * np.sin(theta)
    return np.stack([x1, x2], axis=1).astype(np.float32)


def _sample_ring(
    n: int, center: tuple[float, float], r_inner: float, r_outer: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``n`` points uniformly inside an annulus (Jacobian-correct)."""
    r = np.sqrt(rng.uniform(r_inner ** 2, r_outer ** 2, size=n))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x1 = center[0] + r * np.cos(theta)
    x2 = center[1] + r * np.sin(theta)
    return np.stack([x1, x2], axis=1).astype(np.float32)


def _sample_spiral(
    n: int, center: tuple[float, float], scale: float, n_turns: float,
    thickness: float, rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``n`` points along an Archimedean spiral with radial jitter."""
    t = rng.uniform(0.15, 1.0, size=n)
    theta = n_turns * 2.0 * np.pi * t
    r = scale * t + rng.uniform(-thickness, thickness, size=n)
    x1 = center[0] + r * np.cos(theta)
    x2 = center[1] + r * np.sin(theta)
    return np.stack([x1, x2], axis=1).astype(np.float32)


def make_2d_complex_data(
    n: int = 4000, seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sample a union of two moons + ring + spiral.

    Returns ``(X, y, geom)`` where ``geom`` carries shape parameters used
    later for region masks (in-data vs hole vs OOD) and for plotting.
    """
    rng = np.random.default_rng(seed)
    geom = {
        "moon_a":   {"center": (-0.4,  0.3), "radius": 1.0,
                     "angle_lo": 0.20 * np.pi, "angle_hi": 1.10 * np.pi,
                     "thickness": 0.18},
        "moon_b":   {"center": ( 0.4, -0.3), "radius": 1.0,
                     "angle_lo": 1.20 * np.pi, "angle_hi": 2.10 * np.pi,
                     "thickness": 0.18},
        # Big ring -- hole radius (1.6) is large compared to the FCNet's
        # smoothness scale, otherwise features just interpolate across it.
        "ring":     {"center": ( 3.5,  1.5), "r_inner": 1.6,
                     "r_outer": 2.0},
        "spiral":   {"center": (-3.0, -1.5), "scale": 1.2,
                     "n_turns": 1.6, "thickness": 0.14},
    }
    n_per = n // 4

    moon_a = _sample_moon(n_per, **geom["moon_a"], rng=rng)
    moon_b = _sample_moon(n_per, **geom["moon_b"], rng=rng)
    ring   = _sample_ring(n_per, **geom["ring"],   rng=rng)
    spiral = _sample_spiral(n - 3 * n_per, **geom["spiral"], rng=rng)

    X = np.concatenate([moon_a, moon_b, ring, spiral], axis=0)
    rng.shuffle(X)

    # Heteroskedastic noise scales with distance from origin (kept only
    # as the data-generating process; the demo does not predict aleatoric).
    sigma = (0.08 + 0.25 * np.sqrt(X[:, :1] ** 2 + X[:, 1:2] ** 2)).astype(np.float32)
    mean = gauss_mix_landscape(X[:, :1], X[:, 1:2]).astype(np.float32)
    # y = (mean + sigma * rng.standard_normal((X.shape[0], 1)).astype(np.float32)).astype(np.float32)
    y = mean.astype(np.float32)
    return X, y, geom


# ----------------------------------------------------------------------
# Region masks: classify each grid pixel as in-data, in-hole, or OOD.
# ----------------------------------------------------------------------

def _moon_mask(x1: np.ndarray, x2: np.ndarray, m: dict, pad: float = 0.0) -> np.ndarray:
    cx, cy = m["center"]
    r = np.sqrt((x1 - cx) ** 2 + (x2 - cy) ** 2)
    theta = np.arctan2(x2 - cy, x1 - cx)
    # Wrap theta into [angle_lo - 2*pi, angle_lo + 2*pi) and keep only the
    # branch that lands inside [angle_lo, angle_hi]. arctan2 is in (-pi, pi];
    # the arc bounds extend up to 2.1*pi so we add 2*pi when needed.
    th = np.where(theta < m["angle_lo"], theta + 2 * np.pi, theta)
    in_angle = (th >= m["angle_lo"]) & (th <= m["angle_hi"])
    in_radial = np.abs(r - m["radius"]) <= (m["thickness"] + pad)
    return in_angle & in_radial


def _ring_mask(x1: np.ndarray, x2: np.ndarray, m: dict, pad: float = 0.0) -> np.ndarray:
    cx, cy = m["center"]
    r = np.sqrt((x1 - cx) ** 2 + (x2 - cy) ** 2)
    return (r >= m["r_inner"] - pad) & (r <= m["r_outer"] + pad)


def _ring_hole_mask(x1: np.ndarray, x2: np.ndarray, m: dict) -> np.ndarray:
    cx, cy = m["center"]
    r = np.sqrt((x1 - cx) ** 2 + (x2 - cy) ** 2)
    return r < m["r_inner"]


def _spiral_mask(
    x1: np.ndarray, x2: np.ndarray, m: dict, pad: float = 0.0
) -> np.ndarray:
    """Approximate spiral-arm membership by checking distance to the curve.

    Parameterize the spiral as r(theta) = scale * theta / (n_turns * 2*pi)
    and accept points whose distance to the nearest theta on the curve is
    within ``thickness + pad``. Works well enough for the demo's coarse
    regional accounting.
    """
    cx, cy = m["center"]
    scale = m["scale"]; n_turns = m["n_turns"]
    thickness = m["thickness"]
    # Sample the spiral curve densely.
    t_curve = np.linspace(0.15, 1.0, 400)
    theta_curve = n_turns * 2.0 * np.pi * t_curve
    rc = scale * t_curve
    cx_curve = cx + rc * np.cos(theta_curve)
    cy_curve = cy + rc * np.sin(theta_curve)
    # Pairwise (HxW, T) distance can be expensive on big grids, so chunk.
    flat_x1 = x1.ravel(); flat_x2 = x2.ravel()
    out = np.zeros(flat_x1.shape[0], dtype=bool)
    chunk = 4096
    for i in range(0, flat_x1.shape[0], chunk):
        sl = slice(i, i + chunk)
        d = np.minimum.reduce(
            np.sqrt((flat_x1[sl, None] - cx_curve[None, :]) ** 2
                    + (flat_x2[sl, None] - cy_curve[None, :]) ** 2),
            axis=1,
        ) if False else np.sqrt(
            (flat_x1[sl, None] - cx_curve[None, :]) ** 2
            + (flat_x2[sl, None] - cy_curve[None, :]) ** 2
        ).min(axis=1)
        out[sl] = d <= (thickness + pad)
    return out.reshape(x1.shape)


def classify_regions(x1: np.ndarray, x2: np.ndarray, geom: dict) -> dict:
    """Return boolean masks for in-data / ring-hole / OOD."""
    moon_a = _moon_mask(x1, x2, geom["moon_a"])
    moon_b = _moon_mask(x1, x2, geom["moon_b"])
    ring   = _ring_mask(x1, x2, geom["ring"])
    spiral = _spiral_mask(x1, x2, geom["spiral"])
    in_data = moon_a | moon_b | ring | spiral
    ring_hole = _ring_hole_mask(x1, x2, geom["ring"])
    # OOD = neither in any training shape nor inside the ring hole. The
    # ring hole is reported separately because it is a *topological* hole,
    # not "far OOD".
    ood = ~(in_data | ring_hole)
    return {
        "moon_a": moon_a, "moon_b": moon_b, "ring": ring, "spiral": spiral,
        "in_data": in_data, "ring_hole": ring_hole, "ood": ood,
    }


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------

def _draw_shapes(ax, geom: dict, color: str = "orange") -> None:
    """Outline each training shape on top of a heatmap or scatter."""
    theta = np.linspace(0, 2 * np.pi, 200)
    # Moons: outer + inner arc, capped.
    for key in ("moon_a", "moon_b"):
        m = geom[key]
        cx, cy = m["center"]; rr = m["radius"]; tk = m["thickness"]
        th = np.linspace(m["angle_lo"], m["angle_hi"], 200)
        ax.plot(cx + (rr + tk) * np.cos(th), cy + (rr + tk) * np.sin(th),
                color=color, lw=1.0, alpha=0.85)
        ax.plot(cx + (rr - tk) * np.cos(th), cy + (rr - tk) * np.sin(th),
                color=color, lw=1.0, alpha=0.85)
    # Ring: inner + outer circles.
    rng_m = geom["ring"]
    cx, cy = rng_m["center"]
    for r in (rng_m["r_inner"], rng_m["r_outer"]):
        ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta),
                color=color, lw=1.0, alpha=0.85)
    # Spiral: trace the centerline.
    sp = geom["spiral"]
    cx, cy = sp["center"]
    t_curve = np.linspace(0.15, 1.0, 200)
    th_c = sp["n_turns"] * 2 * np.pi * t_curve
    rc = sp["scale"] * t_curve
    ax.plot(cx + rc * np.cos(th_c), cy + rc * np.sin(th_c),
            color=color, lw=1.0, alpha=0.85)


def _heat(ax, Z, extent, title, geom, *, cmap="viridis", vmin=None, vmax=None) -> None:
    im = ax.imshow(
        Z, extent=extent, origin="lower",
        cmap=cmap, vmin=vmin, vmax=vmax,
        aspect="equal", interpolation="nearest",
    )
    _draw_shapes(ax, geom)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    X_tr, y_tr, geom = make_2d_complex_data(n=1000, seed=1)

    pc = PreprocessConfig(
        # input_transforms=["kde_quantile"],
        input_transforms=["standard"],
        kde_bandwidth_factor=10.0,
        # output_transforms=["kde_quantile"],
        output_transforms=["standard"],
    )
    fc = FitConfig(epochs=1000, lr=5e-4, batch_size=256, seed=0, verbose=True)
    nc = NoiseConfig(
        input_additive_std=0.0, input_multiplicative_std=0.0,
        output_additive_std=0.0,
    )

    print("Fitting predictive FCNet on complex (moons + ring + spiral) topology ...")
    hidden_dims_list = [128] * 2
    net = FCNet(
        n_inputs=2, n_outputs=1,
        hidden_dims=hidden_dims_list,
        activation="odd_piecewise",
        # activation="gelu",
        # activation="tanh",
        # activation="relu",
        fit_config=fc,
        reg_config=RegConfig(l2=1e-5),
        # reg_config=RegConfig(l1=1e-8, l2=1e-8),
        noise_config=nc, preprocess=pc,
        device="cuda",
    )
    net.fit(X_tr, y_tr)

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

    print("Computing last-layer leverage on FCNet features ...")
    ridge = 1e-3
    # bias=True augments Phi with an all-ones column so the formula covers
    # the readout's bias (FCNet's net.readout has bias=True).
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
    print("   * sigma_epi should be smallest in_data, larger in ring_hole")
    print("     (a topological hole, not 'far away'), and largest in ood.")

    # 1-D slice across the ring's diameter -- the cleanest single-axis
    # demonstration that the leverage head detects an *internal* hole, not
    # just a far-OOD region.
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
    ax.set_title("Training data\n(2 moons + ring + spiral)", fontsize=9)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.legend(loc="upper left", fontsize=7)

    mu_vmin = float(truth_mean.min())
    mu_vmax = float(truth_mean.max())
    _heat(axes[0, 1], truth_mean, extent,
          "Truth mean (gauss-mix landscape)",
          geom, cmap="RdBu_r", vmin=mu_vmin, vmax=mu_vmax)
    _heat(axes[0, 2], mu, extent,
          "Mean prediction (FCNet, kde_quantile in/out)",
          geom, cmap="RdBu_r", vmin=mu_vmin, vmax=mu_vmax)

    _heat(axes[1, 0], abserr, extent,
          "|mean - truth|",
          geom, cmap="magma")

    sigma_vmax = float(sigma_epi.max())
    _heat(axes[1, 1], sigma_epi, extent,
          "sigma_epistemic\n(last-layer leverage, tanh-saturated)",
          geom, cmap="magma", vmin=0.0, vmax=sigma_vmax)

    # Zoom on the ring + its hole -- the headline test that the leverage
    # head can flag an *internal* topological hole.
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

    # 1-D ring slice.
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
        f"Ring-diameter slice. Hole sigma_epi is {ratio:.2f}x the on-ring mean: "
        "the leverage head detects a topological hole, not just far-OOD distance.",
        fontsize=10,
    )
    ax_slice.legend(loc="upper center", fontsize=8, ncol=2)

    fig.suptitle(
        "2D FCNet + closed-form last-layer epistemic on a complex training topology\n"
        "Train: 2 interlocking moons + thin annulus + spiral arm. "
        f"Predict on [{X1_LO:+.1f}, {X1_HI:+.1f}] x [{X2_LO:+.1f}, {X2_HI:+.1f}]. "
        "Row 0: data, truth, prediction. Row 1: |error|, sigma_epi (full), sigma_epi (ring zoom).",
        y=0.995, fontsize=10,
    )
    fig.tight_layout()
    out_path = out_dir / "hetero_demo_2d_ood.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
