"""Deep RFF + last-layer leverage on a Gaussian-petal topology.

We sample 2D points from K = 1 + n_petals isotropic Gaussians: one at the
origin and ``n_petals`` arranged uniformly on a circle of radius ``R``.
Only a *subset* of the K clusters is observed; the rest become interior
holes -- compact regions of high leverage embedded in the otherwise
populated petal envelope. The leverage head should light up sharply on
the unobserved petals and on the empty space between clusters.

Run from the repo root:

    python example/hetero/hetero_demo_2d_ood_petal.py
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
# Geometry
# ----------------------------------------------------------------------

def petal_centers(
    n_petals: int, radius: float, include_center: bool = True,
) -> np.ndarray:
    """Centers indexed [center?, petal_0, ..., petal_{n_petals-1}].

    Petal ``k`` sits at angle ``2 pi k / n_petals``, so petal 0 is on
    the +x1 axis.
    """
    centers: list[list[float]] = []
    if include_center:
        centers.append([0.0, 0.0])
    for k in range(n_petals):
        theta = 2.0 * np.pi * k / n_petals
        centers.append([radius * np.cos(theta), radius * np.sin(theta)])
    return np.asarray(centers, dtype=np.float32)


def petal_landscape(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Angular truth with a smooth-at-origin extension.

    ``f(x) = cos(theta) * tanh(r) * exp(-r/3) = (x1 / r) * tanh(r) * exp(-r/3)``.

    The naive ``cos(theta) * exp(-r/3)`` form is genuinely discontinuous
    at the origin -- as ``r -> 0`` from different directions, the limit
    sweeps ``cos(theta)`` across ``[-1, +1]``. No smooth function
    approximator can fit that, so the centre cluster (which sits on the
    singularity) ends up looking weird in the prediction map. The
    ``tanh(r)`` factor smoothly suppresses the angular factor near the
    origin: since ``(x1 / r) * tanh(r) -> x1`` as ``r -> 0``, the
    composite is smooth (C-infinity), evaluates to 0 at the origin, and
    keeps the angular variation around the petal ring.
    """
    r = np.sqrt(x1 ** 2 + x2 ** 2)
    safe_r = np.maximum(r, 1e-12)
    cos_theta = x1 / safe_r
    return (cos_theta * np.tanh(r) * np.exp(-r / 3.0)).astype(x1.dtype)


def make_petal_data(
    n: int = 500, seed: int = 1, n_petals: int = 8,
    radius: float = 3.5, sigma: float = 0.4,
    observed_idx: tuple[int, ...] | None = None,
    include_center: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sample ``n`` points from the *observed* clusters only.

    If ``observed_idx`` is None, the default mask is asymmetric:
    centre (if any) + petals ``{0, 1, 3, 6}`` (out of 8) -- two adjacent
    on the right, isolated petals upper-left and bottom. Picking a
    non-symmetric subset makes the visualization more legible than the
    every-other pattern.
    """
    rng = np.random.default_rng(seed)
    centers = petal_centers(n_petals, radius, include_center=include_center)
    K = len(centers)

    if observed_idx is None:
        # Default asymmetric mask: assumes the standard 8-petal layout.
        # For other ``n_petals``, fall back to "every other petal" so the
        # function still produces something sensible.
        obs: list[int] = []
        if include_center:
            obs.append(0)
            petal_offset = 1
        else:
            petal_offset = 0
        if n_petals == 8:
            obs.extend(petal_offset + k for k in (0, 1, 3, 6))
        else:
            obs.extend(petal_offset + k for k in range(0, n_petals, 2))
        observed_idx = tuple(obs)

    observed_mask = np.zeros(K, dtype=bool)
    observed_mask[list(observed_idx)] = True

    obs_centers = centers[observed_mask]
    n_obs = len(obs_centers)
    n_per = n // n_obs
    extra = n - n_per * n_obs

    pts: list[np.ndarray] = []
    for k, c in enumerate(obs_centers):
        m = n_per + (1 if k < extra else 0)
        local = rng.normal(loc=c, scale=sigma, size=(m, 2)).astype(np.float32)
        pts.append(local)
    X = np.concatenate(pts, axis=0)
    rng.shuffle(X)

    geom = {
        "centers": centers,
        "observed_mask": observed_mask,
        "sigma": float(sigma),
        "radius": float(radius),
        "n_petals": int(n_petals),
        "include_center": bool(include_center),
    }
    y = petal_landscape(X[:, :1], X[:, 1:2]).astype(np.float32)
    return X, y, geom


# ----------------------------------------------------------------------
# Region masks: each grid pixel is in an observed cluster ("in_data"),
# in an unobserved cluster ("interior_hole"), or off-board (far OOD).
# Membership uses a soft cluster radius = ``threshold_factor * sigma``
# around each Gaussian centre.
# ----------------------------------------------------------------------

def classify_regions(
    x1: np.ndarray, x2: np.ndarray, geom: dict,
    threshold_factor: float = 2.5,
) -> dict:
    centers = geom["centers"]
    observed = geom["observed_mask"]
    sigma = float(geom["sigma"])
    threshold = threshold_factor * sigma

    pts = np.stack([x1.ravel(), x2.ravel()], axis=1).astype(np.float32)
    dists = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=-1)
    nearest = np.argmin(dists, axis=1)
    nearest_dist = dists[np.arange(len(pts)), nearest]
    nearest_observed = observed[nearest]
    in_cluster = nearest_dist < threshold

    in_data = (in_cluster & nearest_observed).reshape(x1.shape)
    interior_hole = (in_cluster & ~nearest_observed).reshape(x1.shape)
    ood = (~in_cluster).reshape(x1.shape)
    return {"in_data": in_data, "interior_hole": interior_hole, "ood": ood}


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------

def _draw_shapes(
    ax, geom: dict, *, observed_color: str = "orange",
    unobserved_color: str = "tab:gray", threshold_factor: float = 2.0,
) -> None:
    centers = geom["centers"]
    observed_mask = geom["observed_mask"]
    sigma = float(geom["sigma"])
    r_circle = threshold_factor * sigma
    for c, obs in zip(centers, observed_mask):
        color = observed_color if obs else unobserved_color
        ls = "-" if obs else "--"
        ax.add_patch(plt.Circle(
            (float(c[0]), float(c[1])), r_circle, fill=False,
            edgecolor=color, lw=0.9, ls=ls, alpha=0.9,
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
    n_petals = 8
    radius = 3.5
    sigma = 0.4
    X_tr, y_tr, geom = make_petal_data(
        n=1500, seed=1, n_petals=n_petals, radius=radius, sigma=sigma,
    )
    n_obs = int(geom["observed_mask"].sum())
    n_unobs = len(geom["centers"]) - n_obs
    print(
        f"Dataset: {len(X_tr)} pts from {n_obs} observed clusters; "
        f"{n_unobs} clusters held out as interior holes."
    )

    pc = PreprocessConfig(
        input_transforms=["standard"], output_transforms=["standard"],
    )
    fc = FitConfig(epochs=1500, lr=5e-4, batch_size=128, seed=0, verbose=True)
    nc = NoiseConfig(
        input_additive_std=0.0, input_multiplicative_std=0.0,
        output_additive_std=0.0,
    )

    M_in, M_out = 32, 64
    print(
        f"Fitting deep RFF FCNet (input M={M_in}, residual MLP, "
        f"output M={M_out}) on petal topology ..."
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

    G = 160
    PAD = 1.0
    L = radius + 2.0 * sigma + PAD
    grid_x1 = np.linspace(-L, L, G).astype(np.float32)
    grid_x2 = np.linspace(-L, L, G).astype(np.float32)
    x1g, x2g = np.meshgrid(grid_x1, grid_x2)
    X_grid = np.stack([x1g.ravel(), x2g.ravel()], axis=1)
    extent = (-L, L, -L, L)

    with torch.no_grad():
        Phi_train = net.features(X_tr)
        Phi_grid = net.features(X_grid)
        mu_train = net.predict(X_tr).cpu().numpy()
        mu = net.predict(X_grid).cpu().numpy()[:, 0].reshape(G, G)

    print("Computing last-layer leverage on output-RFF features ...")
    ridge = 1e-3
    h_train = feature_leverage(Phi_train, Phi_train, ridge=ridge,
                                bias=True).cpu().numpy()
    h_grid = feature_leverage(Phi_train, Phi_grid, ridge=ridge,
                               bias=True).cpu().numpy().reshape(G, G)
    h_char = float(np.percentile(h_train, 95))
    sigma_noise = float(np.std(y_tr - mu_train))
    lam = sigma_noise / np.sqrt(max(h_char, 1e-12))
    sigma_max = 5.0 * sigma_noise
    sigma_epi = sigma_max * np.tanh(
        lam * np.sqrt(np.clip(h_grid, 0.0, None)) / sigma_max
    )

    truth_mean = petal_landscape(x1g, x2g).astype(np.float32)
    abserr = np.abs(mu - truth_mean)

    masks = classify_regions(x1g, x2g, geom)
    in_data = masks["in_data"]
    interior_hole = masks["interior_hole"]
    ood = masks["ood"]

    def _stat(name: str, arr: np.ndarray) -> None:
        print(
            f"  {name:18s}  in_data={float(arr[in_data].mean()):.4f}  "
            f"interior_hole={float(arr[interior_hole].mean()):.4f}  "
            f"ood={float(arr[ood].mean()):.4f}"
        )

    print(
        f"\nGrid cells per region: in_data={int(in_data.sum())}, "
        f"interior_hole={int(interior_hole.sum())}, ood={int(ood.sum())}"
    )
    print("\nMean over region:")
    _stat("sigma_epistemic", sigma_epi)
    _stat("|mean - truth|", abserr)
    print("  Notes:")
    print("   * Unobserved petals (dashed circles) are surrounded by observed")
    print("     neighbours but contain zero training data -- the leverage head")
    print("     should rise sharply inside them.")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    out_dir = Path(__file__).resolve().parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    ax = axes[0, 0]
    ax.scatter(X_tr[:, 0], X_tr[:, 1], s=4, alpha=0.5,
               color="tab:gray", label="train")
    _draw_shapes(ax, geom)
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal")
    ax.set_title(
        f"Training data\n(observed: {n_obs} clusters, unobserved: {n_unobs})",
        fontsize=9,
    )
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.legend(loc="upper left", fontsize=7)

    mu_vmin = float(truth_mean.min()); mu_vmax = float(truth_mean.max())
    _heat(axes[0, 1], truth_mean, extent,
          "Truth mean (cos(theta) * tanh(r) * exp(-r/3))",
          geom, cmap="RdBu_r", vmin=mu_vmin, vmax=mu_vmax)
    _heat(axes[0, 2], mu, extent,
          f"Mean prediction (deep RFF)\n"
          f"in M={M_in} ell_in={ell_in} | out M={M_out} ell_out={ell_out}",
          geom, cmap="RdBu_r", vmin=mu_vmin, vmax=mu_vmax)

    _heat(axes[1, 0], abserr, extent, "|mean - truth|", geom, cmap="magma")
    sigma_vmax = float(sigma_epi.max())
    _heat(axes[1, 1], sigma_epi, extent,
          "sigma_epistemic\n(last-layer leverage)",
          geom, cmap="magma", vmin=0.0, vmax=sigma_vmax)
    _heat(axes[1, 2], h_grid, extent,
          "raw leverage h(x)",
          geom, cmap="magma", vmin=0.0,
          vmax=float(np.percentile(h_grid, 99)))

    fig.suptitle(
        f"2D deep RFF on petal topology: 1 + {n_petals} Gaussians, "
        f"only {n_obs} observed.\n"
        f"Solid orange: observed clusters. Dashed gray: unobserved (interior holes). "
        f"radius={radius}, sigma={sigma}.",
        y=0.995, fontsize=10,
    )
    fig.tight_layout()
    out_path = out_dir / "hetero_demo_2d_ood_petal.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
