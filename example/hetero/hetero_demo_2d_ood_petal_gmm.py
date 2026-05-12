"""Deep z + GMM-mixed per-cluster leverage on the petal topology.

Sibling of ``hetero_demo_2d_ood_petal.py``. The original computes
last-layer leverage on output-RFF features -- a single global kernel
``Phi^T Phi + lam I`` over all training points. That covariance smooths
posterior variance *across* modes: in the petal topology, empty space
between two observed petals can look in-distribution because the global
Gram matrix mixes training points from every cluster into one
ellipsoid, so directions aligned with the petal-ring principal axis are
treated as well-supported.

Here we swap the output RFF for a Gaussian mixture model fit on the raw
penultimate features ``z = features(X_train)`` (``hidden_dims[-1]``,
*no* output RFF). We then build a *per-cluster* leverage diagonal
weighted by the GMM responsibilities ``gamma_nk``:

    h_k(x*) = phi(x*)^T (Phi^T diag(gamma_.k) Phi + lam I)^-1 phi(x*)
    h_mix(x*) = sum_k gamma_k(x*) * h_k(x*)

Soft assignment (responsibilities, not hard cluster labels) keeps
``h_mix`` continuous, while the per-cluster Gram matrix stops one
cluster's training points from "supporting" queries that live near a
different cluster. As a complementary signal we also report
``-log p(z*)`` under the fitted GMM -- a direct density-based energy.

We mark which spatial regions each GMM component owns by colouring grid
pixels (and training points) by their argmax responsibility, and by
back-projecting each component's centroid to ``(x1, x2)`` as a
responsibility-weighted average of training inputs.

Run from the repo root:

    python example/hetero/hetero_demo_2d_ood_petal_gmm.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from sklearn.mixture import GaussianMixture

from ebmify.models import (
    FCNet,
    FitConfig,
    NoiseConfig,
    PreprocessConfig,
    RegConfig,
)

from hetero_demo_2d_ood_petal import (
    _draw_shapes,
    _fmt_ell,
    _heat,
    classify_regions,
    make_petal_data,
    petal_landscape,
)


# ----------------------------------------------------------------------
# Per-cluster weighted leverage
# ----------------------------------------------------------------------

def weighted_feature_leverage(
    Phi_train: torch.Tensor,
    weights: torch.Tensor,
    Phi_query: torch.Tensor,
    ridge: float = 1e-3,
    *,
    bias: bool = True,
) -> torch.Tensor:
    """``phi*^T (Phi^T diag(w) Phi + r I)^-1 phi*`` with optional bias column.

    Mirrors :func:`ebmify.models.feature_leverage` but with a per-row
    weight vector on the Gram matrix. ``weights[n]`` is the
    responsibility of training point ``n`` for the cluster whose
    leverage we are computing. Setting ``weights=ones`` recovers the
    standard formula.
    """
    if bias:
        ones_tr = torch.ones(
            Phi_train.shape[0], 1,
            device=Phi_train.device, dtype=Phi_train.dtype,
        )
        ones_qy = torch.ones(
            Phi_query.shape[0], 1,
            device=Phi_query.device, dtype=Phi_query.dtype,
        )
        Phi_train = torch.cat([Phi_train, ones_tr], dim=1)
        Phi_query = torch.cat([Phi_query, ones_qy], dim=1)
    F = Phi_train.shape[1]
    pen = ridge * torch.eye(F, device=Phi_train.device, dtype=Phi_train.dtype)
    if bias:
        pen[-1, -1] = 0.0
    W_Phi = weights.unsqueeze(1) * Phi_train
    A = Phi_train.T @ W_Phi + pen
    sol = torch.linalg.solve(A, Phi_query.T)
    return (Phi_query * sol.T).sum(dim=1)


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
    fc = FitConfig(epochs=1000, lr=5e-4, batch_size=128, seed=0, verbose=True)
    nc = NoiseConfig(
        input_additive_std=0.0, input_multiplicative_std=0.0,
        output_additive_std=0.0,
    )

    M_in = None#32
    print(
        f"Fitting FCNet (input M={M_in}, residual MLP, NO output RFF) "
        f"on petal topology ..."
    )
    net = FCNet(
        n_inputs=2, n_outputs=1,
        hidden_dims=(256, 256,),
        # activation="odd_piecewise",
        activation="gelu",
        fit_config=fc, reg_config=RegConfig(l2=1e-8),
        noise_config=nc, preprocess=pc,
        input_rff=M_in, input_rff_length_scale=[0.15, 0.3, 0.6],
        output_rff=None,
        rff_seed=0,
    )
    net.fit(X_tr, y_tr)
    # ell_in = _fmt_ell(net.net.input_rff.length_scale)

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

    z_train = Phi_train.cpu().numpy()
    z_grid = Phi_grid.cpu().numpy()
    print(f"Penultimate features: z_train shape {z_train.shape}")

    # ------------------------------------------------------------------
    # Fit GMM on raw z. Diagonal covariance: cheap, plenty for 4 modes,
    # avoids brittle full-covariance fits at 64-dim with ~1500 points.
    # ------------------------------------------------------------------
    K = 5
    print(f"Fitting GMM (K={K}, diag covariance) on raw z ...")
    gmm = GaussianMixture(
        n_components=K, covariance_type="diag",
        reg_covar=1e-3, max_iter=200, n_init=4,
        random_state=0,
    )
    gmm.fit(z_train)

    gamma_train = gmm.predict_proba(z_train)  # [N, K]
    gamma_grid = gmm.predict_proba(z_grid)    # [G*G, K]
    logp_train = gmm.score_samples(z_train)   # [N]
    logp_grid = gmm.score_samples(z_grid)     # [G*G]

    # ------------------------------------------------------------------
    # Per-cluster weighted leverage. We compute h_k(x) for both train
    # and grid, then mix by responsibilities for the final signal.
    # ------------------------------------------------------------------
    print("Computing per-cluster weighted leverage ...")
    ridge = 1e-3
    gamma_train_t = torch.from_numpy(gamma_train.astype(np.float32))
    h_k_train = np.zeros((z_train.shape[0], K), dtype=np.float32)
    h_k_grid = np.zeros((z_grid.shape[0], K), dtype=np.float32)
    for k in range(K):
        w_k = gamma_train_t[:, k]
        h_k_train[:, k] = weighted_feature_leverage(
            Phi_train, w_k, Phi_train, ridge=ridge, bias=True,
        ).cpu().numpy()
        h_k_grid[:, k] = weighted_feature_leverage(
            Phi_train, w_k, Phi_grid, ridge=ridge, bias=True,
        ).cpu().numpy()

    h_mix_train = (gamma_train * h_k_train).sum(axis=1)
    h_mix_grid = (gamma_grid * h_k_grid).sum(axis=1).reshape(G, G)

    # Calibrate sigma_epi using the mixed leverage just like the original
    # demo: 95th percentile of train leverage sets a characteristic
    # scale, then squash with tanh into a noise-relative amplitude.
    h_char = float(np.percentile(h_mix_train, 95))
    sigma_noise = float(np.std(y_tr - mu_train))
    lam = sigma_noise / np.sqrt(max(h_char, 1e-12))
    sigma_max = 5.0 * sigma_noise
    sigma_epi = sigma_max * np.tanh(
        lam * np.sqrt(np.clip(h_mix_grid, 0.0, None)) / sigma_max
    )

    # GMM neg-log-density as an energy. Clip the extreme tails so the
    # heatmap stays readable -- far-OOD points produce -inf-ish logp.
    nll_grid = -logp_grid.reshape(G, G)
    nll_train_p95 = float(np.percentile(-logp_train, 95))
    nll_vmax = float(np.percentile(nll_grid, 99))
    nll_vmin = float(np.percentile(nll_grid, 1))

    # ------------------------------------------------------------------
    # Cluster assignments in (x1, x2) space.
    # ------------------------------------------------------------------
    cluster_train = gamma_train.argmax(axis=1)
    cluster_grid = gamma_grid.argmax(axis=1).reshape(G, G)
    # Responsibility-weighted centroid in input space per cluster.
    xy_centroids = np.zeros((K, 2), dtype=np.float32)
    for k in range(K):
        w = gamma_train[:, k]
        s = float(w.sum())
        if s > 0:
            xy_centroids[k] = (w[:, None] * X_tr).sum(axis=0) / s

    masks = classify_regions(x1g, x2g, geom)
    in_data = masks["in_data"]
    interior_hole = masks["interior_hole"]
    ood = masks["ood"]
    truth_mean = petal_landscape(x1g, x2g).astype(np.float32)
    abserr = np.abs(mu - truth_mean)

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
    _stat("h_mix", h_mix_grid)
    _stat("-log p(z*)", nll_grid)
    _stat("|mean - truth|", abserr)
    print("  Notes:")
    print(
        "   * h_mix mixes per-cluster leverages by GMM responsibility, so "
        "between-mode\n     queries are not 'supported' by a single global "
        "Gram matrix the way the\n     output-RFF leverage in "
        "hetero_demo_2d_ood_petal.py is."
    )
    print(
        f"   * 95th-pct -log p on train = {nll_train_p95:.3f}; OOD region "
        f"mean = {float(nll_grid[ood].mean()):.3f}."
    )

    # ------------------------------------------------------------------
    # Plot. 2x4 grid: data + cluster geometry on top row, model
    # prediction + uncertainty signals on bottom row.
    # ------------------------------------------------------------------
    out_dir = Path(__file__).resolve().parent.parent / "out" / "hetero"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))

    # Categorical colormap for clusters (K=4).
    base_colors = plt.cm.tab10(np.linspace(0, 1, 10))[:K]
    cluster_cmap = ListedColormap(base_colors)

    # [0,0] Training points, coloured by argmax GMM cluster.
    ax = axes[0, 0]
    for k in range(K):
        m = cluster_train == k
        ax.scatter(
            X_tr[m, 0], X_tr[m, 1], s=6, alpha=0.6,
            color=base_colors[k], label=f"cluster {k}",
        )
    _draw_shapes(ax, geom)
    ax.scatter(
        xy_centroids[:, 0], xy_centroids[:, 1],
        marker="*", s=180, c="black", edgecolors="white", linewidths=1.0,
        zorder=5, label="GMM centroid (x1,x2)",
    )
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal")
    ax.set_title(
        f"Training data + GMM (K={K})\n"
        f"colour = argmax cluster on z; stars = resp-weighted (x1,x2) means",
        fontsize=9,
    )
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.legend(loc="upper left", fontsize=6, framealpha=0.8)

    # [0,1] Cluster assignment map in (x1, x2).
    ax = axes[0, 1]
    im = ax.imshow(
        cluster_grid, extent=extent, origin="lower",
        cmap=cluster_cmap, vmin=-0.5, vmax=K - 0.5,
        aspect="equal", interpolation="nearest",
    )
    _draw_shapes(ax, geom)
    ax.scatter(
        xy_centroids[:, 0], xy_centroids[:, 1],
        marker="*", s=180, c="black", edgecolors="white", linewidths=1.0,
        zorder=5,
    )
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_title("Cluster argmax on grid\n(maps z-clusters to (x1,x2))",
                 fontsize=9)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        ticks=list(range(K)))
    cbar.set_label("cluster id")

    # [0,2] Truth.
    mu_vmin = float(truth_mean.min()); mu_vmax = float(truth_mean.max())
    _heat(axes[0, 2], truth_mean, extent,
          "Truth mean (cos(theta) * tanh(r) * exp(-r/3))",
          geom, cmap="RdBu_r", vmin=mu_vmin, vmax=mu_vmax)

    # [0,3] Mean prediction (note: no output RFF).
    _heat(axes[0, 3], mu, extent,
          f"Mean prediction (FCNet, no output RFF)\n"
          f"in M={M_in}", #ell_in={ell_in}",
          geom, cmap="RdBu_r", vmin=mu_vmin, vmax=mu_vmax)

    # [1,0] |mean - truth|.
    _heat(axes[1, 0], abserr, extent, "|mean - truth|", geom, cmap="magma")

    # [1,1] sigma_epi from mixed per-cluster leverage.
    sigma_vmax = float(sigma_epi.max())
    _heat(axes[1, 1], sigma_epi, extent,
          "sigma_epistemic\n(GMM-mixed per-cluster leverage)",
          geom, cmap="magma", vmin=0.0, vmax=sigma_vmax)

    # [1,2] raw h_mix(x).
    _heat(axes[1, 2], h_mix_grid, extent,
          "raw h_mix(x) = sum_k gamma_k(x) h_k(x)",
          geom, cmap="magma", vmin=0.0,
          vmax=float(np.percentile(h_mix_grid, 99)))

    # [1,3] -log p(z*) under the GMM.
    _heat(axes[1, 3], nll_grid, extent,
          "-log p(z*) under GMM\n(density-based energy)",
          geom, cmap="magma", vmin=nll_vmin, vmax=nll_vmax)

    fig.suptitle(
        f"2D GMM-mixed leverage on petal topology: 1 + {n_petals} Gaussians, "
        f"only {n_obs} observed.\n"
        f"Penultimate z (no output RFF) -> GMM(K={K}) -> per-cluster "
        f"weighted leverage mixed by responsibility. "
        f"radius={radius}, sigma={sigma}.",
        y=0.995, fontsize=10,
    )
    fig.tight_layout()
    out_path = out_dir / "hetero_demo_2d_ood_petal_gmm.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
