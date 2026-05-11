"""Annealed Langevin on the petal-topology leverage energy.

Same dynamics as ``hetero_demo_2d_ood_checkerboard_langevin.py`` -- the
Langevin update, the geometric temperature schedule, the leverage-energy
construction are all reused. Only the data topology changes: a central
Gaussian + ``n_petals`` peripheral Gaussians, with only a subset
observed.

We run two batches of particles:

* ``off_board`` -- spawned uniformly in the bounding square, rejected if
  too close to *any* cluster centre, so they start in the empty space
  outside the petal envelope.
* ``in_gap`` -- spawned inside the *unobserved* clusters (the dashed
  circles) with a small Gaussian jitter, so they sit on a leverage
  plateau bounded on all sides by data-bearing observed clusters.

Run from the repo root:

    python example/hetero/hetero_demo_2d_ood_petal_langevin.py
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
)

from hetero_demo_2d_ood_checkerboard_langevin import (
    build_leverage_energy,
    geometric_anneal,
    langevin_sample,
)
from hetero_demo_2d_ood_petal import (
    make_petal_data,
    _heat,
)


# ----------------------------------------------------------------------
# Particle initializers (petal-specific)
# ----------------------------------------------------------------------

def init_off_board(
    n_part: int, geom: dict, rng: np.random.Generator,
    x1_lo: float, x1_hi: float, x2_lo: float, x2_hi: float,
    threshold_factor: float = 2.5,
) -> np.ndarray:
    """Sample uniformly in the bbox, reject points within
    ``threshold_factor * sigma`` of *any* cluster centre."""
    centers = geom["centers"]
    sigma = float(geom["sigma"])
    threshold = threshold_factor * sigma
    pts: list[np.ndarray] = []
    while len(pts) < n_part:
        cand = rng.uniform(
            low=[x1_lo + 0.2, x2_lo + 0.2],
            high=[x1_hi - 0.2, x2_hi - 0.2],
            size=(n_part * 8, 2),
        )
        dists = np.linalg.norm(
            cand[:, None, :] - centers[None, :, :], axis=-1,
        )
        outside = dists.min(axis=1) > threshold
        for p in cand[outside]:
            pts.append(p)
            if len(pts) == n_part:
                break
    return np.asarray(pts, dtype=np.float32)


def init_in_gap(
    n_part: int, geom: dict, rng: np.random.Generator,
    jitter_scale: float = 0.4,
) -> np.ndarray:
    """Sample inside the *unobserved* clusters with a small Gaussian jitter.

    ``jitter_scale`` is in units of the cluster sigma; 0.4 places
    particles well inside the cluster, comfortably below the saddle to
    the nearest observed neighbour.
    """
    centers = geom["centers"]
    observed = geom["observed_mask"]
    sigma = float(geom["sigma"])
    unobs = centers[~observed]
    if len(unobs) == 0:
        raise ValueError("no unobserved clusters")
    idx = rng.integers(0, len(unobs), size=n_part)
    pts = (
        unobs[idx]
        + rng.normal(0.0, jitter_scale * sigma, size=(n_part, 2)).astype(np.float32)
    )
    return pts.astype(np.float32)


def classify_final(
    x: np.ndarray, geom: dict, threshold_factor: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = geom["centers"]
    observed = geom["observed_mask"]
    sigma = float(geom["sigma"])
    threshold = threshold_factor * sigma
    dists = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=-1)
    nearest = np.argmin(dists, axis=1)
    nearest_dist = dists[np.arange(len(x)), nearest]
    in_cluster = nearest_dist < threshold
    nearest_observed = observed[nearest]
    return (
        in_cluster & nearest_observed,    # on observed cluster
        in_cluster & ~nearest_observed,   # on unobserved cluster
        ~in_cluster,                       # off-board
    )


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
        f"{n_unobs} clusters held out."
    )

    pc = PreprocessConfig(
        input_transforms=["standard"], output_transforms=["standard"],
    )
    fc = FitConfig(epochs=1000, lr=5e-4, batch_size=128, seed=0, verbose=True)
    nc = NoiseConfig(
        input_additive_std=0.0, input_multiplicative_std=0.0,
        output_additive_std=0.0,
    )

    M_in, M_out = 32, 64
    print(f"Fitting deep RFF FCNet on petal topology (M_in={M_in}, M_out={M_out}) ...")
    net = FCNet(
        n_inputs=2, n_outputs=1,
        hidden_dims=(64, 64),
        activation="odd_piecewise",
        fit_config=fc, reg_config=RegConfig(l2=1e-5),
        noise_config=nc, preprocess=pc,
        input_rff=M_in, input_rff_length_scale=[1.0],
        output_rff=M_out, output_rff_length_scale=[0.1],
        rff_seed=0,
    )
    net.fit(X_tr, y_tr)
    net.eval()
    print("  fit complete.")

    energy_fn, raw_h_fn, h_char = build_leverage_energy(
        net, X_tr, ridge=1e-3, bias=True,
    )
    print(f"  h_char (95th percentile of training leverage) = {h_char:.4e}")

    G = 160
    PAD = 1.0
    L = radius + 2.0 * sigma + PAD
    X1_LO, X1_HI = -L, L
    X2_LO, X2_HI = -L, L
    grid_x1 = np.linspace(X1_LO, X1_HI, G).astype(np.float32)
    grid_x2 = np.linspace(X2_LO, X2_HI, G).astype(np.float32)
    x1g, x2g = np.meshgrid(grid_x1, grid_x2)
    X_grid = np.stack([x1g.ravel(), x2g.ravel()], axis=1)
    extent = (X1_LO, X1_HI, X2_LO, X2_HI)

    with torch.no_grad():
        X_grid_t = torch.as_tensor(X_grid, dtype=torch.float32, device=net.device)
        h_grid = raw_h_fn(X_grid_t).cpu().numpy().reshape(G, G)
        mu_train = net.predict(X_tr).cpu().numpy()

    sigma_noise = float(np.std(y_tr - mu_train))
    lam = sigma_noise / np.sqrt(max(h_char, 1e-12))
    sigma_max = 5.0 * sigma_noise
    sigma_epi_grid = sigma_max * np.tanh(
        lam * np.sqrt(np.clip(h_grid, 0.0, None)) / sigma_max
    )
    sigma_vmax = float(sigma_epi_grid.max())
    print(f"  sigma_noise = {sigma_noise:.4f}, sigma_max = {sigma_max:.4f}")

    rng = np.random.default_rng(0)
    n_part = 64
    x0_off = init_off_board(
        n_part, geom, rng, X1_LO, X1_HI, X2_LO, X2_HI,
        threshold_factor=2.5,
    )
    x0_gap = init_in_gap(n_part, geom, rng, jitter_scale=0.4)

    eta = 0.05
    n_steps = 1200
    # Larger T_hi than the checkerboard variant: the petal off-board region
    # is a wide empty annulus and a few stragglers got stuck in shallow
    # corners with the gentler 0.05 -> 0.001 schedule.
    T_hi, T_lo = 0.15, 0.001
    temperature = geometric_anneal(T_hi, T_lo, n_steps)

    print(
        f"\nRunning Langevin (off-board start, n={n_part}, eta={eta}, "
        f"T:{T_hi}->{T_lo} (geom), steps={n_steps}) ..."
    )
    x_final_off_t, traj_off, energies_off, temps = langevin_sample(
        energy_fn,
        torch.as_tensor(x0_off, dtype=torch.float32, device=net.device),
        n_steps=n_steps, eta=eta, temperature=temperature,
        grad_clip=5.0, record_every=5,
    )
    x_final_off = x_final_off_t.numpy()

    print(
        f"Running Langevin (in-gap start, n={n_part}, eta={eta}, "
        f"T:{T_hi}->{T_lo} (geom), steps={n_steps}) ..."
    )
    x_final_gap_t, traj_gap, energies_gap, _ = langevin_sample(
        energy_fn,
        torch.as_tensor(x0_gap, dtype=torch.float32, device=net.device),
        n_steps=n_steps, eta=eta, temperature=temperature,
        grad_clip=5.0, record_every=5,
    )
    x_final_gap = x_final_gap_t.numpy()

    on_obs_off, on_unobs_off, off_off = classify_final(x_final_off, geom)
    on_obs_gap, on_unobs_gap, off_gap = classify_final(x_final_gap, geom)

    def _report(label: str, w: np.ndarray, b: np.ndarray, o: np.ndarray) -> None:
        print(
            f"  [{label}] observed={int(w.sum())}/{n_part} ({100*w.mean():.1f}%)  "
            f"unobserved={int(b.sum())}/{n_part} ({100*b.mean():.1f}%)  "
            f"off-board={int(o.sum())}/{n_part} ({100*o.mean():.1f}%)"
        )

    print(f"\nFinal positions after {n_steps} Langevin steps:")
    _report("off-board start", on_obs_off, on_unobs_off, off_off)
    _report("in-gap   start ", on_obs_gap, on_unobs_gap, off_gap)
    print("  (Off-board particles must drift in across the empty annulus to")
    print("   reach a cluster. In-gap particles need to escape the unobserved")
    print("   cluster's leverage plateau and land on a neighbouring observed one.)")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    out_dir = Path(__file__).resolve().parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.7])
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_off = fig.add_subplot(gs[0, 1])
    ax_gap = fig.add_subplot(gs[0, 2])
    ax_curve = fig.add_subplot(gs[1, :])

    _heat(ax_heat, sigma_epi_grid, extent,
          "sigma_epi (training data overlay)\n"
          "solid: observed cluster | dashed: unobserved (interior hole)",
          geom, cmap="magma", vmin=0.0, vmax=sigma_vmax)
    ax_heat.scatter(X_tr[:, 0], X_tr[:, 1], s=2, alpha=0.4,
                     color="tab:cyan", label="train")
    ax_heat.legend(loc="upper right", fontsize=7)

    def _plot_run(ax, title, x0, traj, x_final, w, b, o, start_label, start_marker):
        _heat(ax, sigma_epi_grid, extent, title, geom, cmap="magma",
              vmin=0.0, vmax=sigma_vmax)
        for k in range(traj.shape[1]):
            path = traj[:, k, :]
            ax.plot(path[:, 0], path[:, 1], color="tab:cyan",
                    lw=0.5, alpha=0.45)
        ax.scatter(x0[:, 0], x0[:, 1], s=18, color="white",
                    edgecolor="black", lw=0.5, marker=start_marker,
                    label=start_label, zorder=5)
        if w.any():
            ax.scatter(x_final[w, 0], x_final[w, 1], s=22,
                        color="tab:green", edgecolor="black", lw=0.4,
                        label=f"end on observed ({int(w.sum())})", zorder=6)
        if b.any():
            ax.scatter(x_final[b, 0], x_final[b, 1], s=22,
                        color="tab:orange", edgecolor="black", lw=0.4,
                        label=f"end on unobserved ({int(b.sum())})", zorder=6)
        if o.any():
            ax.scatter(x_final[o, 0], x_final[o, 1], s=22,
                        color="tab:red", edgecolor="black", lw=0.4,
                        label=f"end off-board ({int(o.sum())})", zorder=6)
        ax.legend(loc="upper right", fontsize=6)

    sched_str = f"T:{T_hi:.3g}->{T_lo:.3g} (geom), eta={eta}, steps={n_steps}"
    _plot_run(
        ax_off,
        f"Off-board start -> annealed Langevin\n({sched_str})",
        x0_off, traj_off, x_final_off, on_obs_off, on_unobs_off, off_off,
        "start (off-board)", "s",
    )
    _plot_run(
        ax_gap,
        f"In-gap start -> annealed Langevin\n({sched_str})",
        x0_gap, traj_gap, x_final_gap, on_obs_gap, on_unobs_gap, off_gap,
        "start (in gap)", "D",
    )

    record_steps = energies_off.shape[0]
    step_axis = np.arange(record_steps) * 5
    ax_curve.plot(step_axis, energies_off, color="tab:blue", lw=0.4, alpha=0.20)
    ax_curve.plot(step_axis, energies_gap, color="tab:orange", lw=0.4, alpha=0.20)
    ax_curve.plot(step_axis, energies_off.mean(axis=1), color="tab:blue",
                   lw=2.0, label="off-board start (mean E)")
    ax_curve.plot(step_axis, energies_gap.mean(axis=1), color="tab:orange",
                   lw=2.0, label="in-gap start (mean E)")
    ax_curve.axhline(1.0, color="tab:gray", ls=":", lw=0.8,
                      label="E = 1 (training-leverage 95th pct)")
    ax_curve.set_xlabel("Langevin step")
    ax_curve.set_ylabel("Energy h(x) / h_char  (descended quantity)")
    ax_curve.set_yscale("log")
    ax_curve.set_title(
        "Per-particle energy descent under the annealed schedule. "
        "Dashed green: temperature T(t) on the right axis.",
        fontsize=10,
    )

    ax_temp = ax_curve.twinx()
    ax_temp.plot(np.arange(len(temps)), temps, color="tab:green", lw=1.5,
                  ls="--", label="T(t)")
    ax_temp.set_ylabel("Temperature T(t)", color="tab:green")
    ax_temp.set_yscale("log")
    ax_temp.tick_params(axis="y", colors="tab:green")

    lines, labels = ax_curve.get_legend_handles_labels()
    lines2, labels2 = ax_temp.get_legend_handles_labels()
    ax_curve.legend(lines + lines2, labels + labels2,
                     loc="upper right", fontsize=8)

    fig.suptitle(
        f"Annealed Langevin on petal topology "
        f"(1 + {n_petals} Gaussians, only {n_obs} observed).\n"
        "Off-board start (empty annulus) vs in-gap start (inside dashed clusters). "
        "Heatmap is sigma_epi from the leverage head.",
        y=0.995, fontsize=11,
    )
    fig.tight_layout()
    out_path = out_dir / "hetero_demo_2d_ood_petal_langevin.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
