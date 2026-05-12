"""Annealed Langevin on the deep-RFF leverage energy.

Same dynamics as ``hetero_demo_2d_ood_checkerboard_langevin.py`` and
``hetero_demo_2d_ood_petal_langevin.py`` -- the Langevin update,
``geometric_anneal``, and ``build_leverage_energy`` are reused. Only
the data topology and the model trunk change: the FCNet here is the
same deep RFF (input-RFF + residual MLP + output-RFF) used in
``hetero_demo_2d_ood_deep_rff.py``, fit on the 2-moons + ring + spiral
landscape.

What this stresses:

* The ring's empty inner disc is a *topological* hole bounded on all
  sides by training data. Particles spawned inside it sit on a
  leverage plateau with low local gradient -- they must diffuse
  thermally across the annulus saddle to escape.
* Far-OOD ``off_data`` particles need to drift across a wide low-data
  region toward one of the four training shapes; the bounded
  output-RFF features keep the leverage gradient well-behaved out
  there (the whole reason we use deep RFF on this topology).

We run two batches of particles:

* ``off_data`` -- spawned uniformly in the bounding rectangle and
  rejected if they land inside any training shape or inside the ring
  hole, so they start in the far-OOD region surrounding the data.
* ``in_hole`` -- spawned uniformly inside the ring's inner disc with
  a small margin off the annulus, so they sit on the topological-hole
  plateau bounded by the annulus on all sides.

Run from the repo root:

    python example/hetero/hetero_demo_2d_ood_deep_rff_langevin.py
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

from hetero_demo_2d_ood import (
    make_2d_complex_data,
    classify_regions,
    _heat,
)
from hetero_demo_2d_ood_checkerboard_langevin import (
    build_leverage_energy,
    geometric_anneal,
    langevin_sample,
)


# ----------------------------------------------------------------------
# Particle initializers (complex-topology-specific)
# ----------------------------------------------------------------------

def init_off_data(
    n_part: int, geom: dict, rng: np.random.Generator,
    x1_lo: float, x1_hi: float, x2_lo: float, x2_hi: float,
) -> np.ndarray:
    """Uniform in the bbox, rejected if inside any training shape or the
    ring's inner disc -- so all survivors start in the far-OOD region."""
    pts: list[np.ndarray] = []
    while len(pts) < n_part:
        cand = rng.uniform(
            low=[x1_lo + 0.1, x2_lo + 0.1],
            high=[x1_hi - 0.1, x2_hi - 0.1],
            size=(n_part * 8, 2),
        ).astype(np.float32)
        masks = classify_regions(cand[:, 0], cand[:, 1], geom)
        ok = masks["ood"]
        for p in cand[ok]:
            pts.append(p)
            if len(pts) == n_part:
                break
    return np.asarray(pts, dtype=np.float32)


def init_in_hole(
    n_part: int, geom: dict, rng: np.random.Generator,
    margin: float = 0.15,
) -> np.ndarray:
    """Uniform inside the ring's inner disc with a small inward margin so
    we start on the plateau, not right against the annulus.
    """
    cx, cy = geom["ring"]["center"]
    r_inner = float(geom["ring"]["r_inner"])
    r_safe = max(r_inner - margin, 0.0)
    # Uniform disc sampling: r = R * sqrt(u), theta uniform.
    u = rng.uniform(0.0, 1.0, size=n_part)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_part)
    r = r_safe * np.sqrt(u)
    x1 = cx + r * np.cos(theta)
    x2 = cy + r * np.sin(theta)
    return np.stack([x1, x2], axis=1).astype(np.float32)


def classify_final(
    x: np.ndarray, geom: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(on_data, in_hole, off_data)`` masks for final particle
    positions."""
    masks = classify_regions(x[:, 0], x[:, 1], geom)
    return masks["in_data"], masks["ring_hole"], masks["ood"]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    X_tr, y_tr, geom = make_2d_complex_data(n=1000, seed=1)
    print(f"Dataset: {len(X_tr)} pts (2 moons + ring + spiral).")

    pc = PreprocessConfig(
        # input_transforms=["kde_quantile"],
        input_transforms=["standard"],
        # kde_bandwidth_factor=10.0,
        # output_transforms=["kde_quantile"],
        output_transforms=["standard"],
    )
    fc = FitConfig(epochs=1000, lr=5e-4, batch_size=128, seed=0, verbose=True)
    nc = NoiseConfig(
        input_additive_std=0.0, input_multiplicative_std=0.0,
        output_additive_std=0.0,
    )

    M_in, M_out = None, None
    print(
        f"Fitting deep RFF FCNet (residual MLP) on complex topology ..."
    )
    net = FCNet(
        n_inputs=2, n_outputs=1,
        hidden_dims=(32, 32, 32, 32),
        activation="odd_piecewise",
        fit_config=fc, reg_config=RegConfig(l2=1e-5),
        noise_config=nc, preprocess=pc,
        input_rff=M_in, input_rff_length_scale=[0.25, 0.5, 1.0],
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
    X1_LO, X1_HI = -5.5, 6.0
    X2_LO, X2_HI = -4.0, 4.0
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
    x0_off = init_off_data(
        n_part, geom, rng, X1_LO, X1_HI, X2_LO, X2_HI,
    )
    x0_hole = init_in_hole(n_part, geom, rng, margin=0.15)

    eta = 0.05
    n_steps = 1200
    # Wider OOD region than the petal demo (the bbox is bigger than the
    # ring's outer radius), so we use the same hotter starting T as petal
    # to give off-data particles enough early thermal kick to migrate
    # toward a shape.
    T_hi, T_lo = 0.3, 0.01
    temperature = geometric_anneal(T_hi, T_lo, n_steps)

    print(
        f"\nRunning Langevin (off-data start, n={n_part}, eta={eta}, "
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
        f"Running Langevin (in-hole start, n={n_part}, eta={eta}, "
        f"T:{T_hi}->{T_lo} (geom), steps={n_steps}) ..."
    )
    x_final_hole_t, traj_hole, energies_hole, _ = langevin_sample(
        energy_fn,
        torch.as_tensor(x0_hole, dtype=torch.float32, device=net.device),
        n_steps=n_steps, eta=eta, temperature=temperature,
        grad_clip=5.0, record_every=5,
    )
    x_final_hole = x_final_hole_t.numpy()

    on_data_off, in_hole_off, off_off = classify_final(x_final_off, geom)
    on_data_hole, in_hole_hole, off_hole = classify_final(x_final_hole, geom)

    def _report(label: str, d: np.ndarray, h: np.ndarray, o: np.ndarray) -> None:
        print(
            f"  [{label}] on-data={int(d.sum())}/{n_part} ({100*d.mean():.1f}%)  "
            f"in-hole={int(h.sum())}/{n_part} ({100*h.mean():.1f}%)  "
            f"off-data={int(o.sum())}/{n_part} ({100*o.mean():.1f}%)"
        )

    print(f"\nFinal positions after {n_steps} Langevin steps:")
    _report("off-data start", on_data_off, in_hole_off, off_off)
    _report("in-hole start ", on_data_hole, in_hole_hole, off_hole)
    print("  (Off-data particles drift across the empty OOD region to reach a")
    print("   training shape. In-hole particles need thermal diffusion to escape")
    print("   the ring's interior plateau and cross the annulus saddle.)")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    out_dir = Path(__file__).resolve().parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.7])
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_off = fig.add_subplot(gs[0, 1])
    ax_hole = fig.add_subplot(gs[0, 2])
    ax_curve = fig.add_subplot(gs[1, :])

    _heat(ax_heat, sigma_epi_grid, extent,
          "sigma_epi (training data overlay)\n"
          "shape outlines: training manifolds | inner ring circle: topological hole",
          geom, cmap="magma", vmin=0.0, vmax=sigma_vmax)
    ax_heat.scatter(X_tr[:, 0], X_tr[:, 1], s=2, alpha=0.4,
                    color="tab:cyan", label="train")
    ax_heat.legend(loc="upper right", fontsize=7)

    def _plot_run(ax, title, x0, traj, x_final, d, h, o, start_label, start_marker):
        _heat(ax, sigma_epi_grid, extent, title, geom, cmap="magma",
              vmin=0.0, vmax=sigma_vmax)
        for k in range(traj.shape[1]):
            path = traj[:, k, :]
            ax.plot(path[:, 0], path[:, 1], color="tab:cyan",
                    lw=0.5, alpha=0.45)
        ax.scatter(x0[:, 0], x0[:, 1], s=18, color="white",
                   edgecolor="black", lw=0.5, marker=start_marker,
                   label=start_label, zorder=5)
        if d.any():
            ax.scatter(x_final[d, 0], x_final[d, 1], s=22,
                       color="tab:green", edgecolor="black", lw=0.4,
                       label=f"end on data ({int(d.sum())})", zorder=6)
        if h.any():
            ax.scatter(x_final[h, 0], x_final[h, 1], s=22,
                       color="tab:orange", edgecolor="black", lw=0.4,
                       label=f"end in hole ({int(h.sum())})", zorder=6)
        if o.any():
            ax.scatter(x_final[o, 0], x_final[o, 1], s=22,
                       color="tab:red", edgecolor="black", lw=0.4,
                       label=f"end off-data ({int(o.sum())})", zorder=6)
        ax.legend(loc="upper right", fontsize=6)

    sched_str = f"T:{T_hi:.3g}->{T_lo:.3g} (geom), eta={eta}, steps={n_steps}"
    _plot_run(
        ax_off,
        f"Off-data start -> annealed Langevin\n({sched_str})",
        x0_off, traj_off, x_final_off, on_data_off, in_hole_off, off_off,
        "start (off-data)", "s",
    )
    _plot_run(
        ax_hole,
        f"In-hole start -> annealed Langevin\n({sched_str})",
        x0_hole, traj_hole, x_final_hole, on_data_hole, in_hole_hole, off_hole,
        "start (in ring hole)", "D",
    )

    record_steps = energies_off.shape[0]
    step_axis = np.arange(record_steps) * 5
    ax_curve.plot(step_axis, energies_off, color="tab:blue", lw=0.4, alpha=0.20)
    ax_curve.plot(step_axis, energies_hole, color="tab:orange", lw=0.4, alpha=0.20)
    ax_curve.plot(step_axis, energies_off.mean(axis=1), color="tab:blue",
                  lw=2.0, label="off-data start (mean E)")
    ax_curve.plot(step_axis, energies_hole.mean(axis=1), color="tab:orange",
                  lw=2.0, label="in-hole start (mean E)")
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
        "Annealed Langevin on deep RFF leverage energy "
        "(2 moons + ring + spiral).\n"
        "Off-data start (far OOD) vs in-hole start (ring interior). "
        "Heatmap is sigma_epi from the bounded output-RFF leverage head.",
        y=0.995, fontsize=11,
    )
    fig.tight_layout()
    out_path = out_dir / "hetero_demo_2d_ood_deep_rff_langevin.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
