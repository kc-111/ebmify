"""Overdamped Langevin dynamics on the leverage-based posterior
variance, started from outside the board AND from inside the gaps.

Treat the last-layer leverage

    h(x*) = phi(x*)^T (Phi_train^T Phi_train + r I)^-1 phi(x*)

as a (learned) energy function E(x*) = h(x*) / h_char. It is large on
points the trained model has never seen and small on training data, so
the negative gradient ``-grad_x E`` should pull a particle toward a
training-data manifold. The overdamped Langevin update is

    x_{t+1} = x_t - eta * grad_x E(x_t) + sqrt(2 * eta * T) * xi

with ``xi ~ N(0, I)``. Low temperature ``T`` biases the dynamics toward
minima of ``E`` (the data manifold); higher ``T`` produces samples
roughly proportional to ``exp(-E(x) / T)``.

We run two batches of particles:

* ``off_board`` -- spawned uniformly in the padded square outside
  ``[0, n_cells]^2``. They have to drift in toward the board to land on
  any training-data cell.
* ``in_gap`` -- spawned uniformly inside the *black* cells of the
  checkerboard. They start near a local energy plateau bordered on all
  four sides by white-cell minima; the gradient at a gap interior is
  ambiguous, so the only way out is via thermal diffusion across the
  saddle into a neighbouring white cell.

The heatmap shown is the *same* tanh-saturated ``sigma_epi`` used in
``hetero_demo_2d_ood_checkerboard.py`` so the visual contrast matches.
The Langevin still descends raw ``h(x) / h_char`` -- the per-particle
energy curves track that quantity.

Run from the repo root:

    python example/hetero/hetero_demo_2d_ood_checkerboard_langevin.py
"""

from __future__ import annotations

import math
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

from hetero_demo_2d_ood_checkerboard import (
    make_checkerboard_data,
    checkerboard_landscape,
    classify_regions,
    _draw_shapes,
    _heat,
)


# ----------------------------------------------------------------------
# Differentiable feature pass + leverage energy
# ----------------------------------------------------------------------

def features_with_grad(net: FCNet, x: torch.Tensor) -> torch.Tensor:
    """Penultimate-layer features as a differentiable function of ``x``.

    ``FCNet.features`` wraps the same computation in ``torch.no_grad``;
    here we re-call the underlying modules so autograd can flow back to
    ``x`` for Langevin / gradient-based exploration.
    """
    x_proc = net.input_pipeline(x)
    return net.net.trunk(x_proc)


def build_leverage_energy(
    net: FCNet, X_train: np.ndarray, *, ridge: float = 1e-3, bias: bool = True,
    normalize: bool = False, max_jitter_bumps: int = 8,
):
    """Build a callable ``energy(x) -> h(x) / h_char`` and return calibration.

    Precomputes ``A_inv = (Phi_train^T Phi_train + r I)^-1`` once. If the
    Cholesky fails (e.g., features are nearly collinear because the user
    chose a single wide RFF length scale), the ridge is bumped by 10x
    and retried -- standard "adaptive jitter" trick from GP regression.

    ``normalize=True``: L2-normalize the feature vector phi(x) to the unit
    sphere before (optionally) appending bias. Removes the ||phi||^2
    scaling from h(x), making leverage purely directional. Both the
    training Gram and the test-time scoring are normalized identically.
    Gradient w.r.t. x still flows through the normalization via autograd.
    """
    device = net.device
    X_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    with torch.no_grad():
        Phi = features_with_grad(net, X_t)
        if normalize:
            Phi = Phi / Phi.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        if bias:
            Phi = torch.cat(
                [Phi, torch.ones(Phi.shape[0], 1, device=Phi.device,
                                  dtype=Phi.dtype)],
                dim=1,
            )
        Md = Phi.shape[1]
        gram = Phi.T @ Phi
        eye = torch.eye(Md, device=Phi.device, dtype=Phi.dtype)

        jitter = float(ridge)
        L = None
        for _ in range(max_jitter_bumps):
            try:
                L = torch.linalg.cholesky(gram + jitter * eye)
                break
            except torch._C._LinAlgError:
                jitter *= 10.0
        if L is None:
            raise RuntimeError(
                f"build_leverage_energy: Cholesky failed even at "
                f"jitter={jitter:.3e}; the trained features may be "
                f"degenerate (e.g., RFF length scale too wide)."
            )
        if jitter > ridge:
            print(
                f"  build_leverage_energy: bumped ridge {ridge:.1e} -> "
                f"{jitter:.1e} to make Phi^T Phi + r*I positive-definite."
            )
        A_inv = torch.cholesky_solve(eye, L)
        h_train = (Phi @ A_inv * Phi).sum(dim=1).clamp(min=0.0)
        h_char = float(torch.quantile(h_train, 0.95).item())

    def raw_h(x: torch.Tensor) -> torch.Tensor:
        phi = features_with_grad(net, x)
        if normalize:
            phi = phi / phi.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        if bias:
            phi = torch.cat(
                [phi, torch.ones(phi.shape[0], 1, device=phi.device,
                                  dtype=phi.dtype)],
                dim=1,
            )
        return (phi @ A_inv * phi).sum(dim=1).clamp(min=0.0)

    def energy(x: torch.Tensor) -> torch.Tensor:
        return raw_h(x) / max(h_char, 1e-12)

    return energy, raw_h, h_char


# ----------------------------------------------------------------------
# Langevin sampler
# ----------------------------------------------------------------------

def geometric_anneal(T_hi: float, T_lo: float, n_steps: int):
    """Return a callable t -> T(t) that decays geometrically from T_hi to T_lo.

    Geometric (log-linear) schedules are the standard choice for annealed
    Langevin: large early-stage thermal kicks let particles hop out of
    shallow spurious minima of ``h(x)``, then the late phase cools to a
    near-deterministic descent that locks into the deepest basin.
    """
    log_hi = math.log(max(T_hi, 1e-30))
    log_lo = math.log(max(T_lo, 1e-30))
    denom = max(1, n_steps - 1)

    def schedule(t: int) -> float:
        frac = t / denom
        return math.exp(log_hi + (log_lo - log_hi) * frac)

    return schedule


def langevin_sample(
    energy_fn,
    x0: torch.Tensor,
    *,
    n_steps: int = 600,
    eta: float = 0.04,
    temperature=0.01,
    grad_clip: float | None = 5.0,
    record_every: int = 5,
    project=None,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """Overdamped Langevin on ``energy_fn``.

    ``temperature`` may be a float (constant) or a callable
    ``t -> T(t)`` for an annealing schedule.

    Returns ``(x_final, trajectories, energies, temps)`` where
    ``trajectories`` has shape ``[record_steps, B, D]``, ``energies``
    has shape ``[record_steps, B]``, and ``temps`` has shape ``[n_steps]``
    so we can plot the schedule alongside the descent.
    """
    if callable(temperature):
        T_fn = temperature
    else:
        T_const = float(temperature)
        T_fn = lambda _t: T_const  # noqa: E731

    x = x0.clone().detach().requires_grad_(True)
    traj_list: list[np.ndarray] = [x.detach().cpu().numpy().copy()]
    energy_list: list[np.ndarray] = []
    temp_list: list[float] = []

    for t in range(n_steps):
        T_t = max(T_fn(t), 0.0)
        sigma = math.sqrt(2.0 * eta * T_t)
        temp_list.append(T_t)

        e = energy_fn(x)
        e_sum = e.sum()
        grad = torch.autograd.grad(e_sum, x)[0]
        if grad_clip is not None:
            g_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = (grad_clip / g_norm).clamp(max=1.0)
            grad = grad * scale
        with torch.no_grad():
            noise = sigma * torch.randn_like(x)
            x_new = x - eta * grad + noise
            if project is not None:
                x_new = project(x_new)
            x.copy_(x_new)
        x.requires_grad_(True)
        if (t % record_every) == 0 or t == n_steps - 1:
            traj_list.append(x.detach().cpu().numpy().copy())
            energy_list.append(e.detach().cpu().numpy().copy())

    return (
        x.detach().cpu(),
        np.stack(traj_list, axis=0),
        np.stack(energy_list, axis=0),
        np.asarray(temp_list, dtype=np.float32),
    )


# ----------------------------------------------------------------------
# Particle initializers
# ----------------------------------------------------------------------

def init_off_board(
    n_part: int, n_cells: int, x1_lo: float, x1_hi: float,
    x2_lo: float, x2_hi: float, rng: np.random.Generator,
) -> np.ndarray:
    """Sample uniformly in the padded square, reject points inside the board."""
    pts: list[np.ndarray] = []
    while len(pts) < n_part:
        cand = rng.uniform(
            low=[x1_lo + 0.2, x2_lo + 0.2],
            high=[x1_hi - 0.2, x2_hi - 0.2],
            size=(n_part * 4, 2),
        )
        outside = ~(
            (cand[:, 0] >= 0.0) & (cand[:, 0] < n_cells)
            & (cand[:, 1] >= 0.0) & (cand[:, 1] < n_cells)
        )
        cand = cand[outside]
        for p in cand:
            pts.append(p)
            if len(pts) == n_part:
                break
    return np.asarray(pts, dtype=np.float32)


def init_in_gap(
    n_part: int, n_cells: int, rng: np.random.Generator,
    margin: float = 0.15,
) -> np.ndarray:
    """Sample uniformly inside the black cells (the interior gaps).

    Pulls each point ``margin`` away from the cell boundary so we start
    well inside the energy plateau, not on the white/black saddle.
    """
    black_cells = [
        (i, j) for i in range(n_cells) for j in range(n_cells)
        if (i + j) % 2 == 1
    ]
    if not black_cells:
        raise ValueError("no black cells on this board (n_cells too small)")
    cells = rng.choice(len(black_cells), size=n_part, replace=True)
    pts = np.empty((n_part, 2), dtype=np.float32)
    for k, c in enumerate(cells):
        i, j = black_cells[c]
        pts[k, 0] = i + rng.uniform(margin, 1.0 - margin)
        pts[k, 1] = j + rng.uniform(margin, 1.0 - margin)
    return pts


def classify_final(x: np.ndarray, n_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    in_bbox = (
        (x[:, 0] >= 0.0) & (x[:, 0] < n_cells)
        & (x[:, 1] >= 0.0) & (x[:, 1] < n_cells)
    )
    i = np.floor(np.clip(x[:, 0], 0.0, n_cells - 1e-6)).astype(int)
    j = np.floor(np.clip(x[:, 1], 0.0, n_cells - 1e-6)).astype(int)
    parity = (i + j) % 2
    return in_bbox & (parity == 0), in_bbox & (parity == 1), ~in_bbox


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    n_cells = 4
    X_tr, y_tr, geom = make_checkerboard_data(n=500, seed=1, n_cells=n_cells)

    pc = PreprocessConfig(
        input_transforms=["standard"],
        output_transforms=["standard"],
    )
    fc = FitConfig(epochs=1500, lr=5e-4, batch_size=128, seed=0, verbose=False)
    nc = NoiseConfig(
        input_additive_std=0.0, input_multiplicative_std=0.0,
        output_additive_std=0.0,
    )

    M_in, M_out = 32, 64
    print(
        f"Fitting deep RFF FCNet on {n_cells}x{n_cells} checkerboard "
        f"(M_in={M_in}, M_out={M_out}) ..."
    )
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

    # ------------------------------------------------------------------
    # Build the leverage energy and the matching sigma_epi grid.
    # ------------------------------------------------------------------
    energy_fn, raw_h_fn, h_char = build_leverage_energy(
        net, X_tr, ridge=1e-3, bias=True,
    )
    print(f"  h_char (95th percentile of training leverage) = {h_char:.4e}")

    # Match the sister demo's PAD so the heatmap framing is identical.
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
        X_grid_t = torch.as_tensor(X_grid, dtype=torch.float32, device=net.device)
        h_grid = raw_h_fn(X_grid_t).cpu().numpy().reshape(G, G)
        mu_train = net.predict(X_tr).cpu().numpy()

    # tanh-saturated sigma_epi -- same calibration as the sister demo so
    # the visual contrast (white-cell dark, gaps + far-OOD bright) matches.
    sigma_noise = float(np.std(y_tr - mu_train))
    lam = sigma_noise / np.sqrt(max(h_char, 1e-12))
    sigma_max = 5.0 * sigma_noise
    sigma_epi_grid = sigma_max * np.tanh(
        lam * np.sqrt(np.clip(h_grid, 0.0, None)) / sigma_max
    )
    sigma_vmax = float(sigma_epi_grid.max())
    print(f"  sigma_noise = {sigma_noise:.4f}, sigma_max = {sigma_max:.4f}")

    # ------------------------------------------------------------------
    # Two batches of particles: off-board start, in-gap start.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)
    n_part = 64

    x0_off = init_off_board(n_part, n_cells, X1_LO, X1_HI, X2_LO, X2_HI, rng)
    x0_gap = init_in_gap(n_part, n_cells, rng, margin=0.15)

    eta = 0.05
    n_steps = 1200
    # Annealed schedule: hot enough early to escape shallow leverage
    # plateaus that off-board particles otherwise get stuck on, then
    # cools by ~50x to settle into a basin.
    T_hi, T_lo = 0.05, 0.001
    temperature = geometric_anneal(T_hi, T_lo, n_steps)

    print(
        f"\nRunning Langevin (off-board start, n_particles={n_part}, "
        f"eta={eta}, T: {T_hi} -> {T_lo} (geometric), steps={n_steps}) ..."
    )
    x_final_off_t, traj_off, energies_off, temps = langevin_sample(
        energy_fn,
        torch.as_tensor(x0_off, dtype=torch.float32, device=net.device),
        n_steps=n_steps, eta=eta, temperature=temperature,
        grad_clip=5.0, record_every=5,
    )
    x_final_off = x_final_off_t.numpy()

    print(
        f"Running Langevin (in-gap start, n_particles={n_part}, "
        f"eta={eta}, T: {T_hi} -> {T_lo} (geometric), steps={n_steps}) ..."
    )
    x_final_gap_t, traj_gap, energies_gap, _ = langevin_sample(
        energy_fn,
        torch.as_tensor(x0_gap, dtype=torch.float32, device=net.device),
        n_steps=n_steps, eta=eta, temperature=temperature,
        grad_clip=5.0, record_every=5,
    )
    x_final_gap = x_final_gap_t.numpy()

    on_white_off, on_black_off, off_off = classify_final(x_final_off, n_cells)
    on_white_gap, on_black_gap, off_gap = classify_final(x_final_gap, n_cells)

    def _report(label: str, w: np.ndarray, b: np.ndarray, o: np.ndarray) -> None:
        print(f"  [{label}] white={int(w.sum())}/{n_part} ({100*w.mean():.1f}%)  "
              f"black={int(b.sum())}/{n_part} ({100*b.mean():.1f}%)  "
              f"off-board={int(o.sum())}/{n_part} ({100*o.mean():.1f}%)")

    print(f"\nFinal positions after {n_steps} Langevin steps:")
    _report("off-board start", on_white_off, on_black_off, off_off)
    _report("in-gap   start ", on_white_gap, on_black_gap, off_gap)
    print("  (Off-board particles flow in across the boundary into white cells.")
    print("   In-gap particles need thermal diffusion to escape the plateau and")
    print("   slide down into a neighbouring white-cell minimum.)")

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

    # Panel 1: sigma_epi heatmap with training data overlay (matches
    # the sister demo's calibration so the contrast lines up).
    _heat(ax_heat, sigma_epi_grid, extent,
          "sigma_epi = sigma_max * tanh(lam * sqrt(h)/sigma_max)\n"
          "(training data overlay)",
          geom, cmap="magma", vmin=0.0, vmax=sigma_vmax)
    ax_heat.scatter(X_tr[:, 0], X_tr[:, 1], s=1, alpha=0.25,
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
                       label=f"end on white ({int(w.sum())})", zorder=6)
        if b.any():
            ax.scatter(x_final[b, 0], x_final[b, 1], s=22,
                       color="tab:orange", edgecolor="black", lw=0.4,
                       label=f"end on black ({int(b.sum())})", zorder=6)
        if o.any():
            ax.scatter(x_final[o, 0], x_final[o, 1], s=22,
                       color="tab:red", edgecolor="black", lw=0.4,
                       label=f"end off-board ({int(o.sum())})", zorder=6)
        ax.legend(loc="upper right", fontsize=6)

    sched_str = f"T:{T_hi:.3g}->{T_lo:.3g} (geom), eta={eta}, steps={n_steps}"
    _plot_run(
        ax_off,
        f"Off-board start -> annealed Langevin\n({sched_str})",
        x0_off, traj_off, x_final_off, on_white_off, on_black_off, off_off,
        "start (off-board)", "s",
    )
    _plot_run(
        ax_gap,
        f"In-gap start -> annealed Langevin\n({sched_str})",
        x0_gap, traj_gap, x_final_gap, on_white_gap, on_black_gap, off_gap,
        "start (in gap)", "D",
    )

    # Bottom: energy curves for each batch + temperature schedule overlay.
    record_steps = energies_off.shape[0]
    step_axis = np.arange(record_steps) * 5  # record_every=5
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
        "Langevin dynamics on the leverage-based posterior variance.\n"
        f"Off-board start ([{X1_LO:+.1f}, {X1_HI:+.1f}]^2 minus the board) "
        f"vs in-gap start (uniform inside the black cells). "
        f"Heatmap is sigma_epi (same calibration as the sister demo).",
        y=0.995, fontsize=11,
    )
    fig.tight_layout()
    out_path = out_dir / "hetero_demo_2d_ood_checkerboard_langevin.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
