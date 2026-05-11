"""Leverage density on 2D moons — what does concatenating raw x do?

Sweeps 8 feature maps for h(x) = phi(x)^T (Phi^T Phi + lam I)^{-1} phi(x):

  (1) raw                  phi(x) = x                             (D = 2)
  (2) RFF only             phi(x) = RFF(x)                        (D = M_in)
  (3) [x; RFF(x)]          phi(x) = [x; RFF(x)]                   (D = 2+M_in)
  (4) random FCNet trunk   phi(x) = FCNet_trunk(x)                (untrained)
  (5) trained FCNet trunk  phi(x) = FCNet_trunk(x)                (n epochs)
  (6) [x; trained trunk]   phi(x) = [x; FCNet_trunk(x)]           (trained)
  (7) [x; h_pre_out_rff]   phi(x) = [x; last block output]        (trained, no RFF_out)
  (8) [x; all hidden+RFF]  phi(x) = [x; h_0; ...; h_last; RFF_out](trained, multi-depth)

(1) and (2) isolate the two ingredients of (3); the contrast shows the
raw-x quadratic bowl `z^T A_zz^{-1} z` against the RFF kernel-density
floor. (4)–(6) repeat the comparison at the FCNet feature level.
(7) drops the output-RFF (pre-last-layer features only). (8) stacks every
intermediate hidden state with the final RFF — probes whether
multi-depth concatenation tightens the density estimate.

Run:
    python example/hetero/random_features_2d_density.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons

from ebmify.models import (
    FCNet, FitConfig, NoiseConfig, PreprocessConfig, RegConfig,
)
from ebmify.models.fc import RFFLayer

# Reuse the moons-region classifier so we get clean "in-data" vs OOD masks.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hetero_demo_2d_ood_moons import classify_regions  # noqa: E402


# ----------------------------------------------------------------------
# Generic leverage builder for any feature map.
# ----------------------------------------------------------------------

def build_leverage(
    phi_fn, X_train_t: torch.Tensor, *, ridge: float = 1e-3, bias: bool = True,
):
    """Return h(x) = phi(x)^T (Phi^T Phi + lam I)^{-1} phi(x).

    Uses a Cholesky of A = Phi^T Phi + lam I so leverage is
    h(x) = ||L^{-1} phi(x)||^2. Bumps ridge if cholesky fails.
    """
    with torch.no_grad():
        Phi = phi_fn(X_train_t)
        if bias:
            Phi = torch.cat(
                [Phi, torch.ones(Phi.shape[0], 1, device=Phi.device)], dim=-1,
            )
        D = Phi.shape[1]
        for bump in (1.0, 10.0, 100.0, 1000.0):
            try:
                A = Phi.T @ Phi + ridge * bump * torch.eye(D, device=Phi.device)
                L = torch.linalg.cholesky(A)
                break
            except torch.linalg.LinAlgError:
                continue
        else:
            raise RuntimeError("Cholesky failed even with 1000x ridge bump.")

    def h_fn(X_t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            phi = phi_fn(X_t)
            if bias:
                phi = torch.cat(
                    [phi, torch.ones(phi.shape[0], 1, device=phi.device)],
                    dim=-1,
                )
            v = torch.linalg.solve_triangular(L, phi.T, upper=False)
            return v.pow(2).sum(dim=0)

    return h_fn, D


# ----------------------------------------------------------------------
# Feature extractors.
# ----------------------------------------------------------------------

def make_fcnet(device: str, *, M_in: int, M_out: int, ell_in: float,
                ell_out: float, hidden=(64, 64), seed: int = 0) -> FCNet:
    pc = PreprocessConfig(
        input_transforms=["standard"], output_transforms=["standard"],
    )
    nc = NoiseConfig(
        input_additive_std=0.0, input_multiplicative_std=0.0,
        output_additive_std=0.0,
    )
    fc = FitConfig(epochs=0, lr=1e-3, batch_size=256, seed=seed, verbose=False)
    return FCNet(
        n_inputs=2, n_outputs=1,
        hidden_dims=hidden,
        activation="odd_piecewise",
        fit_config=fc, reg_config=RegConfig(l2=1e-5),
        noise_config=nc, preprocess=pc,
        input_rff=M_in, input_rff_length_scale=[ell_in],
        output_rff=M_out, output_rff_length_scale=[ell_out],
        block_type="rff",
        block_rff_features=64,
        block_rff_length_scale=[0.5],
        rff_seed=seed,
        device=device,
    )


def make_input_rff_only(device: str, *, M_in: int, ell_in: float,
                         seed: int = 0) -> RFFLayer:
    rff = RFFLayer(
        in_dim=2, n_features=M_in, length_scale=[ell_in], rff_seed=seed,
    ).to(device)
    return rff


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def grid_eval(h_fn, x1, x2, device, batch=4096) -> np.ndarray:
    XY = np.stack(
        [np.repeat(x1, len(x2)), np.tile(x2, len(x1))], axis=1,
    ).astype(np.float32)
    XY_t = torch.as_tensor(XY, device=device)
    out = np.empty(XY.shape[0], dtype=np.float32)
    for i in range(0, XY.shape[0], batch):
        out[i : i + batch] = h_fn(XY_t[i : i + batch]).cpu().numpy()
    return out.reshape(len(x1), len(x2)).T  # row=x2, col=x1


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    n = 1000
    X, _ = make_moons(n_samples=n, noise=0.05, random_state=1)
    X = X.astype(np.float32)
    geom = {
        "moon_top":    {"center": (0.0, 0.0), "radius": 1.0,
                         "angle_lo": 0.0, "angle_hi": np.pi, "thickness": 0.19},
        "moon_bottom": {"center": (1.0, 0.5), "radius": 1.0,
                         "angle_lo": np.pi, "angle_hi": 2.0 * np.pi,
                         "thickness": 0.19},
    }
    # Dummy y so FCNet.fit doesn't complain — used only for preprocessing.
    y_dummy = np.zeros((n, 1), dtype=np.float32)

    # Grid + region masks for quantitative scoring.
    x1_lo, x1_hi = -2.5, 3.5
    x2_lo, x2_hi = -2.0, 2.5
    x1g = np.linspace(x1_lo, x1_hi, 240)
    x2g = np.linspace(x2_lo, x2_hi, 200)
    XX, YY = np.meshgrid(x1g, x2g)
    masks = classify_regions(
        XX, YY, geom,
        x1_lo=x1_lo + 0.5, x1_hi=x1_hi - 0.5,
        x2_lo=x2_lo + 0.4, x2_hi=x2_hi - 0.4,
    )
    extent = (x1_lo, x1_hi, x2_lo, x2_hi)

    # ------------------------------------------------------------------
    # Feature extractors.
    # ------------------------------------------------------------------
    M_in, M_out = 256, 256
    ell_in, ell_out = 1.0, 0.1
    hidden = (64, 64)

    X_t = torch.as_tensor(X, device=device)

    # (1) raw 2D features.
    def phi_raw(x: torch.Tensor) -> torch.Tensor:
        return x

    # (2) RFF only — phi(x) = RFF(x).
    rff = make_input_rff_only(device, M_in=M_in, ell_in=ell_in, seed=0)
    with torch.no_grad():
        rff.init_bandwidth(X_t)

    def phi_rff_only(x: torch.Tensor) -> torch.Tensor:
        return rff(x)

    # (3) [x; RFF(x)] — adds the raw-x quadratic bowl on top.
    def phi_rff_with_x(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, rff(x)], dim=-1)

    # (3) Random-init FCNet (epochs=0). Calls fit to run preprocessing
    #     fits and RFF bandwidth init; skips training loop.
    net_rand = make_fcnet(
        device, M_in=M_in, M_out=M_out, ell_in=ell_in, ell_out=ell_out,
        hidden=hidden, seed=0,
    )
    net_rand.fit(X, y_dummy)

    def phi_rand(x: torch.Tensor) -> torch.Tensor:
        was_training = net_rand.training
        net_rand.eval()
        try:
            with torch.no_grad():
                X_proc = net_rand.input_pipeline(x)
                return net_rand.net.trunk(X_proc)
        finally:
            if was_training:
                net_rand.train()

    # (4) Trained FCNet (regression on a simple smooth target).
    net_trn = make_fcnet(
        device, M_in=M_in, M_out=M_out, ell_in=ell_in, ell_out=ell_out,
        hidden=hidden, seed=0,
    )
    net_trn.fit_config.epochs = 500
    net_trn.fit_config.verbose = False
    # Smooth target: distance to nearest moon center as a non-trivial signal.
    y_tr = np.minimum(
        np.linalg.norm(X - np.array([[0.0, 0.0]], dtype=np.float32), axis=1),
        np.linalg.norm(X - np.array([[1.0, 0.5]], dtype=np.float32), axis=1),
    ).astype(np.float32)[:, None]
    print("Training the trained-baseline FCNet ...")
    net_trn.fit(X, y_tr)

    def phi_trn(x: torch.Tensor) -> torch.Tensor:
        was_training = net_trn.training
        net_trn.eval()
        try:
            with torch.no_grad():
                X_proc = net_trn.input_pipeline(x)
                return net_trn.net.trunk(X_proc)
        finally:
            if was_training:
                net_trn.train()

    def phi_trn_with_x(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, phi_trn(x)], dim=-1)

    def _trn_intermediate(x: torch.Tensor) -> list[torch.Tensor]:
        """Capture all intermediate hidden states of the trained trunk."""
        net = net_trn.net  # _Trunk module
        was_training = net_trn.training
        net_trn.eval()
        try:
            with torch.no_grad():
                X_proc = net_trn.input_pipeline(x)
                if net.input_rff is not None:
                    h_in = torch.cat([X_proc, net.input_rff(X_proc)], dim=-1)
                else:
                    h_in = X_proc
                hs = [net.in_proj(h_in)]
                for b in net.blocks:
                    hs.append(b(hs[-1]))
                # hs[-1] is the pre-output-RFF hidden; trunk output adds RFF_out.
                return hs
        finally:
            if was_training:
                net_trn.train()

    def phi_trn_prelast(x: torch.Tensor) -> torch.Tensor:
        """phi = [x; h_pre_output_rff] — drop the output_rff and the prior blocks."""
        hs = _trn_intermediate(x)
        return torch.cat([x, hs[-1]], dim=-1)

    def phi_trn_allmid(x: torch.Tensor) -> torch.Tensor:
        """phi = [x; h_0; h_1; ...; h_last; RFF_out(h_last)] — full multi-depth stack."""
        hs = _trn_intermediate(x)
        out_rff = net_trn.net.output_rff
        last = hs[-1]
        tail = [last, out_rff(last)] if out_rff is not None else [last]
        return torch.cat([x, *hs[:-1], *tail], dim=-1)

    n_ep = net_trn.fit_config.epochs
    feature_specs = [
        ("(1) raw  phi=x",                                phi_raw),
        (f"(2) RFF only  M={M_in}",                       phi_rff_only),
        (f"(3) [x; RFF]  M={M_in}",                       phi_rff_with_x),
        ("(4) random FCNet trunk (epochs=0)",             phi_rand),
        (f"(5) trained FCNet trunk ({n_ep} ep)",          phi_trn),
        (f"(6) [x; trained FCNet trunk]  ({n_ep} ep)",    phi_trn_with_x),
        (f"(7) [x; h_pre_output_rff]  ({n_ep} ep)",       phi_trn_prelast),
        (f"(8) [x; all hidden; RFF_out]  ({n_ep} ep)",    phi_trn_allmid),
    ]

    # ------------------------------------------------------------------
    # Build leverage h(x) per feature map and evaluate on grid.
    # ------------------------------------------------------------------
    panels = []
    for name, phi in feature_specs:
        h_fn, D = build_leverage(phi, X_t, ridge=1e-3, bias=True)
        H = grid_eval(h_fn, x1g, x2g, device)
        # Normalise: h_char = 95th percentile of in-data leverage.
        h_train = h_fn(X_t).cpu().numpy()
        h_char = float(np.quantile(h_train, 0.95))
        Hn = H / max(h_char, 1e-30)
        in_med = float(np.median(Hn[masks["in_data"]]))
        ood_med = float(np.median(Hn[masks["ood"]]))
        slot_med = float(np.median(Hn[masks["slot"]]))
        signal = ood_med / max(in_med, 1e-30)
        slot_sig = slot_med / max(in_med, 1e-30)
        print(
            f"  {name:<40} D={D:<5}  h/h_char  "
            f"in={in_med:.3f}  slot={slot_med:.3f}  ood={ood_med:.3e}  "
            f"OOD/in={signal:.2e}  slot/in={slot_sig:.2e}"
        )
        panels.append((name, Hn, in_med, slot_med, ood_med))

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()
    for ax, (name, Hn, in_med, slot_med, ood_med) in zip(axes, panels):
        Lh = np.log10(Hn + 1.0)
        im = ax.imshow(
            Lh, extent=extent, origin="lower", cmap="viridis",
            aspect="equal", interpolation="nearest",
            vmin=0.0, vmax=float(np.quantile(Lh, 0.99)),
        )
        # Overlay moon shape outlines.
        for key in ("moon_top", "moon_bottom"):
            m = geom[key]
            cx, cy = m["center"]; rr = m["radius"]; tk = m["thickness"]
            th = np.linspace(m["angle_lo"], m["angle_hi"], 200)
            for r in (rr - tk, rr + tk):
                ax.plot(cx + r * np.cos(th), cy + r * np.sin(th),
                        color="orange", lw=1.0, alpha=0.85)
        ax.scatter(X[:, 0], X[:, 1], s=2, c="white", alpha=0.4)
        ax.set_xlim(x1_lo, x1_hi); ax.set_ylim(x2_lo, x2_hi)
        ax.set_title(
            f"{name}\nlog10(h/h_char + 1); "
            f"in={in_med:.3f}  slot={slot_med:.3f}  ood={ood_med:.2e}",
            fontsize=9,
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Leverage h(x) under different feature maps on sklearn moons. "
        "h_char = 95th pct of in-data leverage. "
        "Larger OOD/in ratio = better density signal.",
        fontsize=11, y=0.995,
    )
    fig.tight_layout()
    out_path = (
        Path(__file__).resolve().parent.parent / "out"
        / "random_features_2d_density.png"
    )
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
