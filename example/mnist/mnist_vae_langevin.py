"""z-space Langevin sampling on a cached β-VAE.

Loads the cached VAE from ``mnist_vae_train.py`` and runs SamAdams
Langevin on a leverage energy built from `phi(z) = [z, RFF(z)]`.
Concatenating raw z adds a quadratic bowl `z^T A_zz^{-1} z` from the
(z,z) block of the precision matrix — empirically this *improves*
generation quality, so it's on by default. Pass `--no-raw-z` to drop
back to the pure-RFF energy.

Run (after training is cached):
    python example/mnist/mnist_vae_langevin.py
    python example/mnist/mnist_vae_langevin.py --no-raw-z --T 1e-4 --anneal off
"""

from __future__ import annotations

import argparse
import sys
from argparse import BooleanOptionalAction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hetero"))
from hetero_demo_2d_ood_checkerboard_langevin import geometric_anneal  # noqa: E402

from ebmify.models.fc import RFFLayer
from ebmify.sampler import SamAdamsConfig, samadams_sample

from mnist_vae_train import VAE, load_mnist_train, vae_ckpt_path


# ----------------------------------------------------------------------
# Leverage on RFF(z)
# ----------------------------------------------------------------------

def build_z_leverage(
    rff: RFFLayer, Z_train: torch.Tensor, *,
    ridge: float = 1e-3, bias: bool = True, include_raw_z: bool = False,
):
    """phi(z) = RFF(z) [+ z if include_raw_z] [+ 1 if bias]. Returns h(z), h_char."""
    device = Z_train.device

    def phi(z: torch.Tensor) -> torch.Tensor:
        parts = [rff(z)]
        if include_raw_z:
            parts = [z] + parts
        return torch.cat(parts, dim=-1)

    with torch.no_grad():
        Phi = phi(Z_train)
        if bias:
            Phi = torch.cat(
                [Phi, torch.ones(Phi.shape[0], 1, device=device)], dim=-1,
            )
        D = Phi.shape[1]
        for bump in (1.0, 10.0, 100.0, 1000.0):
            try:
                A = Phi.T @ Phi + ridge * bump * torch.eye(D, device=device)
                L = torch.linalg.cholesky(A)
                break
            except torch.linalg.LinAlgError:
                continue
        else:
            raise RuntimeError("Cholesky failed")

    def h_fn(z: torch.Tensor) -> torch.Tensor:
        p = phi(z)
        if bias:
            p = torch.cat([p, torch.ones(p.shape[0], 1, device=p.device)], dim=-1)
        v = torch.linalg.solve_triangular(L, p.T, upper=False)
        return v.pow(2).sum(dim=0)

    with torch.no_grad():
        h_train = h_fn(Z_train).cpu().numpy()
    h_char = float(np.quantile(h_train, 0.95))
    return h_fn, h_char, D


def build_phi_leverage(
    phi_fn, Z_train: torch.Tensor, *,
    ridge: float = 1e-3, bias: bool = True,
):
    """Generic h(z) = phi(z)^T (Phi^T Phi + lam I)^{-1} phi(z)."""
    device = Z_train.device
    with torch.no_grad():
        Phi = phi_fn(Z_train)
        if bias:
            Phi = torch.cat(
                [Phi, torch.ones(Phi.shape[0], 1, device=device)], dim=-1,
            )
        D = Phi.shape[1]
        for bump in (1.0, 10.0, 100.0, 1000.0):
            try:
                A = Phi.T @ Phi + ridge * bump * torch.eye(D, device=device)
                L = torch.linalg.cholesky(A)
                break
            except torch.linalg.LinAlgError:
                continue
        else:
            raise RuntimeError("Cholesky failed")

    def h_fn(z: torch.Tensor) -> torch.Tensor:
        p = phi_fn(z)
        if bias:
            p = torch.cat(
                [p, torch.ones(p.shape[0], 1, device=p.device)], dim=-1,
            )
        v = torch.linalg.solve_triangular(L, p.T, upper=False)
        return v.pow(2).sum(dim=0)

    with torch.no_grad():
        h_train = h_fn(Z_train).cpu().numpy()
    h_char = float(np.quantile(h_train, 0.95))
    return h_fn, h_char, D


def plot_z_leverage_separation(
    Z_train: torch.Tensor, rff: RFFLayer, device: str, out_path: Path,
    *, M_rff: int, n_eval: int = 4096,
) -> None:
    """Histograms of log10(h(z)/h_char) for in-data / N(0,I) / N(0,25)
    under three phi maps: z only, RFF(z) only, [z; RFF(z)]."""
    z_dim = Z_train.shape[1]
    z_prior = torch.randn(n_eval, z_dim, device=device)
    z_far = 5.0 * torch.randn(n_eval, z_dim, device=device)

    specs = [
        ("phi = z",           lambda z: z),
        ("phi = RFF(z)",      lambda z: rff(z)),
        ("phi = [z; RFF(z)]", lambda z: torch.cat([z, rff(z)], dim=-1)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    print("\nz-leverage separation (median h/h_char):")
    print(f"  {'phi':<22} {'D':>5}  {'in-data':>10}  "
          f"{'N(0,I)':>10}  {'N(0,25)':>12}  "
          f"{'prior/in':>10}  {'far/in':>10}")
    for ax, (name, phi_fn) in zip(axes, specs):
        h_fn, h_char, D = build_phi_leverage(phi_fn, Z_train)
        with torch.no_grad():
            h_in = h_fn(Z_train).cpu().numpy() / h_char
            h_pr = h_fn(z_prior).cpu().numpy() / h_char
            h_fa = h_fn(z_far).cpu().numpy() / h_char
        med_in = float(np.median(h_in))
        med_pr = float(np.median(h_pr))
        med_fa = float(np.median(h_fa))
        print(f"  {name:<22} {D:>5}  {med_in:>10.3f}  "
              f"{med_pr:>10.3f}  {med_fa:>12.3e}  "
              f"{med_pr/max(med_in,1e-30):>10.2e}  "
              f"{med_fa/max(med_in,1e-30):>10.2e}")
        all_vals = np.concatenate([h_in, h_pr, h_fa])
        lo = float(np.log10(max(all_vals.min(), 1e-6)))
        hi = float(np.log10(all_vals.max() + 1.0))
        bins = np.linspace(lo, hi, 60)
        ax.hist(np.log10(h_in + 1e-6), bins=bins, alpha=0.55,
                label="in-data", color="C0", density=True)
        ax.hist(np.log10(h_pr + 1e-6), bins=bins, alpha=0.55,
                label="z ~ N(0,I)", color="C1", density=True)
        ax.hist(np.log10(h_fa + 1e-6), bins=bins, alpha=0.55,
                label="z ~ N(0,25)", color="C2", density=True)
        ax.axvline(0.0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("log10(h(z) / h_char)")
        ax.set_title(
            f"{name}  (D={D})\n"
            f"med in={med_in:.2f}  prior={med_pr:.2f}  far={med_fa:.2e}",
            fontsize=9,
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"VAE z-space leverage separation  z_dim={z_dim}  M_rff={M_rff}  "
        f"ell={rff.length_scale.tolist()}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved {out_path}")


def build_ood_x_sources(
    X_in_t: torch.Tensor, device: str, *, n_eval: int = 2048, seed: int = 42,
    in_name: str = "MNIST", gauss_mean: float | None = None,
    gauss_std: float | None = None,
):
    """Build (name, x_tensor, color) tuples for in-data and OOD x distributions.

    Image shape is inferred from ``X_in_t.shape[1:]`` so the same helper
    serves both MNIST (1x28x28) and CIFAR (3x32x32). ``gauss_mean`` and
    ``gauss_std`` default to the empirical pixel mean/std of ``X_in_t``;
    the Gaussian OOD source uses those, then clamps to [0, 1].
    """
    img_shape = tuple(X_in_t.shape[1:])
    n_pix = int(np.prod(img_shape))
    gen = torch.Generator(device=device).manual_seed(seed)
    idx = torch.randperm(X_in_t.shape[0], generator=gen, device=device)[:n_eval]
    x_in = X_in_t[idx]

    if gauss_mean is None:
        gauss_mean = float(x_in.mean().item())
    if gauss_std is None:
        gauss_std = float(x_in.std().item())

    x_unif = torch.rand(n_eval, *img_shape, generator=gen, device=device)
    x_sp = (
        torch.rand(n_eval, *img_shape, generator=gen, device=device) > 0.5
    ).float()
    x_gauss = (
        gauss_mean
        + gauss_std
        * torch.randn(n_eval, *img_shape, generator=gen, device=device)
    ).clamp(0.0, 1.0)

    keys = torch.rand(n_eval, n_pix, generator=gen, device=device)
    perms = keys.argsort(dim=1)
    x_shuf = torch.gather(
        x_in.reshape(n_eval, n_pix), 1, perms
    ).reshape(n_eval, *img_shape)

    x_inv = 1.0 - x_in
    x_black = torch.zeros(n_eval, *img_shape, device=device)
    x_white = torch.ones(n_eval, *img_shape, device=device)

    return [
        (in_name,       x_in,    "C0"),
        ("uniform",     x_unif,  "C1"),
        ("Bernoulli",   x_sp,    "C2"),
        ("Gaussian",    x_gauss, "C3"),
        ("shuffled",    x_shuf,  "C4"),
        ("inverted",    x_inv,   "C5"),
        ("black",       x_black, "C6"),
        ("white",       x_white, "C7"),
    ]


def plot_x_to_z_leverage_separation(
    vae, X_in_t: torch.Tensor, Z_train: torch.Tensor,
    rff: RFFLayer, device: str, out_path: Path,
    *, M_rff: int, n_eval: int = 2048, ridge: float = 1e-3,
    seed: int = 42, in_name: str = "MNIST",
    x_sources=None,
) -> None:
    """For in-data X_in and OOD x distributions, encode through the VAE
    (z = mu) and compute leverage under phi in {z, RFF(z), [z; RFF(z)]}.

    Pass ``x_sources`` explicitly to override the default set (e.g. to
    add a cross-dataset source for CIFAR); otherwise the default set is
    built via :func:`build_ood_x_sources` from ``X_in_t``'s shape.
    """
    z_dim = Z_train.shape[1]
    if x_sources is None:
        x_sources = build_ood_x_sources(
            X_in_t, device, n_eval=n_eval, seed=seed, in_name=in_name,
        )
    img_shape = tuple(X_in_t.shape[1:])
    C, H, W = img_shape if len(img_shape) == 3 else (1, img_shape[0], img_shape[1])

    # Encode each x source to z via the VAE encoder (use mu, not sample).
    z_sources = []
    with torch.no_grad():
        for name, x, color in x_sources:
            mu, _ = vae.encode(x)
            z_sources.append((name, mu, color))

    specs = [
        ("phi = z",           lambda z: z),
        ("phi = RFF(z)",      lambda z: rff(z)),
        ("phi = [z; RFF(z)]", lambda z: torch.cat([z, rff(z)], dim=-1)),
    ]

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.0, 1.0])
    hist_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_strip = fig.add_subplot(gs[1, :])

    print("\nx -> z leverage separation (median h/h_char):")
    header = f"  {'phi':<22} {'D':>5}"
    for name, _, _ in z_sources:
        header += f"  {name:>11}"
    print(header)

    for ax, (spec_name, phi_fn) in zip(hist_axes, specs):
        h_fn, h_char, D = build_phi_leverage(phi_fn, Z_train, ridge=ridge)
        row_data = []
        row_str = f"  {spec_name:<22} {D:>5}"
        for src_name, z_src, color in z_sources:
            with torch.no_grad():
                h_vals = h_fn(z_src).cpu().numpy() / h_char
            row_data.append((src_name, h_vals, color))
            row_str += f"  {np.median(h_vals):>11.3e}"
        print(row_str)

        all_vals = np.concatenate([d[1] for d in row_data])
        lo = float(np.log10(max(all_vals.min(), 1e-6)))
        hi = float(np.log10(all_vals.max() + 1.0))
        bins = np.linspace(lo, hi, 60)
        for src_name, h_vals, color in row_data:
            med = float(np.median(h_vals))
            if h_vals.std() < 1e-5 * (abs(med) + 1e-12):
                ax.axvline(
                    np.log10(med + 1e-6), color=color, lw=2.0,
                    alpha=0.8, ls="-",
                    label=f"{src_name} ({med:.1e})",
                )
            else:
                ax.hist(np.log10(h_vals + 1e-6), bins=bins, alpha=0.5,
                        label=f"{src_name} ({med:.1e})", color=color,
                        density=True)
        ax.axvline(0.0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("log10(h(z) / h_char)")
        ax.set_title(f"{spec_name}  (D={D})", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    # Bottom strip: show a sample of each x source.
    n_per_src = 4
    n_src = len(x_sources)
    pad = 2
    img_w = n_per_src * W + (n_per_src - 1) * pad
    sep = 8
    canvas_w = n_src * img_w + (n_src - 1) * sep
    if C == 1:
        sample_canvas = np.ones((H, canvas_w), dtype=np.float32)
    else:
        sample_canvas = np.ones((H, canvas_w, C), dtype=np.float32)
    centers = []
    for i, (name, x, _) in enumerate(x_sources):
        start = i * (img_w + sep)
        for j in range(n_per_src):
            col = start + j * (W + pad)
            img_chw = x[j].cpu().numpy()
            if C == 1:
                sample_canvas[:, col : col + W] = img_chw[0]
            else:
                sample_canvas[:, col : col + W, :] = np.transpose(img_chw, (1, 2, 0))
        centers.append(start + img_w / 2)
    if C == 1:
        ax_strip.imshow(sample_canvas, cmap="gray", vmin=0, vmax=1)
    else:
        ax_strip.imshow(np.clip(sample_canvas, 0, 1))
    ax_strip.set_xticks(centers)
    ax_strip.set_xticklabels([s[0] for s in x_sources])
    ax_strip.set_yticks([])
    ax_strip.set_title("Example x from each source", fontsize=10)

    fig.suptitle(
        f"x -> z OOD classification on VAE  z_dim={z_dim}  M_rff={M_rff}  "
        f"ell={rff.length_scale.tolist()}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved {out_path}")


def imgrid(ax, imgs: np.ndarray, title: str, n: int = 8) -> None:
    B = imgs.shape[0]
    rows = (B + n - 1) // n
    canvas = np.ones((rows * 28, n * 28), dtype=np.float32)
    for i in range(B):
        r, c = divmod(i, n)
        canvas[r * 28 : (r + 1) * 28, c * 28 : (c + 1) * 28] = imgs[i]
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=int, default=64, dest="z_dim")
    ap.add_argument("--length_scale", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--M", type=int, default=1024, dest="M_rff")
    ap.add_argument("--ell", type=float, default=1.0,
                     help="RFF length scale; default = median heuristic")
    ap.add_argument("--raw-z", action=BooleanOptionalAction, default=True,
                     dest="include_raw_z",
                     help="phi(z) = [z, RFF(z)] (adds a confining quadratic). "
                          "Default: on; use --no-raw-z to drop.")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--n-part", type=int, default=64, dest="n_part")
    ap.add_argument("--dtau", type=float, default=5e-3,
                     help="hot stepsize dtau_hi")
    ap.add_argument("--dtau-lo", type=float, default=1e-3, dest="dtau_lo",
                     help="cold stepsize; default = dtau (no decay)")
    ap.add_argument("--T", type=float, default=1e0,
                     help="hot temperature T_hi")
    ap.add_argument("--T-lo", type=float, default=1e-3, dest="T_lo",
                     help="cold temperature; default = T * 1e-3")
    ap.add_argument("--hot-frac", type=float, default=0.0, dest="hot_frac",
                     help="fraction of steps to hold at T_hi before annealing")
    ap.add_argument("--anneal", choices=("on", "off"), default="on",
                     help="on -> T: T_hi -> T_lo (after hot-frac); off -> constant T")
    ap.add_argument("--tag", type=str, default="",
                     help="suffix for output plot filename")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    ckpt = vae_ckpt_path(args.z_dim, args.beta)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No cached VAE at {ckpt}. Run mnist_vae_train.py first."
        )
    vae = VAE(z_dim=args.z_dim).to(device)
    vae.load_state_dict(torch.load(ckpt, map_location=device))
    vae.eval()
    print(f"Loaded VAE from {ckpt}")

    X_tr, _ = load_mnist_train()
    rng = np.random.default_rng(0)
    sub_idx = rng.choice(len(X_tr), size=8192, replace=False)
    X_sub_t = torch.as_tensor(
        X_tr[sub_idx].reshape(-1, 1, 28, 28), dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        Z_train, _ = vae.encode(X_sub_t)
    print(f"  Z_train: {tuple(Z_train.shape)}  "
          f"||z||₂ median={Z_train.norm(dim=1).median().item():.3f}")

    # ------------------------------------------------------------------
    # Build leverage: phi(z) = RFF(z) only (no raw-z concat by default).
    # ------------------------------------------------------------------
    length_scale = args.length_scale if args.length_scale is not None else "median"
    if args.ell is None:
        rff = RFFLayer(
            in_dim=args.z_dim, n_features=args.M_rff,
            length_scale=length_scale, rff_seed=0,
        ).to(device)
        with torch.no_grad():
            rff.init_bandwidth(Z_train)
    else:
        rff = RFFLayer(
            in_dim=args.z_dim, n_features=args.M_rff,
            length_scale=[args.ell], rff_seed=0,
        ).to(device)
    print(f"  RFF length_scale = {rff.length_scale.tolist()}, M = {args.M_rff}")

    # --- Diagnostic: leverage separation under different phi maps. ----
    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = Path(__file__).resolve().parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    plot_x_to_z_leverage_separation(
        vae, X_sub_t, Z_train, rff, device,
        out_dir / f"mnist_vae_x_to_z_ood{suffix}.png",
        M_rff=args.M_rff,
    )

    h_fn, h_char, D = build_z_leverage(
        rff, Z_train, ridge=1e-3, bias=True, include_raw_z=args.include_raw_z,
    )
    print(f"  phi-dim D = {D}, include_raw_z = {args.include_raw_z}")
    print(f"  h_char (95th pct of in-data h(z)) = {h_char:.4e}")

    with torch.no_grad():
        z_prior = torch.randn(2048, args.z_dim, device=device)
        z_far = 5.0 * torch.randn(2048, args.z_dim, device=device)
        h_in = h_fn(Z_train).cpu().numpy() / h_char
        h_prior = h_fn(z_prior).cpu().numpy() / h_char
        h_far = h_fn(z_far).cpu().numpy() / h_char
    print(f"  h/h_char  in-data  median = {np.median(h_in):.3f}")
    print(f"  h/h_char  N(0,I)   median = {np.median(h_prior):.3f}")
    print(f"  h/h_char  N(0,25)  median = {np.median(h_far):.3f}")

    def E(z: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(h_fn(z) / h_char + 1.0)

    z0 = torch.randn(args.n_part, args.z_dim, device=device)
    z_ref = z0.clone().detach().requires_grad_(True)
    g_ref = torch.autograd.grad(E(z_ref).sum(), z_ref)[0]
    omega = float(g_ref.pow(2).sum(dim=-1).median().item())
    print(f"  Omega (median ||grad E||^2 at prior) = {omega:.4e}")

    dtau_hi = args.dtau
    dtau_lo = args.dtau_lo if args.dtau_lo is not None else args.dtau
    if args.anneal == "on":
        T_hi = args.T
        T_lo = args.T_lo if args.T_lo is not None else args.T * 1e-3
        hot_steps = int(args.hot_frac * args.steps)
        cool_steps = max(1, args.steps - hot_steps)
        T_cool = geometric_anneal(T_hi, T_lo, cool_steps)
        d_cool = geometric_anneal(dtau_hi, dtau_lo, cool_steps)
        def temperature(t: int) -> float:
            return T_hi if t < hot_steps else T_cool(t - hot_steps)
        def dtau_schedule(t: int) -> float:
            return dtau_hi if t < hot_steps else d_cool(t - hot_steps)
    else:
        T_hi = T_lo = args.T
        temperature = float(args.T)
        dtau_schedule = float(args.dtau)
        hot_steps = args.steps
    cfg = SamAdamsConfig(
        dtau=args.dtau, alpha=1.0, s=2.0, Omega=omega,
        m=0.05, M=10.0, r=0.5, kernel="psi1", grad_clip=10.0,
    )
    print(f"  SamAdams n_part={args.n_part}  steps={args.steps}  "
          f"dtau:{dtau_hi}->{dtau_lo}  T:{T_hi}->{T_lo}  "
          f"hot_steps={hot_steps if args.anneal == 'on' else 'all'}")

    out = samadams_sample(
        E, z0, n_steps=args.steps, temperature=temperature,
        dtau_schedule=dtau_schedule, config=cfg,
        record_every=50, zeta_init="g", log_every=1000,
    )
    z_final = out["x_final"].to(device)
    energies = out["energies"]
    print(f"  E start median = {np.median(energies[0]):.4f}")
    print(f"  E end   median = {np.median(energies[-1]):.4f}")

    with torch.no_grad():
        x_final = vae.decode(z_final).cpu().numpy().reshape(-1, 28, 28)
        x_init = vae.decode(z0).cpu().numpy().reshape(-1, 28, 28)
        ref_idx = rng.choice(len(X_tr), size=args.n_part, replace=False)
        X_ref_t = torch.as_tensor(
            X_tr[ref_idx].reshape(-1, 1, 28, 28),
            dtype=torch.float32, device=device,
        )
        mu_ref, _ = vae.encode(X_ref_t)
        x_ref = vae.decode(mu_ref).cpu().numpy().reshape(-1, 28, 28)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
    ax_e = fig.add_subplot(gs[0, :])
    rec_steps = np.linspace(0, args.steps, energies.shape[0])
    ax_e.plot(rec_steps, energies[:, :16], color="gray", alpha=0.3, lw=0.8)
    ax_e.plot(rec_steps, np.median(energies, axis=1),
              color="C3", lw=2.0, label="median")
    ax_e.set_xlabel("step"); ax_e.set_ylabel("E(z)")
    ax_e.set_title(
        f"z-space SamAdams  z_dim={args.z_dim}  M_rff={args.M_rff}  "
        f"ell={rff.length_scale.tolist()}  raw_z={args.include_raw_z}  "
        f"T:{T_hi}->{T_lo}"
    )
    ax_e.legend(); ax_e.grid(alpha=0.3)

    ax1 = fig.add_subplot(gs[1, 0])
    imgrid(ax1, x_init, "decode(z0 ~ N(0,I))")
    ax2 = fig.add_subplot(gs[1, 1])
    imgrid(ax2, x_final, "decode(z_final) — after Langevin")
    ax3 = fig.add_subplot(gs[2, 0])
    imgrid(ax3, x_ref, "decode(encode(MNIST)) — reconstruction ceiling")
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.hist(np.log10(h_in + 1e-6), bins=60, alpha=0.5,
             label="in-data", color="C0")
    ax4.hist(np.log10(h_prior + 1e-6), bins=60, alpha=0.5,
             label="z~N(0,I)", color="C1")
    ax4.set_xlabel("log10(h(z)/h_char)"); ax4.legend()
    ax4.set_title("Leverage separation")

    fig.tight_layout()
    out_path = (Path(__file__).resolve().parent.parent / "out"
                / f"mnist_vae_langevin{suffix}.png")
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
