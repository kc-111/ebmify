"""Diagnostic plots for coreset selections.

Compares the *selected* sub-population to the *full* feature pool along
statistics derived from the eigendecomposition:

1. Leverage-score histogram (log-y) of all samples with the selected
   subset overlaid per algorithm.
2. Home-bucket histogram of selected samples vs population, per algorithm.
3. Top-2 spectral-coord scatter of all samples with selected samples
   highlighted per algorithm.
4. Per-bucket rank spread per algorithm: median line plus 25-75% and
   5-95% bands. The bands should sit at the corresponding Uniform[0,1]
   target quantiles (flat lines at 0.5 / [0.25, 0.75] / [0.05, 0.95])
   when the selection is rank-balanced both across and within buckets.

These are statistics *about* the selection, not raw images — fits the
"no samples written" output contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


_ALGO_STYLE = {
    "greedy":        ("C3", "Greedy max-variance"),
    "leverage":      ("C0", "Ridge leverage sample"),
    "spectral_rank": ("C2", "Spectral-rank coverage"),
}


def _load_selected(art: Path, algo: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Load indices and (optionally) the leverage aux for one algorithm.

    Args:
        art: Top-level artifacts directory containing per-algo subdirs.
        algo: Subdirectory name (algorithm tag).

    Returns:
        ``(indices_np, leverage_np_or_None)``. ``leverage_np_or_None`` is
        ``None`` when ``aux_leverage_score.pt`` was not written.
    """
    idx = torch.load(art / algo / "indices.pt").cpu().numpy()
    lev = None
    p = art / algo / "aux_leverage_score.pt"
    if p.exists():
        lev = torch.load(p).cpu().numpy()
    return idx, lev


def _full_population_leverage(art: Path,
                              compute_if_missing: bool = False) -> np.ndarray | None:
    """Load the population leverage from a side cache next to artifacts.

    The CLI does not write the full-N leverage by default (it's ``N`` floats
    and we want zero-bloat outputs). Wrappers that want the diagnostic plots
    pass it in via ``plot_selection_diagnostics(..., population_leverage=...)``.

    Args:
        art: Top-level artifacts directory.
        compute_if_missing: Currently unused; kept for forward-compat.

    Returns:
        ``(N,)`` numpy array if ``population_leverage.pt`` is present,
        otherwise ``None``.
    """
    cache = art / "population_leverage.pt"
    if cache.exists():
        return torch.load(cache).cpu().numpy()
    return None


def plot_selection_diagnostics(
    art: Path,
    *,
    algorithms: list[str] | None = None,
    population_leverage: np.ndarray | None = None,
    population_S: np.ndarray | None = None,
    population_ranks: np.ndarray | None = None,
    population_coords: np.ndarray | None = None,
    out_path: Path | None = None,
    suptitle: str = "Coreset selection diagnostics",
) -> Path:
    """Render the four-panel diagnostic figure.

    Panels:

    1. Leverage histogram of the full pool with each algorithm's selected
       leverages overlaid (step histograms).
    2. Home-bucket fraction comparison (population vs each algorithm).
    3. Top-2 spectral-coord scatter of population with the selected
       samples highlighted per algorithm.
    4. Per-bucket rank-spread plot: for each algorithm, plot the
       median, 25-75% IQR band, and 5-95% band of the selected-sample
       ranks vs bucket id. Uniform target quantiles are drawn as
       dotted gray lines; flat bands hugging those lines indicate the
       selection is rank-uniform across and within buckets.

    Args:
        art: Artifacts root directory written by the CLI; must contain
            ``feature_stats.json`` and one subdir per algorithm.
        algorithms: List of algorithm subdirs to include. If ``None``,
            auto-detect by scanning ``art`` for subdirs with ``indices.pt``.
        population_leverage: Optional ``(N,)`` array of full-pool leverage
            scores. Panel 1 falls back to selected-only if absent.
        population_S: Optional ``(N, B)`` per-bucket alignment matrix.
            Used to compute the population home-bucket distribution for
            panel 2 when present.
        population_ranks: Optional ``(N, B)`` per-bucket ranks array
            (currently unused but accepted for API stability).
        population_coords: Optional ``(N, >=2)`` top-eigvec coordinates;
            forms the gray scatter backdrop in panel 3.
        out_path: Optional override for the output PNG path. Defaults to
            ``art / "selection_diagnostics.png"``.
        suptitle: Figure supertitle.

    Returns:
        Path to the written PNG file.
    """
    art = Path(art)
    if algorithms is None:
        algorithms = [d.name for d in art.iterdir() if d.is_dir() and (d / "indices.pt").exists()]
        algorithms.sort()

    with open(art / "feature_stats.json") as f:
        feat_stats = json.load(f)
    N = int(feat_stats["N"])

    if population_leverage is None:
        population_leverage = _full_population_leverage(art)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_lev, ax_bucket, ax_scatter, ax_rank = axes.flat

    # ---- 1. Leverage histogram with selected overlaid ----
    if population_leverage is not None:
        bins = np.linspace(0, float(np.percentile(population_leverage, 99.5)), 60)
        ax_lev.hist(population_leverage, bins=bins, color="lightgray",
                    label=f"population (N={N})", alpha=0.9)
    for algo in algorithms:
        idx, lev = _load_selected(art, algo)
        if lev is None:
            continue
        color, label = _ALGO_STYLE.get(algo, ("C5", algo))
        bins = np.linspace(0, float(np.percentile(lev, 99.5)), 40)
        ax_lev.hist(lev, bins=bins, color=color, alpha=0.6, label=label,
                    histtype="step", lw=2.0)
    ax_lev.set_xlabel("leverage h(x)")
    ax_lev.set_ylabel("count")
    ax_lev.set_yscale("log")
    ax_lev.set_title("Leverage of selected vs full pool")
    ax_lev.legend(fontsize=8)

    # ---- 2. Home-bucket histogram ----
    hb_population = None
    if population_S is not None:
        hb_population = np.asarray(population_S).argmax(axis=1)
        n_buckets = population_S.shape[1]
    else:
        # Fallback: try to read stats.json from any algo with home_bucket info
        n_buckets = None
        for algo in algorithms:
            p = art / algo / "stats.json"
            if p.exists():
                d = json.load(open(p))
                if "home_bucket_histogram" in d:
                    hb_population = np.asarray(d["home_bucket_histogram"], dtype=float)
                    n_buckets = len(hb_population)
                    break
    if n_buckets is not None:
        bx = np.arange(n_buckets)
        if hb_population is not None and hb_population.ndim == 1 \
                and hb_population.shape[0] == n_buckets and hb_population.sum() > 0:
            # treat as histogram already
            hb_pop_hist = hb_population
            ax_bucket.bar(bx - 0.3, hb_pop_hist / hb_pop_hist.sum(),
                          width=0.3, color="lightgray", label="population")
        elif hb_population is not None:
            counts = np.bincount(hb_population, minlength=n_buckets).astype(float)
            ax_bucket.bar(bx - 0.3, counts / counts.sum(),
                          width=0.3, color="lightgray", label="population")
        offset_step = 0.25
        for i_a, algo in enumerate(algorithms):
            p = art / algo / "aux_home_bucket.pt"
            if not p.exists():
                continue
            hb = torch.load(p).cpu().numpy()
            counts = np.bincount(hb, minlength=n_buckets).astype(float)
            color, label = _ALGO_STYLE.get(algo, ("C5", algo))
            ax_bucket.bar(bx + i_a * offset_step,
                          counts / counts.sum(),
                          width=offset_step * 0.9, color=color,
                          alpha=0.85, label=label)
        ax_bucket.axhline(1.0 / n_buckets, color="gray", lw=0.8, ls="--",
                          label=f"uniform coverage (1/B = {1.0 / n_buckets:.3f})")
        ax_bucket.set_xlabel("home bucket id")
        ax_bucket.set_ylabel("fraction")
        ax_bucket.set_title("Home-bucket distribution: selected vs population")
        ax_bucket.legend(fontsize=8)
    else:
        ax_bucket.set_title("Home-bucket — not provided")
        ax_bucket.axis("off")

    # ---- 3. Top-2 spectral coords scatter ----
    if population_coords is not None and population_coords.shape[1] >= 2:
        pc = np.asarray(population_coords)
        ax_scatter.scatter(pc[:, 0], pc[:, 1], s=2, c="lightgray",
                           alpha=0.35, rasterized=True, label="population")
    for algo in algorithms:
        p = art / algo / "aux_spectral_coords.pt"
        if not p.exists():
            continue
        coords = torch.load(p).cpu().numpy()
        if coords.shape[1] < 2:
            continue
        color, label = _ALGO_STYLE.get(algo, ("C5", algo))
        ax_scatter.scatter(coords[:, 0], coords[:, 1], s=6, c=color,
                           alpha=0.7, label=label, rasterized=True)
    ax_scatter.set_xlabel("$V_0^T \\phi$  (top eigvec coord)")
    ax_scatter.set_ylabel("$V_1^T \\phi$")
    ax_scatter.set_title("Selected samples in top-2 eigvec coords")
    ax_scatter.legend(fontsize=8)

    # ---- 4. Per-bucket rank spread (median + IQR + 5-95% band) ----
    ranks_by_algo: dict[str, np.ndarray] = {}
    for algo in algorithms:
        p = art / algo / "aux_bucket_ranks.pt"
        if not p.exists():
            continue
        ranks_by_algo[algo] = torch.load(p).cpu().numpy()
    if ranks_by_algo:
        qs = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        B = next(iter(ranks_by_algo.values())).shape[1]
        bx = np.arange(B)
        for algo, r in ranks_by_algo.items():
            color, label = _ALGO_STYLE.get(algo, ("C5", algo))
            Q = np.quantile(r, qs, axis=0)  # (5, B)
            ax_rank.fill_between(bx, Q[0], Q[4], color=color, alpha=0.10)
            ax_rank.fill_between(bx, Q[1], Q[3], color=color, alpha=0.22)
            ax_rank.plot(bx, Q[2], color=color, lw=1.6, label=label)
        # Uniform[0,1] target quantiles -- selected marginals should hug these.
        for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
            ax_rank.axhline(q, color="gray", lw=0.5, ls=":", alpha=0.5)
        ax_rank.set_xlabel("bucket id")
        ax_rank.set_ylabel("rank in [0, 1]")
        ax_rank.set_ylim(-0.02, 1.02)
        ax_rank.set_xlim(-0.5, B - 0.5)
        ax_rank.set_title("Per-bucket rank spread (median, IQR, 5-95% band)"
                          "\nuniform target: flat at 0.05/0.25/0.5/0.75/0.95")
        ax_rank.legend(fontsize=8, loc="center right")
    else:
        ax_rank.text(0.5, 0.5, "bucket_ranks aux not enabled",
                     ha="center", va="center", transform=ax_rank.transAxes)
        ax_rank.set_title("Per-bucket rank spread (aux not written)")

    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    if out_path is None:
        out_path = art / "selection_diagnostics.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_spectrum(art: Path, out_path: Path | None = None) -> Path:
    """Eigenvalue spectrum + cumulative effective dimension.

    Reads ``art / "eig.pt"`` (produced by :func:`coreset.eig.compute_eig`)
    and writes a two-panel PNG: left = ``log(sigma2 + lam)`` vs eigvec
    index, right = cumulative ``sigma2 / (sigma2 + lam)`` (the
    effective-dimension curve).

    Args:
        art: Artifacts root directory.
        out_path: Optional override for the output PNG. Defaults to
            ``art / "spectrum.png"``.

    Returns:
        Path to the written PNG file.
    """
    art = Path(art)
    obj = torch.load(art / "eig.pt", map_location="cpu")
    sigma2 = obj["sigma2"].numpy()
    lam = float(obj["lam"])
    eff = sigma2 / (sigma2 + lam)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(sigma2 + lam, lw=1.5, color="C0")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("eigvec index (desc)")
    axes[0].set_ylabel("$\\sigma^2 + \\lambda$")
    axes[0].set_title("Feature covariance spectrum")
    axes[1].plot(np.cumsum(eff), lw=1.5, color="C2")
    axes[1].set_xlabel("eigvec index (desc)")
    axes[1].set_ylabel("cumulative $\\sigma^2 / (\\sigma^2 + \\lambda)$")
    axes[1].set_title(f"effective dim @ $\\lambda$={lam:.1e}: {eff.sum():.1f}")
    fig.tight_layout()
    if out_path is None:
        out_path = art / "spectrum.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
