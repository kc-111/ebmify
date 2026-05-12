"""Animated demo: greedy max-coverage of rank strata (the spectral_rank pick).

Generates a GIF showing how :func:`coreset.spectral_rank._stratified_rank_select`
picks samples one at a time, each step greedily maximizing the number of
newly-covered ``(bucket, stratum)`` cells.

Setup:

- ``N`` synthetic samples, each represented by a per-bucket rank vector
  ``R[i] in [0, 1]^B`` with mild cross-bucket correlation so picks
  actually have to *reach* for under-covered strata.
- Each bucket's rank axis is partitioned into ``k`` equal strata of
  width ``1/k``. Sample ``i`` occupies stratum
  ``s_b(i) = min(floor(R[i, b] * k), k - 1)`` in bucket ``b``, so each
  sample is a length-``B`` *set* on the universe of ``B * k`` cells.
- Greedy max-coverage: at step ``t`` pick the available sample whose
  set covers the most currently-uncovered cells. This mirrors
  ``_stratified_rank_select`` exactly.

The animation has two panels:

- Left: 2-D scatter of bucket 0 vs bucket 1 ranks. Already-picked
  samples are green; the current pick is circled in orange. Grid lines
  show the stratum boundaries for those two buckets.
- Right: the full ``B x k`` strata grid. Cells already covered are
  green; the current pick's new cells flash orange before fading into
  the covered set. The title tracks ``coverage = covered / (B * k)``.

Usage:
    python example/cifar/coreset/demo_spectral_rank_greedy.py
    python example/cifar/coreset/demo_spectral_rank_greedy.py --B 6 --k 12 --N 150
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import REPO_ROOT  # noqa: F401, E402


def make_demo_ranks(
    N: int, B: int, *, corr: float = 0.3, seed: int = 0,
) -> np.ndarray:
    """Synthetic ``(N, B)`` rank matrix in ``[0, 1]``.

    ``corr in [0, 1]`` controls cross-bucket correlation of the raw scores
    via ``a*z + b*eps`` with ``a = corr``, ``b = sqrt(1 - corr^2)``:

    - ``corr = 0`` : per-bucket scores are independent. Per-stratum cells
      are nearly independent across buckets, so full coverage of the
      ``B * k`` universe is achievable with high probability when
      ``N >> k``.
    - ``corr -> 1`` : the buckets collapse onto a shared latent.
      Sample ``i``'s rank is roughly the same in every bucket, so
      reachable cells lie on the *diagonal* of the strata grid and most
      off-diagonal cells are simply uncoverable -- no algorithm can fill
      those gaps.
    """
    if not 0.0 <= corr <= 1.0:
        raise ValueError(f"corr must be in [0, 1]; got {corr}")
    a = float(corr)
    b = math.sqrt(max(0.0, 1.0 - corr * corr))
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((N, 1))                  # shared latent
    eps = rng.standard_normal((N, B))                # per-bucket noise
    raw = a * z + b * eps
    order = np.argsort(raw, axis=0)
    ranks = np.empty_like(order, dtype=np.float64)
    arange = np.arange(1, N + 1).reshape(-1, 1)
    np.put_along_axis(ranks, order, np.broadcast_to(arange, order.shape), axis=0)
    return ranks / N


def greedy_trace(ranks: np.ndarray, k: int) -> list[dict]:
    """Run greedy max-coverage and snapshot per-step state for animation."""
    N, B = ranks.shape
    strata = np.minimum(np.floor(ranks * k).astype(np.int64), k - 1)  # (N, B)
    covered = np.zeros((B, k), dtype=bool)
    available = np.ones(N, dtype=bool)
    history: list[dict] = []
    for step in range(k):
        # hits[i, b] = True iff sample i's bucket-b cell is already covered.
        hits = covered[np.arange(B)[None, :], strata]                # (N, B)
        score = (~hits).sum(axis=1).astype(np.int32)
        score[~available] = -1
        i_star = int(score.argmax())
        # Snapshot the gain *before* marking the new cells covered.
        new_cells = [(b, int(strata[i_star, b])) for b in range(B)
                     if not covered[b, strata[i_star, b]]]
        for b, s in new_cells:
            covered[b, s] = True
        available[i_star] = False
        history.append({
            "step": step,
            "i_star": i_star,
            "gain": len(new_cells),
            "new_cells": new_cells,
            "covered_after": covered.copy(),
            "total_covered": int(covered.sum()),
        })
    return history


def render_gif(ranks: np.ndarray, k: int, history: list[dict],
               out_path: Path, *, fps: int) -> None:
    N, B = ranks.shape
    total_cells = B * k

    fig, (ax_sc, ax_gr) = plt.subplots(
        1, 2, figsize=(11, 5.2),
        gridspec_kw={"width_ratios": [1.0, 1.2]},
    )

    # Left: 2-D rank scatter for buckets 0 and 1.
    ax_sc.set_xlim(0, 1); ax_sc.set_ylim(0, 1); ax_sc.set_aspect("equal")
    ax_sc.set_xlabel("rank in bucket 0")
    ax_sc.set_ylabel("rank in bucket 1")
    for q in np.linspace(0, 1, k + 1):
        ax_sc.axvline(q, color="gray", lw=0.3, alpha=0.35)
        ax_sc.axhline(q, color="gray", lw=0.3, alpha=0.35)
    ax_sc.scatter(ranks[:, 0], ranks[:, 1], c="#cccccc",
                  s=14, edgecolors="none", zorder=1)
    sel_sc = ax_sc.scatter([], [], c="#2ca25f", s=36,
                           edgecolors="black", linewidths=0.7, zorder=3)
    pick_sc = ax_sc.scatter([], [], c="#fdae6b", s=140,
                            edgecolors="black", linewidths=1.6, zorder=4)
    ax_sc.set_title("rank scatter (buckets 0, 1 shown)", fontsize=10)

    # Right: B x k strata grid.
    ax_gr.set_xlim(-0.5, B - 0.5)
    ax_gr.set_ylim(-0.5, k - 0.5)
    ax_gr.set_xticks(np.arange(B))
    ax_gr.set_yticks(np.arange(0, k, max(1, k // 8)))
    ax_gr.set_xlabel("bucket b")
    ax_gr.set_ylabel("stratum s = floor(R[:, b] * k)")
    ax_gr.invert_yaxis()
    ax_gr.set_aspect("equal")
    cells: list[list[Rectangle]] = []
    for b in range(B):
        col: list[Rectangle] = []
        for s in range(k):
            r = Rectangle((b - 0.5, s - 0.5), 1, 1,
                          facecolor="white", edgecolor="#dddddd", lw=0.4)
            ax_gr.add_patch(r)
            col.append(r)
        cells.append(col)
    ax_gr.set_title(f"strata grid: B x k = {B} x {k} = {total_cells} cells",
                    fontsize=10)

    title = fig.suptitle("", fontsize=12, y=0.98)

    def draw_frame(t: int):
        h = history[t]
        sel_idx = [history[s]["i_star"] for s in range(t + 1)]
        sel_sc.set_offsets(ranks[sel_idx, :2])
        pick_sc.set_offsets(ranks[[h["i_star"]], :2])

        cov = h["covered_after"]
        new_set = set(h["new_cells"])
        for b in range(B):
            for s in range(k):
                if (b, s) in new_set:
                    cells[b][s].set_facecolor("#fdae6b")  # this step's gain
                elif cov[b, s]:
                    cells[b][s].set_facecolor("#2ca25f")  # previously covered
                else:
                    cells[b][s].set_facecolor("white")

        pct = 100.0 * h["total_covered"] / total_cells
        title.set_text(
            f"step {t + 1}/{k}   pick i = {h['i_star']}   "
            f"gain = {h['gain']} new cells   "
            f"coverage = {h['total_covered']}/{total_cells} ({pct:.0f}%)"
        )
        return []

    anim = animation.FuncAnimation(
        fig, draw_frame, frames=len(history),
        interval=1000 // fps, blit=False, repeat=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--N", type=int, default=200,
        help="Number of synthetic samples in the demo (the candidate pool).",
    )
    ap.add_argument(
        "--B", type=int, default=8,
        help="Number of buckets (columns in the strata grid).",
    )
    ap.add_argument(
        "--k", type=int, default=16,
        help=("Coreset size to pick. Also the number of strata per bucket, "
              "so the universe has B * k cells."),
    )
    ap.add_argument(
        "--fps", type=int, default=2,
        help="GIF frame rate. One frame per greedy pick.",
    )
    ap.add_argument(
        "--corr", type=float, default=0.3,
        help=("Cross-bucket correlation of the synthetic ranks, in [0, 1]. "
              "0 = independent buckets (full coverage usually reachable); "
              "near 1 = ranks collapse onto a shared latent, so off-diagonal "
              "strata cells become uncoverable."),
    )
    ap.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for the synthetic rank matrix.",
    )
    ap.add_argument(
        "--out", type=str, default=None,
        help=("Output GIF path. Default: "
              "example/out/coreset/demo_spectral_rank_greedy.gif"),
    )
    args = ap.parse_args()

    ranks = make_demo_ranks(args.N, args.B, corr=args.corr, seed=args.seed)
    history = greedy_trace(ranks, args.k)
    out_path = (
        Path(args.out) if args.out else
        REPO_ROOT / "example" / "out" / "coreset" / "demo_spectral_rank_greedy.gif"
    )
    render_gif(ranks, args.k, history, out_path, fps=args.fps)

    final = history[-1]
    universe = args.B * args.k
    pct = 100 * final["total_covered"] / universe
    print(f"gif -> {out_path}")
    print(f"final coverage = {final['total_covered']}/{universe} ({pct:.1f}%)")
    print(f"per-step gains = {[h['gain'] for h in history]}")


if __name__ == "__main__":
    main()
