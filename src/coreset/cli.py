"""CLI orchestrator: preprocess -> eig -> algorithms -> aux targets.

Usage:
    python -m coreset.cli --config cfg.yaml --out artifacts/

Config keys (yaml or json):
    phi_path:        str (path to .pt/.npy/.npz containing Phi (N, D) float32)
    phi_key:         str (key inside the file; ignored for .pt/.npy)
    budget_k:        int
    ridge_lambda:    float
    n_buckets:       int
    n_top_eigvecs:   int
    standardize:     "center_l2" | "center" | "none"
    algorithms:      [greedy, leverage, spectral_rank]
    aux_targets:     [spectral_coords, bucket_ranks, leverage_score, home_bucket]
    seed:            int
    device:          "cuda" | "cpu"
    chunk_size:      int
    low_rank_eig:    bool
    low_rank_r:      int

Output layout (under ``--out``)::

    out/
      preprocessing.pt
      eig.pt
      bucket_assignment.pt
      feature_stats.json
      <algo>/
        indices.pt
        weights.pt
        stats.json
        config.json
        aux_*.pt

One ``<algo>/`` directory is written for each entry of ``algorithms``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from coreset.preprocess import fit_view
from coreset.eig import compute_eig
from coreset.greedy import greedy_max_variance
from coreset.leverage import ridge_leverage_sample, compute_leverage
from coreset.spectral_rank import (
    spectral_rank_coverage,
    bucket_assignment,
    per_bucket_alignment,
    _ranks_per_column,
)
from coreset.aux_targets import compute_aux_targets


_VALID_ALGOS = ("greedy", "leverage", "spectral_rank")
_VALID_AUX = ("spectral_coords", "bucket_ranks", "leverage_score",
              "home_bucket", "feature_distill")
_VALID_STD = ("center_l2", "center", "none")


@dataclasses.dataclass
class Config:
    """Validated CLI configuration.

    Attributes:
        phi_path: Path to a ``.pt``, ``.npy``, or ``.npz`` file holding the
            ``(N, D)`` float32 feature matrix.
        budget_k: Coreset size for every algorithm in this run.
        ridge_lambda: Ridge ``lambda`` used by leverage and spectral
            formulas.
        n_buckets: Number of eigenvector buckets for spectral-rank
            coverage and the ``bucket_ranks`` / ``home_bucket`` aux
            targets.
        n_top_eigvecs: How many leading eigenvectors to expose via the
            ``spectral_coords`` aux target.
        standardize: ``"none"``, ``"center"``, or ``"center_l2"``.
        algorithms: Subset of ``("greedy", "leverage", "spectral_rank")``.
        aux_targets: Subset of ``("spectral_coords", "bucket_ranks",
            "leverage_score", "home_bucket")``.
        seed: Global RNG seed for the run.
        device: ``"cuda"`` or ``"cpu"``.
        chunk_size: Row block size used by the streaming view.
        low_rank_eig: If True, use randomized subspace iteration.
        low_rank_r: Number of eigenpairs requested in the low-rank path.
        phi_key: Key inside the input file (required for dict-style
            ``.pt``; first array used if omitted for ``.npz``).
    """

    phi_path: str
    budget_k: int
    ridge_lambda: float
    n_buckets: int
    n_top_eigvecs: int
    standardize: str
    algorithms: list[str]
    aux_targets: list[str]
    seed: int
    device: str
    chunk_size: int
    low_rank_eig: bool
    low_rank_r: int
    phi_key: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Build and validate a Config from a YAML/JSON-derived dict.

        Args:
            d: Mapping with the CLI config keys (see module docstring).

        Returns:
            A populated :class:`Config`.

        Raises:
            ValueError: For unknown ``standardize`` / ``algorithms`` /
                ``aux_targets`` values, or missing required keys.
        """
        if d["standardize"] not in _VALID_STD:
            raise ValueError(f"standardize must be one of {_VALID_STD}")
        for a in d["algorithms"]:
            if a not in _VALID_ALGOS:
                raise ValueError(f"unknown algorithm: {a}; valid={_VALID_ALGOS}")
        for a in d["aux_targets"]:
            if a not in _VALID_AUX:
                raise ValueError(f"unknown aux target: {a}; valid={_VALID_AUX}")
        return cls(
            phi_path=str(d["phi_path"]),
            phi_key=d.get("phi_key"),
            budget_k=int(d["budget_k"]),
            ridge_lambda=float(d["ridge_lambda"]),
            n_buckets=int(d["n_buckets"]),
            n_top_eigvecs=int(d["n_top_eigvecs"]),
            standardize=str(d["standardize"]),
            algorithms=list(d["algorithms"]),
            aux_targets=list(d["aux_targets"]),
            seed=int(d.get("seed", 0)),
            device=str(d.get("device", "cuda" if torch.cuda.is_available() else "cpu")),
            chunk_size=int(d.get("chunk_size", 8192)),
            low_rank_eig=bool(d.get("low_rank_eig", False)),
            low_rank_r=int(d.get("low_rank_r", 512)),
        )


def _load_cfg(path: Path) -> Config:
    """Parse a YAML or JSON config file into a validated :class:`Config`.

    Args:
        path: Path to ``.yaml``/``.yml`` (requires PyYAML) or ``.json``.

    Returns:
        Validated :class:`Config`.
    """
    text = Path(path).read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml  # type: ignore
        d = yaml.safe_load(text)
    else:
        d = json.loads(text)
    return Config.from_dict(d)


def _load_phi(path: str, key: str | None) -> np.ndarray | torch.Tensor:
    """Load the feature matrix referenced by the config.

    Args:
        path: File path. Suffix selects the loader:
            ``.pt`` -> torch.load (dict requires ``key``);
            ``.npy`` -> ``np.load(mmap_mode="r")``;
            ``.npz`` -> ``np.load`` then index by ``key`` (or first array).
        key: Optional key inside dict-style ``.pt`` or ``.npz``.

    Returns:
        ``(N, D)`` feature matrix as either a numpy array (possibly memmap)
        or a torch tensor — :class:`coreset.preprocess.StandardizedView`
        accepts both.

    Raises:
        ValueError: For unsupported suffixes or missing required keys.
    """
    p = Path(path)
    if p.suffix == ".pt":
        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, dict):
            if key is None:
                raise ValueError(f"phi_key required for dict-style .pt: keys={list(obj.keys())}")
            return obj[key]
        return obj
    if p.suffix == ".npy":
        return np.load(p, mmap_mode="r")
    if p.suffix == ".npz":
        z = np.load(p, allow_pickle=False)
        if key is None:
            key = z.files[0]
        return z[key]
    raise ValueError(f"unsupported Phi file format: {p.suffix}")


def _run_one_algo(
    name: str,
    view,
    sigma2,
    V,
    cfg: Config,
    leverage_cache: torch.Tensor | None,
    ranks_cache: torch.Tensor | None,
    S_cache: torch.Tensor | None,
    algo_dir: Path,
) -> dict[str, Any]:
    """Run a single algorithm and persist its outputs under ``algo_dir``.

    Args:
        name: One of ``"greedy"``, ``"leverage"``, ``"spectral_rank"``.
        view: Standardized view over ``Phi``.
        sigma2: ``(r,)`` eigenvalues (without ridge).
        V: ``(D, r)`` eigenvectors.
        cfg: Run config supplying ``budget_k``, ``ridge_lambda``, etc.
        leverage_cache: Optional precomputed leverage tensor reused across
            algorithms.
        ranks_cache: Optional precomputed per-bucket ranks tensor.
        S_cache: Optional precomputed per-bucket alignment matrix.
        algo_dir: Output directory (created if missing). Files written:
            ``indices.pt``, ``weights.pt``, ``stats.json``, ``config.json``,
            and one ``aux_*.pt`` per requested aux target.

    Returns:
        Dict of merged stats (algorithm stats plus ``aux_saved`` mapping
        and ``n_selected``).

    Raises:
        ValueError: For unknown ``name``.
    """
    algo_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if name == "greedy":
        idx, stats = greedy_max_variance(
            view, lam=cfg.ridge_lambda, k=cfg.budget_k, seed=cfg.seed
        )
        weights = torch.ones(idx.numel(), dtype=torch.float32)
    elif name == "leverage":
        idx, weights, stats = ridge_leverage_sample(
            view, sigma2=sigma2, V=V, lam=cfg.ridge_lambda, k=cfg.budget_k,
            seed=cfg.seed, leverage=leverage_cache,
        )
    elif name == "spectral_rank":
        idx, _ranks_at_sel, stats = spectral_rank_coverage(
            view, V=V, sigma2=sigma2, lam=cfg.ridge_lambda, k=cfg.budget_k,
            n_buckets=cfg.n_buckets, seed=cfg.seed,
        )
        weights = torch.ones(idx.numel(), dtype=torch.float32)
    else:
        raise ValueError(name)

    torch.save(idx.to(torch.long), algo_dir / "indices.pt")
    torch.save(weights.to(torch.float32), algo_dir / "weights.pt")

    aux_saved = compute_aux_targets(
        view, sigma2, V, cfg.ridge_lambda, idx,
        targets=cfg.aux_targets,
        n_buckets=cfg.n_buckets,
        n_top_eigvecs=cfg.n_top_eigvecs,
        out_dir=algo_dir,
        leverage_cache=leverage_cache,
        ranks_cache=ranks_cache,
        S_cache=S_cache,
    )

    full_stats = {**stats, "aux_saved": aux_saved, "n_selected": int(idx.numel())}
    with open(algo_dir / "stats.json", "w") as f:
        json.dump(full_stats, f, indent=2)
    with open(algo_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    return full_stats


def run(cfg: Config, out: Path) -> dict[str, Any]:
    """End-to-end pipeline: load Phi, fit view, eig, run algorithms, write aux.

    Steps:

    1. Load ``Phi`` from ``cfg.phi_path`` (numpy memmap or torch tensor).
    2. Fit a :class:`StandardizedView`; persist ``preprocessing.pt``.
    3. Eigendecompose ``Phi^T Phi`` (full or low-rank); persist ``eig.pt``.
    4. Persist the contiguous bucket assignment.
    5. Pre-compute shared caches (leverage / S / ranks) if any consumer
       needs them, so each is computed at most once.
    6. Stream once more to get ``mean_row_norm`` and write
       ``feature_stats.json``.
    7. For each algorithm in ``cfg.algorithms``, call :func:`_run_one_algo`.

    Args:
        cfg: Validated :class:`Config`.
        out: Output directory; created if missing.

    Returns:
        Dict with ``setup`` (feature stats) and ``algorithms`` (map of
        algorithm name -> stats dict).
    """
    out.mkdir(parents=True, exist_ok=True)
    art = out
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    Phi = _load_phi(cfg.phi_path, cfg.phi_key)
    if Phi.ndim != 2:
        raise ValueError(f"Phi must be 2D; got shape {Phi.shape}")
    N, D = int(Phi.shape[0]), int(Phi.shape[1])
    t0 = time.time()

    view = fit_view(
        Phi, mode=cfg.standardize, chunk_size=cfg.chunk_size,
        device=cfg.device, out_path=art / "preprocessing.pt",
    )

    sigma2, V = compute_eig(
        view, lam=cfg.ridge_lambda,
        low_rank=cfg.low_rank_eig, r=cfg.low_rank_r, seed=cfg.seed,
        out_path=art / "eig.pt",
    )

    bucket_of = bucket_assignment(sigma2, cfg.n_buckets, mode="equal_mass")
    torch.save(bucket_of, art / "bucket_assignment.pt")

    needs_leverage = ("leverage" in cfg.algorithms) or ("leverage_score" in cfg.aux_targets)
    needs_S = (
        "spectral_rank" in cfg.algorithms
        or "bucket_ranks" in cfg.aux_targets
        or "home_bucket" in cfg.aux_targets
    )
    leverage_cache = None
    S_cache = None
    ranks_cache = None
    if needs_leverage:
        leverage_cache = compute_leverage(view, sigma2, V, cfg.ridge_lambda)
    if needs_S:
        S_cache = per_bucket_alignment(view, V, bucket_of, cfg.n_buckets)
        ranks_cache = _ranks_per_column(S_cache)

    norms_sq_acc = 0.0
    nrows = 0
    for _, chunk in view.stream():
        norms_sq_acc += float((chunk * chunk).sum().item())
        nrows += int(chunk.shape[0])
    mean_norm = float(np.sqrt(norms_sq_acc / max(nrows, 1)))
    eff_dim = float((sigma2.to(cfg.device) /
                     (sigma2.to(cfg.device) + cfg.ridge_lambda)).sum().item())

    feat_stats = {
        "N": N, "D": D,
        "mean_row_norm": mean_norm,
        "effective_dim_lambda": eff_dim,
        "standardize": cfg.standardize,
        "ridge_lambda": cfg.ridge_lambda,
        "low_rank_eig": cfg.low_rank_eig,
        "low_rank_r": cfg.low_rank_r if cfg.low_rank_eig else None,
        "runtime_setup_sec": time.time() - t0,
    }
    with open(art / "feature_stats.json", "w") as f:
        json.dump(feat_stats, f, indent=2)

    all_stats: dict[str, Any] = {"setup": feat_stats, "algorithms": {}}
    for name in cfg.algorithms:
        algo_dir = art / name
        stats = _run_one_algo(
            name, view, sigma2, V, cfg,
            leverage_cache=leverage_cache,
            ranks_cache=ranks_cache,
            S_cache=S_cache,
            algo_dir=algo_dir,
        )
        all_stats["algorithms"][name] = stats
        print(f"[{name}] selected {stats['n_selected']} idx in {stats.get('runtime_sec', 0):.1f}s")
    return all_stats


def main(argv: list[str] | None = None) -> None:
    """Argparse entry point: ``python -m coreset.cli --config cfg --out dir``.

    Args:
        argv: Optional argument list (default: ``sys.argv[1:]``).
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args(argv)
    cfg = _load_cfg(Path(args.config))
    run(cfg, Path(args.out))


if __name__ == "__main__":
    main()
