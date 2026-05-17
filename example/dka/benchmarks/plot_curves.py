"""Plot training curves from JSONL logs under example/dka/logs/.

By default scans every ``*.jsonl`` under logs/ that matches one of the
known stems (cifar denoising or MLM) and produces:

    plots/cifar_<dataset>_<sigma>__val_psnr.png
    plots/cifar_<dataset>_<sigma>__val_mse.png
    plots/mlm__val_loss.png

Each curve is labeled by the run's `tag` (the part after the dataset/noise
stem). Use ``--glob`` to filter; ``--tag-rename`` to relabel one or more
runs in the legend.

Useful for quick "DKA vs softmax" comparisons after a sweep finishes:

    python example/dka/benchmarks/plot_curves.py
    python example/dka/benchmarks/plot_curves.py --glob 'dka*'
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import DKA_ROOT  # noqa: F401, E402


CIFAR_STEM = re.compile(r"^(dka|softmax)_cifar_(?P<dataset>cifar10|cifar100)_s(?P<sigma>[\d.]+)_(?P<tag>.+)$")
MLM_STEM = re.compile(r"^(dka|softmax)_mlm_(?P<tag>.+)$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="*",
                    help="filename glob applied to logs/*.jsonl (without extension)")
    ap.add_argument("--out-dir", default=str(DKA_ROOT / "plots"),
                    help="where to write PNGs")
    ap.add_argument("--tag-rename", nargs="*", default=[],
                    help="entries 'oldtag=newtag' to relabel legend entries")
    return ap.parse_args()


def _load_jsonl(p: Path) -> list[dict]:
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _plot(curves: dict[str, list[tuple[float, float]]], *,
          out_path: Path, xlabel: str, ylabel: str, title: str,
          ylog: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not curves:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in sorted(curves):
        pts = sorted(curves[name])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=name, markersize=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylog:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def main() -> None:
    args = parse_args()
    log_dir = DKA_ROOT / "logs"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    rename = dict(s.split("=", 1) for s in args.tag_rename if "=" in s)

    cifar_groups: dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    cifar_groups_mse: dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    mlm_curves: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for p in sorted(log_dir.glob("*.jsonl")):
        stem = p.stem
        if not fnmatch.fnmatch(stem, args.glob):
            continue
        rows = _load_jsonl(p)
        if not rows:
            continue

        m = CIFAR_STEM.match(stem)
        if m:
            dataset, sigma, tag = m.group("dataset"), m.group("sigma"), m.group("tag")
            arch = stem.split("_", 1)[0]
            label = rename.get(tag, f"{arch}/{tag}")
            for r in rows:
                if "val_psnr_db" in r:
                    cifar_groups[(arch, dataset, sigma)][label].append((r["epoch"], r["val_psnr_db"]))
                if "val_mse" in r:
                    cifar_groups_mse[(arch, dataset, sigma)][label].append((r["epoch"], r["val_mse"]))
            continue
        m = MLM_STEM.match(stem)
        if m:
            tag = m.group("tag")
            arch = stem.split("_", 1)[0]
            label = rename.get(tag, f"{arch}/{tag}")
            for r in rows:
                if "val_loss" in r:
                    mlm_curves[label].append((r["step"], r["val_loss"]))

    # CIFAR: merge across archs (the dataset/sigma group is the natural axis;
    # the legend distinguishes arch via the label prefix).
    merged_psnr: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    merged_mse: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for (_arch, dataset, sigma), curves in cifar_groups.items():
        for label, pts in curves.items():
            merged_psnr[(dataset, sigma)][label].extend(pts)
    for (_arch, dataset, sigma), curves in cifar_groups_mse.items():
        for label, pts in curves.items():
            merged_mse[(dataset, sigma)][label].extend(pts)

    for (dataset, sigma), curves in merged_psnr.items():
        _plot(curves,
              out_path=out_dir / f"cifar_{dataset}_s{sigma}__val_psnr.png",
              xlabel="epoch", ylabel="val PSNR (dB)",
              title=f"{dataset} σ={sigma}: val PSNR")
    for (dataset, sigma), curves in merged_mse.items():
        _plot(curves,
              out_path=out_dir / f"cifar_{dataset}_s{sigma}__val_mse.png",
              xlabel="epoch", ylabel="val MSE (patch)",
              title=f"{dataset} σ={sigma}: val MSE",
              ylog=True)
    if mlm_curves:
        _plot(mlm_curves,
              out_path=out_dir / "mlm__val_loss.png",
              xlabel="step", ylabel="masked CE",
              title="MLM val loss")

    if not (merged_psnr or mlm_curves):
        print(f"no matching logs under {log_dir} for glob={args.glob!r}", flush=True)


if __name__ == "__main__":
    main()
