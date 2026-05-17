"""Forward/backward latency benchmark: DKA vs softmax across sequence lengths.

Sweeps L over a small grid for each of:

    * DKA non-causal, n_heads=1
    * DKA non-causal, n_heads=4
    * DKA causal,     n_heads=4
    * softmax non-causal,
    * softmax causal

Reports median forward/backward/total ms per iteration (median of N timed
iterations after warmup) and writes a JSONL log plus plots to ``../plots/``.

Notes:

* Causal DKA materialises ``(B, H, L, M, M)`` cumulative Grams, so its memory
  grows linearly in L (and the constant has an M² factor). Keep L modest
  unless you have headroom — the default grid maxes at L=512 to stay friendly.
* This is a *micro*-benchmark of the layer in isolation, not an end-to-end
  throughput claim. Use it to compare scaling shapes, not absolute numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import DKA_ROOT  # noqa: F401, E402

from ebmify.models.dka import DKABlock, SoftmaxAttentionBlock  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--d-feat", type=int, default=64)
    ap.add_argument("--kernel-strides", type=int, nargs="+", default=[1, 2, 4],
                    help="DKA multi-kernel stride factors (default: 1 2 4)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument(
        "--return-info",
        action="store_true",
        help="Compute and return info dict during timing (off by default).",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="Wrap each block with torch.compile for timed runs.",
    )
    ap.add_argument("--tag", default="default")
    return ap.parse_args()


def _build_configs(args):
    """Return list of (name, build_fn) for the benchmark grid."""
    d_model, d_feat = args.d_model, args.d_feat
    kernel_strides = tuple(sorted({max(1, int(s)) for s in args.kernel_strides}))
    configs = [
        ("dka_h1_bidir",  lambda: DKABlock(d_model, d_feat, n_heads=1, causal=False, kernel_strides=kernel_strides)),
        ("dka_h4_bidir",  lambda: DKABlock(d_model, d_feat, n_heads=4, causal=False, kernel_strides=kernel_strides)),
        ("dka_h4_causal",  lambda: DKABlock(d_model, d_feat, n_heads=4, causal=True, kernel_strides=kernel_strides)),
        ("soft_h4_bidir",  lambda: SoftmaxAttentionBlock(d_model, n_heads=4, causal=False)),
        ("soft_h4_causal", lambda: SoftmaxAttentionBlock(d_model, n_heads=4, causal=True)),
    ]
    return configs


def _median_ms(values):
    values.sort()
    return values[len(values) // 2]


def time_one(block, x, *, warmup, iters, device, return_info):
    """Median per-iter forward/backward/total ms. Sync on CUDA boundaries."""
    is_cuda = device.type == "cuda"
    block.train()
    # Warmup
    for _ in range(warmup):
        y, _ = block(x, return_info=return_info)
        loss = y.float().pow(2).mean()
        loss.backward()
        block.zero_grad(set_to_none=True)
    if is_cuda:
        torch.cuda.synchronize()
    fwd_times = []
    bwd_times = []
    total_times = []
    for _ in range(iters):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y, _ = block(x, return_info=return_info)
        loss = y.float().pow(2).mean()
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss.backward()
        if is_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        fwd_times.append((t1 - t0) * 1000.0)
        bwd_times.append((t2 - t1) * 1000.0)
        total_times.append((t2 - t0) * 1000.0)
        block.zero_grad(set_to_none=True)
    return {
        "ms_forward": _median_ms(fwd_times),
        "ms_backward": _median_ms(bwd_times),
        "ms_total": _median_ms(total_times),
    }


def maybe_plot(rows, *, out_path, metric_key, metric_label, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"matplotlib not available ({e!r}); skipping plot.")
        return
    by_name: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        by_name.setdefault(r["config"], []).append((r["L"], r[metric_key]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in sorted(by_name):
        pts = sorted(by_name[name])
        L = [p[0] for p in pts]
        ms = [p[1] for p in pts]
        ax.plot(L, ms, marker="o", label=name)
    ax.set_xlabel("seq length L")
    ax.set_ylabel(metric_label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote plot: {out_path}", flush=True)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if (args.dtype == "bf16" and device.type == "cuda") else torch.float32

    log_dir = DKA_ROOT / "logs"
    plot_dir = DKA_ROOT / "plots"
    log_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"speed_{args.tag}.jsonl"
    plot_total_path = plot_dir / f"speed_{args.tag}.png"
    plot_bwd_path = plot_dir / f"speed_{args.tag}_backward.png"
    log_path.write_text("")

    configs = _build_configs(args)
    rows: list[dict] = []
    print(
        f"device={device}  dtype={dtype}  B={args.batch}  d_model={args.d_model}  "
        f"d_feat={args.d_feat}  kernel_strides={tuple(args.kernel_strides)}  "
        f"return_info={args.return_info}  compile={args.compile}",
          flush=True)
    print(f"{'config':<20} {'L':>5}  {'fwd_ms':>10}  {'bwd_ms':>10}  {'total_ms':>10}", flush=True)
    print("-" * 66, flush=True)
    for name, builder in configs:
        for L in args.lengths:
            try:
                torch.manual_seed(0)
                block = builder().to(device=device, dtype=dtype)
                if args.compile:
                    block = torch.compile(block)
                x = torch.randn(args.batch, L, args.d_model, device=device, dtype=dtype)
                timing = time_one(
                    block,
                    x,
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                    return_info=args.return_info,
                )
            except RuntimeError as e:
                print(f"{name:<18} {L:>5}  OOM/err: {e!s}", flush=True)
                with log_path.open("a") as f:
                    f.write(json.dumps({"config": name, "L": L, "error": str(e)[:200]}) + "\n")
                continue
            row = {
                "config": name,
                "L": L,
                "ms_forward": timing["ms_forward"],
                "ms_backward": timing["ms_backward"],
                "ms_total": timing["ms_total"],
                # Back-compat for older plotting scripts.
                "ms_per_iter": timing["ms_total"],
                "batch": args.batch,
                "d_model": args.d_model,
                "dtype": str(dtype),
                "device": str(device),
                "return_info": bool(args.return_info),
                "compile": bool(args.compile),
            }
            rows.append(row)
            print(
                f"{name:<20} {L:>5}  {row['ms_forward']:>10.2f}  "
                f"{row['ms_backward']:>10.2f}  {row['ms_total']:>10.2f}",
                flush=True,
            )
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            del block, x
            if device.type == "cuda":
                torch.cuda.empty_cache()

    maybe_plot(
        rows,
        out_path=plot_total_path,
        metric_key="ms_total",
        metric_label="ms / iter (forward + backward, median)",
        title="DKA vs softmax: total step latency",
    )
    maybe_plot(
        rows,
        out_path=plot_bwd_path,
        metric_key="ms_backward",
        metric_label="ms / iter (backward only, median)",
        title="DKA vs softmax: reverse-AD latency",
    )
    print(f"log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
