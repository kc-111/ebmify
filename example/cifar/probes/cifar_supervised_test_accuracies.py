"""Evaluate every supervised ResNet18-CIFAR checkpoint on the test set.

Walks ``example/cifar/cache/`` for all supervised checkpoints written by
``cifar_resnet18_train.py`` and ``coreset/cifar_train_from_artifacts.py``,
then evaluates each on the chosen dataset's test split. Every saved
checkpoint carries both EMA weights (under ``state_dict``) and raw
non-EMA weights (under ``raw_state_dict``); this script reports test
accuracy for both heads so you can see the EMA delta directly, side by
side with the train-time accuracies recorded in the checkpoint metadata.

Discovery globs (relative to ``example/cifar/cache/``):

* ``<dataset>_resnet18*.pt``               -- full-train baseline(s),
                                              including tagged variants
                                              produced via
                                              ``--tag <name>``.
* ``coreset_models/*/<algo>/model.pt``     -- per-coreset checkpoints
                                              from
                                              ``cifar_train_from_artifacts.py``.

Expected checkpoint schema (set by both training scripts):

* ``state_dict``       -- EMA weights, with BN running stats already
                          refreshed against the train set before save.
* ``raw_state_dict``   -- non-EMA weights from the same training run.
* ``config``           -- ``{"arch": str, "num_classes": int}``.
* ``best_acc``         -- raw-weight peak test accuracy across epochs.
* ``best_ema_acc``     -- EMA-weight peak test accuracy across epochs
                          (EMA is only evaluated every 10 epochs +
                          final epoch, so this lags ``best_acc`` in
                          granularity).
* ``final_ema_acc``    -- EMA-weight test accuracy after the last
                          BN-stat refresh, matching what is in
                          ``state_dict``.
* ``algorithm``, ``tag``, ``epochs``, ``seed``
                       -- present only on coreset checkpoints.

We do **not** rebuild BN running stats here: ``cifar_resnet18_train.py``
already calls ``_update_bn_stats`` on the EMA shadow before saving, so
the EMA buffers in ``state_dict`` are already aligned with the EMA
weights. Raw weights' BN stats come from the training loop directly.

Note on EMA-vs-saved drift: the EMA shadow's parameters keep updating
after the last BN-stat refresh that produced ``best_ema_acc``, so the
EMA ``state_dict`` we load here corresponds to ``final_ema_acc`` (the
post-training snapshot), not to ``best_ema_acc``. A small gap between
the freshly-computed ``EMA test_acc`` and ``saved_best_ema_acc`` is
normal -- they are different epochs.

Usage:
    python example/cifar/probes/cifar_supervised_test_accuracies.py
    python example/cifar/probes/cifar_supervised_test_accuracies.py \\
        --dataset cifar10 --json out.json
    python example/cifar/probes/cifar_supervised_test_accuracies.py \\
        --paths example/cifar/cache/cifar10_resnet18.pt --no-raw
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# ``_paths`` registers cifar/ and its subdirs on sys.path so the
# ``cifar_data`` / ``cifar_resnet18_train`` imports below resolve even
# though they live in sibling directories. The symbol itself is unused
# (the import is for its side effects).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import REPO_ROOT  # noqa: F401, E402

from cifar_data import load_cifar_test  # noqa: E402
from cifar_resnet18_train import evaluate, make_resnet18_cifar  # noqa: E402

# All cached checkpoints, features, and intermediate artifacts live
# here. The training scripts use the same constant indirectly via
# ``Path(__file__).resolve().parent.parent / "cache"``.
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def discover_checkpoints(dataset: str) -> list[Path]:
    """Find every supervised-ResNet18 checkpoint we know how to load.

    Top-level baselines live at ``cache/<dataset>_resnet18*.pt`` and
    coreset checkpoints live one level deeper under
    ``cache/coreset_models/<tag>/<algo>/model.pt``. We skip top-level
    SSL backbones (``<dataset>_ssl_*``) because they don't carry a
    classification head -- the SSL pretraining replaces ``fc`` with
    Identity, so loading them through ``make_resnet18_cifar`` would
    leave the readout randomly initialized and the test accuracy would
    be near chance.

    Args:
        dataset: Either ``"cifar10"`` or ``"cifar100"``. Used only as a
            filename prefix for top-level baselines; coreset
            checkpoints are not dataset-prefixed and are returned
            regardless (their ``config['num_classes']`` is checked at
            load time).

    Returns:
        A list of absolute ``Path`` objects, sorted: baselines first
        (alphabetical), then coreset checkpoints by tag then algorithm.
        Empty if no matches.
    """
    found: list[Path] = []
    # Top-level baselines: ``cifar10_resnet18.pt`` and any tagged
    # variants saved with ``--tag <name>`` (``cifar10_resnet18_tag.pt``).
    for p in sorted(CACHE_DIR.glob(f"{dataset}_resnet18*.pt")):
        found.append(p)
    # Coreset variants: one checkpoint per (artifact tag, algorithm)
    # pair. The ``coreset_models/`` root may not exist on fresh
    # installs; that is not an error, just an empty result.
    coreset_root = CACHE_DIR / "coreset_models"
    if coreset_root.is_dir():
        for p in sorted(coreset_root.glob("*/*/model.pt")):
            found.append(p)
    return found


def short_name(p: Path) -> str:
    """Display name for a checkpoint -- path relative to ``cache/``.

    Falls back to the bare filename for paths outside ``cache/`` (e.g.
    when the caller passes ``--paths`` with an arbitrary location).

    Args:
        p: Absolute (or relative-cwd) path to a checkpoint file.

    Returns:
        A short, log-friendly string. For
        ``cache/coreset_models/tag/algo/model.pt`` this is
        ``coreset_models/tag/algo/model.pt``; for an arbitrary path it
        is ``p.name``.
    """
    try:
        return str(p.relative_to(CACHE_DIR))
    except ValueError:
        return p.name


def _eval_weights(model: torch.nn.Module, state: dict,
                  X_te: torch.Tensor, y_te: torch.Tensor, *,
                  batch_size: int, device: str) -> float:
    """Load ``state`` into ``model`` and run a single test-set eval pass.

    Mutates ``model`` in place. ``channels_last`` matches the memory
    format used at train time, so we avoid a tensor reshape per forward
    pass when the GPU supports it.

    Args:
        model: A freshly built ``make_resnet18_cifar`` instance. Reused
            across the EMA/raw evals to skip a second arch
            instantiation.
        state: A state_dict from the checkpoint -- either
            ``raw["state_dict"]`` (EMA) or ``raw["raw_state_dict"]``.
        X_te: ``(N, 3, 32, 32)`` float32 test images in ``[0, 1]``,
            already on ``device``.
        y_te: ``(N,)`` int64 labels, already on ``device``.
        batch_size: Eval batch size (no gradient memory, can be larger
            than the train batch size).
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        Top-1 accuracy on the test set as a float in ``[0, 1]``.
    """
    model.load_state_dict(state)
    model.to(device).to(memory_format=torch.channels_last)
    return evaluate(model, X_te, y_te, batch_size=batch_size, device=device)


def evaluate_checkpoint(ckpt_path: Path, X_te: torch.Tensor,
                        y_te: torch.Tensor, *, batch_size: int,
                        device: str, eval_raw: bool) -> dict:
    """Load one checkpoint, evaluate both EMA and raw heads, return a record.

    Args:
        ckpt_path: Path to a ``.pt`` file matching the checkpoint
            schema documented in the module docstring.
        X_te: ``(N, 3, 32, 32)`` float32 test images on ``device``.
        y_te: ``(N,)`` int64 labels on ``device``.
        batch_size: Forwarded to ``_eval_weights``.
        device: ``"cuda"`` or ``"cpu"``.
        eval_raw: If ``False``, skip the raw-weights eval and leave
            ``raw_test_acc=None`` in the returned dict. Useful when
            only the EMA head matters (matches the upstream OOD-eval
            convention).

    Returns:
        A flat dict suitable for JSON dumping. Keys:

        * ``checkpoint``           -- absolute checkpoint path as str.
        * ``name``                 -- ``short_name`` of the checkpoint.
        * ``num_classes``          -- from ``config['num_classes']``,
                                       defaulting to 10.
        * ``arch``                 -- from ``config['arch']``, or
                                       ``None`` if missing.
        * ``ema_test_acc``         -- freshly computed top-1 in
                                       ``[0, 1]`` for the EMA weights.
        * ``raw_test_acc``         -- top-1 for raw weights, or
                                       ``None`` if skipped / absent.
        * ``saved_best_acc``       -- train-time raw-weight peak
                                       (across all epochs).
        * ``saved_best_ema_acc``   -- train-time EMA-weight peak
                                       (only sampled every 10 epochs +
                                       final).
        * ``saved_final_ema_acc``  -- train-time EMA accuracy after the
                                       last BN refresh; matches
                                       ``ema_test_acc`` up to floating
                                       point (use this to sanity-check
                                       BN-stat alignment).
        * ``algorithm``/``tag``/``epochs``/``seed``
                                   -- coreset-specific metadata or
                                      ``None`` for baselines.

    Raises:
        KeyError: if the checkpoint has no ``state_dict`` key. Other
            torch errors (wrong arch, mismatched shapes) bubble up to
            the caller, which catches them and skips the entry.
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = raw.get("config", {})
    # Defaulting to 10 matches the only currently-supervised dataset
    # (cifar10). If the checkpoint was for cifar100 we will fail with a
    # shape mismatch when loading state_dict; the caller catches that.
    num_classes = int(cfg.get("num_classes", 10))
    model = make_resnet18_cifar(num_classes=num_classes)

    ema_acc = _eval_weights(
        model, raw["state_dict"], X_te, y_te,
        batch_size=batch_size, device=device,
    )
    raw_acc: float | None = None
    if eval_raw and "raw_state_dict" in raw:
        # Reuse the same ``model`` -- load_state_dict overwrites every
        # parameter and buffer, so the previous EMA eval leaves no
        # residue.
        raw_acc = _eval_weights(
            model, raw["raw_state_dict"], X_te, y_te,
            batch_size=batch_size, device=device,
        )

    return {
        "checkpoint": str(ckpt_path),
        "name": short_name(ckpt_path),
        "num_classes": num_classes,
        "arch": cfg.get("arch"),
        "ema_test_acc": float(ema_acc),
        "raw_test_acc": None if raw_acc is None else float(raw_acc),
        "saved_best_acc": _maybe_float(raw.get("best_acc")),
        "saved_best_ema_acc": _maybe_float(raw.get("best_ema_acc")),
        "saved_final_ema_acc": _maybe_float(raw.get("final_ema_acc")),
        # Coreset checkpoints carry these; baselines leave them as
        # ``None`` so the table column for those rows simply prints as
        # missing.
        "algorithm": raw.get("algorithm"),
        "tag": raw.get("tag"),
        "epochs": raw.get("epochs"),
        "seed": raw.get("seed"),
    }


def _maybe_float(v) -> float | None:
    """Coerce ``v`` to float, returning ``None`` for missing or unparseable.

    Used to normalise the saved-metadata fields, which may be Python
    floats, numpy scalars, or absent entirely depending on which
    training script produced the checkpoint.

    Args:
        v: Anything ``float(...)`` might accept, ``None``, or some
            other unconvertible value.

    Returns:
        ``float(v)`` on success; ``None`` on ``v is None`` or any
        ``TypeError`` / ``ValueError`` during conversion.
    """
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pct(x: float | None) -> str:
    """Format a fraction in ``[0, 1]`` as a fixed-width percent string.

    A right-aligned width of 7 chars (e.g. ``" 95.72%"``) keeps the
    summary table columns flush. ``None`` becomes a dash placeholder of
    the same visible width so missing cells line up with present ones.

    Args:
        x: A fraction in ``[0, 1]`` or ``None``.

    Returns:
        ``" 95.72%"``-style string, or ``"    -   "`` (8 chars,
        matching the field width) when ``x is None``.
    """
    return "    -   " if x is None else f"{x * 100:6.2f}%"


def main() -> None:
    """CLI entry point: discover, evaluate, summarise.

    Side effects:
        * Prints a per-checkpoint block plus a final summary table to
          stdout.
        * Optionally writes ``args.json_out`` (a dict with keys
          ``dataset`` and ``results``).

    Exits:
        With ``SystemExit`` if discovery returns zero checkpoints
        (likely a fresh repo where ``cifar_resnet18_train.py`` has not
        been run yet).
    """
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--dataset", default="cifar10",
        choices=["cifar10", "cifar100"],
        help=("test split to evaluate against. Only cifar10 is "
              "currently produced by the supervised training scripts; "
              "cifar100 is accepted for forward compatibility and will "
              "fail-loud at state_dict load time if a discovered "
              "checkpoint was trained for cifar10 (default: cifar10)"),
    )
    ap.add_argument(
        "--batch", type=int, default=256,
        help=("eval batch size. Test-time only, so we can be more "
              "aggressive than training (default: 256)"),
    )
    ap.add_argument(
        "--no-raw", action="store_true", dest="no_raw",
        help=("skip evaluating raw (non-EMA) weights. Halves the "
              "eval work and matches the downstream OOD-eval convention "
              "which only consumes ``state_dict``"),
    )
    ap.add_argument(
        "--json", type=str, default=None, dest="json_out",
        help=("optional path to dump results as JSON. Schema: "
              "``{dataset: str, results: list[record]}`` where each "
              "record matches the dict returned by "
              "``evaluate_checkpoint``"),
    )
    ap.add_argument(
        "--paths", nargs="+", default=None, type=Path,
        help=("explicit checkpoint paths (overrides discovery). "
              "Useful to spot-check a single file or to evaluate "
              "checkpoints outside ``cache/``"),
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    # Load test set once and lift to device; every checkpoint reuses
    # the same tensors. Float32 matches the train-time normalisation
    # path in ``cifar_resnet18_train.evaluate``.
    X_te, y_te = load_cifar_test(args.dataset)
    X_te_t = torch.as_tensor(X_te, dtype=torch.float32, device=device)
    y_te_t = torch.as_tensor(y_te, dtype=torch.long, device=device)
    print(f"{args.dataset} test set: {X_te.shape[0]} images, "
          f"{int(y_te.max()) + 1} classes")

    # ``--paths`` is the explicit-override mode; otherwise scan the
    # cache directory using the documented globs.
    ckpts = ([Path(p) for p in args.paths] if args.paths
             else discover_checkpoints(args.dataset))
    if not ckpts:
        raise SystemExit(
            f"no supervised checkpoints found for {args.dataset} under "
            f"{CACHE_DIR}. Run cifar_resnet18_train.py first."
        )
    print(f"\ndiscovered {len(ckpts)} checkpoint(s):")
    for p in ckpts:
        print(f"  - {short_name(p)}")

    results: list[dict] = []
    for p in ckpts:
        if not p.exists():
            # ``--paths`` can name nonexistent files; discovery cannot,
            # but the symmetric check keeps both branches safe.
            print(f"\n[skip] {short_name(p)}: file missing")
            continue
        print(f"\n=== {short_name(p)} ===")
        try:
            r = evaluate_checkpoint(
                p, X_te_t, y_te_t,
                batch_size=args.batch, device=device,
                eval_raw=not args.no_raw,
            )
        except Exception as e:
            # We deliberately swallow every exception per checkpoint
            # so one bad file (wrong arch, missing state_dict, dataset
            # mismatch) doesn't abort the whole sweep. The error type
            # plus message is enough to debug afterwards.
            print(f"  [error] {type(e).__name__}: {e}")
            continue
        results.append(r)
        print(f"  arch       = {r['arch']}  num_classes = {r['num_classes']}")
        if r.get("algorithm"):
            # Only coreset checkpoints carry these fields; the baseline
            # row stays compact.
            print(f"  algorithm  = {r['algorithm']}  tag = {r['tag']}  "
                  f"epochs = {r['epochs']}  seed = {r['seed']}")
        print(f"  EMA test_acc = {_pct(r['ema_test_acc'])}")
        print(f"  raw test_acc = {_pct(r['raw_test_acc'])}")
        print(f"  saved metadata:")
        print(f"    best_acc      = {_pct(r['saved_best_acc'])}")
        print(f"    best_ema_acc  = {_pct(r['saved_best_ema_acc'])}")
        print(f"    final_ema_acc = {_pct(r['saved_final_ema_acc'])}")

    if not results:
        print("\nno checkpoints evaluated.")
        return

    # Width the ``model`` column to the widest name actually printed,
    # bounded below by the header text so the layout never collapses.
    name_w = max(len(r["name"]) for r in results)
    name_w = max(name_w, len("model"))
    print("\n=== summary ===")
    header = (f"  {'model':<{name_w}}  {'ncls':>4}  "
              f"{'ema_test':>9}  {'raw_test':>9}  "
              f"{'saved_ema_best':>14}  {'saved_raw_best':>14}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        print(
            f"  {r['name']:<{name_w}}  {r['num_classes']:>4}  "
            f"{_pct(r['ema_test_acc']):>9}  {_pct(r['raw_test_acc']):>9}  "
            f"{_pct(r['saved_best_ema_acc']):>14}  "
            f"{_pct(r['saved_best_acc']):>14}"
        )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"dataset": args.dataset, "results": results},
                      f, indent=2)
        print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()
