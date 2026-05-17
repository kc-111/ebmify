"""Tokenize OpenWebText parquet shards into uint16 ``train.bin`` / ``val.bin``.

Reads the 80 parquet shards under ``openwebtext/plain_text/``, encodes
text with the GPT-2 BPE via tiktoken (vocab=50257, fits uint16),
inserts an end-of-text token between documents, and streams the result
into numpy memmaps. The last shard (``train-00079-of-00080.parquet``)
is held out as the validation split.

Run once before training:

    python example/openwebtext_lm/prepare.py [--workers K] [--data-dir DIR]

Outputs (in the same directory as this script):
    train.bin, val.bin, meta.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken

OWT_LM_DIR = Path(__file__).resolve().parent
REPO_ROOT = OWT_LM_DIR.parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "openwebtext" / "plain_text"
VAL_SHARD_INDEX = 79
N_SHARDS = 80
CHUNK_DOCS = 1024  # docs per encode_ordinary_batch call


def _shard_path(data_dir: Path, i: int) -> Path:
    return data_dir / f"train-{i:05d}-of-{N_SHARDS:05d}.parquet"


def _iter_texts(path: Path, chunk: int = CHUNK_DOCS):
    table = pq.read_table(str(path), columns=["text"])
    texts = table.column("text").to_pylist()
    for i in range(0, len(texts), chunk):
        yield texts[i : i + chunk]


def _encode_chunk(args):
    """Encode one shard, returning a flat uint16 numpy array of tokens.

    Runs in a worker process — re-creates the encoder there (tiktoken
    encoders are not always pickle-friendly across processes).
    """
    path_str, eot = args
    enc = tiktoken.get_encoding("gpt2")
    out = []
    for batch in _iter_texts(Path(path_str)):
        ids_per_doc = enc.encode_ordinary_batch(batch)
        for ids in ids_per_doc:
            ids.append(eot)
            out.extend(ids)
    arr = np.asarray(out, dtype=np.uint32)
    if arr.max(initial=0) >= 1 << 16:
        raise ValueError(f"token id overflow in shard {path_str}; check tokenizer")
    return arr.astype(np.uint16)


def _process(shards: list[Path], out_path: Path, workers: int, eot: int, label: str) -> int:
    """Tokenize shards in parallel and stream tokens to ``out_path``.

    Two passes: count tokens to size the memmap, then write them.
    """
    args = [(str(p), eot) for p in shards]

    t0 = time.time()
    print(f"[{label}] pass 1/2: tokenize + count over {len(shards)} shard(s)...", flush=True)
    encoded: list[np.ndarray] = [None] * len(shards)  # type: ignore[list-item]
    with Pool(processes=max(1, workers)) as pool:
        for i, arr in enumerate(pool.imap_unordered(_encode_chunk, args)):
            # imap_unordered loses ordering, but the order of tokens across
            # shards does not matter for next-token prediction. We still
            # report progress.
            encoded[i] = arr
            print(
                f"  [{label}] shard {i + 1}/{len(shards)} tokenized "
                f"({arr.size:,} tokens, {time.time() - t0:.1f}s elapsed)",
                flush=True,
            )

    total = sum(a.size for a in encoded)
    print(f"[{label}] total tokens: {total:,}", flush=True)

    print(f"[{label}] pass 2/2: writing uint16 memmap -> {out_path}", flush=True)
    mm = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total,))
    cursor = 0
    for a in encoded:
        mm[cursor : cursor + a.size] = a
        cursor += a.size
    mm.flush()
    del mm
    print(f"[{label}] done in {time.time() - t0:.1f}s, wrote {total:,} tokens", flush=True)
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="parallel worker processes for tokenization",
    )
    args = ap.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(
            f"openwebtext data dir not found: {args.data_dir}\n"
            "Expected 80 parquet shards under openwebtext/plain_text/."
        )

    train_shards = [
        _shard_path(args.data_dir, i) for i in range(N_SHARDS) if i != VAL_SHARD_INDEX
    ]
    val_shards = [_shard_path(args.data_dir, VAL_SHARD_INDEX)]
    for p in train_shards + val_shards:
        if not p.exists():
            raise FileNotFoundError(f"missing parquet shard: {p}")

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    train_path = OWT_LM_DIR / "train.bin"
    val_path = OWT_LM_DIR / "val.bin"

    n_train = _process(train_shards, train_path, args.workers, eot, label="train")
    n_val = _process(val_shards, val_path, args.workers, eot, label="val")

    meta = {
        "tokenizer": "tiktoken/gpt2",
        "vocab_size": 50257,
        "eot": eot,
        "train_tokens": int(n_train),
        "val_tokens": int(n_val),
        "val_shard": VAL_SHARD_INDEX,
        "n_shards": N_SHARDS,
        "data_dir": str(args.data_dir),
    }
    (OWT_LM_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print("meta.json:")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
