# openwebtext_lm

Small-LM training and eval on OpenWebText. Two architectural baselines
behind a single `--arch` toggle:

- `llama` — RMSNorm, RoPE, SwiGLU, no biases, tied embeddings.
- `gpt2`  — LayerNorm, learned positional embeddings, GeLU, biases on,
  tied embeddings.

Three size presets in `model.py`:

| preset       | layers | heads | d_model | seq_len | ~params |
| ------------ | -----: | ----: | ------: | ------: | ------: |
| `tiny`       |      6 |     6 |     384 |     512 |    ~10M |
| `small`      |      8 |     8 |     512 |    1024 |    ~30M |
| `gpt2-small` |     12 |    12 |     768 |    1024 |   ~125M |

## Setup

The dataset is expected as 80 parquet shards under
`openwebtext/plain_text/train-NNNNN-of-00080.parquet` (already present
in this repo). Tokenisation uses GPT-2 BPE via tiktoken; `prepare.py`
writes `train.bin` + `val.bin` (uint16) and a `meta.json` next to itself.

```bash
# install (adds tiktoken + pyarrow to repo deps)
uv sync          # or: pip install -e .

# one-time tokenisation (~30-60 min on a 16-core box, ~18 GB train.bin)
python example/openwebtext_lm/prepare.py
```

> **Disk:** `train.bin` is ~18 GB and is read with random ~4 KB seeks
> during training. Keep it on an SSD; on an HDD the data loader will
> bottleneck the GPU.

## Train

```bash
# smoke test (~2-3 min on a single 4090)
python example/openwebtext_lm/train/owt_lm_train.py \
    --arch llama --model tiny --max-iters 100 --eval-interval 50 \
    --batch 16 --grad-accum 1

# real "tiny" run (~2-4 h on a 4090)
python example/openwebtext_lm/train/owt_lm_train.py \
    --arch llama --model tiny --max-iters 20000 \
    --batch 32 --grad-accum 4

# A/B the architecture at iso-params
python example/openwebtext_lm/train/owt_lm_train.py \
    --arch gpt2  --model tiny --max-iters 20000 \
    --batch 32 --grad-accum 4
```

Logs land in `logs/owt_lm_{arch}_{model}.jsonl` (one JSON line per eval
event). Checkpoints land in `cache/owt_lm_{arch}_{model}.pt`.

## Evaluate

```bash
python example/openwebtext_lm/eval/owt_lm_eval.py \
    --ckpt example/openwebtext_lm/cache/owt_lm_llama_tiny.pt \
    --n-batches 200

python example/openwebtext_lm/eval/owt_lm_sample.py \
    --ckpt example/openwebtext_lm/cache/owt_lm_llama_tiny.pt \
    --prompt "The capital of France is"
```

## Expected val PPL ballparks

These are single-GPU numbers, not headline figures.

| preset       | iters | llama         | gpt2          |
| ------------ | ----: | :------------ | :------------ |
| `tiny`       |  20 k | ~40–50        | ~45–55        |
| `small`      |  50 k | ~25–30        | ~28–34        |
| `gpt2-small` | ≥100k | depends on budget |              |

Published GPT-2 small reaches ~29–35 OWT-val PPL after ~300 B tokens of
training; on a single GPU we generally stop well short of that.

## Files

```
_paths.py                   sys.path bootstrap (mirrors cifar/_paths.py)
owt_data.py                 np.memmap loader, get_batch, lm_ckpt_path
model.py                    GPT, GPTConfig, PRESETS, RoPE/RMSNorm/SwiGLU
prepare.py                  parquet -> train.bin / val.bin / meta.json
train/owt_lm_train.py       training loop (AdamW + cosine + warmup, bf16)
eval/owt_lm_eval.py         val loss + perplexity + bits/token
eval/owt_lm_sample.py       prompted top-k sampling
cache/                      checkpoints (gitignored)
logs/                       JSONL training logs (gitignored)
train.bin, val.bin          token streams (gitignored)
meta.json                   tokenizer contract + token counts (tracked)
```
