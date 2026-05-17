# dka — Denoising Kalman Attention

A small, focused implementation of the **denoising** Kalman-attention
primitive from `DKA.md §14.2`, plus a softmax-transformer baseline so
both can be benchmarked at the same shape interface on two demos
(image-patch denoising on CIFAR, masked-LM on OpenWebText tokens) with
the four §C.1 ablations.

## What it is

For an input `(B, L, d_model)`:

    phi   = rmsnorm(MLP_phi(x))              # (B, L, M)   ← 2-layer GELU MLP
    Y     = x                                # denoising target
    G     = phi.T @ phi                      # (B, M, M)  per-batch Gram
    Λ, V  = eigh(G)                          # detached  (§3.3 stop-grad)
    V_r   = V[..., :r]                       # truncated eigenbasis
    Φ_r   = phi @ V_r                        # gradient flows here
    hat   = Φ_r · (1/(Λ_r+λ)) · Φ_r^T Y
    v     = (1/λ) · (1 - Σ_k (Λ_k/(Λ_k+λ)) · Φ_{r,k}²)   # per-token
    K     = 1 / (1 + v)
    out   = x + K · (hat - x)

Every line has a Bayesian justification (closed-form ridge posterior +
Kalman-gain fusion). `M ≪ L` makes per-batch `eigh` cheap, and freezing
the eigenbasis keeps gradients away from the ill-conditioned `eigh`
backward.

### Why φ is an MLP and there's no FFN

A standard transformer block is *attention → FFN* because attention is
linear in the inputs and needs a separate nonlinear pointwise stage.
DKA's φ is the only place nonlinearity is needed — once features are
mapped to `(B, L, M)`, the rest is a closed-form regression. So:

- `W_phi` defaults to a **SwiGLU MLP** (`d_model → 4·d_feat → d_feat`)
  to match the modern transformer FFN convention. Swap to the GELU
  2-layer with `phi_type="mlp"` or `--phi-type mlp`.
- `DKABlock` ships **without** a SwiGLU FFN by default. Opt back in
  with `use_ffn=True` if you want to A/B that against the no-FFN case.

The `softmax` baseline (`SoftmaxAttentionBlock`) keeps the standard
pre-norm MHA + SwiGLU FFN structure — that's the right comparison for
the conventional architecture.

### Multi-head and causal

- **Multi-head** (`n_heads > 1`): φ and Y are split into heads of size
  `d_feat` / `d_model/n_heads`, each head runs its own Gram + eigh +
  Kalman fusion, and the per-head outputs are concatenated and mixed
  by a learned `W_out` (MHA-style). `n_heads = 1` reproduces §14.2
  exactly with no `W_out`.
- **Causal** (`causal=True`): the per-token regression uses only
  positions `≤ l`. Implemented via cumulative `φφᵀ` / `φyᵀ` and a
  single batched `linalg.solve`, which is well-conditioned with `λ>0`
  and has stable backward (no eigh, no rank truncation in this mode).
  Memory is `O(B·H·L·M²)` so keep `M` small if you push `L`. Long-context
  Sherman-Morrison streaming (Appendix B) is left as future work.

## Files

```
_paths.py                      sys.path bootstrap
sanity.py                      42 numeric tests on DKALayer / DKABlock / Softmax
cifar/dka_cifar_data.py        patchify + add_gaussian_noise (wraps cifar_data.py)
cifar/train.py                 stack of blocks → patch-space MSE denoiser
cifar/eval.py                  PSNR / MSE at multiple noise levels
seq/dka_mlm_data.py            BERT-style masked-token batches over OWT bins
seq/train_mlm.py               short MLM demo
benchmarks/speed.py            fwd+bwd latency: DKA vs softmax × {h1,h4} × {bidir,causal}
benchmarks/plot_curves.py      read logs/*.jsonl → plots/ (val PSNR / val loss)
ablations/run_cifar.sh         §C.1 sweep (rank, output_form, recurrent_depth)
                               + softmax baseline + multi-head + causal
cache/                         checkpoints  (gitignored)
logs/                          JSONL logs   (gitignored)
plots/                         PNG plots    (gitignored)
```

The reusable primitive lives in `src/ebmify/models/dka.py` and is
exported via `from ebmify.models import DKALayer, DKABlock,
SoftmaxAttentionBlock`.

## Quickstart

```bash
# 0. Sanity (no GPU / dataset needed; ~5 s)
python example/dka/sanity.py

# 1. CIFAR denoising — DKA vs softmax at the same shape
python example/dka/cifar/train.py --arch dka     --epochs 5 --tag dka_5e
python example/dka/cifar/train.py --arch softmax --epochs 5 --tag soft_5e
python example/dka/cifar/eval.py \
    --ckpt example/dka/cache/dka_cifar_cifar10_s0.1_dka_5e.pt
python example/dka/cifar/eval.py \
    --ckpt example/dka/cache/softmax_cifar_cifar10_s0.1_soft_5e.pt

# 2. The four §C.1 ablations + softmax baseline (~20 min on a 4090)
bash example/dka/ablations/run_cifar.sh

# 3. Short MLM demo on OpenWebText tokens
#    Requires train.bin / val.bin from example/openwebtext_lm/prepare.py
python example/dka/seq/train_mlm.py --arch dka     --max-iters 1000 --tag dka
python example/dka/seq/train_mlm.py --arch softmax --max-iters 1000 --tag soft

# 4. Latency micro-bench + plot training curves
python example/dka/benchmarks/speed.py --lengths 64 128 256 512
python example/dka/benchmarks/plot_curves.py   # reads logs/*.jsonl
```

## Causal / multi-head examples

```bash
# Multi-head DKA on CIFAR (n_heads must divide d_model)
python example/dka/cifar/train.py --arch dka --n-heads 4 --tag dka_h4

# Causal DKA on the MLM stack (NB: BERT-style MLM is bidirectional by
# design — only use --causal if you actually want causal masking, e.g.
# for an LM-style ablation against the softmax causal baseline).
python example/dka/seq/train_mlm.py --arch dka     --causal --tag dka_causal
python example/dka/seq/train_mlm.py --arch softmax --causal --tag soft_causal
```

## Ablation knobs (§C.1)

All four are constructor/CLI flags on `DKALayer` / `DKABlock`:

| knob              | values                              | what it tests              |
| ----------------- | ----------------------------------- | -------------------------- |
| `rank_rule`       | `full` / `fixed_r` / `adaptive`     | feature-rank truncation    |
| `output_form`     | `kalman` / `none` / `gated`         | §2.2 anti-doubling fusion  |
| `recurrent_depth` | 1, 2, 4, …                          | §5.2 recurrent DKA         |
| `feature_norm`    | True / False                        | §2.3 QK-Norm fix           |

Extra knobs added here:

| knob          | values             | what it tests                                |
| ------------- | ------------------ | -------------------------------------------- |
| `phi_hidden`  | `0` / int          | 0 = single Linear; int = MLP hidden dim      |
| `phi_type`    | `swiglu` / `mlp`   | SwiGLU (default) vs GELU 2-layer for φ       |
| `use_ffn`     | bool               | add a SwiGLU FFN per block                   |
| `n_heads`     | 1, 2, 4, …         | multi-head DKA (`n_heads=1` ≡ §14.2 exactly) |
| `causal`      | bool               | causal cumulative-solve variant              |
| `--arch`      | `dka` / `softmax`  | DKA block vs standard transformer            |

## Notes

- **Bidirectional or causal.** Bidirectional uses per-batch eigh with
  the §3.3 stop-grad (fast, low memory). Causal uses cumulative
  `linalg.solve` (well-conditioned, stable backward, but materialises
  `(B, H, L, M, M)` so prefer small `M` when pushing `L`). Long-context
  Sherman-Morrison streaming (Appendix B) is left as future work.
- **MLM mask token** is GPT-2 EOT (`50256`). Pragmatic shortcut for the
  demo so the existing 50257-vocab tokenisation is reused unchanged.
- **`output_form="none"` will underperform** — that's the §2.2
  doubling pathology the ablation is supposed to surface.
- **Param parity:** at iso-`d_model` and iso-`n_layers`, the DKA block
  with default `phi_hidden = 4·d_feat` and no FFN has *substantially
  fewer* parameters than a softmax+FFN block. To match params, bump
  `phi_hidden`, raise `n_heads`, or set `--use-ffn`.
