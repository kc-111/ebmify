#!/usr/bin/env bash
# §C.1 DKA ablation sweep on CIFAR-10 denoising + softmax baseline.
# Each run writes a JSONL log under example/dka/logs/ and a checkpoint
# under example/dka/cache/. Short budget so the whole sweep finishes in
# ~20 min on a single 4090. Bump --epochs / --iters-per-epoch for real
# numbers.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN="${HERE}/../cifar/train.py"
COMMON=(--dataset cifar10 --noise-sigma 0.1 --patch-size 4
        --d-model 192 --n-layers 4
        --epochs 3 --iters-per-epoch 200 --batch 128)
DKA_COMMON=(--arch dka --d-feat 64)
SOFTMAX_COMMON=(--arch softmax --n-heads 4)

# 0. Softmax baseline (same d_model, n_layers, training budget)
python "$TRAIN" "${COMMON[@]}" "${SOFTMAX_COMMON[@]}" \
    --tag baseline

# 1. Default DKA
python "$TRAIN" "${COMMON[@]}" "${DKA_COMMON[@]}" \
    --rank-rule adaptive --output-form kalman --recurrent-depth 1 \
    --tag sanity_default

# 2. Rank-rule ablation
python "$TRAIN" "${COMMON[@]}" "${DKA_COMMON[@]}" \
    --rank-rule full     --output-form kalman --recurrent-depth 1 \
    --tag rank_full
python "$TRAIN" "${COMMON[@]}" "${DKA_COMMON[@]}" \
    --rank-rule fixed_r --fixed-r 16 --output-form kalman --recurrent-depth 1 \
    --tag rank_fixed16
python "$TRAIN" "${COMMON[@]}" "${DKA_COMMON[@]}" \
    --rank-rule adaptive --rho 1e-2 --r-max 64 --output-form kalman --recurrent-depth 1 \
    --tag rank_adaptive

# 3. Output-form ablation (§2.2 anti-doubling)
python "$TRAIN" "${COMMON[@]}" "${DKA_COMMON[@]}" \
    --rank-rule adaptive --output-form kalman --recurrent-depth 1 \
    --tag out_kalman
python "$TRAIN" "${COMMON[@]}" "${DKA_COMMON[@]}" \
    --rank-rule adaptive --output-form none   --recurrent-depth 1 \
    --tag out_none
python "$TRAIN" "${COMMON[@]}" "${DKA_COMMON[@]}" \
    --rank-rule adaptive --output-form gated  --recurrent-depth 1 \
    --tag out_gated

# 4. Recurrent-depth ablation
python "$TRAIN" "${COMMON[@]}" "${DKA_COMMON[@]}" \
    --rank-rule adaptive --output-form kalman --recurrent-depth 1 \
    --tag rec_1
python "$TRAIN" "${COMMON[@]}" "${DKA_COMMON[@]}" \
    --rank-rule adaptive --output-form kalman --recurrent-depth 2 \
    --tag rec_2
python "$TRAIN" "${COMMON[@]}" "${DKA_COMMON[@]}" \
    --rank-rule adaptive --output-form kalman --recurrent-depth 4 \
    --tag rec_4

# 5. Multi-head ablation (n_heads must divide d_model=192 → use 1, 3, 4, 6)
python "$TRAIN" "${COMMON[@]}" "${DKA_COMMON[@]}" \
    --rank-rule adaptive --output-form kalman --n-heads 1 \
    --tag heads_1
python "$TRAIN" "${COMMON[@]}" "${DKA_COMMON[@]}" \
    --rank-rule adaptive --output-form kalman --n-heads 4 \
    --tag heads_4

# 6. Phi-type ablation (SwiGLU vs GELU 2-layer MLP)
python "$TRAIN" "${COMMON[@]}" "${DKA_COMMON[@]}" \
    --rank-rule adaptive --output-form kalman --phi-type swiglu \
    --tag phi_swiglu
python "$TRAIN" "${COMMON[@]}" "${DKA_COMMON[@]}" \
    --rank-rule adaptive --output-form kalman --phi-type mlp \
    --tag phi_mlp

echo
echo "Sweep complete. Final-epoch val_psnr by tag:"
for f in "${HERE}/../logs/"{dka,softmax}_cifar_cifar10_s0.1_*.jsonl; do
    [ -e "$f" ] || continue
    fname="$(basename "$f" .jsonl)"
    tag="$(echo "$fname" | sed -E 's/^(dka|softmax)_cifar_cifar10_s0.1_(.*)$/\1::\2/')"
    last="$(tail -n 1 "$f")"
    printf "  %-25s  %s\n" "$tag" "$last"
done
