# coreset

Feature-spectrum coreset selection from a pre-extracted feature matrix
`Phi` of shape `(N, D)`. The pipeline is dataset- and backbone-agnostic:
the caller produces `Phi` once and we select indices via three algorithms,
each emitting indices, weights, and optional auxiliary targets derived
from the eigendecomposition of `Phi^T Phi`. No raw samples are written.

## Install

The package lives at `src/coreset/` and is picked up by the repo's
editable install:

```bash
pip install -e .
```

After that, `import coreset` and `python -m coreset.cli` work from any
working directory.

## Run

```bash
python -m coreset.cli --config cfg.yaml --out artifacts/
```

CIFAR end-to-end wrappers (feature extraction + this CLI + diagnostic
plots + downstream training + test eval) live in `example/cifar/coreset/`:

| stage          | supervised                              | SSL (LeJEPA-recon)                            |
| -------------- | --------------------------------------- | --------------------------------------------- |
| build coresets | `cifar_build_coreset_supervised.py`     | `cifar_build_coreset_ssl.py`                  |
| train models   | `cifar_train_from_artifacts.py`         | `cifar_ssl_train_from_artifacts.py`           |
| eval on test   | `cifar_eval_from_artifacts.py`          | `cifar_ssl_eval_from_artifacts.py`            |

Each CIFAR script accepts `--help` for its flag list; see
`example/cifar/coreset/README.md` for the full pipeline walkthrough.

## Config example

```yaml
phi_path: example/cifar/cache/phi_supervised_resnet18.pt
phi_key: Phi                       # required when the .pt is a dict
budget_k: 5000                     # coreset size k
ridge_lambda: 1.0e-3               # lambda used by leverage / spectral_rank
n_buckets: 64                      # B: equal-mass eigvec buckets (spectral_rank)
n_top_eigvecs: 64                  # width of aux_spectral_coords (NOT a selection knob)
standardize: center_l2             # required for the spectral algorithms
algorithms: [greedy, leverage, spectral_rank]
aux_targets: [spectral_coords, bucket_ranks, leverage_score, home_bucket, feature_distill]
seed: 0
device: cuda
chunk_size: 8192
low_rank_eig: false                # true for D > 4096
low_rank_r: 512
```

## Algorithms

- `greedy` (`greedy_max_variance`): pick `argmax h(x)` greedily under a
  Sherman-Morrison-maintained `A_inv = (lam I + Phi_S^T Phi_S)^{-1}`. Seeded
  by farthest-point sampling in feature space; `A_inv` is refactored every
  500 picks to control drift.
- `leverage` (`ridge_leverage_sample`): per-sample ridge leverage
  `h_i = sum_j (V_j^T phi_i)^2 / (sigma^2_j + lam)`, then sample `k` with
  inclusion probability mixed with a uniform prior of weight `alpha`.
  Returns importance weights `w_i = 1 / sqrt(k * p_i_used)`.
- `spectral_rank` (`spectral_rank_coverage`): bucket eigenvectors into
  `B` equal-variance-mass contiguous groups, compute per-bucket squared
  alignment `S[i, b] = sum_{j in bucket(b)} (V_j^T phi_i)^2` and its
  per-column uniform rank `R[i, b] in [0, 1]`. Picking `k` samples is
  then a maximum-coverage problem on the `B * k` rank strata of width
  `1/k`; the textbook greedy (pick the sample covering the most
  uncovered cells) attains the `(1 - 1/e)` bound and drives every
  per-bucket marginal of selected ranks toward `Uniform[0, 1]`
  (mean ~ 0.5, std ~ `1/sqrt(12)`). See `spectral_rank.py` for the
  full description.

## Aux targets

Per-coreset supervision tensors written under `<algo>/aux_<name>.pt` and
indexed by *coreset position* `[0, k)`:

- `spectral_coords`: `(k, n_top_eigvecs)` projections onto the top
  eigenvectors. Paired with `spectral_weights` = `1 / sqrt(sigma2[:n] + lam)`
  for per-coordinate ridge-weighted MSE downstream.
- `bucket_ranks`: `(k, n_buckets)` per-bucket rank in `[0, 1]` for the
  selected samples (MSE).
- `leverage_score`: `(k,)` ridge leverage of selected samples (scalar MSE).
- `home_bucket`: `(k,)` `argmax`-bucket id of selected samples
  (cross-entropy over `n_buckets` classes).
- `feature_distill`: `(k, D)` standardized backbone features
  $\phi_i$ for the selected samples — a frozen-teacher target for
  MSE feature-distillation downstream.

### Consumer pattern

These targets are intentionally small (no raw samples) and ready to be
plugged into a downstream training loop. The CIFAR pipeline ships a
reference consumer at
`example/cifar/coreset/_aux_losses.py` that does the standard plumbing:

1. **`discover_aux_targets(<algo_dir>)`** loads whichever `aux_*.pt`
   files exist for the algorithm (older artifact bundles missing newer
   aux files are silently skipped). For each present target it returns
   an `AuxSpec(name, target_dim, loss_kind, per_coord_weights)`.
2. **`build_aux_heads(emb_dim, specs)`** returns an
   `nn.ModuleDict({name: nn.Linear(emb_dim, target_dim)})`.
3. **`aux_loss_terms(emb, heads, targets_batch, specs, lambdas)`**
   forwards each active head on the batch's embeddings, applies the
   per-name loss kind (`mse`, `weighted_mse`, `ce`), weights by
   `lambdas[name]`, sums, and returns `(total_loss, per_aux_log)`.

The CIFAR supervised + SSL trainers use this helper directly. Other
projects can either reuse it or follow the same recipe with their own
indexing of coreset positions into the on-disk `aux_<name>.pt` rows.

## Output layout

```
artifacts/
  preprocessing.pt          # mu, mode, N, D
  eig.pt                    # sigma2, V, lam
  bucket_assignment.pt      # (D,) int
  feature_stats.json
  {algo}/
    indices.pt              # (k,) int64
    weights.pt              # (k,) float32
    aux_spectral_coords.pt  # (k, n_top_eigvecs)   if requested
    aux_spectral_weights.pt # (n_top_eigvecs,)     paired with spectral_coords
    aux_bucket_ranks.pt     # (k, n_buckets)       if requested
    aux_leverage_score.pt   # (k,)                 if requested
    aux_home_bucket.pt      # (k,)                 if requested
    aux_feature_distill.pt  # (k, D)               if requested
    stats.json
    config.json
```

Outputs contain only indices, weights, and feature-spectrum-derived aux
targets. No images and no Phi rows are written. Diagnostic plots (which
*are* derived statistics, not samples) are emitted to
`example/out/coreset/` by the CIFAR wrappers.

## Tests

```bash
pytest src/coreset/tests
```
