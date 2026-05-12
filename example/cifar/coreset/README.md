# CIFAR-10 coreset pipeline

End-to-end wrappers around the `coreset` library (`src/coreset/`) for
CIFAR-10. Two backbones are supported (supervised ResNet-18 and the
LeJEPA-recon SSL ResNet-18) and three coreset algorithms are run on each
backbone's features $\Phi \in \mathbb{R}^{N \times D}$. Outputs land in
`example/out/coreset/`.

## Glossary

| symbol            | meaning                                                            |
| ----------------- | ------------------------------------------------------------------ |
| $N$               | number of training samples (50 000 for CIFAR-10 train)             |
| $D$               | feature dimension produced by the backbone (e.g. 512 for ResNet-18) |
| $\Phi \in \mathbb{R}^{N \times D}$ | row $i$ is the feature vector $\phi_i$ for image $i$               |
| $\lambda$         | ridge regularizer (`ridge_lambda` in the config)                   |
| $k$               | **coreset budget**: the number of indices the algorithm picks. The whole point of a coreset is to summarize the dataset with $k \ll N$ samples while preserving as much information about $\Phi^\top \Phi$ as possible. |
| $\sigma_j^2, V_j$ | eigenvalue / eigenvector $j$ of $\Phi^\top \Phi$, sorted descending |
| $B$               | number of **buckets** (`n_buckets` in the config). A bucket is a *contiguous group of eigenvectors* with similar variance. Buckets are how `spectral_rank_coverage` enforces that the coreset isn't dominated by one frequency band. |
| $h_i$             | ridge leverage of sample $i$ — how much "unique" information row $i$ adds to the regularized Gram matrix |
| $S_{i,b}$         | total squared alignment of sample $i$ with bucket $b$'s eigenvectors |
| $b^\star_i$       | **home bucket** of sample $i$: the bucket whose eigenvectors sample $i$ aligns with most ($\arg\max_b S_{i,b}$) |

### What is a bucket, exactly?

Eigenvectors are sorted by descending $\sigma_j^2$, then partitioned
into $B$ contiguous groups. We use **equal-mass** bucketing: each
bucket holds a range $[j_b, j_{b+1})$ such that
$\sum_{j = j_b}^{j_{b+1} - 1} \sigma_j^2 \approx \text{total} / B$.

Why not equal *count*? Natural-image features have heavy-tailed spectra
($\sigma_1^2 \gg \sigma_2^2 \gg \ldots$). With equal-count bucketing,
bucket $0$ holds the leading eigenvectors *and* most of the variance,
so $S_{i, 0}$ dominates every other $S_{i, b}$ and
$\arg\max_b S_{i, b} = 0$ for nearly every sample — the home-bucket
distribution collapses to a single bin and the `home_bucket` aux target
becomes uninformative. Equal-mass bucketing puts every bucket on the
same expected alignment scale, so home buckets distribute meaningfully
and the bucketing carves the spectrum into bands of comparable
"importance" rather than equal count.

## Pipeline

```
       train backbone                 build coreset                 train on coreset                 eval on test
       ──────────────                 ─────────────                 ────────────────                 ────────────
sup:   cifar_resnet18_train.py   ──►  cifar_build_coreset_     ──►  cifar_train_from_           ──►  cifar_eval_from_
                                      supervised.py                 artifacts.py                     artifacts.py
ssl:   ssl_pretrain_recon.py     ──►  cifar_build_coreset_     ──►  cifar_ssl_train_from_       ──►  cifar_ssl_eval_from_
                                      ssl.py                        artifacts.py                     artifacts.py
```

Each "build coreset" script:

1. Loads its frozen backbone and encodes all 50k CIFAR-10 train images
   into $\Phi \in \mathbb{R}^{N \times D}$ (cached to
   `example/cifar/cache/phi_<tag>.pt`).
2. Invokes `coreset.cli.run` (the library entry point) on $\Phi$, which
   runs **all three** selection algorithms and writes per-algorithm
   artifacts to `example/cifar/cache/coreset/<tag>/<algo>/`.
3. Emits diagnostic plots to `example/out/coreset/`.

The shared encoding + CLI-invocation + diagnostics logic lives in
`_extract_common.py::extract_and_select` — this is where the three
coreset algorithms actually execute (via `coreset.cli`).

## Algorithms

All three operate on the eigendecomposition of the feature covariance,

$$
\Phi^\top \Phi \;=\; V \,\operatorname{diag}(\sigma^2)\, V^\top,
\qquad \sigma_1^2 \ge \sigma_2^2 \ge \cdots \ge \sigma_D^2 \ge 0,
$$

with the ridge $\lambda$ **not** baked into $\sigma^2$ — callers use
$\sigma_j^2 + \lambda$ as the regularized eigenvalue.

### 1. Greedy max-variance (`greedy_max_variance`)

Selects samples by greedily maximizing posterior variance reduction
under a ridge prior. Maintains the selected-Gram inverse

$$
A^{-1} \;=\; \Bigl(\lambda I + \sum_{i \in S} \phi_i \phi_i^\top\Bigr)^{-1}
\quad \in \mathbb{R}^{D \times D}
$$

and per-sample leverages $h_i = \phi_i^\top A^{-1} \phi_i$. Each
iteration:

1. Pick $i^\star = \arg\max_i h_i$.
2. Sherman–Morrison rank-1 update:

$$
u = A^{-1} \phi_{i^\star}, \qquad
\beta = 1 + \phi_{i^\star}^\top u, \qquad
A^{-1} \;\mathrel{-}=\; \frac{u u^\top}{\beta}.
$$

3. Streamed update of $h$: for each chunk,
   $h_\text{chunk} \mathrel{-}= (\text{chunk} \cdot u)^2 / \beta$.
4. Mark $h_{i^\star} \leftarrow -\infty$ so it is never re-picked.

Seeded by farthest-point sampling in feature space (Gonzalez seed) so
the initial subset isn't dominated by one eigendirection. $A^{-1}$ is
rebuilt from scratch via Cholesky every 500 picks to control round-off
drift. The trajectory $\{\max_i h_i\}$ is recorded in `stats.json` and
should trend monotonically downward.

### 2. Ridge-leverage sampling (`ridge_leverage_sample`)

Per-sample ridge leverage

$$
h_i \;=\; \phi_i^\top \bigl(\Phi^\top \Phi + \lambda I\bigr)^{-1} \phi_i
\;=\; \sum_{j=1}^{D} \frac{\bigl(V_j^\top \phi_i\bigr)^2}{\sigma_j^2 + \lambda}
$$

is streamed once, then $k$ indices are drawn by either

- **with replacement**: `torch.multinomial` on the mixed distribution

$$
p_i \;=\; (1 - \alpha)\,\frac{h_i}{\sum_{i'} h_{i'}} \;+\; \frac{\alpha}{N},
$$

  the uniform tail keeps samples with $h_i = 0$ reachable when $\Phi$
  has near-redundant rows;
- **without replacement** (default): independent Bernoulli inclusion
  with $q_i = \min(1, c\, h_i)$ for $c$ binary-searched so
  $\sum_i q_i \approx k$. The actual sampled set is padded or truncated
  to land at exactly $k$.

Importance weights

$$
w_i \;=\; \frac{1}{\sqrt{k \cdot p_i^{\text{used}}}}
$$

are returned so downstream weighted-least-squares estimators on the
coreset are unbiased. The effective dimension
$d_\text{eff}(\lambda) = \sum_j \sigma_j^2 / (\sigma_j^2 + \lambda)$ is
recorded in `stats.json`.

### 3. Spectral-rank coverage (`spectral_rank_coverage`)

Coverage-style selection driven by per-bucket eigenvector alignment.

1. **Bucket eigvecs by equal variance mass.** Split $[0, D)$ into $B$
   contiguous buckets such that each bucket carries roughly equal
   $\sum_{j \in b} \sigma_j^2$ — boundaries are placed at uniform
   quantiles of the cumulative variance, so the leading bucket is
   *narrow but high-variance* and trailing buckets are wider. See the
   glossary above for the rationale.

2. **Per-bucket alignment.** Stream

$$
S_{i,b} \;=\; \sum_{j \in \mathrm{bucket}(b)} \bigl(V_j^\top \phi_i\bigr)^2.
$$

3. **Per-bucket rank.** Convert each column to its uniform-in-$[0, 1]$ rank,

$$
R_{i,b} \;=\; \frac{\#\{i' : S_{i',b} \le S_{i,b}\}}{N}.
$$

   Row $i$ of $R$ is the *rank vector* of sample $i$ in $[0, 1]^B$.

4. **Greedy max-coverage of rank strata.** Partition each bucket's
   rank axis into $k$ equal strata of width $1/k$, so each sample $i$
   becomes a length-$B$ set $\{(b, s_b(i))\}_{b=1}^{B}$ with
   $s_b(i) = \min(\lfloor k\,R_{i,b} \rfloor,\ k - 1)$. Picking $k$
   samples is then the classical *maximum-coverage* problem on a
   universe of $B \cdot k$ cells; the textbook greedy — repeatedly
   pick the sample covering the most currently-uncovered cells —
   attains the $(1 - 1/e)$ approximation (Hochbaum 1996):

$$
i_t \;=\; \arg\max_{i \notin S_{t-1}}\; \bigl|\{b : (b,\, s_b(i)) \notin C_{t-1}\}\bigr|,
\quad C_t \;=\; C_{t-1} \cup \{(b,\, s_b(i_t))\}_{b=1}^{B}.
$$

   *When* a perfect transversal exists in the candidate pool — $k$
   samples whose stratum tuples partition the entire $B \cdot k$ cell
   universe — the per-bucket marginal of selected ranks reaches
   $\mathrm{Uniform}[0, 1]$ exactly. In practice neither condition
   strictly holds: (i) the $(1 - 1/e)$ approximation gap means greedy
   can leave cells uncovered even when a transversal exists, and (ii)
   for some joint stratum distributions no transversal exists at all
   (Hall-type condition). So the algorithm's actual guarantee is
   weaker: it *drives* the marginal *toward* Uniform[0, 1] — mean
   $\approx 0.5$ and std $\approx 1/\sqrt{12}$ in every bucket — and
   in practice gets within $\sim 0.05$ on both at $B = 128, k = 256$.
   What matters is that it avoids the *systematic* corner-bias
   $\ell_\infty$ farthest-first develops as $B$ grows.

   **Animated demo:** run
   `python example/cifar/coreset/demo_spectral_rank_greedy.py`
   to render
   `example/out/coreset/demo_spectral_rank_greedy.gif`. Each frame is
   one greedy pick. Orange cells flash as a sample's new strata are
   claimed; green cells stay covered. The per-step gain prints as
   ``[8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 6, 5, 5, 4, 4]`` for the default
   $(N, B, k) = (200, 8, 16)$: the first eight picks each cover a fresh
   cell in every bucket, then the gain drops as the strata fill up.

The home bucket $b^\star_i = \arg\max_b S_{i,b}$ is still computed and
emitted as the `home_bucket` aux target (a coarse cluster label), but
it is no longer used by the selection.

Deterministic given $(\Phi, V, B)$, modulo a $\mathcal{O}(10^{-9})$
jitter that breaks exact-tie argmax/argmin.

## Auxiliary targets

For each selection, the CLI also writes spectrum-derived targets to the
same `<algo>/` directory. They are downstream-trainable supervision
signals (the model can predict them in addition to the main task):

| target            | shape          | meaning                                                  |
| ----------------- | -------------- | -------------------------------------------------------- |
| `spectral_coords` | $(k, n_\text{top})$ | $c_{i,j} = V_j^\top \phi_i$ for top-$n_\text{top}$ eigvecs |
| `bucket_ranks`    | $(k, B)$       | per-bucket uniform rank in $[0, 1]$                       |
| `leverage_score`  | $(k,)$         | scalar ridge leverage $h_i$                               |
| `home_bucket`     | $(k,)$         | $b^\star_i = \arg\max_b S_{i,b}$ (coarse cluster label)   |
| `feature_distill` | $(k, D)$       | standardized backbone feature $\phi_i$ (frozen teacher)   |

`spectral_coords` is paired with weights

$$
w_j \;=\; \frac{1}{\sqrt{\sigma_j^2 + \lambda}}, \qquad j = 1, \ldots, n_\text{top}
$$

(`aux_spectral_weights.pt`) so a downstream head can ridge-weight its
prediction loss across coordinates.

### Consuming aux targets at train time

Both training scripts can attach a small linear head per aux name on top
of the backbone's 512-D pre-fc feature and add a weighted loss to the
main objective:

- **Supervised** (`cifar_train_from_artifacts.py`): a forward hook on
  `model.avgpool` captures the 512-D embedding. Each `--aux-<name>`
  builds an `nn.Linear(512, target_dim)` head, trained jointly with the
  CE classifier. The per-batch coreset position (the permutation index
  `b`) indexes `aux_<name>.pt` into batch-aligned targets.
- **SSL** (`cifar_ssl_train_from_artifacts.py`): the wrapped LeJEPA-recon
  `forward` reuses the existing `out["embedding"]` (the backbone output
  of view 0) — no hook needed. Coreset position comes from
  `batch["sample_idx"]`, auto-added by `spt.data.FromTorchDataset`.
  Aux losses are added to `out["loss"]` and logged via
  `self.log("{stage}/aux_{name}", ...)`.

| aux name          | loss kind                       | head shape           |
| ----------------- | ------------------------------- | -------------------- |
| `spectral_coords` | per-coord ridge-weighted MSE    | $512 \to n_\text{top}$ |
| `bucket_ranks`    | MSE                             | $512 \to B$          |
| `leverage_score`  | MSE on scalar                   | $512 \to 1$          |
| `home_bucket`     | cross-entropy                   | $512 \to n_\text{buckets}$ |
| `feature_distill` | MSE on standardized features    | $512 \to D$          |

Each aux gets its own `--aux-<name>` flag (default `0.0` = head not
attached, no parameters trained). Missing targets on disk for an
algorithm are silently skipped — older artifact bundles built before
`feature_distill` was added still work.

```bash
# Supervised: classification + ridge-weighted top-eigvec regression
python example/cifar/coreset/cifar_train_from_artifacts.py \
    --algorithms greedy --epochs 60 \
    --aux-spectral-coords 0.1

# SSL: LeJEPA-recon + leverage + feature-distillation
python example/cifar/coreset/cifar_ssl_train_from_artifacts.py \
    --algorithms greedy --epochs 200 \
    --aux-leverage-score 0.05 --aux-feature-distill 0.1
```

Shared aux-loss plumbing (target discovery, head construction, per-batch
indexing, weighted-sum loss) lives in `_aux_losses.py`.

## Files

| script                                | role                                                          |
| ------------------------------------- | ------------------------------------------------------------- |
| `cifar_build_coreset_supervised.py`   | extract supervised features + run all 3 algorithms            |
| `cifar_build_coreset_ssl.py`          | same, but with the LeJEPA-recon SSL backbone                  |
| `cifar_train_from_artifacts.py`       | train ResNet-18 from scratch on each coreset (supervised)     |
| `cifar_ssl_train_from_artifacts.py`   | SSL-pretrain LeJEPA-recon on each coreset                     |
| `cifar_eval_from_artifacts.py`        | load trained supervised models and report CIFAR-10 test top-1 |
| `cifar_ssl_eval_from_artifacts.py`    | load SSL ckpts and report the trained linear-probe test top-1 |
| `_extract_common.py`                  | shared encoder + CLI driver + diagnostics plotter (not run directly) |

Every script accepts `-h / --help` and lists every flag with a one-line
description. The summary table below shows the flags you'll usually touch.

### Build flags (supervised + SSL)

| flag                       | default | what it controls |
| -------------------------- | ------- | ---------------- |
| `--backbone {sup,imagenet}`| `supervised` | (sup only) feature extractor source |
| `--ssl-tag TAG`            | `recon`      | (ssl only) SSL ckpt suffix: `cifar10_ssl_resnet18_<tag>.pt` |
| `--budget-k K`             | `5000`       | coreset size each algo picks |
| `--ridge LAM`              | `1e-3`       | ridge $\lambda$ for leverage + spectral_rank |
| `--n-buckets B`            | `64`         | equal-mass eigvec buckets for spectral_rank |
| `--n-top-eigvecs N`        | `64`         | width of `aux_spectral_coords` (does *not* affect selection) |
| `--batch N`                | `256`        | extraction batch size |
| `--resize N`               | `None`       | (sup only) square resize; required only with `--backbone imagenet` |
| `--seed S`                 | `0`          | RNG seed for sampling / tie-breaking |

### Supervised train + eval flags

| flag                  | default | what it controls |
| --------------------- | ------- | ---------------- |
| `--artifacts TAG_OR_PATH` | `supervised_resnet18` | `<artifacts_root>/` written by the build step; bare tag resolved under `example/cifar/cache/coreset/` |
| `--algorithms ...`    | all     | restrict to a subset of `<algo>/indices.pt` |
| `--epochs E`          | `60`    | per-algo training epochs |
| `--batch N`           | `128`   | training mini-batch |
| `--lr`, `--momentum`, `--weight-decay` | `0.1 / 0.9 / 5e-4` | SGD knobs |
| `--aux-<name> LAM`    | `0.0`   | per-aux loss weight (one flag per aux in the table above) |
| `--seed S`            | `0`     | model init / batch order |
| `--full-acc F`        | `None`  | reference 50k-train accuracy for the bar plot |
| `--models-root PATH`  | (eval)  | override the `coreset_models/<tag>/` lookup path |

### SSL train + eval flags

| flag                  | default | what it controls |
| --------------------- | ------- | ---------------- |
| `--artifacts TAG_OR_PATH` | `ssl_resnet18_recon` | `<artifacts_root>/` written by `cifar_build_coreset_ssl.py`; bare tag resolved under `example/cifar/cache/coreset/` |
| `--algorithms ...`    | all     | restrict subset |
| `--epochs E`          | `200`   | LeJEPA-recon pretrain length |
| `--regularizer {sigreg,w1,w2}` | `sigreg` | uniformity regularizer |
| `--lambd`, `--lambd-recon`, `--inv-tol` | `0.05 / 0.1 / 0.0` | LeJEPA loss weights |
| `--proj-dim`, `--proj-hidden`, `--num-proj`, `--dec-base` | `64 / 2048 / 1024 / 256` | projector / decoder widths |
| `--aux-<name> LAM`    | `0.0`   | per-aux loss weight (one flag per aux in the table above) |
| `--batch-size`, `--lr`, `--weight-decay`, `--num-workers`, `--precision` | `256 / 5e-4 / 5e-4 / 8 / 16-mixed` | trainer knobs |
| `--seed S`            | `0`     | **must match** the seed used at eval time (run-name lookup) |
| `--full-acc F`        | `None`  | reference line on the bar plots |

## Quick start

```bash
# Supervised (assumes example/cifar/cache/cifar10_resnet18.pt exists)
python example/cifar/coreset/cifar_build_coreset_supervised.py --budget-k 5000
python example/cifar/coreset/cifar_train_from_artifacts.py \
    --artifacts example/cifar/cache/coreset/supervised_resnet18 \
    --epochs 60
python example/cifar/coreset/cifar_eval_from_artifacts.py \
    --artifacts example/cifar/cache/coreset/supervised_resnet18

# SSL (assumes example/cifar/cache/cifar10_ssl_resnet18_recon.pt exists)
python example/cifar/coreset/cifar_build_coreset_ssl.py --budget-k 5000
python example/cifar/coreset/cifar_ssl_train_from_artifacts.py \
    --artifacts example/cifar/cache/coreset/ssl_resnet18_recon \
    --epochs 200
python example/cifar/coreset/cifar_ssl_eval_from_artifacts.py \
    --artifacts example/cifar/cache/coreset/ssl_resnet18_recon
```

### Smallest-possible smoke test

```bash
python example/cifar/coreset/cifar_build_coreset_supervised.py --budget-k 2000
python example/cifar/coreset/cifar_train_from_artifacts.py \
    --artifacts example/cifar/cache/coreset/supervised_resnet18 --epochs 5
python example/cifar/coreset/cifar_eval_from_artifacts.py \
    --artifacts example/cifar/cache/coreset/supervised_resnet18
```

## Output layout

```
example/cifar/cache/
  phi_<tag>.pt                                # cached Phi + labels
  coreset/<tag>/
    preprocessing.pt eig.pt bucket_assignment.pt feature_stats.json
    greedy/         indices.pt weights.pt aux_*.pt stats.json config.json
    leverage/       indices.pt weights.pt aux_*.pt stats.json config.json
    spectral_rank/  indices.pt weights.pt aux_*.pt stats.json config.json
    summary.json
  coreset_models/<tag>/<algo>/model.pt        # supervised model per algo
  ../logs/ssl_coreset_<tag>_<algo>_s<seed>/checkpoints/last.ckpt
                                              # SSL Lightning ckpt per algo

example/out/coreset/
  <tag>_coreset_selection_diagnostics.png     # 4-panel plot (build step)
  <tag>_coreset_spectrum.png                  # eigenvalue / eff-dim   (build)
  <tag>_coreset_train_accuracy.png            # bar plot (sup train)
  <tag>_coreset_train_results.json            #          (sup train)
  <tag>_coreset_eval_accuracy.png             # bar plot (sup eval)
  <tag>_coreset_eval_results.json             #          (sup eval)
  <tag>_ssl_coreset_linprobe.png              # bar plot (ssl train)
  <tag>_ssl_coreset_knn.png                   #          (ssl train)
  <tag>_ssl_coreset_results.json              #          (ssl train)
  <tag>_ssl_coreset_eval_linprobe.png         # bar plot (ssl eval)
  <tag>_ssl_coreset_eval_results.json         #          (ssl eval)
```

Diagnostics are *statistics about the selection* (leverage histogram,
home-bucket distribution, top-2 spectral-coord scatter, per-bucket mean
rank), not images.
