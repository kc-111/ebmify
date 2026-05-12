# Linear leverage on CIFAR features: EBM energy, posterior variance, OOD

This is the consolidated note on what `h(z) = phi(z)^T (Phi_tr^T Phi_tr +
ridge·I)^-1 phi(z)` does on CIFAR-10/CIFAR-100 features, when used as
(a) an EBM energy for Langevin sampling, (b) a GP / Bayesian last-layer
posterior variance, and (c) an OOD detector. The three roles are the
same formula evaluated against the same training Gram — the
interpretation changes, the math doesn't.

Supersedes `OOD_NORMALIZE_FINDINGS.md`, which only covered the OOD
role.

---

## Part 0. Backbones and scripts

Four CIFAR-10 feature backbones are compared throughout:

| Tag         | Backbone                                 | dim | Training script                  |
|-------------|------------------------------------------|----:|----------------------------------|
| `resnet18`  | Supervised ResNet18 on CIFAR-10          | 512 | `train/cifar_resnet18_train.py`  |
| `ssl`       | LeJEPA ResNet18 (joint-embedding inv.)   | 512 | `train/ssl_pretrain.py`          |
| `dinov2`    | DINOv2 ViT-B/14 (frozen Meta release)    | 768 | n/a — `torch.hub.load`           |
| `vae`       | β-VAE encoder μ on CIFAR-10              | 256 | `train/cifar_vae_train.py`       |

The SSL backbone is **LeJEPA**, wrapped opaquely through
`stable_pretraining as spt` in `ssl_pretrain.py`. Earlier notes that
called it "SSL SimCLR" or "VICReg" were wrong.

Scripts referenced in this doc:

- Per-backbone OOD thresholding: `ood/cifar_{resnet18,ssl,dinov2,vae}_ood_threshold.py`
- Bandwidth diagnostics: `diagnostics/cifar_resnet18_bandwidth_scan.py`,
  `diagnostics/cifar_memorization_scan.py`
- Preprocessing treatments: `diagnostics/cifar_centering_comparison.py`
- Concat representation: `diagnostics/cifar_concat_features_test.py`,
  `diagnostics/cifar_concat_linear_probe.py`
- z-space EBM / Langevin: `ebm/cifar_vae_langevin.py`
- Linear probe baseline: `probes/ssl_linear_probe.py`

---

## Part I. The three roles of the leverage formula

Throughout, let `Phi_tr ∈ R^{n×D}` be the train feature matrix (rows =
training points, columns = features of `phi(z)`), and

```
H_train = Phi_tr (Phi_tr^T Phi_tr + ridge·I)^-1 Phi_tr^T     ∈ R^{n×n}
h(z)    = phi(z)^T (Phi_tr^T Phi_tr + ridge·I)^-1 phi(z)     ∈ R
```

`H_train` is the kernel-ridge hat matrix; `h(z)` is its diagonal
extended to a new point. In the linear case `phi(z) = z`, `h(z)` is the
ridge-regularized Mahalanobis distance to the train cloud under the
covariance `(1/n) Z_tr^T Z_tr`.

`build_phi_leverage` in `mnist_vae_langevin.py` appends a constant `1`
column to `phi` by default (`bias=True`). The bias column absorbs the
training mean — so the formula automatically uses `(z - μ_train)` for
the linear part, and **raw is bit-identical to "centered" (z - μ)**.
This was a surprise the first time we ran the centering comparison:
only L2 (a nonlinear projection onto the sphere) is a real,
non-redundant preprocessing knob.

### 1.1 Role as EBM energy

Define energy `E(z) = -log h(z) + const`. Then `p(z) ∝ h(z)` after
exp/normalize. The gradient `∇_z E(z) = -∇_z h(z) / h(z)` is a
well-defined force field on `R^D`, so we can run overdamped Langevin
in `z`-space:

```
z_{t+1} = z_t - τ ∇_z E(z_t) + sqrt(2τ) ξ
```

Implemented in `cifar_vae_langevin.py` with SamAdams (preconditioned
adaptive Langevin) and a geometric anneal on `τ`. Particles started
from `z ~ N(0, I)` land on the data manifold; the decoded `g(z)` look
like CIFAR-10 samples. The same `h(z)` doubles as the OOD detector
(Part II): high `h` ⇔ low energy ⇔ in-distribution.

This is the cleanest "leverage as EBM" demonstration in the repo:
no separate score model, no contrastive divergence, just the
ridge-regularized Mahalanobis quadratic on a backbone-features sample.

### 1.2 Role as GP / Bayesian posterior variance

If we treat `phi(z)^T w` as a linear model with Gaussian prior
`w ~ N(0, σ_w^2 I)` and Gaussian likelihood with noise variance
`σ_n^2`, the posterior predictive variance at a new point `z*` is

```
Var[ f(z*) | data ] = σ_n^2 + σ_w^2 · phi(z*)^T (Phi_tr^T Phi_tr + (σ_n^2/σ_w^2) I)^-1 phi(z*)
                    = σ_n^2 + σ_w^2 · h(z*)
```

with `ridge = σ_n^2 / σ_w^2`. So `h(z)` *is* the data-dependent
posterior variance (up to an additive observation-noise floor and a
prior-precision scaling). Reading the OOD interpretation backwards:
"the model is uncertain at z*" and "z* is far from the train cloud
under the feature-induced metric" are the same statement.

In the linear-`phi=z` case this is exactly the closed-form posterior
variance of Bayesian linear regression. With `phi = [z; RFF(z)]` or
any other feature lift, it's the Bayesian last-layer view of a
random-features network. The ridge parameter is not a tuning knob —
it has a calibration meaning (`σ_n^2 / σ_w^2`), and that's what makes
the same number reusable as an OOD score and an energy.

### 1.3 Role as OOD score

Take the train Gram once, then evaluate `h(z*)` on any new point. A
threshold at the `0.95`-quantile of `{h(z_i^train)}` flags ~5% of
training points and (we want) a much higher fraction of OOD points.
The whole inductive Setup A protocol used in
`cifar_{resnet18,ssl,dinov2,vae}_ood_threshold.py` is this one
formula plus thresholding plus a held-out RFF bandwidth tuner.

Everything in Part II below is about making this role behave well
when the backbone features have non-trivial geometry.

---

## Part II. OOD findings

### 2.1 The symptom that started this

Running the LeJEPA backbone with the default (un-normalized) protocol:

```
AUROC vs cifar10 train baseline (phi = z):
    cifar10 test       cifar100        Gaussian       inverted
        0.5037          0.2900          0.9999         0.6500
```

cifar100 came back at **0.29** — the leverage classifier was saying
"cifar100 is *more* in-distribution than cifar10 train itself." Wrong
direction. The story should be `cifar10 ≲ cifar10 test ≲ cifar100 ≲
noise`.

The LeJEPA linear probe (`ssl_linear_probe.py`) hits 87.5% top-1 on
cifar10 and 43.6% on cifar100 (44× chance). The features are
semantically meaningful and transfer; the OOD failure is not "features
collapsed on cifar100."

### 2.2 Why: norm-geometry × leverage scaling

Linear `h(z) = z^T A z ∝ ||z||² · (direction-dependent factor)`. For a
fixed direction, `h` scales with squared norm. Median feature norms on
LeJEPA:

```
                ||z|| median
cifar10 train        6.63
cifar10 test         6.55
cifar100             4.98     ← smaller
MNIST (32×32 RGB)    ~4.8
```

cifar100 features are shorter than cifar10's, so even directionally
different points get a *low* leverage and look more "typical" than
held-out cifar10 test. This is a confound between feature-norm
encoding confidence and the formula's norm-scaling.

### 2.3 First fix: L2-normalize features

Project `z → z / ||z||` before building the Gram and before scoring.
Now `h(ẑ)` depends only on direction. One-line change exposed as
`--normalize` in every OOD threshold script.

Pre/post comparison on LeJEPA:

```
                    pre-normalize     post-normalize
cifar10 test           0.5037              0.5165
cifar100               0.2900              0.8696
Gaussian noise         0.9999              0.9997
MNIST                  0.5000              0.8917
```

Across backbones (post-normalize, linear phi=z, AUROC of cifar100 vs
cifar10 train):

| Backbone               | phi=z linear | phi=RFF (protocol ell*) |
|------------------------|-------------:|------------------------:|
| ResNet18 supervised    |        0.942 |   0.432 (degenerate ell)|
| DINOv2 ViT-B/14        |        0.975 |                   0.965 |
| SSL LeJEPA ResNet18    |        0.870 |                   0.860 |
| VAE z=256, β=1.0       |        0.623 |                   0.488 |

Every backbone whose features actually encode class semantics recovers
strong OOD signal under linear phi=z post-normalize. The VAE remains
weak — its μ-directions encode reconstruction-relevant statistics
(color, low-frequency structure), not class identity.

### 2.4 Second fix: center *then* L2

The four preprocessing treatments — `raw`, `L2`, `centered`,
`centered+L2` — are scanned in `cifar_centering_comparison.py`. Two
surprises:

1. **`raw` ≡ `centered`** because `build_phi_leverage` already appends
   a bias column. Mean-centering is redundant with the bias degree of
   freedom; the leverage quadratic absorbs it for free.
2. **`centered+L2` ≠ `L2`** because L2 is a nonlinear projection that
   does *not* commute with mean subtraction. Plain `L2` projects onto
   the sphere centered at the origin. After projection, training
   points concentrate in a *hemisphere* whose centroid `μ_sphere` has
   nonzero norm:

   ```
   Train hemisphere center norm ||μ_train|| (on the unit sphere):
       ResNet18           0.32
       LeJEPA             0.47
       DINOv2             0.32
       VAE                0.067   ← already near origin
   ```

   Under plain L2, the leverage quadratic is therefore *not* a clean
   directional Mahalanobis from the train centroid — it's
   Mahalanobis-from-origin biased by a non-trivial offset. `centered+L2`
   = `(z - μ_train) / ||z - μ_train||` projects onto the sphere
   *centered on the training distribution*, and that sphere is the
   one whose tangent-plane geometry the leverage quadratic is actually
   reading.

The principled choice on directional features is therefore
`centered+L2`, not `L2`. The VAE is the exception that proves the
rule: its `μ_train` is already near the origin, so `L2` and
`centered+L2` agree to leading order.

### 2.5 LeJEPA invariance signature, unmasked

Under `centered+L2`, LeJEPA's AUROCs on the four probes are:

```
LeJEPA centered+L2:   cifar10 test = 0.52   cifar100 = 0.91
                      Gaussian     = 0.69   inverted = 0.71
```

Gaussian noise — *the pixel-statistically most extreme probe* — is
seen as **less OOD** than cifar100. This is the joint-embedding
invariance objective speaking: LeJEPA learns directions that are
invariant to pixel statistics by construction, so under a purely
directional score Gaussian noise's pixel-stats axis is collapsed away,
while semantic differences (cifar100 vs cifar10 classes) remain
visible.

VAE under `centered+L2` is the mirror image:

```
VAE centered+L2:      cifar10 test = 0.55   cifar100 = 0.55
                      Gaussian     = 0.96   inverted = 0.57
```

The VAE's encoder reads pixel statistics by reconstruction necessity
but is class-blind. So we get a clean **two-axis picture**:

- LeJEPA = semantic-invariants axis (sees cifar100, blind to Gaussian)
- VAE   = pixel-statistics axis (sees Gaussian, blind to cifar100)

The earlier "high-frequency vs low-frequency" framing is misleading;
the cleaner axis is *pixel statistics* (VAE) vs *semantic invariants*
(LeJEPA).

### 2.6 The concat representation: [VAE; LeJEPA]

If the trade-off is feature-coverage, not training-objective, then
`phi(z) = [phi_vae(z); phi_lejepa(z)]` should catch both axes — no
joint training needed, just deploy a concatenated head at inference.
Tested in `cifar_concat_features_test.py`:

```
                       cifar10 test  cifar100  Gaussian  inverted
vae   centered+L2          0.547      0.554     0.962     0.572
ssl   centered+L2          0.522      0.910     0.690     0.708
vae+ssl block-centered+L2  0.555      0.938     0.994     0.747
```

Concat **strictly improves** every axis. The synergy is most visible
on `inverted`: VAE alone 0.57, LeJEPA alone 0.71, concat 0.75 — the
joint feature has more leverage than either piece can deliver. Block
normalization (per-piece centering + per-piece L2 *before* concat) is
mandatory because the natural norms of the two pieces differ
(`||z_vae|| ≈ 12`, `||z_ssl|| ≈ 6`); skipping per-piece L2 lets the
VAE block dominate the joint Gram.

### 2.7 OOD ≠ classification: orthogonal trade-off

`cifar_concat_linear_probe.py` trains a linear probe on the same three
representations (per-feature standardized on train stats, then
concatenated):

```
                cifar10 top1   cifar100 top1
VAE                 44.91%        21.19%
LeJEPA              87.50%        43.59%
VAE+LeJEPA          86.10%        42.20%
```

Concat *hurts* the probe by 1.4–1.4pp. This is not contradictory with
the OOD result: classification is a structured label-mapping task, so
adding lower-quality features (VAE) to good ones (LeJEPA) injects
noise the classifier cannot fully suppress. OOD detection is
structureless ("is this point typical under the train distribution?"),
so adding any orthogonal information *can only help* — the train Gram
just sees more of the data manifold.

In short: **the leverage-OOD score and the classification probe are
not the same quality metric on features**, and they trade off in
opposite directions when pieces are stacked. Treat them as
complementary readings, not redundant ones.

A separate surprise: VAE features alone hit 44.9% top-1 on CIFAR-10
(predicted ~30%) and 21.2% on CIFAR-100 (21× chance). The VAE encoder
carries more semantic signal than the OOD failure on cifar100
suggested — it's just *misaligned* with the leverage quadratic's
directional axis. A nonlinear probe sees the signal that the linear
leverage quadratic misses.

### 2.8 RFF underperforms linear at the protocol-chosen ell*

Post-normalize, the OOD scripts pick ell* by `AUROC(cifar10 test) ≤ 0.55`.
For ResNet18 supervised this gives ell* ≈ 5.28, at which RFF-leverage
cifar100 AUROC drops to 0.43 while linear is 0.94.

`cifar_resnet18_bandwidth_scan.py` sweeps ell across
`logspace(-2, 2.7, 40)` and shows the issue:

| ell    | cifar10 test | cifar100 | Gaussian | inverted |
|-------:|-------------:|---------:|---------:|---------:|
|  0.010 |        0.667 |    0.974 |    1.000 |    0.949 |
|  0.640 |        0.577 |    0.925 |    0.999 |    0.875 |
|  2.57  |        0.571 |    0.883 |    0.973 |    0.827 |
|  5.90  |        0.499 |    0.305 |    0.183 |    0.344 |  ← protocol picks here
| 17.9   |        0.458 |    0.070 |    0.003 |    0.121 |

Two regimes separated by a hard transition around ell ≈ 4–6:

- **Small ell (≲ 3)**: kernel has angular resolution; cifar100 AUROC
  ≈ 0.97 beats linear; train/test gap is real (0.57–0.67).
- **Large ell (≳ 6)**: kernel ≈ constant on the unit sphere,
  `Phi^T Phi` rank-deficient, ridge dominates, leverage degenerates,
  every AUROC drops below 0.5.

The protocol's "smallest ell with AUROC(test) ≤ 0.55" forces ell* into
the degenerate regime because at smaller ell the kernel correctly
reads the train/test gap as 0.55+. The fix is to relax the tune-target
(say 0.6–0.7), use a fixed small ell suited to unit-sphere geometry,
or tune against a known OOD source. Linear `phi = z` has no bandwidth
knob and reads the anisotropic spectrum of train features directly —
it's the right default on directional features.

### 2.9 Memorization scan: linear AUROCs as a feature-geometry probe

`cifar_memorization_scan.py` sweeps `ell` across all four backbones
and records three quantities per `(backbone, ell)`:

- **AUROC(cifar10 test vs train)** — memorization signal. A high value
  means the kernel fits train-specific structure that doesn't
  generalize.
- **AUROC(cifar100 vs train)** — OOD-discrimination signal.
- **eff_dof / n_train** — `(1/n) Σ_i h(z_i^train) = trace(H)/n`. With
  `M_rff = 2048`, `n_train = 8192`, bias on, the delta-kernel ceiling
  is `M / (2n) = 0.25`.

Three regimes appear across every backbone:

1. **Memorization** (small ell, dof/n at 0.25): test and OOD both
   saturate near 1.0 — useless.
2. **Generalization** (mid ell, dof/n falling): test settles at
   0.55–0.65 while OOD stays high — *useful band*.
3. **Degenerate** (large ell, dof/n → 0): both AUROCs collapse near
   or below 0.5.

Per-backbone snapshot at linear `phi = z` (bandwidth-free):

| Backbone               |  AUROC(test) | AUROC(c100) | gap | dof/n |
|------------------------|-------------:|------------:|----:|------:|
| ResNet18 supervised    |        0.601 |       0.940 | 0.34| 0.059 |
| LeJEPA ResNet18        |        0.571 |       0.883 | 0.31| 0.063 |
| DINOv2 ViT-B/14        |        0.604 |       0.971 | 0.37| 0.094 |
| VAE z=256              |        0.578 |       0.622 | 0.04| 0.027 |

The gap `AUROC(c100) − AUROC(test)` is the cleanest single-number
OOD-quality score (controls for memorization). VAE collapses to 0.04;
the others sit at 0.31–0.37.

The RFF sweep adds *memorization capacity*: how easily can the kernel
be pushed into the memorization regime?

| Backbone   | min ell at AUROC(test) ≥ 0.9 | dof/n at AUROC(test) ≈ 0.6 |
|------------|-----------------------------:|---------------------------:|
| ResNet18   |     never (max ≈ 0.665) |                      0.019 |
| LeJEPA     |                     ~0.04 |                      0.068 |
| DINOv2     |                     ~0.02 |                      0.092 |
| VAE        |                     ~0.06 |                      0.030 |

Supervised ResNet18 is uniquely *resistant* to memorization: its
features collapse to ~10 angular cluster centers (one per class),
so even at ell = 0.01 there are many train points within any
angular ball of radius `ell` and the kernel can't isolate
individuals. The other three spread training points per-instance.
Among them, LeJEPA's invariance objective produces the *lowest*
per-image individuation (integrated AUROC(test vs train) ≈ 0.506),
followed by DINOv2 (≈ 0.554), supervised ResNet18 (≈ 0.561), VAE
(≈ 0.592, the most memorizing).

---

## Part III. The "best foundation model resolves OOD" frame

The encoder-progression observation in Section 2.7 (concat helps OOD,
hurts the probe) and Section 2.9 (LeJEPA spreads instances on the
sphere; DINOv2 does too, but more semantically) points at a broader
claim worth stating explicitly:

**With a sufficiently good encoder, OOD detection becomes trivial via
the simplest possible head.** "Sufficient" means the encoder
preserves enough discrimination structure that downstream targets —
OOD against any distribution, classification against any label set,
retrieval against any query — are linearly (or nearly-linearly)
accessible. DINOv2 approaches this for natural images; whatever
replaces it pushes further.

The implication is somewhat anti-method: most OOD work is fighting
feature quality rather than improving it. If foundation models keep
getting better, OOD methods that work by adding machinery on top of
weak features become obsolete. Methods that **read off properties of
good features cleanly** remain useful because they're invariant to
which good encoder you use. The leverage construction here is one of
those — it's a closed-form quadratic on top of whatever encoder you
plug in, and Section 2.3's table shows the OOD signal scaling
monotonically with encoder quality (DINOv2 > ResNet18 > LeJEPA > VAE
on cifar100 AUROC, post-normalize).

This is the "scaling solves it" view applied to a specific problem.
True to the extent that:

1. Foundation models are getting better fast.
2. The relevant downstream tasks are dominated by feature quality.
3. The cost of using a foundation model is acceptable.

Not true to the extent that:

1. Some tasks have fundamentally task-specific information needs that
   no foundation model is incentivized to learn (e.g. fine-grained
   distinctions the pretraining data and objective collapse).
2. Domain-specific tasks (medical, scientific, low-resource) lag
   behind because foundation models are trained on web data.
3. Even good features have failure modes — the norm-confound in
   Section 2.2, the LeJEPA-style augmentation-induced invariance that
   makes Gaussian noise look in-distribution (Section 2.5), etc.

**The honest version:** for tasks where information needs align with
what foundation models learn, the best foundation model + simple
closed-form head solves the problem and method-level innovation is
moot. For tasks orthogonal to foundation-model training, this
doesn't apply and specialized methods (or specialized encoders, or
encoder stacking as in Section 2.6) matter. The frontier is figuring
out which tasks fall in which category and whether the
foundation-model-aligned set is growing.

What this repo is building, framed this way, is **infrastructure for
the closed-form head**: a single leverage quadratic that doubles as
energy, posterior variance, and OOD score, and which works on top of
whatever encoder you point it at. The CIFAR experiments are not
arguing that any particular encoder is best; they're showing that the
quality ordering one would expect from the encoder-progression view
(DINOv2 ≫ supervised > LeJEPA > VAE on semantic OOD) is exactly
what the leverage score reads off.

---

## Part IV. The 2D analogue

`hetero/hetero_demo_2d_ood_petal_langevin.py` exposes the same
`--normalize` flag and shows the directional-vs-norm question
cleanly: petal data sits near the origin while OOD probes are at
larger radius, so un-normalized leverage assigns OOD points *low*
`h`. Normalization moves all probes onto the unit circle and the
contour plot becomes purely angular — what we want for a directional
density estimator.

---

## Part V. Practical recipes

The leverage formula reuses across the three roles, so the same
preprocessing decisions feed all three.

**For OOD detection on directional features:**

1. L2-normalize, then center on the L2-normalized train mean, then
   re-L2 (= `centered+L2`). Or use the bias column and just `L2` if
   your features already have `||μ_train||` near zero.
2. Default to linear `phi = z`. Reach for RFF only if you scan the
   bandwidth and verify a generalization band exists.
3. Reference the train-vs-train threshold at the 0.95-quantile of
   `h(z_i^train)`. Report `AUROC(test)`, `AUROC(OOD)`, and the gap.

**For an EBM via leverage:**

1. Compute the Gram on whatever features you intend to evaluate
   `h(z)` on (typically `[z; RFF(z)]` for VAE-z space; raw `z` if
   you've already L2-normalized backbone features).
2. Run overdamped Langevin with a geometric anneal on the step
   size (see `cifar_vae_langevin.py`). Particles drawn from
   `N(0, I)` should land on the manifold within ≲ 10⁴ steps.
3. The same `h(z)` thresholded gives OOD scores on decoded
   samples — energy and OOD are the same number.

**For posterior-variance / uncertainty:**

1. Choose `ridge = σ_n^2 / σ_w^2`. If you have a noise estimate
   and a prior-variance estimate this is calibrated; if not, treat
   `ridge` as an inverse-confidence prior (small ridge ⇔ confident
   posterior at well-supported inputs, sharp falloff away from them).
2. Report `h(z*)` as relative predictive variance. The training-set
   quantile sets a "typical model uncertainty" reference.

**Stacking representations:** Per-block standardize then concat.
Helps OOD strictly; may hurt classification probes — that's expected,
not a bug.

---

## Summary

1. The leverage formula plays three roles with the same math: EBM
   energy (via Langevin in `cifar_vae_langevin.py`), GP posterior
   variance (Bayesian last-layer view), and OOD score.
2. On directional features, `centered+L2` is the principled
   preprocessing — `L2` alone is biased by the train hemisphere's
   off-origin centroid. `centered` alone is a no-op because the
   bias column in `build_phi_leverage` already absorbs the mean.
3. Norm-confound (Section 2.2) flips the OOD signal pre-normalize.
   L2 normalize is necessary; on features with off-center hemispheres
   it's not quite sufficient and `centered+L2` is the right default.
4. The bandwidth-tuned RFF kernel can degenerate at the protocol's
   ell*. Linear `phi=z` is the safer default on unit-sphere features;
   `cifar_memorization_scan.py` shows that a useful generalization
   band exists for ResNet18, LeJEPA, DINOv2 but not for VAE.
5. VAE features encode pixel statistics, LeJEPA features encode
   semantic invariants. The concat `[VAE; LeJEPA]` covers both axes
   under leverage-OOD; the same concat *hurts* linear classification
   by ~1.4pp. OOD-as-leverage and classification probing are
   complementary metrics, not redundant ones.
