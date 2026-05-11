# ebmify

**ebmify any model.** Pick any feature extractor `phi(x)` — a trained
MLP's penultimate activations, random Fourier features, even raw `x` —
and the Bayesian last-layer posterior variance

```
    h(x) = phi(x)^T (Phi^T Phi + lambda*I)^-1 phi(x)
```

is a ready-made energy function. The same scalar serves as:

* an **OOD score** (it blows up where the training set never put mass),
* a **density estimate** (`exp(-h)` is the predictive Gaussian-process
  variance under a linear-in-features Bayesian model — the GP analog
  drops out for free),
* and an **energy for Langevin sampling** (`-grad h` flows toward the
  data manifold).

The design lever is the feature map `phi`: pick it to match the
geometry of the data, and the same formula yields OOD scores, density
level-sets, and sampler drift fields.

Three families of experiments are included:

1. **2D toy data** — random Fourier features (RFFs) on the raw plane,
   with explicit comparisons against deeper FCNet feature maps. Shows
   when concatenating raw `x` helps and when a deep stack helps.
2. **MNIST** — train a small beta-VAE, then build leverage on
   `phi(z) = [z; RFF(z)]` in the latent space. The same `h(z)`
   doubles as an OOD detector (encode strange `x`, watch `h(z)`
   explode) and as the energy for a SamAdams Langevin sampler that
   regenerates diverse digits.
3. **CIFAR-10 / CIFAR-100** — same recipe as MNIST, with a deeper
   conv-VAE and cross-dataset OOD evaluation.

## Quick start

```bash
git clone <repo-url> ebmify
cd ebmify
pip install -e .

# fetch datasets (MNIST = ~12 MB, CIFAR = ~330 MB)
python download_mnist.py
python download_cifar.py
```

Every script writes its plot to `example/out/`.

## Toy 2D experiments

Four self-contained scripts, each illustrating a different facet of the
leverage signal on synthetic 2D data:

| Script | What it shows |
|---|---|
| `example/hetero/hetero_demo_2d_ood_deep_rff.py` | Deep RFF stack (input-RFF + MLP trunk + output-RFF) on a moons+ring+spiral topology with *internal holes*. Bounded output-RFF activations keep the leverage signal sharp across both the empty disc inside the ring and the slot between the moons. |
| `example/hetero/hetero_demo_2d_ood_checkerboard_langevin.py` | Overdamped Langevin on the leverage energy `E(x) = h(x) / h_char` for a checkerboard. Particles started outside the board drift in; particles started *inside* black cells must cross thermal saddles into the neighboring white cells. |
| `example/hetero/hetero_demo_2d_ood_petal_langevin.py` | Same Langevin dynamics on a petal-topology dataset (center cluster + `n_petals` peripheral clusters, only a subset observed). Tests whether annealed sampling can find the unobserved clusters from inside the leverage plateau. |
| `example/hetero/random_features_2d_density.py` | Sweeps **8 feature maps** for `h(x)` on moons: raw `x`, RFF only, `[x; RFF]`, random FCNet trunk, trained FCNet trunk, `[x; trunk]`, `[x; pre-out]`, `[x; every hidden state]`. Isolates how concatenating raw `x` adds a quadratic bowl on top of an RFF kernel-density floor. |

Run any of them directly:

```bash
python example/hetero/hetero_demo_2d_ood_deep_rff.py
python example/hetero/random_features_2d_density.py
```

## MNIST experiments

Three-step pipeline.

```bash
# 1) train the beta-VAE (cached under example/mnist/cache/)
python example/mnist/mnist_vae_train.py

# 2) z-space SamAdams Langevin — generates digits from h(z) energy
python example/mnist/mnist_vae_langevin.py --T 10 --T-lo 1e-7 --steps 100000

# 3) x -> z OOD evaluation under phi = z, RFF(z), and [z; RFF(z)]
python example/mnist/mnist_vae_ood_eval.py
```

The Langevin recipe `--T 10 --T-lo 1e-7 --steps 100000` reproduces a
diverse set of digits (all 10 classes). Lower temperatures collapse to
a few modes; higher temperatures over-explore and produce blurry
samples.

The OOD eval reports leverage separation versus 7 OOD `x` sources:
uniform, Bernoulli, Gaussian (clamped to `[0,1]`), pixel-shuffled MNIST,
inverted MNIST, all-black, and all-white. The encoder's behavior on
constant images (black ~5x, white ~100x+) reflects the asymmetric
"MNIST = mostly dark + bright strokes" prior it has internalized.

## CIFAR experiments

Same recipe, deeper VAE (4 conv blocks, `z_dim=64`), one VAE per
dataset.

```bash
# CIFAR-10
python example/cifar/cifar_vae_train.py    --dataset cifar10
python example/cifar/cifar_vae_langevin.py --dataset cifar10
python example/cifar/cifar_vae_ood_eval.py --dataset cifar10

# CIFAR-100
python example/cifar/cifar_vae_train.py    --dataset cifar100
python example/cifar/cifar_vae_langevin.py --dataset cifar100
python example/cifar/cifar_vae_ood_eval.py --dataset cifar100
```

The OOD eval includes a **cross-dataset** column: the CIFAR-10 VAE
scoring CIFAR-100 inputs and vice versa. This is the standard hard case
in the OOD-detection literature.

## Reading the code

```
src/ebmify/
    models/
        _base.py      h(x), train/eval scaffolding, FitConfig wiring
        _config.py    FitConfig, RegConfig, NoiseConfig, PreprocessConfig
        _scaler.py    StandardScale, YeoJohnson, KDEQuantile, ...
        fc.py         FCNet, RFFLayer (random Fourier features module)
    sampler/
        samadams.py   SamAdams overdamped Langevin sampler (adaptive)
```

The public surface is small — `FCNet`, `RFFLayer`, `FitConfig`,
`feature_leverage`, the SamAdams sampler — re-exported at
`ebmify.models` and `ebmify.sampler`.

## License

MIT. See `LICENSE`.
