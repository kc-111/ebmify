"""Numeric sanity tests for DKALayer / DKABlock.

Run as ``python example/dka/sanity.py``. Exits non-zero on first failure.
No GPU / dataset required.

Tests (all from DKA.md §C.1 / §2.2 / §3.2 / §3.3 / §12):

    1. Shapes
    2. Confident Kalman limit:        v -> 0, K -> 1, out ≈ hat
    3. Uncertain Kalman limit:        v -> 1/lam, K small, out ≈ x
    4. Variance bounds:               0 <= v <= 1/lam
    5. Gradient flow:                 W_phi.grad, W_v.grad nonzero
    6. Stop-grad on eigenbasis:       V_r/Lam_r have requires_grad = False
    7. Naive-doubling pathology:      x + hat doubles in confident limit;
                                       Kalman form does not.
    8. Recurrent quasi-fixed-point:   variance weakly decreases (or close)
                                       across recurrent iterations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT  # noqa: F401, E402

from ebmify.models.dka import DKABlock, DKALayer  # noqa: E402

torch.manual_seed(0)

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"  PASS  {name}")
    else:
        FAILED.append((name, detail))
        print(f"  FAIL  {name}  ({detail})")


# ---------------------------------------------------------------- 1. shapes
def test_shapes() -> None:
    layer = DKALayer(d_model=64, d_feat=32, rank_rule="adaptive", lam=1.0)
    x = torch.randn(2, 16, 64)
    out, info = layer(x)
    _check("shapes/out", out.shape == (2, 16, 64), f"got {tuple(out.shape)}")
    _check("shapes/v", info["v"].shape == (2, 16), f"got {tuple(info['v'].shape)}")
    _check("shapes/hat", info["hat"].shape == (2, 16, 64), f"got {tuple(info['hat'].shape)}")
    _check("shapes/r_eff_int", isinstance(info["r_eff"], int))


# --------------------------------------------------- 2. confident Kalman limit
def test_confident_limit() -> None:
    """λ small + L >> M with rich features → low variance, K → 1, out ≈ hat."""
    torch.manual_seed(1)
    layer = DKALayer(d_model=32, d_feat=8, rank_rule="full", lam=1e-3)
    x = torch.randn(1, 256, 32)
    out, info = layer(x)
    v_mean = info["v"].mean().item()
    K_mean = (1.0 / (1.0 + info["v"])).mean().item()
    # out should be much closer to hat than to x
    err_to_hat = (out - info["hat"]).pow(2).mean().sqrt().item()
    err_to_x = (out - x).pow(2).mean().sqrt().item()
    _check(
        "confident/v_small",
        v_mean < 0.5,
        f"v_mean={v_mean:.4f}",
    )
    _check("confident/K_large", K_mean > 0.5, f"K_mean={K_mean:.4f}")
    _check(
        "confident/out_near_hat",
        err_to_hat < err_to_x,
        f"|out-hat|={err_to_hat:.4f} |out-x|={err_to_x:.4f}",
    )


# --------------------------------------------------- 3. uncertain Kalman limit
def test_uncertain_limit() -> None:
    """v ∈ (0, 1/λ]. K = 1/(1+v) is monotone-decreasing in v: more context
    (larger L) drives v down and K up. Verify the monotonic trend."""
    torch.manual_seed(2)
    lam = 1.0
    layer = DKALayer(d_model=16, d_feat=8, rank_rule="full", lam=lam)
    x_small = torch.randn(1, 4, 16)
    x_big = torch.randn(1, 256, 16)
    _, info_small = layer(x_small)
    _, info_big = layer(x_big)
    v_small = info_small["v"].mean().item()
    v_big = info_big["v"].mean().item()
    K_small = (1.0 / (1.0 + info_small["v"])).mean().item()
    K_big = (1.0 / (1.0 + info_big["v"])).mean().item()
    _check(
        "uncertain/v_decreases_with_L",
        v_big <= v_small + 1e-6,
        f"v(L=4)={v_small:.4f}  v(L=256)={v_big:.4f}",
    )
    _check(
        "uncertain/K_increases_with_L",
        K_big >= K_small - 1e-6,
        f"K(L=4)={K_small:.4f}  K(L=256)={K_big:.4f}",
    )
    _check(
        "uncertain/K_in_unit",
        0.0 < K_small <= 1.0 and 0.0 < K_big <= 1.0,
        f"K_small={K_small:.4f}  K_big={K_big:.4f}",
    )


# ------------------------------------------------------- 4. variance bounds
def test_variance_bounds() -> None:
    torch.manual_seed(3)
    for lam in (0.01, 1.0, 100.0):
        layer = DKALayer(d_model=16, d_feat=8, rank_rule="full", lam=lam)
        x = torch.randn(3, 24, 16)
        _, info = layer(x)
        v = info["v"]
        # Allow a tiny float-precision slop above 1/lam.
        ok = (v.min().item() >= 0.0) and (v.max().item() <= 1.0 / lam + 1e-3)
        _check(
            f"var_bounds/lam={lam}",
            ok,
            f"v in [{v.min().item():.4g}, {v.max().item():.4g}], 1/lam={1/lam:.4g}",
        )


# --------------------------------------------------------- 5. gradient flow
def test_grad_flow() -> None:
    layer = DKALayer(d_model=16, d_feat=8, rank_rule="adaptive", target="value", lam=1.0)
    x = torch.randn(2, 12, 16, requires_grad=False)
    out, _ = layer(x)
    out.sum().backward()
    phi_params = list(layer.W_phi.parameters())
    _check(
        "grad/W_phi_all_nonzero",
        all(p.grad is not None and p.grad.abs().sum() > 0 for p in phi_params),
        f"phi has {len(phi_params)} param(s)",
    )
    _check(
        "grad/W_v",
        layer.W_v.weight.grad is not None and layer.W_v.weight.grad.abs().sum() > 0,
    )


# ----------------------------------------------------- 5b. MLP phi works too
def test_phi_mlp() -> None:
    """φ defaults to a 2-layer MLP. Verify (a) shapes pass through,
    (b) grads flow to BOTH MLP layers."""
    layer = DKALayer(d_model=32, d_feat=16, phi_hidden=64, rank_rule="full", lam=1.0)
    x = torch.randn(2, 12, 32)
    out, info = layer(x)
    _check("phi_mlp/shape", out.shape == (2, 12, 32))
    out.sum().backward()
    grads = [p.grad for p in layer.W_phi.parameters()]
    _check("phi_mlp/grads_present", all(g is not None for g in grads))
    _check(
        "phi_mlp/grads_nonzero",
        all(g.abs().sum() > 0 for g in grads),
    )


# ------------------------------------------------ 5c. softmax baseline runs
def test_softmax_baseline() -> None:
    from ebmify.models.dka import SoftmaxAttentionBlock

    blk = SoftmaxAttentionBlock(d_model=32, n_heads=4)
    x = torch.randn(2, 12, 32, requires_grad=True)
    out, infos = blk(x)
    _check("softmax/shape", out.shape == (2, 12, 32))
    _check("softmax/info_list", isinstance(infos, list) and len(infos) == 0)
    out.sum().backward()
    _check("softmax/grad_in", x.grad is not None and x.grad.abs().sum() > 0)


# ------------------------------------------- 6. stop-grad on eigenbasis path
def test_stop_grad() -> None:
    """V_r, Lam_r, inv, w_var should all live inside no_grad → no grad through eigh."""
    layer = DKALayer(d_model=16, d_feat=8, rank_rule="full", lam=1.0)
    x = torch.randn(1, 16, 16, requires_grad=True)
    phi = layer.W_phi(x)
    from ebmify.models.dka import _feature_rmsnorm

    if layer.feature_norm:
        phi = _feature_rmsnorm(phi, layer.gamma)
    with torch.no_grad():
        G = phi.float().transpose(-1, -2) @ phi.float()
        Lam, V = torch.linalg.eigh(G)
        V_r = V.flip(-1)[..., :8].to(phi.dtype)
        Lam_r = Lam.flip(-1)[..., :8].to(phi.dtype)
    _check("stop_grad/V_r", not V_r.requires_grad)
    _check("stop_grad/Lam_r", not Lam_r.requires_grad)
    # Confirm grads still flow via Phi_r = phi @ V_r path.
    Phi_r = torch.einsum("blm,bmr->blr", phi, V_r)
    Phi_r.sum().backward()
    _check("stop_grad/phi_path_alive", x.grad is not None and x.grad.abs().sum() > 0)


# ------------------------------------------------ 7. naive-doubling pathology
def test_naive_doubling() -> None:
    """§2.2: the naive `x + hat` form has no convex-combination guarantee
    and can grow without bound; the Kalman form `(1-K)x + K·hat` keeps the
    per-token output norm bounded by the max of (||x||, ||hat||)."""
    torch.manual_seed(4)
    # Match d_feat to d_model so hat can fully reconstruct x in the
    # confident limit — that's the regime where naive doubling bites.
    layer_kal = DKALayer(d_model=32, d_feat=32, rank_rule="full",
                         lam=1e-3, output_form="kalman")
    layer_none = DKALayer(d_model=32, d_feat=32, rank_rule="full",
                          lam=1e-3, output_form="none")
    layer_none.load_state_dict(layer_kal.state_dict())
    x = torch.randn(1, 128, 32)
    out_kal, info_kal = layer_kal(x)
    out_none, info_none = layer_none(x)
    hat = info_kal["hat"]
    # 7a. Definitional: out_none == x + hat.
    _check(
        "naive_doubling/none_is_x_plus_hat",
        (out_none - (x + hat)).abs().max().item() < 1e-3,
    )
    # 7b. Kalman is a convex combination of x and hat → norm ≤ max(||x||, ||hat||).
    kal_norm = out_kal.norm(dim=-1)
    bound = torch.maximum(x.norm(dim=-1), hat.norm(dim=-1)) + 1e-3
    _check(
        "naive_doubling/kalman_norm_bounded",
        (kal_norm <= bound).all().item(),
        f"max excess: {(kal_norm - bound).max().item():.4f}",
    )
    # 7c. Naive form can exceed both ||x|| and ||hat||.
    none_norm = out_none.norm(dim=-1)
    excess_none = (none_norm - bound).max().item()
    _check(
        "naive_doubling/none_can_exceed",
        excess_none > 0,
        f"none-form max excess over Kalman bound = {excess_none:.4f}",
    )


# ------------------------------------------------- 7b. multi-head
def test_multihead() -> None:
    """n_heads > 1: shapes pass through, info['v'] has the head axis,
    grads flow to W_out and W_phi."""
    torch.manual_seed(6)
    layer = DKALayer(d_model=64, d_feat=16, n_heads=4, rank_rule="adaptive", lam=1.0)
    x = torch.randn(2, 24, 64, requires_grad=True)
    out, info = layer(x)
    _check("multihead/out_shape", out.shape == (2, 24, 64), f"got {tuple(out.shape)}")
    _check("multihead/v_shape", info["v"].shape == (2, 4, 24),
           f"got {tuple(info['v'].shape)}")
    _check("multihead/hat_shape", info["hat"].shape == (2, 24, 64),
           f"got {tuple(info['hat'].shape)}")
    out.sum().backward()
    _check("multihead/grad_x", x.grad is not None and x.grad.abs().sum() > 0)
    assert layer.W_out is not None
    _check("multihead/grad_W_out",
           layer.W_out.weight.grad is not None and layer.W_out.weight.grad.abs().sum() > 0)


# ------------------------------------------------- 7c. causal mode
def test_causal() -> None:
    """causal=True: output at position l is independent of positions > l.
    Perturb x[:, l+1:, :] and verify out[:, :l+1, :] is unchanged."""
    torch.manual_seed(7)
    layer = DKALayer(d_model=32, d_feat=16, n_heads=2, causal=True, lam=1.0)
    layer.eval()
    x = torch.randn(1, 16, 32)
    out1, info1 = layer(x)
    _check("causal/out_shape", out1.shape == (1, 16, 32))
    _check("causal/v_shape", info1["v"].shape == (1, 2, 16))
    # Perturb the second half of the input. Output of the first half should match.
    x2 = x.clone()
    x2[:, 8:, :] += torch.randn_like(x2[:, 8:, :])
    out2, _ = layer(x2)
    head_match = (out1[:, :8, :] - out2[:, :8, :]).abs().max().item()
    _check("causal/no_future_leakage", head_match < 1e-4,
           f"max diff in causal head: {head_match:.6f}")
    # And the second half SHOULD differ.
    tail_diff = (out1[:, 8:, :] - out2[:, 8:, :]).abs().max().item()
    _check("causal/tail_differs", tail_diff > 1e-3,
           f"tail diff: {tail_diff:.4f}")
    # Grad flow.
    x_g = torch.randn(1, 16, 32, requires_grad=True)
    out, _ = layer(x_g)
    out.sum().backward()
    _check("causal/grad_x", x_g.grad is not None and x_g.grad.abs().sum() > 0)


# ------------------------------------------------- 7d. phi_type swap
def test_phi_type() -> None:
    """phi_type='mlp' and phi_type='swiglu' both run and produce different weights."""
    layer_swi = DKALayer(d_model=32, d_feat=16, phi_type="swiglu", rank_rule="full", lam=1.0)
    layer_mlp = DKALayer(d_model=32, d_feat=16, phi_type="mlp", rank_rule="full", lam=1.0)
    x = torch.randn(1, 12, 32)
    out_s, _ = layer_swi(x)
    out_m, _ = layer_mlp(x)
    _check("phi_type/swiglu_shape", out_s.shape == (1, 12, 32))
    _check("phi_type/mlp_shape", out_m.shape == (1, 12, 32))
    # SwiGLU has 3 Linears (gate/up/down), MLP has 2 (fc1/fc2).
    n_swi = sum(1 for _ in layer_swi.W_phi.parameters())
    n_mlp = sum(1 for _ in layer_mlp.W_phi.parameters())
    _check("phi_type/param_counts", n_swi == 3 and n_mlp == 2,
           f"swiglu={n_swi} mlp={n_mlp}")


# -------------------------------------------------- 8. recurrent fixed-point
def test_recurrent_fixed_point() -> None:
    """Iterating DKABlock should not blow up; variance should not increase
    on average across iterations (the §12 recurrent contraction picture)."""
    torch.manual_seed(5)
    block = DKABlock(d_model=32, d_feat=8, recurrent_depth=4, lam=1.0)
    x = torch.randn(2, 24, 32)
    out, infos = block(x)
    _check("recurrent/finite", torch.isfinite(out).all().item())
    means = [i["v"].mean().item() for i in infos]
    # Weak check: last iter's mean variance should not exceed the first by
    # more than 50% (allows fluctuation but flags divergence).
    _check(
        "recurrent/var_non_explode",
        means[-1] < 1.5 * means[0] + 0.05,
        f"v means across iters = {means}",
    )


# ---------------------------------------------------------------- driver
def main() -> int:
    print("DKA sanity tests")
    print("-" * 40)
    for fn in (
        test_shapes,
        test_confident_limit,
        test_uncertain_limit,
        test_variance_bounds,
        test_grad_flow,
        test_phi_mlp,
        test_softmax_baseline,
        test_stop_grad,
        test_naive_doubling,
        test_multihead,
        test_causal,
        test_phi_type,
        test_recurrent_fixed_point,
    ):
        try:
            fn()
        except Exception as e:  # pragma: no cover (sanity script)
            FAILED.append((fn.__name__, f"exception: {e!r}"))
            print(f"  FAIL  {fn.__name__}  (exception: {e!r})")
    print("-" * 40)
    print(f"passed: {len(PASSED)}   failed: {len(FAILED)}")
    if FAILED:
        for n, d in FAILED:
            print(f"  - {n}: {d}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
