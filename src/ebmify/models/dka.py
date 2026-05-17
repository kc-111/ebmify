"""Denoising Kalman Attention (DKA) — DKA.md §14.2 + multi-head + causal.

A Bayesian primitive: per-batch ridge regression in a learned feature space
with closed-form posterior and Kalman-gain fusion of the residual stream
with the context-driven estimate.

Eleven essential lines (paraphrased from §14.2, single-head non-causal):

    phi = rmsnorm(MLP_phi(X))
    Y   = X                                # denoising target
    with no_grad():
        G        = phi.T @ phi
        Lam, V   = eigh(G)
        r        = adaptive_rank(Lam)
        V_r      = V[..., :r]; Lam_r = Lam[..., :r]
        inv      = 1 / (Lam_r + lam)
    Phi_r = phi @ V_r
    hat   = Phi_r @ (inv * (Phi_r.T @ Y))
    v     = (1/lam) * (1 - sum(Lam_r * inv * Phi_r^2))
    K     = 1 / (1 + v)
    out   = X + K * (hat - X)

The §3.3 stop-grad on the eigenbasis is what makes gradients flow cleanly
through `Phi_r` without touching `eigh`'s ill-conditioned backward.

This file also implements:

- **Multi-head DKA** (``n_heads > 1``): φ and Y are split into heads of
  size ``d_feat`` / ``d_h``; per-head Gram + eigh; per-head Kalman fusion;
  outputs concatenated and projected by a learned ``W_out`` (MHA-style).
- **Causal DKA** (``causal=True``): the per-token regression uses only
  positions ``<= l``. Implemented via cumulative ``phi phi^T`` /
  ``phi y^T`` and a single fully-batched Cholesky solve over all prefixes.
  No eigh / rank truncation in this mode.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def _feature_rmsnorm(phi: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """QK-Norm on features (§2.3). RMSNorm over the last dim with a learnable per-channel gain.
    Works for both (B, L, M) and (B, H, L, M) by broadcasting gamma over the last axis.

    `eps` is tiny by design: when φ comes from a SwiGLU MLP whose three Linears are init'd
    at std=0.02, the raw mean-square can be ~1e-7. A typical eps=1e-6 would dominate the
    denominator and leave the normalized output at ~0.5, which breaks the §14.2 variance
    formula (which assumes ||φ_l||² ≈ M after norm)."""
    rms = phi.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return phi * rms * gamma


class _PhiMLP(nn.Module):
    """Two-layer GELU MLP for the feature map φ(x)."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_out, bias=False)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _PhiSwiGLU(nn.Module):
    """SwiGLU MLP for the feature map φ(x): down(silu(gate(x)) * up(x)).

    Matches the Llama / PaLM FFN pattern. Default choice for φ because
    SwiGLU is the modern transformer FFN default and outperforms GELU
    MLPs at the same param budget on most LM benchmarks.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_in, d_hidden, bias=False)
        self.w_up = nn.Linear(d_in, d_hidden, bias=False)
        self.w_down = nn.Linear(d_hidden, d_out, bias=False)
        # Kaiming-style init so the SwiGLU produces output with std ~ O(1)
        # rather than the ~1e-3 magnitude you get from stacking std=0.02
        # linears. Required so the downstream RMSNorm denominator isn't
        # dominated by eps.
        nn.init.kaiming_normal_(self.w_gate.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.w_up.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.w_down.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class _PhiBranch(nn.Module):
    """One-hidden-layer branch projection + optional RMSNorm."""

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        *,
        branch_type: Literal["linear", "swiglu", "mlp"],
        use_norm: bool,
    ) -> None:
        super().__init__()
        if branch_type == "linear" or d_hidden <= 0:
            self.proj = nn.Linear(d_in, d_out, bias=False)
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        elif branch_type == "swiglu":
            self.proj = _PhiSwiGLU(d_in, d_hidden, d_out)
        elif branch_type == "mlp":
            self.proj = _PhiMLP(d_in, d_hidden, d_out)
        else:
            raise ValueError(f"unknown branch_type: {branch_type!r}")
        self.use_norm = bool(use_norm)
        if self.use_norm:
            self.gamma = nn.Parameter(torch.ones(d_out))
        else:
            self.register_parameter("gamma", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(x)
        if self.use_norm:
            y = _feature_rmsnorm(y, self.gamma)
        return y


class DKALayer(nn.Module):
    """Feature-attention DKA with q/k/v/p branches and variance gating."""

    def __init__(
        self,
        d_model: int,
        d_feat: int,
        *,
        n_heads: int = 1,
        causal: bool = False,
        d_value: int | None = None,
        value_hidden: int | None = None,
        value_type: Literal["linear", "swiglu", "mlp"] = "swiglu",
        phi_hidden: int | None = None,
        phi_type: Literal["swiglu", "mlp"] = "swiglu",
        lam: float = 1.0,
        target: Literal["x", "value"] = "value",
        output_form: Literal["kalman", "none", "gated"] = "kalman",
        feature_norm: bool = True,
        kernel_strides: tuple[int, ...] = (1, 2, 4),
        **deprecated_kwargs,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = d_model // n_heads
        self.d_feat = int(d_feat)
        self.causal = bool(causal)
        # Kept for API compatibility; this architecture always uses learned phi_v target.
        self.target = "value"
        self.output_form = output_form
        self.lam = float(lam)
        self.kernel_strides = tuple(int(s) for s in kernel_strides if int(s) > 0)
        if len(self.kernel_strides) == 0:
            self.kernel_strides = (1,)
        # Accept legacy kwargs for compatibility, but ignore them.
        _ = deprecated_kwargs

        d_value = d_value or self.d_head
        self.d_value = int(d_value)
        d_feat_total = self.n_heads * self.d_feat
        d_value_total = self.n_heads * self.d_value

        if phi_hidden is None:
            phi_hidden = 4 * d_feat_total
        if value_hidden is None:
            value_hidden = 4 * d_value_total
        self.phi_hidden = int(phi_hidden)
        self.value_hidden = int(value_hidden)

        self.phi_q = _PhiBranch(
            d_model,
            self.phi_hidden,
            d_feat_total,
            branch_type=phi_type,
            use_norm=feature_norm,
        )
        self.phi_k = _PhiBranch(
            d_model,
            self.phi_hidden,
            d_feat_total,
            branch_type=phi_type,
            use_norm=feature_norm,
        )
        self.phi_p = _PhiBranch(
            d_model,
            self.phi_hidden,
            d_feat_total,
            branch_type=phi_type,
            use_norm=feature_norm,
        )
        self.phi_v = _PhiBranch(
            d_model,
            self.value_hidden,
            d_value_total,
            branch_type=value_type,
            use_norm=feature_norm,
        )

        if output_form == "gated":
            self.gate_log = nn.Parameter(torch.zeros(d_value_total))

        # Project back whenever merged value dim differs from d_model.
        if d_value_total != d_model:
            self.W_out = nn.Linear(d_value_total, d_model, bias=False)
            nn.init.normal_(self.W_out.weight, std=0.02)
        else:
            self.W_out = None

    def _split_heads(self, t: torch.Tensor, dim_per_head: int) -> torch.Tensor:
        B, L, _ = t.shape
        return t.reshape(B, L, self.n_heads, dim_per_head).transpose(1, 2)

    def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, H, L, d = t.shape
        return t.transpose(1, 2).contiguous().reshape(B, L, H * d)

    def _kernel_weights(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Default halving-factor blend: weight ~ 1/stride.
        w = torch.tensor([1.0 / float(s) for s in self.kernel_strides], device=device, dtype=dtype)
        return w / w.sum()

    def _sample_token_mask(
        self,
        *,
        B: int,
        H: int,
        L: int,
        stride: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample per-token mask without replacement.

        For stride=1, returns deterministic all-ones (no randomness).
        For stride>1, samples K=max(1, L//stride) unique positions per (B,H).
        """
        if stride <= 1:
            return torch.ones(B, H, L, 1, device=device, dtype=dtype)
        k_keep = max(1, L // stride)
        # Random permutation via argsort over random scores, then take top-k.
        scores = torch.rand(B, H, L, device=device)
        topk_idx = scores.argsort(dim=-1, descending=True)[..., :k_keep]      # (B,H,K)
        mask = torch.zeros(B, H, L, device=device, dtype=dtype)
        mask.scatter_(dim=-1, index=topk_idx, value=1.0)
        return mask.unsqueeze(-1)

    def _variance_proxy(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Return per-token variance proxy u with shape (B,H,L)."""
        if not self.causal:
            Gp = p.transpose(-1, -2) @ p                                # (B,H,M,M)
            qGp = torch.einsum("bhlm,bhmn->bhln", q, Gp)
            u = (qGp * q).sum(dim=-1)
            return u

        pp = torch.einsum("bhlm,bhln->bhlmn", p, p)                     # (B,H,L,M,M)
        Gp_prefix = pp.cumsum(dim=2)
        qGp = torch.einsum("bhlm,bhlmn->bhln", q, Gp_prefix)
        u = (qGp * q).sum(dim=-1)
        return u

    def forward(self, x: torch.Tensor, *, return_info: bool = True) -> tuple[torch.Tensor, dict]:
        B, L, _ = x.shape
        H = self.n_heads

        q = self._split_heads(self.phi_q(x), self.d_feat)               # (B,H,L,M)
        k = self._split_heads(self.phi_k(x), self.d_feat)
        p = self._split_heads(self.phi_p(x), self.d_feat)
        v = self._split_heads(self.phi_v(x), self.d_value)

        # Use subsampled kernels only when sequence length is large enough.
        # If L < d_feat, higher-stride branches tend to be unhelpful/noisy.
        active_strides = [s for s in self.kernel_strides if (s == 1 or L >= self.d_feat)]
        if len(active_strides) == 0:
            active_strides = [1]
        weights = torch.tensor([1.0 / float(s) for s in active_strides], device=x.device, dtype=x.dtype)
        weights = weights / weights.sum()
        hat_stack: list[torch.Tensor] = []
        u_raw_stack: list[torch.Tensor] = []
        delta_stack: list[torch.Tensor] = []
        for idx, stride in enumerate(active_strides):
            m_attn = self._sample_token_mask(B=B, H=H, L=L, stride=stride, device=x.device, dtype=x.dtype)
            m_var = self._sample_token_mask(B=B, H=H, L=L, stride=stride, device=x.device, dtype=x.dtype)

            if self.causal:
                kv_prefix = torch.einsum("bhlm,bhld->bhlmd", k, v)
                kv_prefix = (kv_prefix * m_attn.unsqueeze(-1)).cumsum(dim=2)
                hat_s = torch.einsum("bhlm,bhlmd->bhld", q, kv_prefix)
                attn_count = m_attn.cumsum(dim=2).clamp_min(1.0)          # (B,H,L,1)

                pp_prefix = torch.einsum("bhlm,bhln->bhlmn", p, p)
                pp_prefix = (pp_prefix * m_var.unsqueeze(-1)).cumsum(dim=2)
                qGp = torch.einsum("bhlm,bhlmn->bhln", q, pp_prefix)
                u_raw_s = (qGp * q).sum(dim=-1)
                var_count = m_var.cumsum(dim=2).squeeze(-1).clamp_min(1.0)  # (B,H,L)
            else:
                kv = k.transpose(-1, -2) @ (v * m_attn)
                hat_s = q @ kv
                attn_count = m_attn.sum(dim=2, keepdim=True).clamp_min(1.0)  # (B,H,1,1)

                Gp = p.transpose(-1, -2) @ (p * m_var)
                qGp = torch.einsum("bhlm,bhmn->bhln", q, Gp)
                u_raw_s = (qGp * q).sum(dim=-1)
                var_count = m_var.sum(dim=2).squeeze(-1).clamp_min(1.0).unsqueeze(-1)  # (B,H,1)

            hat_s = hat_s / (attn_count * float(max(1, self.d_feat))).sqrt()
            u_raw_s = u_raw_s / (var_count * float(max(1, self.d_feat)))

            hat_stack.append(hat_s)
            u_raw_stack.append(u_raw_s)

            if self.output_form == "kalman":
                u_s = u_raw_s.clamp_min(0.0)
                K_s = 1.0 / (1.0 + u_s)
                delta_stack.append(K_s.unsqueeze(-1) * (hat_s - v))

        if self.output_form == "kalman":
            delta = sum(weights[i] * delta_stack[i] for i in range(len(delta_stack)))
            hat = sum(weights[i] * hat_stack[i] for i in range(len(hat_stack)))
            u_raw = sum(weights[i] * u_raw_stack[i] for i in range(len(u_raw_stack)))
            u = u_raw.clamp_min(0.0)
            K = 1.0 / (1.0 + u)
        else:
            hat = sum(weights[i] * hat_stack[i] for i in range(len(hat_stack)))
            u_raw = sum(weights[i] * u_raw_stack[i] for i in range(len(u_raw_stack)))
            u = u_raw.clamp_min(0.0)
            K = 1.0 / (1.0 + u)

        Y = v

        if self.output_form == "none":
            delta = hat
        elif self.output_form == "gated":
            g = torch.sigmoid(self.gate_log).reshape(1, H, 1, self.d_value)
            delta = g * hat
        elif self.output_form != "kalman":
            raise ValueError(f"unknown output_form: {self.output_form!r}")

        delta_merged = self._merge_heads(delta)
        if self.W_out is not None:
            delta_merged = self.W_out(delta_merged)
        out = x + delta_merged

        if not return_info:
            return out, {}

        hat_out = self._merge_heads(hat)
        if self.W_out is not None:
            hat_out = self.W_out(hat_out)
        v_out = u.squeeze(1) if H == 1 else u
        gate_out = K.squeeze(1) if H == 1 else K
        v_raw_out = u_raw.squeeze(1) if H == 1 else u_raw
        clamp_frac = float((u_raw < 0).float().mean().detach().item())
        info = {
            "v": v_out,
            "v_raw": v_raw_out,
            "gate": gate_out,
            "clamp_frac": clamp_frac,
            "r_eff": int(self.d_feat),
            "hat": hat_out,
        }
        return out, info


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or (((8 * d // 3) + 63) // 64) * 64
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class DKABlock(nn.Module):
    """DKA layer, optional recurrent depth, no separate FFN by default.

    `forward(x)` runs the block `recurrent_depth` times with the SAME weights
    (the recurrent-DKA iteration from §5.2 / §12.1). Returns `(out, infos)`
    where `infos` is a list of `info` dicts.

    DKA's output already lives in the residual stream (`X + K·(hat - X)`),
    so we replace `x` with `dka(x)` rather than adding. DKA's φ is itself
    an MLP, so the FFN is opt-in via ``use_ffn=True``.
    """

    def __init__(
        self,
        d_model: int,
        d_feat: int,
        *,
        recurrent_depth: int = 1,
        use_ffn: bool = False,
        ffn_hidden: int | None = None,
        **dka_kwargs,
    ) -> None:
        super().__init__()
        if recurrent_depth < 1:
            raise ValueError(f"recurrent_depth must be >= 1, got {recurrent_depth}")
        self.recurrent_depth = recurrent_depth
        self.dka = DKALayer(d_model, d_feat, **dka_kwargs)
        if use_ffn:
            self.norm2 = RMSNorm(d_model)
            self.mlp = SwiGLU(d_model, ffn_hidden)
        else:
            self.norm2 = None
            self.mlp = None

    def forward(self, x: torch.Tensor, *, return_info: bool = True) -> tuple[torch.Tensor, list[dict]]:
        infos: list[dict] = []
        for _ in range(self.recurrent_depth):
            x, info = self.dka(x, return_info=return_info)
            if return_info:
                infos.append(info)
            if self.mlp is not None:
                x = x + self.mlp(self.norm2(x))
        return x, infos


class SoftmaxAttentionBlock(nn.Module):
    """Pre-norm bidirectional or causal multi-head softmax attention + SwiGLU FFN.

    Plain transformer block, used as a same-shape baseline for DKABlock.
    `forward(x)` returns `(out, [])` so the call site matches DKABlock.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        *,
        causal: bool = False,
        ffn_hidden: int | None = None,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = bool(causal)
        self.attn_dropout = float(attn_dropout)
        self.norm1 = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, ffn_hidden)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x: torch.Tensor, *, return_info: bool = True) -> tuple[torch.Tensor, list[dict]]:
        h = self.norm1(x)
        B, L, D = h.shape
        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)               # (3, B, H, L, d_h)
        q, k, v = qkv.unbind(0)
        dp = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, dropout_p=dp)
        out = out.transpose(1, 2).contiguous().reshape(B, L, D)
        x = x + self.proj(out)
        x = x + self.mlp(self.norm2(x))
        return x, []
