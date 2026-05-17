"""Small GPT for OpenWebText with a Llama-style / GPT-2-style toggle.

A single ``GPT`` class is parameterised by ``arch in {"llama", "gpt2"}``:

                   llama                       gpt2
    norm           RMSNorm                     LayerNorm
    position       RoPE on q,k                 learned positional embed
    MLP            SwiGLU (hidden ~ 8/3 * d)   GeLU (hidden = 4 * d)
    Linear bias    off                         on
    tied embed     yes                         yes

Causal self-attention runs through ``F.scaled_dot_product_attention``
with ``is_causal=True`` so PyTorch 2.x auto-selects FlashAttention for
bf16 + contiguous q/k/v on Ampere+.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


PRESETS: dict[str, dict] = {
    "tiny":       dict(n_layer=6,  n_head=6,  d_model=384, seq_len=512),
    "small":      dict(n_layer=8,  n_head=8,  d_model=512, seq_len=1024),
    "gpt2-small": dict(n_layer=12, n_head=12, d_model=768, seq_len=1024),
}


@dataclass
class GPTConfig:
    n_layer: int
    n_head: int
    d_model: int
    seq_len: int
    vocab_size: int = 50257
    arch: Literal["llama", "gpt2"] = "llama"
    rope_base: float = 10000.0
    mlp_hidden: int | None = field(default=None)

    def __post_init__(self) -> None:
        if self.mlp_hidden is None:
            if self.arch == "llama":
                # SwiGLU: ~ (8/3) * d, rounded up to multiple of 64
                h = int(8 * self.d_model / 3)
                self.mlp_hidden = ((h + 63) // 64) * 64
            else:
                self.mlp_hidden = 4 * self.d_model
        assert self.d_model % self.n_head == 0


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).to(x.dtype) * self.weight


def _build_rope_cache(seq_len: int, head_dim: int, base: float, device, dtype):
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (T, half)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin  # (T, half) each


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D) where D is even. Split D into (D/2, D/2) halves.
    T = x.shape[-2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D/2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x.chunk(2, dim=-1)
    r1 = x1 * cos - x2 * sin
    r2 = x1 * sin + x2 * cos
    return torch.cat([r1, r2], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        use_bias = cfg.arch == "gpt2"
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=use_bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=use_bias)
        self.use_rope = cfg.arch == "llama"

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None,
        rope_sin: torch.Tensor | None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.use_rope:
            q = _apply_rope(q, rope_cos, rope_sin)
            k = _apply_rope(k, rope_cos, rope_sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class GeluMLP(nn.Module):
    def __init__(self, d: int, hidden: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d, hidden, bias=True)
        self.proj = nn.Linear(hidden, d, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


def _make_norm(d: int, arch: str) -> nn.Module:
    return RMSNorm(d) if arch == "llama" else nn.LayerNorm(d)


def _make_mlp(cfg: GPTConfig) -> nn.Module:
    if cfg.arch == "llama":
        return SwiGLU(cfg.d_model, cfg.mlp_hidden)
    return GeluMLP(cfg.d_model, cfg.mlp_hidden)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.norm1 = _make_norm(cfg.d_model, cfg.arch)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = _make_norm(cfg.d_model, cfg.arch)
        self.mlp = _make_mlp(cfg)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = (
            nn.Embedding(cfg.seq_len, cfg.d_model) if cfg.arch == "gpt2" else None
        )
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = _make_norm(cfg.d_model, cfg.arch)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # tied embeddings
        self.lm_head.weight = self.tok_emb.weight

        if cfg.arch == "llama":
            head_dim = cfg.d_model // cfg.n_head
            cos, sin = _build_rope_cache(
                cfg.seq_len, head_dim, cfg.rope_base, device="cpu", dtype=torch.float32
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.rope_cos = None
            self.rope_sin = None

        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 * n_layer) for stable depth init.
        scale = 1.0 / math.sqrt(2 * cfg.n_layer)
        for blk in self.blocks:
            with torch.no_grad():
                blk.attn.proj.weight.mul_(scale)
                proj = blk.mlp.w_down if cfg.arch == "llama" else blk.mlp.proj
                proj.weight.mul_(scale)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    @classmethod
    def from_preset(
        cls,
        name: str,
        arch: Literal["llama", "gpt2"] = "llama",
        vocab_size: int = 50257,
        seq_len: int | None = None,
    ) -> "GPT":
        if name not in PRESETS:
            raise KeyError(f"unknown preset {name!r}; choose from {list(PRESETS)}")
        kwargs = dict(PRESETS[name])
        if seq_len is not None:
            kwargs["seq_len"] = seq_len
        cfg = GPTConfig(arch=arch, vocab_size=vocab_size, **kwargs)
        return cls(cfg)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # tok_emb is tied with lm_head; pos_emb (gpt2) is also embedding.
            n -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                n -= self.pos_emb.weight.numel()
        return n

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.seq_len, f"sequence length {T} exceeds {self.cfg.seq_len}"
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None]
        rope_cos = self.rope_cos.to(x.device) if self.rope_cos is not None else None
        rope_sin = self.rope_sin.to(x.device) if self.rope_sin is not None else None
        for blk in self.blocks:
            x = blk(x, rope_cos, rope_sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.seq_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
