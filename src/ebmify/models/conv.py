"""Residual convolutional building blocks + a residual conv-VAE.

Mirrors the design conventions of :mod:`ebmify.models.fc`:

* Pre-norm residual blocks with a learnable skip projection when shapes
  change (channels or spatial size).
* Same activation pool as ``FCNet`` (resolved through ``_ACTIVATIONS``),
  defaulting to ``OddPiecewiseReLU`` to match the FCNet default.
* GroupNorm is used in spatial blocks (the spatial analogue of the
  RMSNorm used in :class:`_ResidualMLPBlock`).
* The FC trunk between the conv stack and the latent heads is the same
  :class:`_ResidualMLP` used by :class:`FCNet`, so MLP + conv layers
  share normalization, activation, and residual conventions.

These pieces compose into :class:`ConvResVAE` --- a beta-VAE whose
encoder and decoder are residual conv stacks bracketed by residual MLP
trunks. The interface mirrors the older plain-conv VAEs:
``encode(x) -> (mu, logvar)``, ``decode(z) -> x_hat``, ``forward(x) ->
(x_hat, mu, logvar)``.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn

from .fc import _ACTIVATIONS, _ResidualMLP, _resolve_activation


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    g = 1
    for cand in range(min(max_groups, channels), 0, -1):
        if channels % cand == 0:
            g = cand
            break
    return nn.GroupNorm(num_groups=g, num_channels=channels)


class ConvResBlock(nn.Module):
    """Pre-norm residual conv block (spatial analogue of ``_ResidualMLPBlock``).

    Forward::

        f(x) = conv2( act( conv1( GN(x) ) ) )
        y    = skip(x) + f(x)

    ``conv1`` is the shape-changing conv (3x3, ``stride=stride``); ``conv2``
    is a 3x3 same conv at ``out_channels``. The skip path is identity when
    both channel count and spatial size match, otherwise a 1x1 conv at
    ``stride=stride``. ``stride < 0`` is interpreted as ``Upsample(2) +
    3x3 conv``, the artifact-free alternative to ``ConvTranspose2d``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_factory: Callable[[], nn.Module],
        *,
        stride: int = 1,
        upsample: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.upsample = bool(upsample)
        if upsample and stride != 1:
            raise ValueError("upsample=True requires stride=1; spatial up uses Upsample(2) before the conv.")

        self.gn = _group_norm(self.in_channels)
        if self.upsample:
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 3, stride=1, padding=1)
        else:
            self.up = None
            self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 3, stride=self.stride, padding=1)
        self.act = act_factory()
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1)

        same_channels = self.in_channels == self.out_channels
        same_spatial = (not self.upsample) and self.stride == 1
        if same_channels and same_spatial:
            self.skip = nn.Identity()
        elif self.upsample:
            self.skip = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(self.in_channels, self.out_channels, 1),
            )
        else:
            self.skip = nn.Conv2d(self.in_channels, self.out_channels, 1, stride=self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.gn(x)
        if self.up is not None:
            h = self.up(h)
        h = self.conv1(h)
        h = self.act(h)
        h = self.conv2(h)
        return self.skip(x) + h


class ConvEncoder(nn.Module):
    """Stem conv + a stack of stride-2 :class:`ConvResBlock` downsamplers.

    Output: ``(B, channels[-1], H // 2**len(channels), W // 2**len(channels))``.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        act_factory: Callable[[], nn.Module],
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.channels = tuple(int(c) for c in channels)
        self.stem = nn.Conv2d(self.in_channels, self.channels[0], 3, stride=1, padding=1)
        blocks = []
        prev = self.channels[0]
        for c in self.channels:
            blocks.append(ConvResBlock(prev, c, act_factory, stride=2))
            prev = c
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))


class ConvDecoder(nn.Module):
    """Stack of upsample :class:`ConvResBlock`s + final 1x1 conv.

    Mirror of :class:`ConvEncoder`: ``channels`` is given top-down (deepest
    first), and a final 1x1 conv to ``out_channels`` produces the
    reconstruction logits (a sigmoid is applied by :class:`ConvResVAE`
    for image data).
    """

    def __init__(
        self,
        channels: Sequence[int],
        out_channels: int,
        act_factory: Callable[[], nn.Module],
    ) -> None:
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        self.out_channels = int(out_channels)
        blocks = []
        prev = self.channels[0]
        for c in self.channels[1:]:
            blocks.append(ConvResBlock(prev, c, act_factory, upsample=True))
            prev = c
        # one more upsample at the same final channel width keeps the
        # spatial-resolution doublings symmetric with the encoder.
        blocks.append(ConvResBlock(prev, prev, act_factory, upsample=True))
        self.blocks = nn.Sequential(*blocks)
        self.head_gn = _group_norm(prev)
        self.head_act = act_factory()
        self.head = nn.Conv2d(prev, self.out_channels, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.blocks(h)
        return self.head(self.head_act(self.head_gn(h)))


class ConvResVAE(nn.Module):
    """beta-VAE with residual conv encoder/decoder + residual MLP trunks.

    Shapes:
        x   : (B, C, H, W)  -- ``H`` and ``W`` should be divisible by
                              ``2**len(channels)``.
        z   : (B, z_dim)

    Args:
        input_shape:   (C, H, W).
        z_dim:         Latent dimension.
        channels:      Conv channel widths, low-to-high (encoder reads
                       left-to-right, decoder reads right-to-left).
        fc_hidden:     Residual-MLP trunk widths between the conv block
                       and the latent heads. The same widths are used
                       both pre-z (in the encoder) and post-z (in the
                       decoder); the latter is reversed.
        activation:    Name in :data:`ebmify.models.fc._ACTIVATIONS`
                       (default ``"silu"``; ``"odd_piecewise"`` matches
                       the FCNet default).
        sigmoid_out:   Apply ``sigmoid`` to decoder output (default
                       ``True``; turn off for Gaussian / MSE losses with
                       unbounded outputs).
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (1, 28, 28),
        z_dim: int = 64,
        channels: Sequence[int] = (32, 64),
        fc_hidden: Sequence[int] = (256,),
        activation: str | Callable[[], nn.Module] = "silu",
        sigmoid_out: bool = True,
    ) -> None:
        super().__init__()
        C, H, W = input_shape
        self.input_shape = (int(C), int(H), int(W))
        self.z_dim = int(z_dim)
        self.channels = tuple(int(c) for c in channels)
        self.fc_hidden = tuple(int(h) for h in fc_hidden)
        self.sigmoid_out = bool(sigmoid_out)

        scale = 2 ** len(self.channels)
        if H % scale != 0 or W % scale != 0:
            raise ValueError(
                f"input_shape spatial dims {(H, W)} must be divisible by "
                f"2**len(channels) = {scale}"
            )
        self.feat_h = H // scale
        self.feat_w = W // scale
        self.feat_c = self.channels[-1]
        flat = self.feat_c * self.feat_h * self.feat_w

        act_factory = _resolve_activation(activation)

        # ---- Encoder: conv stack -> flatten -> residual MLP trunk -> mu/logvar
        self.enc_conv = ConvEncoder(C, self.channels, act_factory)
        self.enc_mlp = _ResidualMLP(
            n_inputs=flat,
            n_outputs=2 * self.z_dim,
            hidden_dims=self.fc_hidden,
            act_factory=act_factory,
        )

        # ---- Decoder: residual MLP trunk on z -> unflatten -> conv up-stack
        self.dec_mlp = _ResidualMLP(
            n_inputs=self.z_dim,
            n_outputs=flat,
            hidden_dims=tuple(reversed(self.fc_hidden)),
            act_factory=act_factory,
        )
        self.dec_conv = ConvDecoder(
            channels=tuple(reversed(self.channels)),
            out_channels=C,
            act_factory=act_factory,
        )

    # ------------------------------------------------------------------
    # VAE API
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_conv(x)
        h = h.flatten(1)
        params = self.enc_mlp(h)
        mu, logvar = params.chunk(2, dim=-1)
        # Keep exp(logvar) finite so KL can't blow up to NaN.
        logvar = logvar.clamp(-10.0, 10.0)
        return mu, logvar

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_mlp(z)
        h = h.view(-1, self.feat_c, self.feat_h, self.feat_w)
        x_hat = self.dec_conv(h)
        return torch.sigmoid(x_hat) if self.sigmoid_out else x_hat

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar


__all__ = ["ConvResBlock", "ConvEncoder", "ConvDecoder", "ConvResVAE"]
