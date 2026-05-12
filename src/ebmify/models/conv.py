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
* Optional RFF feature lifts at three placements, parallel to FCNet:
  - ``mlp_*`` args propagate to the encoder/decoder MLP trunks
    (input lift, output lift, and ``block_type='rff'`` inside each
    residual block);
  - ``conv_block_type='rff'`` swaps the conv-block inner activation for
    a per-pixel :class:`SpatialRFFLayer`;
  - ``dec_pre_readout_rff`` concatenates a frozen spatial RFF to the
    decoder's final feature map before the 1x1 readout.
  All RFF layers with ``"median"`` bandwidth must be calibrated once via
  :meth:`ConvResVAE.init_rff_bandwidths` before the first forward pass.

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

from .fc import (
    RFFLayer,
    _ResidualMLP,
    _ResidualRFFBlock,
    _resolve_activation,
)


class SpatialRFFLayer(nn.Module):
    """Per-pixel :class:`RFFLayer` for spatial features ``(B, C, H, W)``.

    Reshapes ``(B, C, H, W) -> (B*H*W, C)``, applies a frozen RFFLayer,
    and reshapes back to ``(B, M, H, W)``. Equivalent to a 1x1 conv with
    frozen ``omega`` weights followed by ``cos`` and the ``sqrt(2/M)``
    normalization; using :class:`RFFLayer` directly keeps the same buffer
    layout (``omega``, ``phase``, ``length_scale``, ``feature_scale_idx``)
    so save/load round-trips through ``state_dict``.
    """

    def __init__(
        self,
        in_channels: int,
        n_features: int,
        length_scale: str | float | Sequence[float] = "median",
        rff_seed: int = 0,
        median_pairs: int = 1000,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.n_features = int(n_features)
        self.rff = RFFLayer(
            in_dim=self.in_channels,
            n_features=self.n_features,
            length_scale=length_scale,
            rff_seed=int(rff_seed),
            median_pairs=int(median_pairs),
        )

    def is_initialized(self) -> bool:
        return self.rff.is_initialized()

    @torch.no_grad()
    def init_bandwidth(self, x: torch.Tensor) -> None:
        """Calibrate the underlying RFF from a per-pixel sample of ``x``.

        Flattens ``(B, C, H, W) -> (B*H*W, C)``, subsamples to at most
        ``8 * median_pairs`` rows for memory, then delegates to
        :meth:`RFFLayer.init_bandwidth`.
        """
        if self.is_initialized():
            return
        flat = x.permute(0, 2, 3, 1).reshape(-1, self.in_channels)
        cap = max(8 * self.rff.median_pairs, 1024)
        if flat.shape[0] > cap:
            g = torch.Generator(device=flat.device)
            g.manual_seed(self.rff.rff_seed)
            idx = torch.randperm(flat.shape[0], generator=g, device=flat.device)[:cap]
            flat = flat[idx]
        self.rff.init_bandwidth(flat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        out = self.rff(flat)
        return out.reshape(B, H, W, self.n_features).permute(0, 3, 1, 2)


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    g = 1
    for cand in range(min(max_groups, channels), 0, -1):
        if channels % cand == 0:
            g = cand
            break
    return nn.GroupNorm(num_groups=g, num_channels=channels)


class ConvResBlock(nn.Module):
    """Pre-norm residual conv block (spatial analogue of ``_ResidualMLPBlock``).

    Linear variant (``block_type='linear'``, default)::

        f(x) = conv2( act( conv1( GN(x) ) ) )
        y    = skip(x) + f(x)

    where ``conv1`` is the shape-changing 3x3 conv (``stride=stride``) and
    ``conv2`` is a 3x3 same conv at ``out_channels``.

    RFF variant (``block_type='rff'``, spatial analogue of
    :class:`_ResidualRFFBlock`)::

        f(x) = conv2( rff( conv1( GN(x) ) ) )
        y    = skip(x) + f(x)

    where ``rff`` is a frozen :class:`SpatialRFFLayer` mapping each pixel's
    ``out_channels`` features to ``block_rff_features`` bounded RFF features,
    and ``conv2`` is a 1x1 readout (``block_rff_features -> out_channels``).
    The shape-changing 3x3 ``conv1`` is unchanged, so up/downsampling lives
    in the same place as the linear variant.

    The skip path is identity when both channel count and spatial size
    match, otherwise a 1x1 conv at ``stride=stride``. ``upsample=True``
    is interpreted as ``Upsample(2) + 3x3 conv``, the artifact-free
    alternative to ``ConvTranspose2d``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_factory: Callable[[], nn.Module],
        *,
        stride: int = 1,
        upsample: bool = False,
        block_type: str = "linear",
        rff_features: int | None = None,
        rff_length_scale: str | float | Sequence[float] = "median",
        rff_seed: int = 0,
    ) -> None:
        super().__init__()
        if block_type not in ("linear", "rff"):
            raise ValueError(f"block_type must be 'linear' or 'rff', got {block_type!r}")
        if block_type == "rff" and rff_features is None:
            raise ValueError("block_type='rff' requires rff_features.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.upsample = bool(upsample)
        self.block_type = block_type
        if upsample and stride != 1:
            raise ValueError("upsample=True requires stride=1; spatial up uses Upsample(2) before the conv.")

        self.gn = _group_norm(self.in_channels)
        if self.upsample:
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 3, stride=1, padding=1)
        else:
            self.up = None
            self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 3, stride=self.stride, padding=1)
        if block_type == "linear":
            self.act = act_factory()
            self.rff = None
            self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1)
        else:
            self.act = None
            self.rff = SpatialRFFLayer(
                in_channels=self.out_channels,
                n_features=int(rff_features),
                length_scale=rff_length_scale,
                rff_seed=int(rff_seed),
            )
            self.conv2 = nn.Conv2d(int(rff_features), self.out_channels, 1)

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
        if self.rff is not None:
            h = self.rff(h)
        else:
            h = self.act(h)
        h = self.conv2(h)
        return self.skip(x) + h


class ConvEncoder(nn.Module):
    """Stem conv + a stack of stride-2 :class:`ConvResBlock` downsamplers.

    Output: ``(B, channels[-1], H // 2**len(channels), W // 2**len(channels))``.

    Pass ``block_type='rff'`` (with ``block_rff_features``) to use
    :class:`SpatialRFFLayer` in place of the inner activation in every
    block; ``rff_seed_base + i`` is used for block ``i`` so the projections
    are independent.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        act_factory: Callable[[], nn.Module],
        *,
        block_type: str = "linear",
        block_rff_features: int | None = None,
        block_rff_length_scale: str | float | Sequence[float] = "median",
        rff_seed_base: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.channels = tuple(int(c) for c in channels)
        self.stem = nn.Conv2d(self.in_channels, self.channels[0], 3, stride=1, padding=1)
        blocks = []
        prev = self.channels[0]
        for i, c in enumerate(self.channels):
            blocks.append(
                ConvResBlock(
                    prev, c, act_factory, stride=2,
                    block_type=block_type,
                    rff_features=block_rff_features,
                    rff_length_scale=block_rff_length_scale,
                    rff_seed=int(rff_seed_base) + i,
                )
            )
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

    The optional ``pre_readout_rff`` argument inserts a frozen
    :class:`SpatialRFFLayer` between ``head_act`` and ``head``: the RFF
    features are concatenated channel-wise with the post-activation feature
    map (so the linear identity path is preserved alongside the bounded
    kernel features), and the head 1x1 conv's input width grows from
    ``channels[-1]`` to ``channels[-1] + pre_readout_rff``. Same rationale
    as :class:`_ResidualMLP`'s ``output_rff``: the bounded RFF half tames
    OOD behavior of the final readout, while the linear half stays
    available where it helps.
    """

    def __init__(
        self,
        channels: Sequence[int],
        out_channels: int,
        act_factory: Callable[[], nn.Module],
        *,
        block_type: str = "linear",
        block_rff_features: int | None = None,
        block_rff_length_scale: str | float | Sequence[float] = "median",
        pre_readout_rff: int | None = None,
        pre_readout_rff_length_scale: str | float | Sequence[float] = "median",
        rff_seed_base: int = 0,
    ) -> None:
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        self.out_channels = int(out_channels)
        blocks = []
        prev = self.channels[0]
        for i, c in enumerate(self.channels[1:]):
            blocks.append(
                ConvResBlock(
                    prev, c, act_factory, upsample=True,
                    block_type=block_type,
                    rff_features=block_rff_features,
                    rff_length_scale=block_rff_length_scale,
                    rff_seed=int(rff_seed_base) + i,
                )
            )
            prev = c
        # one more upsample at the same final channel width keeps the
        # spatial-resolution doublings symmetric with the encoder.
        blocks.append(
            ConvResBlock(
                prev, prev, act_factory, upsample=True,
                block_type=block_type,
                rff_features=block_rff_features,
                rff_length_scale=block_rff_length_scale,
                rff_seed=int(rff_seed_base) + len(self.channels) - 1,
            )
        )
        self.blocks = nn.Sequential(*blocks)
        self.head_gn = _group_norm(prev)
        self.head_act = act_factory()
        self.pre_readout_rff: SpatialRFFLayer | None
        if pre_readout_rff is not None:
            self.pre_readout_rff = SpatialRFFLayer(
                in_channels=prev,
                n_features=int(pre_readout_rff),
                length_scale=pre_readout_rff_length_scale,
                rff_seed=int(rff_seed_base) + len(self.channels) + 100,
            )
            head_in = prev + int(pre_readout_rff)
        else:
            self.pre_readout_rff = None
            head_in = prev
        self.head = nn.Conv2d(head_in, self.out_channels, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.blocks(h)
        h = self.head_act(self.head_gn(h))
        if self.pre_readout_rff is not None:
            h = torch.cat([h, self.pre_readout_rff(h)], dim=1)
        return self.head(h)


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

    RFF args (all default to off; parallel to :class:`FCNet`):

        mlp_input_rff, mlp_input_rff_length_scale:
            Lift fed into both ``enc_mlp`` (on the flattened conv features)
            and ``dec_mlp`` (on z). ``None`` disables.
        mlp_output_rff, mlp_output_rff_length_scale:
            Pre-readout lift inside both MLP trunks (post-block hidden
            features -> readout sees ``[h ; cos(Omega @ h + b)]``).
        mlp_block_type, mlp_block_rff_features, mlp_block_rff_length_scale:
            ``"rff"`` swaps the inner activation in every MLP residual
            block for a frozen :class:`RFFLayer` (see :class:`_ResidualRFFBlock`).
        conv_block_type, conv_block_rff_features, conv_block_rff_length_scale:
            ``"rff"`` swaps the inner activation in every :class:`ConvResBlock`
            for a frozen :class:`SpatialRFFLayer`.
        dec_pre_readout_rff, dec_pre_readout_rff_length_scale:
            Per-pixel RFF concatenated to the decoder's final feature map
            before the 1x1 readout. Bounded kernel features on the spatial
            "pixel-feature" space; the linear identity path is preserved.
        rff_seed:
            Base RNG seed; every RFF layer gets a distinct offset so the
            projections are independent.

    Any RFF layer constructed with ``length_scale="median"`` must be
    calibrated once via :meth:`init_rff_bandwidths` before the first
    forward pass. Numeric / multi-scale ``length_scale`` is resolved at
    construction and needs no calibration.
    """

    _RFF_SEED_OFFSETS = {
        "enc_conv": 100,
        "dec_conv": 200,
        "dec_pre_readout": 300,
        "enc_mlp_input": 400,
        "enc_mlp_output": 401,
        "enc_mlp_blocks": 410,
        "dec_mlp_input": 500,
        "dec_mlp_output": 501,
        "dec_mlp_blocks": 510,
    }

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (1, 28, 28),
        z_dim: int = 64,
        channels: Sequence[int] = (32, 64),
        fc_hidden: Sequence[int] = (256,),
        activation: str | Callable[[], nn.Module] = "silu",
        sigmoid_out: bool = True,
        *,
        mlp_input_rff: int | None = None,
        mlp_input_rff_length_scale: str | float | Sequence[float] = "median",
        mlp_output_rff: int | None = None,
        mlp_output_rff_length_scale: str | float | Sequence[float] = "median",
        mlp_block_type: str = "linear",
        mlp_block_rff_features: int | None = None,
        mlp_block_rff_length_scale: str | float | Sequence[float] = "median",
        conv_block_type: str = "linear",
        conv_block_rff_features: int | None = None,
        conv_block_rff_length_scale: str | float | Sequence[float] = "median",
        dec_pre_readout_rff: int | None = None,
        dec_pre_readout_rff_length_scale: str | float | Sequence[float] = "median",
        rff_seed: int = 0,
    ) -> None:
        super().__init__()
        C, H, W = input_shape
        self.input_shape = (int(C), int(H), int(W))
        self.z_dim = int(z_dim)
        self.channels = tuple(int(c) for c in channels)
        self.fc_hidden = tuple(int(h) for h in fc_hidden)
        self.sigmoid_out = bool(sigmoid_out)
        self.rff_seed = int(rff_seed)

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

        off = self._RFF_SEED_OFFSETS

        def _make_rff(in_dim: int | None, n_features: int | None,
                      length_scale, offset: int) -> RFFLayer | None:
            if n_features is None or in_dim is None:
                return None
            return RFFLayer(
                in_dim=int(in_dim), n_features=int(n_features),
                length_scale=length_scale, rff_seed=self.rff_seed + offset,
            )

        enc_in_rff = _make_rff(flat, mlp_input_rff,
                                mlp_input_rff_length_scale, off["enc_mlp_input"])
        enc_out_rff = _make_rff(self.fc_hidden[-1], mlp_output_rff,
                                 mlp_output_rff_length_scale, off["enc_mlp_output"])
        dec_in_rff = _make_rff(self.z_dim, mlp_input_rff,
                                mlp_input_rff_length_scale, off["dec_mlp_input"])
        # dec_mlp's hidden_dims are reversed(self.fc_hidden), so the last
        # hidden width fed to the readout is self.fc_hidden[0].
        dec_out_rff = _make_rff(self.fc_hidden[0], mlp_output_rff,
                                 mlp_output_rff_length_scale, off["dec_mlp_output"])

        # ---- Encoder: conv stack -> flatten -> residual MLP trunk -> mu/logvar
        self.enc_conv = ConvEncoder(
            C, self.channels, act_factory,
            block_type=conv_block_type,
            block_rff_features=conv_block_rff_features,
            block_rff_length_scale=conv_block_rff_length_scale,
            rff_seed_base=self.rff_seed + off["enc_conv"],
        )
        self.enc_mlp = _ResidualMLP(
            n_inputs=flat,
            n_outputs=2 * self.z_dim,
            hidden_dims=self.fc_hidden,
            act_factory=act_factory,
            input_rff=enc_in_rff,
            output_rff=enc_out_rff,
            block_type=mlp_block_type,
            block_rff_features=mlp_block_rff_features,
            block_rff_length_scale=mlp_block_rff_length_scale,
            block_rff_seed_base=self.rff_seed + off["enc_mlp_blocks"],
        )

        # ---- Decoder: residual MLP trunk on z -> unflatten -> conv up-stack
        self.dec_mlp = _ResidualMLP(
            n_inputs=self.z_dim,
            n_outputs=flat,
            hidden_dims=tuple(reversed(self.fc_hidden)),
            act_factory=act_factory,
            input_rff=dec_in_rff,
            output_rff=dec_out_rff,
            block_type=mlp_block_type,
            block_rff_features=mlp_block_rff_features,
            block_rff_length_scale=mlp_block_rff_length_scale,
            block_rff_seed_base=self.rff_seed + off["dec_mlp_blocks"],
        )
        self.dec_conv = ConvDecoder(
            channels=tuple(reversed(self.channels)),
            out_channels=C,
            act_factory=act_factory,
            block_type=conv_block_type,
            block_rff_features=conv_block_rff_features,
            block_rff_length_scale=conv_block_rff_length_scale,
            pre_readout_rff=dec_pre_readout_rff,
            pre_readout_rff_length_scale=dec_pre_readout_rff_length_scale,
            rff_seed_base=self.rff_seed + off["dec_conv"],
        )

    # ------------------------------------------------------------------
    # RFF bandwidth calibration
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_rff_bandwidths(self, x: torch.Tensor) -> None:
        """Calibrate every ``"median"`` RFF layer from a real input sample.

        Walks the full encode/decode path on ``x`` and calls
        :meth:`RFFLayer.init_bandwidth` / :meth:`SpatialRFFLayer.init_bandwidth`
        on each RFF in the order it would be invoked at forward time, so
        every layer's median is taken on the feature space it actually sees.
        Numeric / multi-scale ``length_scale`` layers are no-ops here
        (already resolved at construction).
        """
        was_training = self.training
        self.eval()
        try:
            # ---- Encoder conv stack
            h = self.enc_conv.stem(x)
            for blk in self.enc_conv.blocks:
                h = self._init_conv_block_rff_and_apply(blk, h)
            # ---- Encoder MLP trunk
            flat = h.flatten(1)
            mu_logvar = self._init_mlp_rff_and_apply(self.enc_mlp, flat)
            mu, _ = mu_logvar.chunk(2, dim=-1)
            # Use mu (no reparam noise) so the decoder calibration sees
            # the same statistics as predict() / decode() at eval time.
            z = mu
            # ---- Decoder MLP trunk
            h2 = self._init_mlp_rff_and_apply(self.dec_mlp, z)
            h2 = h2.view(-1, self.feat_c, self.feat_h, self.feat_w)
            # ---- Decoder conv stack
            for blk in self.dec_conv.blocks:
                h2 = self._init_conv_block_rff_and_apply(blk, h2)
            h2 = self.dec_conv.head_act(self.dec_conv.head_gn(h2))
            if self.dec_conv.pre_readout_rff is not None:
                self.dec_conv.pre_readout_rff.init_bandwidth(h2)
        finally:
            if was_training:
                self.train()

    @staticmethod
    def _init_conv_block_rff_and_apply(
        blk: ConvResBlock, x: torch.Tensor,
    ) -> torch.Tensor:
        """Run ``blk`` forward, calibrating its spatial RFF on its actual
        post-conv1 features before evaluating the rest of the block."""
        h = blk.gn(x)
        if blk.up is not None:
            h = blk.up(h)
        h = blk.conv1(h)
        if blk.rff is not None:
            blk.rff.init_bandwidth(h)
            h = blk.rff(h)
        else:
            h = blk.act(h)
        h = blk.conv2(h)
        return blk.skip(x) + h

    @staticmethod
    def _init_mlp_rff_and_apply(
        mlp: _ResidualMLP, x: torch.Tensor,
    ) -> torch.Tensor:
        """Run ``mlp`` forward, calibrating its input/output/block RFFs on
        their actual input features. Mirrors :class:`_ResidualMLP.forward`
        + :class:`_ResidualMLP.trunk`."""
        if mlp.input_rff is not None:
            mlp.input_rff.init_bandwidth(x)
            h = torch.cat([x, mlp.input_rff(x)], dim=-1)
        else:
            h = x
        h = mlp.in_proj(h)
        for b in mlp.blocks:
            if isinstance(b, _ResidualRFFBlock):
                b.rff.init_bandwidth(b.ln(h))
            h = b(h)
        if mlp.output_rff is not None:
            mlp.output_rff.init_bandwidth(h)
            h = torch.cat([h, mlp.output_rff(h)], dim=-1)
        return mlp.readout(h)

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


__all__ = [
    "ConvResBlock",
    "ConvEncoder",
    "ConvDecoder",
    "ConvResVAE",
    "SpatialRFFLayer",
]
