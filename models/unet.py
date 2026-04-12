"""
UNet noise predictor for image DLPM.

Ported from dlpm/models/unet.py (DLPM project, ICLR 2025).
Fully standalone — depends only on models/nn.py.

Input/output:
    x  : (B, C, H, W) noisy image at step t.
    t  : (B,) integer timesteps.
    out: (B, C, H, W) predicted noise ε̂.
"""
from abc import abstractmethod
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import (
    SiLU, conv_nd, avg_pool_nd, zero_module, normalization,
    timestep_embedding, gradient_checkpoint,
)


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(dims, stride)

    def forward(self, x):
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None,
                 use_conv=False, dims=2, use_checkpoint=False):
        super().__init__()
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            nn.Linear(emb_channels, self.out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        return gradient_checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return gradient_checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x_flat = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x_flat))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1.0 / math.sqrt(math.sqrt(ch))
        w = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        w = torch.softmax(w.float(), dim=-1).type(w.dtype)
        h = torch.einsum("bts,bcs->bct", w, v)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x_flat + h).reshape(b, c, *spatial)


class UNetModel(nn.Module):
    """
    Full UNet with attention and sinusoidal timestep embeddings.

    Args:
        in_channels:          Number of input image channels.
        model_channels:       Base channel count.
        out_channels:         Number of output channels (same as in_channels).
        num_res_blocks:       ResBlocks per resolution level.
        attention_resolutions: Downsample rates at which attention is applied.
        dropout:              Dropout rate.
        channel_mult:         Channel multipliers per level.
        conv_resample:        Use learned up/downsampling.
        dims:                 1, 2, or 3 for conv dims.
        use_checkpoint:       Gradient checkpointing on ResBlocks.
        num_heads:            Attention heads.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions,
        dropout: float = 0.0,
        channel_mult=(1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        use_checkpoint: bool = False,
        num_heads: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = set(attention_resolutions)
        self.dropout = dropout
        self.channel_mult = channel_mult

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))
        ])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, dropout,
                                   out_channels=mult * model_channels,
                                   dims=dims, use_checkpoint=use_checkpoint)]
                ch = mult * model_channels
                if ds in self.attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads,
                                                 use_checkpoint=use_checkpoint))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint),
            AttentionBlock(ch, num_heads=num_heads, use_checkpoint=use_checkpoint),
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                skip_ch = input_block_chans.pop()
                layers = [ResBlock(ch + skip_ch, time_embed_dim, dropout,
                                   out_channels=model_channels * mult,
                                   dims=dims, use_checkpoint=use_checkpoint)]
                ch = model_channels * mult
                if ds in self.attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads,
                                                 use_checkpoint=use_checkpoint))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        hs = []
        h = x.float()
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = module(torch.cat([h, hs.pop()], dim=1), emb)
        return self.out(h).type(x.dtype)
