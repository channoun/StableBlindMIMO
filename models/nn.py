"""
Neural network utilities for the DLPM UNet.

Ported from dlpm/models/nn.py (DLPM project, ICLR 2025).
"""
import math
import torch
import torch.nn as nn


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"Unsupported dims: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"Unsupported dims: {dims}")


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels, num_groups=32):
    # Find largest divisor of channels that is <= num_groups
    g = min(num_groups, channels)
    while channels % g != 0:
        g -= 1
    return GroupNorm32(g, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Sinusoidal timestep embeddings.

    Args:
        timesteps: (N,) integer or float tensor.
        dim:       Embedding dimension.

    Returns:
        (N, dim) embedding tensor.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class _CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_fn, n_inputs, *args):
        ctx.run_fn = run_fn
        ctx.input_tensors = list(args[:n_inputs])
        ctx.input_params = list(args[n_inputs:])
        with torch.no_grad():
            out = ctx.run_fn(*ctx.input_tensors)
        return out

    @staticmethod
    def backward(ctx, *grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow = [x.view_as(x) for x in ctx.input_tensors]
            out = ctx.run_fn(*shallow)
        input_grads = torch.autograd.grad(
            out, ctx.input_tensors + ctx.input_params, grads, allow_unused=True
        )
        del ctx.input_tensors, ctx.input_params, out
        return (None, None) + input_grads


def gradient_checkpoint(func, inputs, params, flag: bool):
    if flag:
        return _CheckpointFunction.apply(func, len(inputs), *inputs, *params)
    return func(*inputs)
