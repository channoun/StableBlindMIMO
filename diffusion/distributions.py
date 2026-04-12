"""
Lévy stable distribution utilities for DLPM.

Ported from DLPM (Denoising Lévy Probabilistic Models, ICLR 2025)
github.com/giulio98/DLPM — adapted to be fully standalone.

Key functions:
  gen_skewed_levy  — sample A ~ PositiveStable(α/2), the variance multiplier
  gen_sas          — sample eps ~ SαS(α) = sqrt(A) * G, symmetric alpha-stable noise
"""
import math
import numpy as np
import scipy.stats
import torch


def match_last_dims(data: torch.Tensor, size: tuple) -> torch.Tensor:
    """
    Broadcast a 1-D tensor to a target shape by repeating along trailing dims.

    Args:
        data: (N,) 1-D tensor.
        size: Target shape (N, d1, d2, ...). size[0] must equal data.shape[0].

    Returns:
        Tensor of shape `size`.
    """
    assert data.dim() == 1, f"data must be 1-D, got {data.shape}"
    for _ in range(len(size) - 1):
        data = data.unsqueeze(-1)
    return data.expand(*size).contiguous()


def gen_skewed_levy(
    alpha: float,
    size: tuple,
    device: torch.device = None,
    isotropic: bool = True,
    clamp_a: float = None,
) -> torch.Tensor:
    """
    Sample A ~ PositiveStable(α/2) (totally skewed, β=1).

    This is the variance multiplier in the sub-Gaussian representation:
        eps = sqrt(A) * G,   G ~ N(0, 1)

    Uses Nolan's parameterization with scale = 2·cos(π·α/4)^{2/α}.

    Args:
        alpha:     Stability index in (0, 2]. alpha=2 returns constant 2.
        size:      Output shape.
        device:    Output device.
        isotropic: If True, sample size[0] scalars and broadcast to `size`.
                   All elements within each batch share the same A.
        clamp_a:   If set, clamp A to [0, clamp_a].

    Returns:
        A: Tensor of shape `size`, dtype float32, all values > 0.
    """
    if alpha > 2.0 or alpha <= 0.0:
        raise ValueError(f"alpha must be in (0, 2], got {alpha}")
    if alpha == 2.0:
        ret = 2.0 * torch.ones(size, dtype=torch.float32)
        return ret if device is None else ret.to(device)

    scale = 2.0 * math.cos(math.pi * alpha / 4.0) ** (2.0 / alpha)

    if isotropic:
        n = size[0]
        samples = scipy.stats.levy_stable.rvs(
            alpha / 2.0, 1.0, loc=0.0, scale=scale, size=n
        )
        ret = torch.tensor(samples, dtype=torch.float32)
        ret = match_last_dims(ret, size)
    else:
        samples = scipy.stats.levy_stable.rvs(
            alpha / 2.0, 1.0, loc=0.0, scale=scale, size=size
        )
        ret = torch.tensor(samples, dtype=torch.float32)

    if clamp_a is not None:
        ret = torch.clamp(ret, 0.0, clamp_a)

    return ret if device is None else ret.to(device)


def gen_sas(
    alpha: float,
    size: tuple,
    a: torch.Tensor = None,
    device: torch.device = None,
    isotropic: bool = True,
    clamp_eps: float = None,
) -> torch.Tensor:
    """
    Sample eps ~ SαS(α) — symmetric alpha-stable noise at unit scale.

    Representation: eps = sqrt(A) * G,  A ~ PositiveStable(α/2),  G ~ N(0,1).

    Args:
        alpha:     Stability index in (0, 2].
        size:      Output shape.
        a:         Optional pre-sampled A values (same shape as `size`).
        device:    Output device.
        isotropic: Passed to gen_skewed_levy if a is None.
        clamp_eps: If set, clamp noise to [-clamp_eps, clamp_eps].

    Returns:
        eps: Tensor of shape `size`, dtype float32.
    """
    if a is None:
        a = gen_skewed_levy(alpha, size, device=device, isotropic=isotropic)

    g = torch.randn(size, device=device, dtype=torch.float32)
    ret = torch.sqrt(a) * g

    if clamp_eps is not None:
        ret = torch.clamp(ret, -clamp_eps, clamp_eps)

    return ret
