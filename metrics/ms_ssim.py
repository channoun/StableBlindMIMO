"""
Multi-Scale Structural Similarity Index (MS-SSIM).
Higher is better. Range: [0, 1].

Based on Wang et al., "Multiscale structural similarity for image quality assessment."
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _gaussian_kernel(kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Create a 1D Gaussian kernel."""
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _ssim_per_channel(
    img1: torch.Tensor,
    img2: torch.Tensor,
    kernel: torch.Tensor,
    data_range: float = 1.0,
    K1: float = 0.01,
    K2: float = 0.03,
) -> tuple:
    """Compute SSIM map per channel. Returns (ssim_map, cs_map)."""
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    B, C, H, W = img1.shape
    kernel_2d = kernel.view(1, 1, kernel.shape[0], 1) * kernel.view(1, 1, 1, kernel.shape[0])
    kernel_2d = kernel_2d.expand(C, 1, -1, -1).to(img1.device)
    pad = kernel.shape[0] // 2

    def filt(x):
        return F.conv2d(x, kernel_2d, padding=pad, groups=C)

    mu1 = filt(img1)
    mu2 = filt(img2)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = filt(img1 ** 2) - mu1_sq
    sigma2_sq = filt(img2 ** 2) - mu2_sq
    sigma12 = filt(img1 * img2) - mu12

    # Luminance and contrast-structure terms
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu12 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    return ssim_map.mean(dim=(2, 3)), cs_map.mean(dim=(2, 3))


def ms_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 1.0,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute MS-SSIM between two batches of images.

    Args:
        img1, img2: (B, C, H, W) tensors, values in [0, data_range].
        data_range: Dynamic range of pixel values.
        weights:    Per-scale weights (default: Wang et al. 2003 weights).

    Returns:
        ms_ssim_val: (B,) tensor of MS-SSIM values.
    """
    if weights is None:
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=img1.device)

    levels = weights.shape[0]
    kernel = _gaussian_kernel(11, 1.5).to(img1.device)

    mcs = []
    x1, x2 = img1.float(), img2.float()
    for i in range(levels):
        ssim_per_c, cs_per_c = _ssim_per_channel(x1, x2, kernel, data_range=data_range)
        if i < levels - 1:
            mcs.append(cs_per_c.mean(dim=1))  # average over channels
            # Downsample
            x1 = F.avg_pool2d(x1, kernel_size=2, stride=2, padding=0)
            x2 = F.avg_pool2d(x2, kernel_size=2, stride=2, padding=0)
        else:
            mcs.append(ssim_per_c.mean(dim=1))

    mcs = torch.stack(mcs, dim=1)  # (B, levels)
    # Clamp to avoid negative values in pow
    mcs = mcs.clamp(0)
    ms_ssim_val = (mcs ** weights.unsqueeze(0)).prod(dim=1)
    return ms_ssim_val


class MSSSIM(nn.Module):
    """MS-SSIM as a torch Module (for use in loss functions)."""

    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return ms_ssim(img1, img2, data_range=self.data_range).mean()
