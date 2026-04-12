"""
Normalized Mean Squared Error (NMSE) in dB.

NMSE_dB = 10 * log10( ||H_hat - H0||^2_F / ||H0||^2_F )

Lower is better.
"""
import torch


def nmse_db(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute NMSE in dB.

    Args:
        estimate: Estimated tensor (any shape).
        target:   Ground-truth tensor (same shape).

    Returns:
        nmse_val: Scalar NMSE in dB (averaged over batch if first dim is batch).
    """
    # Work on real representations
    if estimate.is_complex():
        estimate = torch.view_as_real(estimate)
        target = torch.view_as_real(target)

    num = (estimate - target).pow(2).sum(dim=list(range(1, estimate.dim())))
    den = target.pow(2).sum(dim=list(range(1, target.dim()))).clamp(min=1e-12)
    nmse_linear = num / den
    return 10.0 * torch.log10(nmse_linear.mean() + 1e-12)


def nmse_db_batch(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute per-sample NMSE in dB.

    Returns:
        nmse_vals: (B,) tensor of NMSE values in dB.
    """
    if estimate.is_complex():
        estimate = torch.view_as_real(estimate)
        target = torch.view_as_real(target)

    num = (estimate - target).pow(2).sum(dim=list(range(1, estimate.dim())))
    den = target.pow(2).sum(dim=list(range(1, target.dim()))).clamp(min=1e-12)
    return 10.0 * torch.log10((num / den).clamp(min=1e-12))
