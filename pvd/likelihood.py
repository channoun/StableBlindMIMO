"""
DLPM-PVD likelihood score.

Computes:
    ∇_{H_t} ln q(Y | Ψ_t) = E_{p(A | Y, Ψ̂_0)} [ ∇_{H_t} ln p(Y | Ψ̂_0, A) ]
    ∇_{D_t} ln q(Y | Ψ_t) = E_{p(A | Y, Ψ̂_0)} [ ∇_{D_t} ln p(Y | Ψ̂_0, A) ]

where:
  - Ψ̂_0 = (Ĥ_0, D̂_0) are DLPM Tweedie estimates (not Gaussian score estimates)
  - p(Y | Ψ̂_0, A) = CN(Y; Ĥ_0 @ f_γ(D̂_0), A·σ_n²·I)
  - A ~ PositiveStable(α/2) is the stable noise scale

The gradient flows through:
    H_t → Ĥ_0 = (H_t - σ̄_H[t] · ε_θ_H(H_t, t)) / γ̄_H[t]  → Y - Ĥ_0 @ X̂ → loss
    D_t → D̂_0 = (D_t - σ̄_D[t] · ε_θ_D(D_t, t)) / γ̄_D[t]  → X̂ = f_γ(D̂_0) → loss

The Jacobians ∂ε_θ_H/∂H_t and ∂ε_θ_D/∂D_t + ∂f_γ/∂D̂_0 are computed via autograd.
The f_γ Jacobian is NEVER materialized; it is implicitly computed via reverse-mode AD.
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Tuple

from noise.stable_noise import SubGaussianStableNoise
from diffusion.levy_diffusion import DLPM
from channels.rayleigh import channel_blocks_to_blockdiag


def dlpm_likelihood_score(
    H_t: torch.Tensor,
    D_t: torch.Tensor,
    Y: torch.Tensor,
    f_gamma: nn.Module,
    eps_theta_H: nn.Module,
    eps_theta_D: nn.Module,
    t: int,
    dlpm_H: DLPM,
    dlpm_D: DLPM,
    noise_model: SubGaussianStableNoise,
    Nr: int,
    Nt: int,
    L_A: int = 20,
    use_checkpoint: bool = True,
    use_analytical_channel_prior: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute likelihood gradients for the DLPM-PVD reverse step.

    Args:
        H_t:          (B*K, 2, Nr, Nt) noisy channel blocks (real representation).
        D_t:          (B, 3, H_px, W_px) noisy image.
        Y:            (B, NrK, T) complex received signal.
        f_gamma:      DJSCC encoder (frozen).
        eps_theta_H:  Channel DLPM denoiser (frozen).
        eps_theta_D:  Image DLPM denoiser (frozen).
        t:            Current diffusion step (integer).
        dlpm_H:       DLPM instance for channel.
        dlpm_D:       DLPM instance for image.
        noise_model:  SubGaussianStableNoise instance.
        Nr, Nt:       Block dimensions.
        L_A:          Number of Monte Carlo samples for A posterior.
        use_checkpoint: Use gradient checkpointing on f_gamma.
        use_analytical_channel_prior: If True, use Wiener-filter Tweedie for H
            instead of the learned denoiser. Exact MMSE for Rayleigh CN(0,I)
            under Gaussian approximation.

    Returns:
        grad_H: (B*K, 2, Nr, Nt) gradient w.r.t. H_t (ascent direction).
        grad_D: (B, 3, H_px, W_px) gradient w.r.t. D_t (ascent direction).
    """
    B = D_t.shape[0]
    BK = H_t.shape[0]
    K = BK // B
    device = D_t.device

    # --- Leaf tensors that require grad ---
    H_t_leaf = H_t.detach().requires_grad_(True)
    D_t_leaf = D_t.detach().requires_grad_(True)

    # --- DLPM Tweedie estimate for H ---
    t_H = torch.full((BK,), t, device=device, dtype=torch.long)

    if use_analytical_channel_prior:
        # Wiener-filter MMSE for Rayleigh CN(0,I) under Gaussian approx:
        #   H_hat_0 = bargamma_t / (bargamma_t^2 + barsigma_t^2) * H_t
        # Autograd flows through this division, so grad_H is still computed.
        bargamma_t = dlpm_H.bargammas[t].item()
        barsigma_t = dlpm_H.barsigmas[t].item()
        scale = bargamma_t / (bargamma_t ** 2 + barsigma_t ** 2 + 1e-8)
        H_hat_blocks = scale * H_t_leaf  # (B*K, 2, Nr, Nt)
    else:
        eps_H = eps_theta_H(H_t_leaf, t_H)              # (B*K, 2, Nr, Nt)
        H_hat_blocks = dlpm_H.predict_xstart(H_t_leaf, t_H, eps_H)  # (B*K, 2, Nr, Nt)

    # Convert blocks to complex block-diagonal: (B, NrK, NtK)
    H_hat_real = H_hat_blocks[:, 0].reshape(B, K, Nr, Nt)
    H_hat_imag = H_hat_blocks[:, 1].reshape(B, K, Nr, Nt)
    H_hat_c_blocks = torch.complex(H_hat_real, H_hat_imag)  # (B, K, Nr, Nt)
    H_hat_0 = channel_blocks_to_blockdiag(H_hat_c_blocks, Nr, Nt)  # (B, NrK, NtK)

    # --- DLPM Tweedie estimate for D ---
    t_D = torch.full((B,), t, device=device, dtype=torch.long)
    eps_D = eps_theta_D(D_t_leaf, t_D)              # (B, 3, H, W)
    D_hat_0 = dlpm_D.predict_xstart(D_t_leaf, t_D, eps_D)  # (B, 3, H, W)
    D_hat_0 = D_hat_0.clamp(-1.0, 1.0)

    # --- Encoder forward ---
    if use_checkpoint:
        X_hat = grad_checkpoint(f_gamma, D_hat_0, use_reentrant=False)  # (B, Nu, NtK, T)
    else:
        X_hat = f_gamma(D_hat_0)
    # Take first user
    X_hat_u = X_hat[:, 0] if X_hat.dim() == 4 else X_hat  # (B, NtK, T)

    # --- Residual ---
    HX = torch.bmm(H_hat_0, X_hat_u)                # (B, NrK, T)
    residual = Y - HX                               # (B, NrK, T)
    res_sq = (residual.real ** 2 + residual.imag ** 2).sum(dim=(1, 2))  # (B,)

    # --- A posterior ---
    with torch.no_grad():
        A_posterior = noise_model.sample_A_posterior(res_sq.detach(), L_A=L_A)  # (B, L_A)

    # --- Average Gaussian likelihood scores over A samples ---
    grad_H_accum = torch.zeros_like(H_t_leaf)
    grad_D_accum = torch.zeros_like(D_t_leaf)

    for l_idx in range(L_A):
        A_l = A_posterior[:, l_idx]   # (B,)
        eff_var = A_l * (noise_model.sigma_n ** 2)  # (B,)

        # Re-use the already-computed graph: re-forward for each A sample
        # (cheap: only the scalar weighting changes)
        loss_l = (res_sq / eff_var.clamp(min=1e-8)).sum()

        grads = torch.autograd.grad(
            loss_l, [H_t_leaf, D_t_leaf],
            retain_graph=(l_idx < L_A - 1),
            allow_unused=True,
        )
        gH = grads[0] if grads[0] is not None else torch.zeros_like(H_t_leaf)
        gD = grads[1] if grads[1] is not None else torch.zeros_like(D_t_leaf)

        grad_H_accum = grad_H_accum + (-gH) / L_A
        grad_D_accum = grad_D_accum + (-gD) / L_A

    return grad_H_accum.detach(), grad_D_accum.detach()
