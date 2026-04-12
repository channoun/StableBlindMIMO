"""
Sub-Gaussian Alpha-Stable noise model for the MIMO channel.

Channel noise model:
    N = A^{1/2} G,   G ~ CN(0, sigma_n^2 I),   A ~ PositiveStable(alpha/2)

This is a Gaussian scale mixture. Setting alpha=2 recovers AWGN exactly.

Reference (Taqqu's definition):
  Samoradnitsky & Taqqu (1994). "Stable Non-Gaussian Random Processes."
"""
import math
import warnings
import numpy as np
import torch
from scipy.stats import levy_stable
from typing import Optional


def sample_positive_stable(
    alpha_half: float,
    n_samples: int,
    device: torch.device = torch.device("cpu"),
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample from PositiveStable(alpha/2), i.e. a one-sided stable distribution.

    Uses the Chambers-Mallows-Stuck method via scipy.stats.levy_stable with
    Nolan's parameterization S(alpha/2, beta=1, loc=0, scale).

    Args:
        alpha_half: Index alpha/2, must be in (0, 1).
        n_samples:  Number of samples.
        device:     Output device.
        seed:       Optional numpy RNG seed.

    Returns:
        (n_samples,) float32 tensor, all values > 0.
    """
    if not 0.0 < alpha_half <= 1.0:
        raise ValueError(f"alpha_half must be in (0, 1], got {alpha_half}")
    
    if alpha_half == 1.0:
        # When alpha=2, return Gaussian distribution with scale=1.0
        rng = np.random.default_rng(seed)
        samples = rng.normal(loc=0.0, scale=1.0, size=n_samples)
        return torch.from_numpy(samples.astype(np.float32)).to(device)
    
    rng = np.random.default_rng(seed)
    scale = math.cos(math.pi * alpha_half / 2.0) ** (2.0 / alpha_half)
    samples = levy_stable.rvs(
        alpha=alpha_half, beta=1.0, loc=0.0, scale=scale,
        size=n_samples, random_state=rng,
    )
    samples = np.clip(samples, 1e-8, None)
    return torch.from_numpy(samples.astype(np.float32)).to(device)


def stable_log_density(
    a: torch.Tensor,
    alpha_half: float,
) -> torch.Tensor:
    """
    Evaluate log p_{stable}(a; alpha/2) using scipy's logpdf.

    Args:
        a:          (N,) positive tensor.
        alpha_half: Stability index.

    Returns:
        (N,) log-density tensor.
    """
    if alpha_half == 1.0:
        # When alpha=2, return Gaussian log-density: log N(0, 1)
        return -0.5 * a ** 2 - 0.5 * np.log(2 * np.pi)
    
    a_np = a.cpu().numpy()
    scale = math.cos(math.pi * alpha_half / 2.0) ** (2.0 / alpha_half)
    log_p = levy_stable.logpdf(a_np, alpha=alpha_half, beta=1.0, loc=0.0, scale=scale)
    return torch.from_numpy(log_p.astype(np.float32)).to(a.device)


class SubGaussianStableNoise:
    """
    Sub-Gaussian Alpha-Stable noise N = A^{1/2} G for the MIMO channel.

    Provides:
      - Noise generation: sample_noise()
      - Log-likelihood: log_likelihood()
      - Posterior sampling over A: sample_A_posterior()

    Args:
        alpha:   Stability index in (1, 2]. alpha=2 is pure AWGN.
        sigma_n: Scale of the Gaussian component.
    """

    def __init__(self, alpha: float = 1.5, sigma_n: float = 1.0):
        if not 1.0 < alpha <= 2.0:
            raise ValueError(f"alpha must be in (1, 2], got {alpha}")
        self.alpha = alpha
        print(f"Initialized SubGaussianStableNoise with alpha={alpha}, sigma_n={sigma_n}")
        self.alpha_half = alpha / 2.0
        self.sigma_n = sigma_n
        self._is_gaussian = (alpha == 2.0)

    def sample_noise(
        self,
        shape: tuple,
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample N = A^{1/2} G of given shape (complex).

        Returns:
            N: complex64 tensor of shape `shape`.
        """
        if self._is_gaussian:
            scale = self.sigma_n / math.sqrt(2)
            return torch.complex(
                torch.randn(shape, device=device) * scale,
                torch.randn(shape, device=device) * scale,
            )
        total = int(np.prod(shape))
        A = sample_positive_stable(self.alpha_half, total, device=device, seed=seed)
        A = A.view(shape)
        g_scale = self.sigma_n / math.sqrt(2)
        G = torch.complex(
            torch.randn(shape, device=device) * g_scale,
            torch.randn(shape, device=device) * g_scale,
        )
        return A.sqrt() * G

    def log_likelihood(
        self,
        Y: torch.Tensor,
        mean: torch.Tensor,
        A: Optional[torch.Tensor] = None,
        L_A: int = 20,
    ) -> torch.Tensor:
        """
        Compute log p(Y | mean) = log E_A[ CN(Y; mean, A·sigma_n²·I) ]
        via Monte Carlo over A.

        Args:
            Y:    (B, NrK, T) complex received signal.
            mean: (B, NrK, T) complex signal mean.
            A:    (L_A,) pre-sampled A values (optional).
            L_A:  Number of MC samples.

        Returns:
            (B,) log-likelihood per batch element.
        """
        B, NrK, T = Y.shape
        n_dim = NrK * T
        residual = Y - mean
        res_sq = (residual.abs() ** 2).sum(dim=(1, 2))  # (B,)

        if self._is_gaussian:
            var = self.sigma_n ** 2
            return -res_sq / var - n_dim * math.log(math.pi * var)

        device = Y.device
        if A is None:
            A = sample_positive_stable(self.alpha_half, L_A, device=device)

        var_base = self.sigma_n ** 2
        log_probs = []
        for a in A:
            a_val = a.item()
            lp = -n_dim * math.log(math.pi * a_val * var_base) - res_sq / (a_val * var_base)
            log_probs.append(lp)

        log_probs_stack = torch.stack(log_probs, dim=1)  # (B, L_A)
        return torch.logsumexp(log_probs_stack, dim=1) - math.log(L_A)

    def sample_A_posterior(
        self,
        residual_sq: torch.Tensor,
        L_A: int = 20,
        n_proposals: int = 200,
    ) -> torch.Tensor:
        """
        Draw samples from p(A | Y, H0, D0) via importance sampling.

        p(A | residual) ∝ CN(residual; 0, A·sigma_n²·I) · p_stable(A)

        Args:
            residual_sq:  (B,) squared Frobenius norm ||Y - H@X||²_F.
            L_A:          Number of posterior samples to return.
            n_proposals:  Number of IS proposals (prior draws).

        Returns:
            (B, L_A) posterior samples.
        """
        B = residual_sq.shape[0]
        device = residual_sq.device

        # Proposal: prior p_stable(A)
        A_prop = sample_positive_stable(self.alpha_half, n_proposals, device=device)

        A_post = torch.zeros(B, L_A, device=device)
        var_base = self.sigma_n ** 2

        for b in range(B):
            rsq = residual_sq[b].item()
            log_w = -rsq / (A_prop * var_base + 1e-8)
            w = torch.softmax(log_w, dim=0)
            idx = torch.multinomial(w, L_A, replacement=True)
            A_post[b] = A_prop[idx]

        return A_post
