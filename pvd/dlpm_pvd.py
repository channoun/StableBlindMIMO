"""
DLPM-PVD: Blind MIMO channel estimation with alpha-stable diffusion priors
and alpha-stable channel noise.

This replaces the Gaussian score networks in the original PVD (Algorithm 1,
arXiv:2510.27043) with DLPM (Denoising Lévy Probabilistic Models, ICLR 2025)
as priors for both the channel H and the source image D.

Channel noise model: N = A^{1/2} G, A ~ PositiveStable(α_noise/2), G ~ CN(0, σ_n²I).
Channel prior:       H_k ~ DLPM(α_H) on i.i.d. blocks.
Image prior:         D ~ DLPM(α_D).

Algorithm (DPS-style with DLPM prior):

    Pre-sample: A_H[1:J], A_D[1:J], Σ_H[1:J], Σ_D[1:J]
    Initialize: H_J^blocks ~ SαS(α_H),  D_J ~ SαS(α_D) at max scale

    For t = J-1, ..., 1:
        1. ε_H = ε_θ_H(H_t^blocks, t)           # channel denoiser
           ε_D = ε_θ_D(D_t, t)                  # image denoiser (UNet)

        2. DLPM Tweedie:
           Ĥ_0 = (H_t - σ̄_H[t] ε_H) / γ̄_H[t]  (assembled to block-diagonal)
           D̂_0 = (D_t - σ̄_D[t] ε_D) / γ̄_D[t]  (clamped to [-1,1])

        3. Likelihood gradient (alpha-stable noise):
           ∇_H, ∇_D = dlpm_likelihood_score(...)

        4. DLPM reverse step:
           H_t-1 = DLPM_reverse(H_t, ε_H, t) + (λ_H / ||∇_H||) · ∇_H + noise_H
           D_t-1 = DLPM_reverse(D_t, ε_D, t) + (λ_D / ||∇_D||) · ∇_D + noise_D

    Final Tweedie at t=1:
        H_hat = (H_1 - σ̄_H[1] ε_H) / γ̄_H[1]  (block-diagonal complex)
        D_hat = ((D_1 - σ̄_D[1] ε_D) / γ̄_D[1]).clamp(-1, 1)
"""
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple, Optional

from diffusion.levy_diffusion import DLPM, GenerativeLevyProcess
from noise.stable_noise import SubGaussianStableNoise
from channels.rayleigh import channel_blocks_to_blockdiag
from .likelihood import dlpm_likelihood_score


class DLPMPVDSolver:
    """
    DLPM-PVD blind receiver for joint channel and source recovery.

    Args:
        f_gamma:       DJSCC encoder f_γ (frozen).
        eps_theta_H:   Channel DLPM denoiser ε_θ_H (frozen).
        eps_theta_D:   Image DLPM denoiser ε_θ_D (frozen).
        glp_H:         GenerativeLevyProcess for channel.
        glp_D:         GenerativeLevyProcess for image.
        noise_model:   SubGaussianStableNoise (channel noise).
        Nr, Nt, K:     Channel dimensions.
        T:             Number of transmitted symbols.
        Nu:            Number of users (default 1).
        J:             Number of diffusion steps.
        lambda_H:      Likelihood step size for H (DPS-normalized).
        lambda_D:      Likelihood step size for D (DPS-normalized).
        L_A:           MC samples for A posterior.
        device:        Computation device.
        use_checkpoint: Gradient checkpointing on f_γ.
        use_analytical_channel_prior: Use Wiener-filter Tweedie for H instead of the
                       learned denoiser. Useful for debugging and for Rayleigh channels
                       where the analytical MMSE is exact (under Gaussian approx).
        img_channels:  Number of image channels (3 for RGB, 1 for grayscale MNIST).
        img_size:      Spatial size of the image (height = width, e.g. 256 or 28).
    """

    def __init__(
        self,
        f_gamma: nn.Module,
        eps_theta_H: nn.Module,
        eps_theta_D: nn.Module,
        glp_H: GenerativeLevyProcess,
        glp_D: GenerativeLevyProcess,
        noise_model: SubGaussianStableNoise,
        Nr: int,
        Nt: int,
        K: int,
        T: int,
        Nu: int = 1,
        J: int = 1000,
        lambda_H: float = 1.0,
        lambda_D: float = 1.0,
        L_A: int = 20,
        device: torch.device = torch.device("cpu"),
        use_checkpoint: bool = True,
        use_analytical_channel_prior: bool = False,
        img_channels: int = 3,
        img_size: int = 256,
    ):
        self.f_gamma = f_gamma.eval()
        self.eps_theta_H = eps_theta_H.eval()
        self.eps_theta_D = eps_theta_D.eval()
        self.glp_H = glp_H
        self.glp_D = glp_D
        self.noise_model = noise_model
        self.Nr, self.Nt, self.K = Nr, Nt, K
        self.T, self.Nu = T, Nu
        self.J = J
        self.lambda_H = lambda_H
        self.lambda_D = lambda_D
        self.L_A = L_A
        self.device = device
        self.use_checkpoint = use_checkpoint
        self.use_analytical_channel_prior = use_analytical_channel_prior
        self.img_channels = img_channels
        self.img_size = img_size

        # Freeze all networks
        for net in [self.f_gamma, self.eps_theta_H, self.eps_theta_D]:
            for p in net.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stat(name: str, x: torch.Tensor) -> str:
        """Return a compact statistics string for a tensor."""
        xf = x.float()
        return (f"{name}: mean={xf.mean().item():.3e}  "
                f"std={xf.std().item():.3e}  "
                f"max={xf.abs().max().item():.3e}")

    @staticmethod
    def _analytical_channel_tweedie(
        H_blocks: torch.Tensor,
        bargamma_t: float,
        barsigma_t: float,
    ) -> torch.Tensor:
        """
        Wiener-filter Tweedie for Rayleigh channel H_0 ~ CN(0, I).

        DLPM forward: H_t = bargamma_t * H_0 + barsigma_t * eps_alpha
        Under Gaussian approximation of the alpha-stable noise:
            E[H_0 | H_t] = bargamma_t / (bargamma_t^2 + barsigma_t^2) * H_t

        Args:
            H_blocks:   (B*K, 2, Nr, Nt) noisy channel blocks.
            bargamma_t: Cumulative mean coefficient at step t.
            barsigma_t: Cumulative std coefficient at step t.

        Returns:
            H_hat_blocks: (B*K, 2, Nr, Nt) Tweedie estimate.
        """
        scale = bargamma_t / (bargamma_t ** 2 + barsigma_t ** 2 + 1e-8)
        return scale * H_blocks

    def _init_latents(self, B: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize H_J and D_J from the DLPM prior at maximum noise.

        Returns:
            H_t: (B*K, 2, Nr, Nt) float32.
            D_t: (B, 3, 256, 256) float32.
        """
        shape_H = (B * self.K, 2, self.Nr, self.Nt)
        shape_D = (B, self.img_channels, self.img_size, self.img_size)

        # Initialize DLPM A/Sigma sequences
        self.glp_H.dlpm.sample_A(shape_H, self.J)
        self.glp_H.dlpm.compute_Sigmas()
        self.glp_D.dlpm.sample_A(shape_D, self.J)
        self.glp_D.dlpm.compute_Sigmas()

        # Initialize from the noise level at step J-1, not the end of the full
        # diffusion schedule.  The reverse loop runs t = J-1 → 1, so H_t / D_t
        # must start at the noise scale corresponding to barsigmas[J-1].
        H_t = (self.glp_H.dlpm.barsigmas[self.J - 1].item()
               * self.glp_H.dlpm.gen_eps.generate(size=shape_H))
        D_t = (self.glp_D.dlpm.barsigmas[self.J - 1].item()
               * self.glp_D.dlpm.gen_eps.generate(size=shape_D))

        return H_t.to(self.device), D_t.to(self.device)

    def _blocks_to_complex_blockdiag(self, H_blocks: torch.Tensor, B: int) -> torch.Tensor:
        """
        Convert (B*K, 2, Nr, Nt) real representation to (B, NrK, NtK) complex.
        """
        H_r = H_blocks[:, 0].reshape(B, self.K, self.Nr, self.Nt)
        H_i = H_blocks[:, 1].reshape(B, self.K, self.Nr, self.Nt)
        blocks_c = torch.complex(H_r, H_i)             # (B, K, Nr, Nt)
        return channel_blocks_to_blockdiag(blocks_c, self.Nr, self.Nt)  # (B, NrK, NtK)

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(
        self,
        Y: torch.Tensor,
        verbose: bool = True,
        debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run DLPM-PVD to jointly recover H0 and D0 from Y.

        Args:
            Y:       (B, NrK, T) complex received signal.
            verbose: Show tqdm progress bar.
            debug:   Print per-step diagnostics.

        Returns:
            H_hat: (B, NrK, NtK) complex channel estimate.
            D_hat: (B, 3, 256, 256) reconstructed image in [-1, 1].
        """
        B = Y.shape[0]
        H_t, D_t = self._init_latents(B)
        dlpm_H = self.glp_H.dlpm
        dlpm_D = self.glp_D.dlpm

        if debug:
            print(f"\n[DLPM-PVD] B={B}  Nr={self.Nr}  Nt={self.Nt}  K={self.K}  "
                  f"J={self.J}  analytical_prior={self.use_analytical_channel_prior}")
            print(self._stat("  H_init", H_t))
            print(self._stat("  D_init", D_t))

        indices = range(self.J - 1, 0, -1)
        if verbose:
            indices = tqdm(list(indices), desc="DLPM-PVD", unit="step")

        for t in indices:
            t_H = torch.full((B * self.K,), t, device=self.device, dtype=torch.long)
            t_D = torch.full((B,), t, device=self.device, dtype=torch.long)

            with torch.no_grad():
                eps_D = self.eps_theta_D(D_t, t_D)   # (B, 3, 256, 256)

                if self.use_analytical_channel_prior:
                    bargamma_t = dlpm_H.bargammas[t].item()
                    barsigma_t = dlpm_H.barsigmas[t].item()
                    eps_H = dlpm_H.predict_eps(
                        H_t, t,
                        self._analytical_channel_tweedie(H_t, bargamma_t, barsigma_t)
                    )
                else:
                    eps_H = self.eps_theta_H(H_t, t_H)   # (B*K, 2, Nr, Nt)

            # Likelihood gradient (through DLPM Tweedie)
            with torch.enable_grad():
                grad_H, grad_D = dlpm_likelihood_score(
                    H_t, D_t, Y,
                    self.f_gamma, self.eps_theta_H, self.eps_theta_D,
                    t, dlpm_H, dlpm_D, self.noise_model,
                    self.Nr, self.Nt, self.L_A, self.use_checkpoint,
                    use_analytical_channel_prior=self.use_analytical_channel_prior,
                )

            with torch.no_grad():
                # DLPM stochastic reverse step
                H_mean, H_var = dlpm_H.anterior_mean_variance_dlpm(H_t, t, eps_H)
                D_mean, D_var = dlpm_D.anterior_mean_variance_dlpm(D_t, t, eps_D)

                # DPS-normalized likelihood correction
                lik_norm_H = grad_H.norm() + 1e-8
                lik_norm_D = grad_D.norm() + 1e-8

                # Inject Gaussian noise
                noise_H = torch.randn_like(H_t) * torch.sqrt(H_var.clamp(min=0))
                noise_D = torch.randn_like(D_t) * torch.sqrt(D_var.clamp(min=0))

                # Update with DPS-normalized likelihood correction
                H_t = H_mean + (self.lambda_H / lik_norm_H) * grad_H + noise_H
                D_t = D_mean + (self.lambda_D / lik_norm_D) * grad_D + noise_D

            if debug and (t % max(1, self.J // 10) == 0 or t <= 5):
                bargamma_t = dlpm_H.bargammas[t].item()
                barsigma_t = dlpm_H.barsigmas[t].item()
                print(f"\n  t={t:4d}  bargamma={bargamma_t:.4f}  barsigma={barsigma_t:.4f}")
                print(self._stat("  H_t      ", H_t))
                print(self._stat("  D_t      ", D_t))
                print(self._stat("  eps_H    ", eps_H))
                print(self._stat("  eps_D    ", eps_D))
                print(self._stat("  grad_H   ", grad_H))
                print(self._stat("  grad_D   ", grad_D))
                print(f"  lik_norm_H={lik_norm_H.item():.3e}  "
                      f"lik_norm_D={lik_norm_D.item():.3e}")
                print(f"  lik_step_H={self.lambda_H / lik_norm_H.item():.3e}  "
                      f"lik_step_D={self.lambda_D / lik_norm_D.item():.3e}")

        # --- Final Tweedie at t=1 ---
        with torch.no_grad():
            t_H_final = torch.full((B * self.K,), 1, device=self.device, dtype=torch.long)
            t_D_final = torch.full((B,), 1, device=self.device, dtype=torch.long)

            if self.use_analytical_channel_prior:
                bargamma_1 = dlpm_H.bargammas[1].item()
                barsigma_1 = dlpm_H.barsigmas[1].item()
                H_hat_blocks = self._analytical_channel_tweedie(H_t, bargamma_1, barsigma_1)
            else:
                eps_H_final = self.eps_theta_H(H_t, t_H_final)
                H_hat_blocks = dlpm_H.predict_xstart(H_t, t_H_final, eps_H_final)

            H_hat = self._blocks_to_complex_blockdiag(H_hat_blocks, B)

            eps_D_final = self.eps_theta_D(D_t, t_D_final)
            D_hat = dlpm_D.predict_xstart(D_t, t_D_final, eps_D_final).clamp(-1.0, 1.0)

        if debug:
            print(f"\n[DLPM-PVD] Final estimates:")
            print(self._stat("  H_hat (abs)", H_hat.abs()))
            print(self._stat("  D_hat      ", D_hat))

        return H_hat, D_hat
