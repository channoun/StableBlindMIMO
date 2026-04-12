"""
Standalone DLPM (Denoising Lévy Probabilistic Models) implementation.

Ported from:
  Minshuo Chen et al., "Denoising Lévy Probabilistic Models", ICLR 2025.
  github.com/giulio98/DLPM

Provides:
  - DLPM: discrete-time heavy-tailed diffusion (forward / reverse / training).
  - GenerativeLevyProcess: unified training+sampling API for DLPM.
  - VPSDE: continuous-time VP-SDE with alpha-stable noise (LIM).
  - GenerativeLIMProcess: unified training+sampling API for LIM.

Changes from the original:
  - No dependency on the BEM framework or DLPM project imports.
  - Merged Generator class into simple direct function calls.
  - Works standalone with distributions.py in the same package.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .distributions import gen_skewed_levy, gen_sas, match_last_dims


# ---------------------------------------------------------------------------
# Simple generator wrapper (replaces BEM's Data.Generator)
# ---------------------------------------------------------------------------

class _LevyGenerator:
    """Thin wrapper around gen_skewed_levy and gen_sas."""

    def __init__(self, kind: str, alpha: float, device, isotropic: bool = True,
                 clamp_a: float = None, clamp_eps: float = None):
        assert kind in ("skewed_levy", "sas")
        self.kind = kind
        self.alpha = alpha
        self.device = device
        self.isotropic = isotropic
        self.clamp_a = clamp_a
        self.clamp_eps = clamp_eps

    def setParams(self, clamp_a=None, clamp_eps=None):
        if clamp_a is not None:
            self.clamp_a = clamp_a
        if clamp_eps is not None:
            self.clamp_eps = clamp_eps

    def generate(self, size) -> torch.Tensor:
        if isinstance(size, torch.Size):
            size = tuple(size)
        if self.kind == "skewed_levy":
            return gen_skewed_levy(self.alpha, size, device=self.device,
                                   isotropic=self.isotropic, clamp_a=self.clamp_a)
        else:
            return gen_sas(self.alpha, size, device=self.device,
                           isotropic=self.isotropic, clamp_eps=self.clamp_eps)


# ---------------------------------------------------------------------------
# DLPM core
# ---------------------------------------------------------------------------

class DLPM:
    """
    Discrete-time heavy-tailed diffusion model.

    Forward process:
        x_t = γ̄[t] · x_0 + σ̄[t] · ε,   ε ~ SαS(α)

    Where γ̄ (bargammas) and σ̄ (barsigmas) follow a cosine schedule.

    Adapted from dlpm/methods/dlpm.py (DLPM project).
    """

    def __init__(
        self,
        alpha: float,
        device: torch.device,
        diffusion_steps: int,
        isotropic: bool = True,
        clamp_a: float = None,
        clamp_eps: float = None,
        scale: str = "scale_preserving",
    ):
        self.alpha = alpha
        self.device = device
        self.isotropic = isotropic
        self.scale = scale

        # 1-D noise schedules, shape (diffusion_steps,)
        self.gammas, self.bargammas, self.sigmas, self.barsigmas = (
            x.to(self.device)
            for x in self._gen_noise_schedule(diffusion_steps, scale)
        )

        # Cached broadcast constants (invalidated when shape changes)
        self._constants_cache = None

        # Lévy generators
        self.gen_a = _LevyGenerator("skewed_levy", alpha, device, isotropic, clamp_a=clamp_a)
        self.gen_eps = _LevyGenerator("sas", alpha, device, isotropic, clamp_eps=clamp_eps)

        # Set during reverse sampling by sample_A / compute_Sigmas
        self.A: torch.Tensor = None
        self.Sigmas: torch.Tensor = None

    # ------------------------------------------------------------------
    # Noise schedule
    # ------------------------------------------------------------------

    def _gen_noise_schedule(self, diffusion_steps: int, scale: str):
        if scale == "scale_preserving":
            s = 0.008
            t = torch.arange(diffusion_steps, dtype=torch.float32)
            schedule = torch.cos((t / diffusion_steps + s) / (1 + s) * math.pi / 2) ** 2
            baralphas = schedule / schedule[0]
            betas = 1.0 - baralphas / torch.cat([baralphas[:1], baralphas[:-1]])
            alphas = 1.0 - betas
            gammas = alphas ** (1.0 / self.alpha)
            bargammas = torch.cumprod(gammas, dim=0)
            sigmas = (1.0 - gammas ** self.alpha) ** (1.0 / self.alpha)
            barsigmas = (1.0 - bargammas ** self.alpha) ** (1.0 / self.alpha)
        elif scale == "scale_exploding":
            t = torch.arange(diffusion_steps, dtype=torch.float32)
            sigma_min, sigma_max, rho = 0.002, 80.0, 7.0
            gammas = torch.ones_like(t)
            bargammas = torch.ones_like(t)
            barsigmas = (
                sigma_min ** (1.0 / rho)
                + (t / (diffusion_steps - 1)) * (sigma_max ** (1.0 / rho) - sigma_min ** (1.0 / rho))
            ) ** rho
            barsigmas_alpha = barsigmas ** self.alpha
            sigmas_alpha = torch.zeros_like(barsigmas)
            sigmas_alpha[0] = barsigmas_alpha[0]
            for i in range(1, diffusion_steps):
                sigmas_alpha[i] = barsigmas_alpha[i] - barsigmas_alpha[:i].sum()
            sigmas = sigmas_alpha ** (1.0 / self.alpha)
        else:
            raise ValueError(f"Unknown scale schedule: {scale}")
        return gammas, bargammas, sigmas, barsigmas

    def _get_schedule_broadcast(self, shape: tuple):
        """Return schedule tensors broadcast to `shape` (cached)."""
        if self._constants_cache is None or self._constants_cache[0].shape[1:] != shape[1:]:
            def _bc(v):
                # v shape (J,) → (J, 1, 1, ...) → (J, shape[1], ..., shape[-1])
                v2 = v
                for _ in range(len(shape) - 1):
                    v2 = v2.unsqueeze(-1)
                return v2.expand(-1, *shape[1:]).contiguous()
            self._constants_cache = (
                _bc(self.gammas), _bc(self.bargammas),
                _bc(self.sigmas), _bc(self.barsigmas),
            )
        return self._constants_cache

    def _t_vec(self, x_or_sigma: torch.Tensor, t) -> torch.Tensor:
        """Return t as a (B,) integer tensor on device."""
        B = x_or_sigma.shape[0]
        if isinstance(t, int):
            return torch.full((B,), t, dtype=torch.long, device=self.device)
        return t.long().to(self.device)

    # ------------------------------------------------------------------
    # Forward diffusion helpers
    # ------------------------------------------------------------------

    def predict_xstart(self, x_t: torch.Tensor, t, eps: torch.Tensor) -> torch.Tensor:
        """DLPM Tweedie: x̂_0 = (x_t - σ̄[t] · ε) / γ̄[t]."""
        g, bg, s, bs = self._get_schedule_broadcast(x_t.shape)
        t_vec = self._t_vec(x_t, t)
        return (x_t - eps * bs[t_vec]) / bg[t_vec]

    def predict_eps(self, x_t: torch.Tensor, t, xstart: torch.Tensor) -> torch.Tensor:
        """Invert Tweedie: ε = (x_t - γ̄[t] · x_0) / σ̄[t]."""
        g, bg, s, bs = self._get_schedule_broadcast(x_t.shape)
        t_vec = self._t_vec(x_t, t)
        return (x_t - xstart * bg[t_vec]) / bs[t_vec]

    def sample_x_t_from_xstart(
        self, xstart: torch.Tensor, t, eps: torch.Tensor = None
    ):
        """
        Forward diffuse x_0 to x_t:
            x_t = γ̄[t] · x_0 + σ̄[t] · ε

        Returns (x_t, eps).
        """
        g, bg, s, bs = self._get_schedule_broadcast(xstart.shape)
        t_vec = self._t_vec(xstart, t)
        if eps is None:
            eps = self.gen_eps.generate(size=tuple(xstart.shape))
        x_t = bg[t_vec] * xstart + bs[t_vec] * eps
        return x_t, eps

    # ------------------------------------------------------------------
    # Sampling: pre-compute A sequence and conditional Sigmas
    # ------------------------------------------------------------------

    def sample_A(self, shape: tuple, diffusion_steps: int):
        """Pre-sample A_{1:J} for a full reverse trajectory."""
        self.A = torch.stack([
            self.gen_a.generate(size=shape) for _ in range(diffusion_steps)
        ])  # shape: (J, *shape)

    def compute_Sigmas(self):
        """Compute conditional variances Σ_t from the sampled A sequence."""
        assert self.A is not None, "Call sample_A first."
        g, bg, s, bs = self._get_schedule_broadcast(tuple(self.A[0].shape))
        J = self.A.shape[0]
        sigmas_list = [s[0] ** 2 * self.A[0]]
        for t in range(1, J):
            sigmas_list.append(s[t] ** 2 * self.A[t] + g[t] ** 2 * sigmas_list[-1])
        self.Sigmas = torch.stack(sigmas_list)  # (J, *shape)

    # ------------------------------------------------------------------
    # Reverse step helpers
    # ------------------------------------------------------------------

    def _compute_Gamma_t(self, t: int, Sigma_t_1: torch.Tensor, Sigma_t: torch.Tensor):
        g, bg, s, bs = self._get_schedule_broadcast(Sigma_t_1.shape)
        t_vec = self._t_vec(Sigma_t_1, t)
        return 1.0 - (g[t_vec] ** 2 * Sigma_t_1) / Sigma_t

    def anterior_mean_variance_dlpm(
        self, x_t: torch.Tensor, t: int, eps: torch.Tensor
    ):
        """
        DLPM stochastic reverse step:
            x_{t-1} = (x_t - σ̄[t] · Γ_t · ε) / γ[t]
            Σ_{t-1} = Γ_t · Σ_{t-1}   (posterior variance)

        Returns (mean, variance). Inject noise: x_{t-1} = mean + sqrt(var) * G.
        """
        assert self.Sigmas is not None, "Call compute_Sigmas before reverse sampling."
        g, bg, s, bs = self._get_schedule_broadcast(x_t.shape)
        t_vec = self._t_vec(x_t, t)
        Gamma_t = self._compute_Gamma_t(t, self.Sigmas[t - 1], self.Sigmas[t])
        mean = (x_t - bs[t_vec] * Gamma_t * eps) / g[t_vec]
        var = Gamma_t * self.Sigmas[t - 1]
        return mean, var

    def anterior_mean_variance_dlim(
        self, x_t: torch.Tensor, t: int, eps: torch.Tensor, eta: float = 0.0
    ):
        """
        DLIM deterministic (eta=0) or stochastic (eta>0) reverse step.
        """
        g, bg, s, bs = self._get_schedule_broadcast(x_t.shape)
        t_vec = self._t_vec(x_t, t)
        nonzero = (t_vec != 1).float().view(-1, *([1] * (x_t.dim() - 1)))
        if eta == 0.0:
            mean = (x_t - bs[t_vec] * eps) / g[t_vec] + bs[t_vec - 1] * eps
            return mean, torch.zeros_like(x_t)
        sigma_t = eta * bs[t_vec - 1]
        mean = (x_t - bs[t_vec] * eps) / g[t_vec]
        mean = mean + (bs[t_vec - 1] ** self.alpha - sigma_t ** self.alpha) ** (1.0 / self.alpha) * eps
        var = nonzero * sigma_t ** 2 * self.A[t]
        return mean, var

    # ------------------------------------------------------------------
    # Training: one-RV loss (Proposition 9)
    # ------------------------------------------------------------------

    def get_one_rv_loss_elements(
        self,
        t,
        x_0: torch.Tensor,
        a_t: torch.Tensor = None,
        z_t: torch.Tensor = None,
    ):
        """
        Sample (x_t, ε_t) for training (Proposition 9 of the DLPM paper).

        x_t = γ̄[t] · x_0 + Σ_t^{1/2} · z_t
        Σ_t = a_t · σ̄[t]²

        Returns (x_t, eps_t) where eps_t = (x_t - γ̄[t]·x_0) / σ̄[t].
        """
        if a_t is None:
            a_t = self.gen_a.generate(size=tuple(x_0.shape))
        g, bg, s, bs = self._get_schedule_broadcast(x_0.shape)
        t_vec = self._t_vec(x_0, t)
        Sigma_t = a_t * bs[t_vec] ** 2
        if z_t is None:
            z_t = torch.randn_like(x_0)
        x_t = bg[t_vec] * x_0 + Sigma_t ** 0.5 * z_t
        eps_t = self.predict_eps(x_t, t, x_0)
        return x_t, eps_t


# ---------------------------------------------------------------------------
# High-level process: GenerativeLevyProcess
# ---------------------------------------------------------------------------

class GenerativeLevyProcess:
    """
    Unified training and sampling API for DLPM.

    Ported from dlpm/methods/GenerativeLevyProcess.py (DLPM project),
    simplified to EPSILON prediction + FIXED variance only.

    Usage (training):
        glp = GenerativeLevyProcess(alpha=1.8, device=device, steps=1000)
        loss = glp.training_loss(model, x_batch)

    Usage (sampling):
        samples = glp.sample(model, shape=(B, C, H, W))
    """

    def __init__(
        self,
        alpha: float,
        device: torch.device,
        steps: int,
        isotropic: bool = True,
        scale: str = "scale_preserving",
        clamp_a: float = None,
        clamp_eps: float = None,
    ):
        self.alpha = alpha
        self.device = device
        self.steps = steps

        self.dlpm = DLPM(
            alpha=alpha,
            device=device,
            diffusion_steps=steps,
            isotropic=isotropic,
            scale=scale,
            clamp_a=clamp_a,
            clamp_eps=clamp_eps,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_loss(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        clamp_a: float = None,
        clamp_eps: float = None,
        monte_carlo: int = 1,
        lploss: float = 2.0,
        use_smooth_l1: bool = False,
        smooth_l1_beta: float = 0.1,
    ) -> torch.Tensor:
        """
        DLPM training loss (Proposition 9).

        Args:
            model:           Noise predictor ε_θ(x_t, t) → ε̂.
            x_start:         (B, ...) clean data.
            monte_carlo:     Number of MC samples of a_t per training step.
            lploss:          Loss order (2.0 = MSE, ignored when use_smooth_l1=True).
            use_smooth_l1:   If True, use Huber (Smooth-L1) loss instead of MSE.
                             Recommended for alpha-stable diffusion (α < 2) because
                             the noise has heavier tails than Gaussian — Smooth-L1
                             is more robust to the resulting large-residual outliers.
            smooth_l1_beta:  Transition point between L1 and L2 regimes in Huber loss.
                             Smaller values (e.g. 0.1) make the loss more L1-like.

        Returns:
            scalar loss.
        """
        self.dlpm.gen_a.setParams(clamp_a=clamp_a)
        self.dlpm.gen_eps.setParams(clamp_eps=clamp_eps)

        B = x_start.shape[0]
        t = torch.randint(1, self.steps, size=(B,), device=self.device)

        # MC average over a_t
        outer_shape = list(x_start.shape)
        A = self.dlpm.gen_a.generate(size=tuple(outer_shape))
        A_ext = A.repeat(monte_carlo, *([1] * (A.dim() - 1)))
        x_ext = x_start.repeat(monte_carlo, *([1] * (x_start.dim() - 1)))
        t_ext = t.repeat(monte_carlo)
        z_ext = torch.randn_like(x_ext)

        x_t, eps_t = self.dlpm.get_one_rv_loss_elements(t_ext, x_ext, A_ext, z_ext)
        eps_pred = model(x_t, t_ext)

        if use_smooth_l1:
            losses = nn.functional.smooth_l1_loss(
                eps_pred, eps_t, beta=smooth_l1_beta, reduction="none"
            )
            losses = losses.view(B * monte_carlo, -1).mean(dim=1)
        elif lploss == 2.0:
            losses = nn.functional.mse_loss(eps_pred, eps_t, reduction="none")
            losses = losses.view(B * monte_carlo, -1).mean(dim=1)
        else:
            losses = (eps_pred - eps_t).abs().pow(lploss).view(B * monte_carlo, -1).mean(dim=1)

        return losses.mean()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        deterministic: bool = False,
        eta: float = 0.0,
        clamp_a: float = None,
        clamp_eps: float = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples from the model.

        Args:
            model:         Trained noise predictor.
            shape:         Output shape (B, ...).
            deterministic: Use DLIM (deterministic) reverse if True.
            eta:           Stochastic DLIM noise level (0 = deterministic).
            verbose:       Show tqdm progress bar.

        Returns:
            samples: Tensor of shape `shape`.
        """
        self.dlpm.gen_a.setParams(clamp_a=clamp_a)
        self.dlpm.gen_eps.setParams(clamp_eps=clamp_eps)

        self.dlpm.sample_A(shape, self.steps)
        self.dlpm.compute_Sigmas()

        model.eval()
        with torch.no_grad():
            img = self.dlpm.barsigmas[-1] * self.dlpm.gen_eps.generate(size=shape)

            indices = range(self.steps - 1, 0, -1)
            if verbose:
                indices = tqdm(list(indices), desc="DLPM sampling")

            for t in indices:
                t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
                eps = model(img, t_batch)

                if deterministic:
                    mean, var = self.dlpm.anterior_mean_variance_dlim(img, t, eps, eta=eta)
                else:
                    mean, var = self.dlpm.anterior_mean_variance_dlpm(img, t, eps)

                noise = torch.randn_like(img) if t > 1 else torch.zeros_like(img)
                img = mean + torch.sqrt(var.clamp(min=0)) * noise

        return img

    def tweedie_estimate(
        self,
        x_t: torch.Tensor,
        t: int,
        model: nn.Module,
    ) -> torch.Tensor:
        """
        Compute the DLPM Tweedie estimate x̂_0|t = (x_t - σ̄[t]·ε_θ) / γ̄[t].

        Args:
            x_t:   Noisy tensor at step t.
            t:     Diffusion step (integer).
            model: Noise predictor.

        Returns:
            xhat_0: Denoised estimate, same shape as x_t.
        """
        B = x_t.shape[0]
        t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
        with torch.no_grad():
            eps = model(x_t, t_batch)
        return self.dlpm.predict_xstart(x_t, t_batch, eps)


# ---------------------------------------------------------------------------
# LIM: Lévy Inference Method (continuous-time VP-SDE with alpha-stable noise)
# ---------------------------------------------------------------------------

class VPSDE:
    """
    Continuous-time VP-SDE for LIM.

    SDE: dx = -½ β(t) x dt + β(t)^{1/α} dL_α

    Marginal: x_t = a(t) x_0 + σ(t) ε,  ε ~ SαS(α)
        a(t) = exp(∫₀ᵗ -½ β(s) ds)
        σ(t) = (1 - a(t)^α)^{1/α}

    Supports cosine (default) and linear schedules.

    Ported from dlpm/methods/LIM/functions/sde.py (DLPM project).
    """

    def __init__(self, alpha: float, schedule: str = "cosine", T: float = 0.9946):
        self.alpha = alpha
        self.schedule = schedule
        self.beta_0 = 0.0
        self.beta_1 = 20.0

        self.cosine_s = 0.008
        self.cosine_beta_max = 0.999
        self.cosine_t_max = (
            math.atan(self.cosine_beta_max * (1.0 + self.cosine_s) / math.pi)
            * 2.0 * (1.0 + self.cosine_s) / math.pi
            - self.cosine_s
        )
        self.cosine_log_alpha_0 = math.log(
            math.cos(self.cosine_s / (1.0 + self.cosine_s) * math.pi / 2.0)
        )

        if schedule == "cosine":
            self.T = T
        else:
            self.T = 1.0

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Noise schedule β(t)."""
        if self.schedule == "linear":
            return (self.beta_1 - self.beta_0) * t + self.beta_0
        # cosine
        return (
            math.pi / 2.0 * self.alpha / (self.cosine_s + 1.0)
            * torch.tan((t + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2.0)
        )

    def marginal_log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """log a(t) = ∫₀ᵗ -½ β(s) ds."""
        if self.schedule == "linear":
            return (
                -1.0 / (2.0 * self.alpha) * (t ** 2) * (self.beta_1 - self.beta_0)
                - 1.0 / self.alpha * t * self.beta_0
            )
        log_alpha_fn = lambda s: torch.log(
            torch.cos((s + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2.0)
        )
        return log_alpha_fn(t) - self.cosine_log_alpha_0

    def diffusion_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """a(t) = exp(log_a(t))."""
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t: torch.Tensor) -> torch.Tensor:
        """σ(t) = (1 - a(t)^α)^{1/α}."""
        return torch.pow(
            1.0 - torch.exp(self.marginal_log_mean_coeff(t) * self.alpha),
            1.0 / self.alpha,
        )

    def predict_xstart(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        LIM Tweedie: x̂_0 = (x_t + σ(t)·α·model_output) / a(t)

        model_output is the score target: -ε/α (for α<2) or -ε (for α=2).
        So ε = -α·model_output (for α<2) or ε = -model_output (for α=2),
        and x̂_0 = (x_t - σ(t)·ε) / a(t).
        """
        a_t = match_last_dims(self.diffusion_coeff(t), x_t.shape)
        sigma_t = match_last_dims(self.marginal_std(t), x_t.shape)
        if self.alpha == 2.0:
            eps = -model_output
        else:
            eps = -self.alpha * model_output
        return (x_t - sigma_t * eps) / a_t


class GenerativeLIMProcess:
    """
    Unified training and sampling API for LIM (Lévy Inference Method).

    The model predicts the score target:
        target = -ε         (α = 2, Gaussian)
        target = -ε / α     (α < 2, Lévy)

    Training uses a Smooth-L1 loss on the score target.

    Usage (training):
        glim = GenerativeLIMProcess(alpha=1.8, device=device, steps=1000)
        loss = glim.training_loss(model, x_batch)

    Usage (sampling):
        samples = glim.sample(model, shape=(B, C, H, W))
    """

    def __init__(
        self,
        alpha: float,
        device: torch.device,
        steps: int,
        schedule: str = "cosine",
        isotropic: bool = True,
        clamp_eps: float = 50.0,
        sde_T: float = 0.9946,
    ):
        self.alpha = alpha
        self.device = device
        self.steps = steps
        self.isotropic = isotropic
        self.clamp_eps = clamp_eps

        self.sde = VPSDE(alpha=alpha, schedule=schedule, T=sde_T)

        # Generator for SαS noise
        self.gen_eps = _LevyGenerator(
            "sas", alpha, device, isotropic, clamp_eps=clamp_eps
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_loss(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
    ) -> torch.Tensor:
        """
        LIM training loss (Smooth-L1 on score target).

        Forward: x_t = a(t) x_0 + σ(t) ε,  ε ~ SαS(α)
        Target:  s = -ε   (α=2)  or  s = -ε/α  (α<2)

        Args:
            model:    Score predictor s_θ(x_t, t) → ŝ.
            x_start:  (B, ...) clean data.

        Returns:
            scalar loss.
        """
        B = x_start.shape[0]
        # Sample continuous times uniformly in (0, T]
        t = torch.rand(B, device=self.device) * self.sde.T
        t = t.clamp(min=1e-5)

        a_t = self.sde.diffusion_coeff(t)     # (B,)
        sigma_t = self.sde.marginal_std(t)    # (B,)

        eps = self.gen_eps.generate(size=tuple(x_start.shape))

        a_bc = match_last_dims(a_t, x_start.shape)
        sigma_bc = match_last_dims(sigma_t, x_start.shape)

        x_t = a_bc * x_start + sigma_bc * eps

        if self.alpha == 2.0:
            target = -eps
        else:
            target = -eps / self.alpha

        output = model(x_t, t)
        return F.smooth_l1_loss(output, target, beta=1.0, reduction="mean")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        ddim: bool = False,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples using LIM ODE (ddim=True) or SDE (ddim=False).

        Args:
            model:   Trained score predictor s_θ(x_t, t).
            shape:   Output shape (B, ...).
            ddim:    Use ODE (deterministic) sampler if True, SDE otherwise.
            verbose: Show tqdm progress bar.

        Returns:
            samples: Tensor of shape `shape`.
        """
        model.eval()
        eps_val = 1e-5
        timesteps = torch.linspace(
            self.sde.T, eps_val, self.steps + 1, device=self.device
        )

        # Initialize from max-noise prior
        sigma_max = self.sde.marginal_std(timesteps[:1])  # (1,)
        x = sigma_max.item() * self.gen_eps.generate(size=shape).to(self.device)

        indices = range(self.steps)
        if verbose:
            indices = tqdm(list(indices), desc="LIM sampling")

        with torch.no_grad():
            for i in indices:
                vec_s = torch.ones(shape[0], device=self.device) * timesteps[i]
                vec_t = torch.ones(shape[0], device=self.device) * timesteps[i + 1]

                if ddim:
                    x = self._ode_step(model, x, vec_s, vec_t)
                else:
                    x = self._sde_step(model, x, vec_s, vec_t)

        return x

    def _ode_step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic ODE step (probability flow)."""
        sigma_s = self.sde.marginal_std(s)  # (B,)
        score_s = model(x, s) * match_last_dims(
            torch.pow(sigma_s + 1e-8, -(self.alpha - 1)), x.shape
        )

        a_s = self.sde.diffusion_coeff(s)   # (B,)
        a_t = self.sde.diffusion_coeff(t)   # (B,)
        x_coeff = a_t / a_s                 # (B,)

        if self.alpha == 2.0:
            beta_step = self.sde.beta(s) * (s - t)
            x_coeff_bc = match_last_dims(1.0 + beta_step / self.alpha, x.shape)
            score_coeff_bc = match_last_dims(beta_step / 2.0, x.shape)
        else:
            score_coeff = -self.alpha * (1.0 - x_coeff)  # (B,)
            x_coeff_bc = match_last_dims(x_coeff, x.shape)
            score_coeff_bc = match_last_dims(score_coeff, x.shape)

        return x_coeff_bc * x + score_coeff_bc * score_s

    def _sde_step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Stochastic SDE step (reverse Langevin)."""
        sigma_s = self.sde.marginal_std(s)  # (B,)
        score_s = model(x, s) * match_last_dims(
            torch.pow(sigma_s + 1e-8, -(self.alpha - 1)), x.shape
        )

        a_s = self.sde.diffusion_coeff(s)   # (B,)
        a_t = self.sde.diffusion_coeff(t)   # (B,)
        ratio = a_t / a_s                   # (B,)

        noise_coeff = torch.pow((-1.0 + torch.pow(ratio, self.alpha)).clamp(min=0), 1.0 / self.alpha)

        if self.alpha == 2.0:
            beta_step = self.sde.beta(s) * (s - t)
            e_B = torch.randn_like(x)
            x_coeff_bc = match_last_dims(1.0 + beta_step / self.alpha, x.shape)
            score_coeff_bc = match_last_dims(beta_step, x.shape)
            noise_bc = match_last_dims(
                torch.pow(beta_step.clamp(min=0), 1.0 / self.alpha), x.shape
            )
            return x_coeff_bc * x + score_coeff_bc * score_s + noise_bc * e_B
        else:
            e_L = self.gen_eps.generate(size=x.shape).to(x.device)
            score_coeff = self.alpha ** 2 * (-1.0 + ratio)  # (B,)
            x_coeff_bc = match_last_dims(ratio, x.shape)
            score_coeff_bc = match_last_dims(score_coeff, x.shape)
            noise_bc = match_last_dims(noise_coeff, x.shape)
            return x_coeff_bc * x + score_coeff_bc * score_s + noise_bc * e_L

    def tweedie_estimate(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model: nn.Module,
    ) -> torch.Tensor:
        """
        LIM Tweedie estimate x̂_0|t.

        Args:
            x_t:   Noisy tensor at continuous time t.
            t:     (B,) continuous time values in (0, T].
            model: Score predictor.

        Returns:
            xhat_0: Denoised estimate, same shape as x_t.
        """
        with torch.no_grad():
            output = model(x_t, t)
        return self.sde.predict_xstart(x_t, t, output)
