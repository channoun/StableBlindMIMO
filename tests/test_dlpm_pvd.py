"""Tests for pvd/likelihood.py and pvd/dlpm_pvd.py."""
import math
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.levy_diffusion import GenerativeLevyProcess
from models.channel_net import ChannelDenoiser
from models.unet import UNetModel
from noise.stable_noise import SubGaussianStableNoise
from channels.rayleigh import generate_rayleigh_channel, apply_channel_stable_noise, channel_blocks_to_blockdiag
from pvd.likelihood import dlpm_likelihood_score
from pvd.dlpm_pvd import DLPMPVDSolver
from encoder.swin_jscc import DJSCCEncoder


# ---------------------------------------------------------------------------
# Fixtures: small configs for fast testing
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_config():
    """Tiny system: Nr=2, Nt=1, K=4, T=2, steps=10."""
    return dict(
        Nr=2, Nt=1, K=4, T=2, Nu=1, B=1,
        J=10, alpha_H=1.8, alpha_D=1.8, alpha_noise=1.8,
    )


@pytest.fixture
def channel_net(tiny_config):
    return ChannelDenoiser(Nr=tiny_config["Nr"], Nt=tiny_config["Nt"],
                           hidden_dim=32, depth=2, time_embed_dim=32)


@pytest.fixture
def glp_H(tiny_config):
    return GenerativeLevyProcess(
        alpha=tiny_config["alpha_H"],
        device=torch.device("cpu"),
        steps=tiny_config["J"],
        clamp_a=20.0, clamp_eps=50.0,
    )


@pytest.fixture
def glp_D(tiny_config):
    return GenerativeLevyProcess(
        alpha=tiny_config["alpha_D"],
        device=torch.device("cpu"),
        steps=tiny_config["J"],
        clamp_a=10.0, clamp_eps=50.0,
    )


@pytest.fixture
def image_net():
    return UNetModel(
        in_channels=3, model_channels=16, out_channels=3,
        num_res_blocks=1, attention_resolutions=[], channel_mult=(1, 2),
    )


@pytest.fixture
def encoder(tiny_config):
    cfg = tiny_config
    return DJSCCEncoder(
        embed_dim=48, depths=[1, 1, 1, 1], num_heads=[1, 2, 4, 4],
        window_size=4, Nt=cfg["Nt"], K=cfg["K"], T=cfg["T"], Nu=cfg["Nu"],
    )


@pytest.fixture
def noise_model():
    return SubGaussianStableNoise(alpha=1.8, sigma_n=0.1)


# ---------------------------------------------------------------------------
# Likelihood score
# ---------------------------------------------------------------------------

def test_likelihood_score_shapes(tiny_config, channel_net, image_net, encoder, noise_model,
                                  glp_H, glp_D):
    cfg = tiny_config
    B, Nr, Nt, K, T = cfg["B"], cfg["Nr"], cfg["Nt"], cfg["K"], cfg["T"]
    BK = B * K

    H_t = torch.randn(BK, 2, Nr, Nt)
    D_t = torch.randn(B, 3, 256, 256)
    Y = torch.randn(B, Nr*K, T, dtype=torch.complex64)

    # Minimal encoder for 256x256 (window_size=8 is safe)
    enc = DJSCCEncoder(
        embed_dim=16, depths=[1, 1, 1, 1], num_heads=[1, 2, 4, 8],
        window_size=8, Nt=Nt, K=K, T=T, Nu=1,
    )
    img_net = UNetModel(
        in_channels=3, model_channels=16, out_channels=3,
        num_res_blocks=1, attention_resolutions=[], channel_mult=(1, 2),
    )

    grad_H, grad_D = dlpm_likelihood_score(
        H_t, D_t, Y,
        f_gamma=enc,
        eps_theta_H=channel_net,
        eps_theta_D=img_net,
        t=5,
        dlpm_H=glp_H.dlpm,
        dlpm_D=glp_D.dlpm,
        noise_model=noise_model,
        Nr=Nr, Nt=Nt, L_A=3,
        use_checkpoint=False,
    )
    assert grad_H.shape == H_t.shape
    assert grad_D.shape == D_t.shape
    assert not torch.isnan(grad_H).any()
    assert not torch.isnan(grad_D).any()


# ---------------------------------------------------------------------------
# DLPM-PVD solver
# ---------------------------------------------------------------------------

def test_dlpm_pvd_output_shapes(tiny_config, channel_net, image_net, encoder,
                                  noise_model, glp_H, glp_D):
    """Check that DLPMPVDSolver returns correct output shapes (short run)."""
    cfg = tiny_config
    B, Nr, Nt, K, T = cfg["B"], cfg["Nr"], cfg["Nt"], cfg["K"], cfg["T"]

    # Build small system
    enc = DJSCCEncoder(
        embed_dim=48, depths=[1, 1, 1, 1], num_heads=[1, 2, 4, 4],
        window_size=4, Nt=Nt, K=K, T=T, Nu=1,
    )
    img_net_256 = UNetModel(
        in_channels=3, model_channels=16, out_channels=3,
        num_res_blocks=1, attention_resolutions=[], channel_mult=(1, 2),
    )

    solver = DLPMPVDSolver(
        f_gamma=enc,
        eps_theta_H=channel_net,
        eps_theta_D=img_net_256,
        glp_H=glp_H,
        glp_D=glp_D,
        noise_model=noise_model,
        Nr=Nr, Nt=Nt, K=K, T=T, Nu=1,
        J=3,  # very short run for test
        lambda_H=0.1,
        lambda_D=0.1,
        L_A=2,
        device=torch.device("cpu"),
        use_checkpoint=False,
    )

    Y = torch.randn(B, Nr*K, T, dtype=torch.complex64)
    H_hat, D_hat = solver.solve(Y, verbose=False)

    assert H_hat.shape == (B, Nr*K, Nt*K)
    assert H_hat.is_complex()
    assert D_hat.shape == (B, 3, 256, 256)
    assert D_hat.min() >= -1.0 - 1e-5
    assert D_hat.max() <= 1.0 + 1e-5


def test_dlpm_pvd_deterministic():
    """Two runs with same init should differ (stochastic)."""
    pass  # Stochastic by design; just verify no crash above.
