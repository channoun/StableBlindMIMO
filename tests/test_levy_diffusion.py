"""Tests for diffusion/levy_diffusion.py."""
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.levy_diffusion import DLPM, GenerativeLevyProcess
from models.channel_net import ChannelDenoiser


@pytest.fixture
def dlpm_small():
    return DLPM(alpha=1.8, device=torch.device("cpu"), diffusion_steps=50)


@pytest.fixture
def glp_small():
    return GenerativeLevyProcess(
        alpha=1.8, device=torch.device("cpu"), steps=50,
        clamp_a=20.0, clamp_eps=50.0,
    )


def test_noise_schedule_shapes(dlpm_small):
    assert dlpm_small.gammas.shape == (50,)
    assert dlpm_small.bargammas.shape == (50,)
    assert dlpm_small.barsigmas.shape == (50,)


def test_noise_schedule_monotone(dlpm_small):
    # bargammas should be decreasing (signal fades)
    assert (dlpm_small.bargammas[:-1] >= dlpm_small.bargammas[1:]).all()
    # barsigmas should be increasing (noise grows)
    assert (dlpm_small.barsigmas[:-1] <= dlpm_small.barsigmas[1:]).all()


def test_forward_diffusion_shape(dlpm_small):
    x0 = torch.randn(4, 2, 4, 1)
    t = torch.full((4,), 10, dtype=torch.long)
    x_t, eps = dlpm_small.sample_x_t_from_xstart(x0, t)
    assert x_t.shape == x0.shape
    assert eps.shape == x0.shape


def test_predict_xstart_roundtrip(dlpm_small):
    """predict_xstart(x_t, t, predict_eps(x_t, t, x_0)) ≈ x_0."""
    x0 = torch.randn(4, 2, 4, 1)
    t = torch.full((4,), 5, dtype=torch.long)
    x_t, eps = dlpm_small.sample_x_t_from_xstart(x0, t)
    eps_pred = dlpm_small.predict_eps(x_t, t, x0)
    x0_rec = dlpm_small.predict_xstart(x_t, t, eps_pred)
    assert torch.allclose(x0_rec, x0, atol=1e-4)


def test_sample_A_compute_Sigmas(dlpm_small):
    shape = (3, 2, 4, 1)
    dlpm_small.sample_A(shape, 50)
    assert dlpm_small.A.shape[0] == 50
    dlpm_small.compute_Sigmas()
    assert dlpm_small.Sigmas.shape[0] == 50
    assert (dlpm_small.Sigmas >= 0).all()


def test_anterior_mean_variance_shape(dlpm_small):
    shape = (3, 2, 4, 1)
    dlpm_small.sample_A(shape, 50)
    dlpm_small.compute_Sigmas()
    x_t = torch.randn(*shape)
    eps = torch.randn(*shape)
    mean, var = dlpm_small.anterior_mean_variance_dlpm(x_t, 10, eps)
    assert mean.shape == shape
    assert var.shape == shape
    assert (var >= 0).all()


def test_training_loss(glp_small):
    model = ChannelDenoiser(Nr=4, Nt=1, hidden_dim=64, depth=2)
    x_start = torch.randn(8, 2, 4, 1)
    loss = glp_small.training_loss(model, x_start)
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_glp_sample_shape(glp_small):
    model = ChannelDenoiser(Nr=4, Nt=1, hidden_dim=64, depth=2)
    shape = (2, 2, 4, 1)
    samples = glp_small.sample(model, shape, verbose=False)
    assert samples.shape == torch.Size(shape)


def test_tweedie_estimate_shape(glp_small):
    model = ChannelDenoiser(Nr=4, Nt=1, hidden_dim=64, depth=2)
    x_t = torch.randn(3, 2, 4, 1)
    x0_hat = glp_small.tweedie_estimate(x_t, t=10, model=model)
    assert x0_hat.shape == x_t.shape
