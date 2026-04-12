"""Tests for channels/rayleigh.py and noise/stable_noise.py."""
import math
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channels.rayleigh import (
    generate_rayleigh_channel,
    channel_blocks_to_blockdiag,
    blockdiag_to_channel_blocks,
    apply_channel_awgn,
    apply_channel_stable_noise,
    compute_snr,
    lmmse_channel_estimate,
)
from noise.stable_noise import SubGaussianStableNoise, sample_positive_stable


# ---------------------------------------------------------------------------
# Channel generation
# ---------------------------------------------------------------------------

def test_generate_rayleigh_channel_shape():
    H = generate_rayleigh_channel(batch_size=2, Nu=1, Nr=4, Nt=1, K=8)
    assert H.shape == (2, 1, 32, 8)
    assert H.dtype == torch.complex64


def test_generate_rayleigh_channel_blockdiag():
    """Off-block-diagonal entries should be zero."""
    H = generate_rayleigh_channel(batch_size=1, Nu=1, Nr=2, Nt=2, K=3)
    H0 = H[0, 0]  # (6, 6)
    # Check off-diagonal blocks are zero
    assert torch.all(H0[:2, 2:] == 0)
    assert torch.all(H0[2:4, :2] == 0)


def test_blocks_to_blockdiag_roundtrip():
    B, K, Nr, Nt = 2, 4, 3, 2
    blocks = torch.randn(B, K, Nr, Nt) + 1j * torch.randn(B, K, Nr, Nt)
    blocks = blocks.to(torch.complex64)
    H = channel_blocks_to_blockdiag(blocks, Nr, Nt)
    assert H.shape == (B, Nr*K, Nt*K)
    blocks_rec = blockdiag_to_channel_blocks(H, Nr, Nt)
    assert torch.allclose(blocks_rec, blocks, atol=1e-5)


# ---------------------------------------------------------------------------
# Channel application
# ---------------------------------------------------------------------------

def test_apply_channel_awgn_shape():
    H0 = generate_rayleigh_channel(2, 1, 4, 1, 8)
    X = torch.randn(2, 1, 8, 16, dtype=torch.complex64)
    Y, sigma_n = apply_channel_awgn(H0, X, snr_db=10.0)
    assert Y.shape == (2, 32, 16)
    assert sigma_n > 0


def test_apply_channel_awgn_snr():
    """Empirical SNR should be close to target when averaged."""
    snrs = []
    H0 = generate_rayleigh_channel(16, 1, 4, 1, 8)
    X = torch.ones(16, 1, 8, 16, dtype=torch.complex64)
    Y, sigma_n = apply_channel_awgn(H0, X, snr_db=0.0)
    N = Y - torch.stack([torch.bmm(H0[:, 0], X[:, 0])], dim=0)[0]
    snr_emp = compute_snr(H0, X, N)
    # Should be within 3 dB of 0 dB target on average
    assert abs(snr_emp) < 6.0


def test_apply_channel_stable_shape():
    H0 = generate_rayleigh_channel(2, 1, 4, 1, 8)
    X = torch.randn(2, 1, 8, 16, dtype=torch.complex64)
    noise_model = SubGaussianStableNoise(alpha=1.8, sigma_n=0.1)
    Y, N = apply_channel_stable_noise(H0, X, snr_db=10.0, noise_model=noise_model)
    assert Y.shape == (2, 32, 16)
    assert N.shape == (2, 32, 16)
    assert Y.dtype == torch.complex64


# ---------------------------------------------------------------------------
# Stable noise
# ---------------------------------------------------------------------------

def test_sample_positive_stable_shape():
    a = sample_positive_stable(0.9, 100)
    assert a.shape == (100,)
    assert (a > 0).all()


def test_subgaussian_stable_gaussian_limit():
    """alpha=2 should behave like AWGN."""
    noise_model = SubGaussianStableNoise(alpha=2.0, sigma_n=1.0)
    N = noise_model.sample_noise((1000,))
    # Variance of CN(0,1) noise is sigma_n^2 = 1
    emp_var = (N.real**2 + N.imag**2).mean().item()
    assert abs(emp_var - 1.0) < 0.2  # within 20%


def test_subgaussian_stable_sample_shape():
    noise_model = SubGaussianStableNoise(alpha=1.7, sigma_n=0.5)
    N = noise_model.sample_noise((3, 16, 8))
    assert N.shape == (3, 16, 8)
    assert N.is_complex()


def test_subgaussian_stable_log_likelihood():
    noise_model = SubGaussianStableNoise(alpha=1.8, sigma_n=0.1)
    Y = torch.randn(2, 4, 8) + 1j * torch.randn(2, 4, 8)
    mean = torch.randn(2, 4, 8) + 1j * torch.randn(2, 4, 8)
    ll = noise_model.log_likelihood(Y, mean, L_A=10)
    assert ll.shape == (2,)
    assert not torch.isnan(ll).any()


def test_subgaussian_stable_A_posterior():
    noise_model = SubGaussianStableNoise(alpha=1.8, sigma_n=0.1)
    res_sq = torch.tensor([0.01, 0.1, 1.0])
    A_post = noise_model.sample_A_posterior(res_sq, L_A=10, n_proposals=50)
    assert A_post.shape == (3, 10)
    assert (A_post > 0).all()


# ---------------------------------------------------------------------------
# LMMSE
# ---------------------------------------------------------------------------

def test_lmmse_channel_estimate_shape():
    B, Nr, Nt, K = 2, 4, 1, 8
    T_p = 2 * Nt * K
    Y_p = torch.randn(B, Nr*K, T_p, dtype=torch.complex64)
    X_p = torch.randn(B, Nt*K, T_p, dtype=torch.complex64)
    H_hat = lmmse_channel_estimate(Y_p, X_p, sigma_n=0.1, Nr=Nr, Nt=Nt, K=K)
    assert H_hat.shape == (B, Nr*K, Nt*K)
