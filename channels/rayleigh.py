"""
Rayleigh fading block-diagonal MIMO channel.

Generates i.i.d. complex Gaussian channel blocks and applies the channel:
    Y = sum_{i=1}^{Nu} H0_i @ X_i + N

where N is alpha-stable or Gaussian noise (controlled by noise_model).

Shape conventions:
    H0  : (batch, Nu, Nr*K, Nt*K)  complex64
    X   : (batch, Nu, Nt*K, T)     complex64
    Y   : (batch, Nr*K, T)         complex64
    N   : (batch, Nr*K, T)         complex64

Block structure:
    H0_i = blkdiag(H_1^i, ..., H_K^i),  H_k^i ~ CN(0, I_{Nr} ⊗ I_{Nt})
"""
import math
import torch
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Channel generation
# ---------------------------------------------------------------------------

def generate_rayleigh_channel(
    batch_size: int,
    Nu: int,
    Nr: int,
    Nt: int,
    K: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate block-diagonal Rayleigh fading channels.

    Returns:
        H0: (batch, Nu, Nr*K, Nt*K) complex64.
    """
    real = torch.randn(batch_size, Nu, K, Nr, Nt, device=device)
    imag = torch.randn(batch_size, Nu, K, Nr, Nt, device=device)
    H_blocks = torch.complex(real, imag) / math.sqrt(2)

    H0 = torch.zeros(batch_size, Nu, Nr * K, Nt * K, dtype=torch.complex64, device=device)
    for k in range(K):
        r0, r1 = k * Nr, (k + 1) * Nr
        c0, c1 = k * Nt, (k + 1) * Nt
        H0[:, :, r0:r1, c0:c1] = H_blocks[:, :, k]
    return H0


def channel_blocks_to_blockdiag(
    blocks: torch.Tensor,
    Nr: int,
    Nt: int,
) -> torch.Tensor:
    """
    Assemble K blocks of shape (B, K, Nr, Nt) into block-diagonal (B, Nr*K, Nt*K).

    Args:
        blocks: (B, K, Nr, Nt) complex.
        Nr, Nt: Block dimensions.

    Returns:
        H: (B, Nr*K, Nt*K) complex.
    """
    B, K, _, _ = blocks.shape
    NrK, NtK = Nr * K, Nt * K
    H = torch.zeros(B, NrK, NtK, dtype=blocks.dtype, device=blocks.device)
    for k in range(K):
        r0, r1 = k * Nr, (k + 1) * Nr
        c0, c1 = k * Nt, (k + 1) * Nt
        H[:, r0:r1, c0:c1] = blocks[:, k]
    return H


def blockdiag_to_channel_blocks(
    H: torch.Tensor,
    Nr: int,
    Nt: int,
) -> torch.Tensor:
    """
    Extract K diagonal blocks from block-diagonal H of shape (B, Nr*K, Nt*K).

    Returns:
        blocks: (B, K, Nr, Nt) complex.
    """
    B, NrK, NtK = H.shape
    K = NrK // Nr
    blocks = torch.stack([H[:, k*Nr:(k+1)*Nr, k*Nt:(k+1)*Nt] for k in range(K)], dim=1)
    return blocks


# ---------------------------------------------------------------------------
# Channel application (AWGN)
# ---------------------------------------------------------------------------

def apply_channel_awgn(
    H0: torch.Tensor,
    X: torch.Tensor,
    snr_db: float,
    power: float = 1.0,
) -> Tuple[torch.Tensor, float]:
    """
    Y = sum_i H0_i @ X_i + N,   N ~ CN(0, sigma_n^2 I).

    Args:
        H0:     (batch, Nu, Nr*K, Nt*K) complex channel.
        X:      (batch, Nu, Nt*K, T) complex transmitted signal.
        snr_db: Target SNR in dB.
        power:  Expected signal power per element.

    Returns:
        Y:       (batch, Nr*K, T) received signal.
        sigma_n: Noise standard deviation (float).
    """
    batch_size, Nu, NrK, NtK = H0.shape
    T = X.shape[-1]
    snr_linear = 10.0 ** (snr_db / 10.0)
    sigma_n = math.sqrt(power / snr_linear)

    HX = torch.zeros(batch_size, NrK, T, dtype=torch.complex64, device=H0.device)
    for i in range(Nu):
        HX = HX + torch.bmm(H0[:, i], X[:, i])

    scale = sigma_n / math.sqrt(2)
    noise = torch.complex(
        torch.randn_like(HX.real) * scale,
        torch.randn_like(HX.imag) * scale,
    )
    return HX + noise, sigma_n


def apply_channel_stable_noise(
    H0: torch.Tensor,
    X: torch.Tensor,
    snr_db: float,
    noise_model,
    power: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Y = sum_i H0_i @ X_i + N,   N = A^{1/2} G (sub-Gaussian alpha-stable).

    The noise scale sigma_n in noise_model is adjusted to hit snr_db on average.

    Args:
        H0:          (batch, Nu, Nr*K, Nt*K) complex.
        X:           (batch, Nu, Nt*K, T) complex.
        snr_db:      Target SNR in dB.
        noise_model: SubGaussianStableNoise instance. Its sigma_n is used.
        power:       Expected signal power per element.

    Returns:
        Y:   (batch, Nr*K, T) received signal.
        N:   (batch, Nr*K, T) noise realization.
    """
    batch_size, Nu, NrK, NtK = H0.shape
    T = X.shape[-1]

    HX = torch.zeros(batch_size, NrK, T, dtype=torch.complex64, device=H0.device)
    for i in range(Nu):
        HX = HX + torch.bmm(H0[:, i], X[:, i])

    N = noise_model.sample_noise((batch_size, NrK, T), device=H0.device)
    return HX + N, N


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def compute_snr(
    H0: torch.Tensor,
    X: torch.Tensor,
    N: torch.Tensor,
) -> float:
    """
    Empirical SNR in dB (Eq. 52 in paper):
        SNR = ||sum_i H0_i @ X_i||²_F / ||N||²_F
    """
    Nu = H0.shape[1]
    HX = sum(torch.bmm(H0[:, i], X[:, i]) for i in range(Nu))
    snr = (HX.abs() ** 2).sum() / ((N.abs() ** 2).sum() + 1e-12)
    return 10.0 * torch.log10(snr).item()


def set_sigma_n_for_snr(
    H0: torch.Tensor,
    X: torch.Tensor,
    snr_db: float,
) -> float:
    """
    Compute sigma_n so that the channel SNR equals snr_db.

    For alpha-stable noise, the "average" noise power is E[A] * sigma_n^2
    (E[A] may be infinite for alpha < 2). We use the Gaussian convention:
        sigma_n = sqrt( ||HX||²_F / (NrK * T * 10^{snr_db/10}) )

    This gives a consistent SNR definition across noise models.

    Returns:
        sigma_n: float.
    """
    Nu = H0.shape[1]
    NrK = H0.shape[2]
    T = X.shape[-1]
    HX = sum(torch.bmm(H0[:, i], X[:, i]) for i in range(Nu))
    signal_power = (HX.abs() ** 2).mean().item()
    snr_linear = 10.0 ** (snr_db / 10.0)
    sigma_n = math.sqrt(signal_power / snr_linear)
    return max(sigma_n, 1e-8)


def lmmse_channel_estimate(
    Y: torch.Tensor,
    X_pilot: torch.Tensor,
    sigma_n: float,
    Nr: int,
    Nt: int,
    K: int,
) -> torch.Tensor:
    """
    LMMSE channel estimator for pilot-based baselines.

    H_hat = Y_pilot @ X_pilot^H @ (X_pilot @ X_pilot^H + sigma_n^2 I)^{-1}

    Args:
        Y:       (batch, Nr*K, T_pilot) received pilot signal.
        X_pilot: (batch, Nt*K, T_pilot) known pilot matrix.
        sigma_n: Noise std.

    Returns:
        H_hat: (batch, Nr*K, Nt*K).
    """
    NtK = Nt * K
    G = torch.bmm(X_pilot, X_pilot.conj().transpose(-2, -1))
    reg = (sigma_n ** 2) * torch.eye(NtK, dtype=Y.dtype, device=Y.device).unsqueeze(0)
    C = torch.bmm(Y, X_pilot.conj().transpose(-2, -1))
    H_hat = torch.linalg.solve(
        (G + reg).transpose(-2, -1), C.transpose(-2, -1)
    ).transpose(-2, -1)
    return H_hat
