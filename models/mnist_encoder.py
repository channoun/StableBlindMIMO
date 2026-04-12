"""
Lightweight CNN encoder/decoder for MNIST (28×28, 1-channel) images.

Replaces SwinJSCC for fast experiments. Encodes grayscale images to a
block of complex symbols for MIMO transmission.

f_gamma: (B, 1, 28, 28) → (B, Nu, NtK, T)  complex64
g_beta:  (B, NrK, T)    → (B, 1, 28, 28)   float32  (decoder, optional)
"""
import math
import torch
import torch.nn as nn


class MNISTEncoder(nn.Module):
    """
    Small CNN encoder: image → complex MIMO symbols.

    Architecture:
        Conv2d stack (28→14→7) → Flatten → Linear → split real/imag → normalize power

    Args:
        in_channels: Image channels (1 for MNIST grayscale).
        NtK:         Number of transmit antennas × sub-carriers.
        T:           Number of time slots (transmitted symbols per user).
        Nu:          Number of users (default 1).
        base_ch:     Base CNN channel count.
        power:       Average transmit power per symbol (default 1.0).
    """

    def __init__(
        self,
        in_channels: int = 1,
        NtK: int = 4,
        T: int = 16,
        Nu: int = 1,
        base_ch: int = 32,
        power: float = 1.0,
    ):
        super().__init__()
        self.NtK = NtK
        self.T = T
        self.Nu = Nu
        self.power = power

        # CNN backbone: 28×28 → 14×14 → 7×7
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, stride=2, padding=1),   # 14×14
            nn.GELU(),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),   # 7×7
            nn.GELU(),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=1, padding=1),
            nn.GELU(),
        )
        cnn_out_dim = base_ch * 2 * 7 * 7

        # Project to 2 * NtK * T * Nu (real + imag, per user)
        self.proj = nn.Linear(cnn_out_dim, 2 * NtK * T * Nu, bias=False)

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        """
        Args:
            D: (B, in_channels, 28, 28) image tensor in [-1, 1].

        Returns:
            X: (B, Nu, NtK, T) complex64, power-normalized.
        """
        B = D.shape[0]
        feat = self.encoder(D)                # (B, base_ch*2, 7, 7)
        feat = feat.reshape(B, -1)            # (B, cnn_out_dim)
        out = self.proj(feat)                 # (B, 2*NtK*T*Nu)

        # Split real / imag
        half = self.NtK * self.T * self.Nu
        re = out[:, :half].reshape(B, self.Nu, self.NtK, self.T)
        im = out[:, half:].reshape(B, self.Nu, self.NtK, self.T)
        X = torch.complex(re, im)             # (B, Nu, NtK, T)

        # Power normalize: E[|x|²] = power per symbol
        # Scale so ||X_u||²_F / (NtK * T) = power for each user
        pwr = X.abs().pow(2).mean(dim=(2, 3), keepdim=True).clamp(min=1e-8)
        X = X * math.sqrt(self.power) / pwr.sqrt()

        return X


class MNISTDecoder(nn.Module):
    """
    Small CNN decoder: received signal feature → reconstructed image.

    Takes a flattened real representation of the received signal and
    reconstructs the MNIST image.

    Args:
        NrK:         Number of receive antennas × sub-carriers.
        T:           Number of time slots.
        in_channels: Output image channels (1 for MNIST grayscale).
        base_ch:     Base CNN channel count.
    """

    def __init__(
        self,
        NrK: int = 4,
        T: int = 16,
        in_channels: int = 1,
        base_ch: int = 32,
    ):
        super().__init__()
        self.NrK = NrK
        self.T = T

        feat_dim = 2 * NrK * T   # real + imag flattened

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, base_ch * 2 * 7 * 7),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1),  # 14×14
            nn.GELU(),
            nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1),      # 28×28
            nn.GELU(),
            nn.Conv2d(base_ch, in_channels, 3, padding=1),
            nn.Tanh(),   # output in [-1, 1]
        )
        self._base_ch = base_ch

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Y: (B, NrK, T) complex64 received signal.

        Returns:
            D_hat: (B, in_channels, 28, 28) reconstructed image in [-1, 1].
        """
        B = Y.shape[0]
        feat = torch.cat([Y.real, Y.imag], dim=1)  # (B, 2*NrK, T)
        feat = feat.reshape(B, -1)                  # (B, 2*NrK*T)
        h = self.proj(feat).reshape(B, self._base_ch * 2, 7, 7)
        return self.decoder(h)
