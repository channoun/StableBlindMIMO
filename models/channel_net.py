"""
Channel noise predictor for DLPM prior over H.

The channel H0 has i.i.d. Rayleigh blocks:
    H_k ~ CN(0, I_{Nr} ⊗ I_{Nt}),  k = 1, ..., K

The DLPM prior operates on individual blocks represented as
2-channel real tensors (real and imaginary parts):
    h_k_real ∈ R^{2 × Nr × Nt}

For training: input batch of (B, 2, Nr, Nt).
For inference: operates on (B*K, 2, Nr, Nt) — K blocks per batch element.

Architecture: MLP with sinusoidal time embedding.
Small by design (Nr × Nt is typically 4–64 elements per block).
"""
import math
import torch
import torch.nn as nn

from .nn import timestep_embedding, SiLU


class ChannelDenoiser(nn.Module):
    """
    MLP noise predictor for one channel block h_k ∈ R^{2 × Nr × Nt}.

    forward(x, t) → ε̂ (predicted DLPM noise, same shape as x)

    Args:
        Nr:            Number of receive antennas.
        Nt:            Number of transmit antennas.
        hidden_dim:    MLP hidden width.
        depth:         Number of hidden layers.
        time_embed_dim: Sinusoidal embedding dimension.
    """

    def __init__(
        self,
        Nr: int,
        Nt: int,
        hidden_dim: int = 256,
        depth: int = 4,
        time_embed_dim: int = 128,
    ):
        super().__init__()
        self.Nr = Nr
        self.Nt = Nt
        in_dim = 2 * Nr * Nt
        self.in_dim = in_dim
        self.time_embed_dim = time_embed_dim

        # Sinusoidal time embedding → projection
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Hidden layers with time conditioning
        self.layers = nn.ModuleList()
        self.time_layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
            self.time_layers.append(nn.Linear(time_embed_dim, hidden_dim))

        # Output projection (initialized to zero for stable training)
        self.out_proj = nn.Linear(hidden_dim, in_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, Nr, Nt) noisy block.
            t: (B,) integer timesteps.

        Returns:
            eps_hat: (B, 2, Nr, Nt) predicted noise.
        """
        B = x.shape[0]
        x_flat = x.reshape(B, -1).float()  # (B, 2*Nr*Nt)

        # Time embedding
        t_emb = timestep_embedding(t, self.time_embed_dim)  # (B, time_embed_dim)
        t_emb = self.time_proj(t_emb)  # (B, time_embed_dim)

        h = self.input_proj(x_flat)  # (B, hidden_dim)
        for layer, tl in zip(self.layers, self.time_layers):
            h = h + tl(t_emb)
            h = h + layer(h)  # residual

        out = self.out_proj(h)  # (B, 2*Nr*Nt)
        return out.reshape(B, 2, self.Nr, self.Nt).type(x.dtype)
