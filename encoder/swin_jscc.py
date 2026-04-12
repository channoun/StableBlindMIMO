"""
SwinJSCC: Deep Joint Source-Channel Codec based on Swin Transformer.

Maps source image D0 ∈ R^{B×3×H×W} to complex channel symbols X ∈ C^{B×Nu×NtK×T}.
The decoder g_β reconstructs D̂0 from X (used only during training).

Reference: Yang et al., "SwinJSCC: Taming Swin Transformer for
Deep Joint Source-Channel Coding", IEEE TCCN 2024.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Window attention primitives
# ---------------------------------------------------------------------------

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = coords.flatten(1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", rel.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(self.window_size**2, self.window_size**2, -1).permute(2, 0, 1).unsqueeze(0)
        attn = attn + bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 8,
                 shift_size: int = 0, mlp_ratio: float = 4.,
                 drop: float = 0., attn_drop: float = 0.):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        xw = window_partition(x, self.window_size).view(-1, self.window_size**2, C)
        mask = self._make_mask(H, W, x.device) if self.shift_size > 0 else None
        xw = self.attn(xw, mask=mask)
        xw = xw.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(xw, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = shortcut + x.view(B, H * W, C)
        x = x + self.mlp(self.norm2(x))
        return x

    def _make_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        img = torch.zeros(1, H, W, 1, device=device)
        h_sl = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_sl = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_sl:
            for w in w_sl:
                img[:, h, w, :] = cnt
                cnt += 1
        mw = window_partition(img, self.window_size).view(-1, self.window_size**2)
        mask = mw.unsqueeze(1) - mw.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.).masked_fill(mask == 0, 0.)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 4, in_ch: int = 3, embed_dim: int = 96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W


class PatchMerging(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        B = x.shape[0]
        x = x.view(B, H, W, -1)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        x = self.reduction(self.norm(x.view(B, -1, x.shape[-1])))
        return x, H // 2, W // 2


class EncoderStage(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int = 8,
                 mlp_ratio: float = 4., drop: float = 0., downsample: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, num_heads, window_size,
                      shift_size=0 if i % 2 == 0 else window_size // 2,
                      mlp_ratio=mlp_ratio, drop=drop)
            for i in range(depth)
        ])
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        if self.downsample:
            x, H, W = self.downsample(x, H, W)
        return x, H, W


class PatchExpand(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = self.expand(x).view(B, H, W, 2, 2, C // 2).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = self.norm(x.view(B, 2*H*2*W, C // 2))
        return x, 2*H, 2*W


class FinalExpand(nn.Module):
    def __init__(self, dim: int, patch_size: int = 4, out_ch: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(dim, patch_size * patch_size * out_ch, bias=False)
        self.out_ch = out_ch

    def forward(self, x, H, W):
        B = x.shape[0]
        P = self.patch_size
        x = self.proj(x).view(B, H, W, P, P, self.out_ch)
        return x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, self.out_ch, H*P, W*P)


class DecoderStage(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int = 8,
                 mlp_ratio: float = 4., drop: float = 0., upsample: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, num_heads, window_size,
                      shift_size=0 if i % 2 == 0 else window_size // 2,
                      mlp_ratio=mlp_ratio, drop=drop)
            for i in range(depth)
        ])
        self.upsample = PatchExpand(dim) if upsample else None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        if self.upsample:
            x, H, W = self.upsample(x, H, W)
        return x, H, W


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------

class DJSCCEncoder(nn.Module):
    """
    DJSCC Encoder f_γ: image → complex channel symbols.

    D0 ∈ R^{B×3×256×256}  →  X ∈ C^{B×Nu×NtK×T}

    Power constraint: (1/NtKT) ||X||^2_F = power per user per sample.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 96,
        depths: List[int] = None,
        num_heads: List[int] = None,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        patch_size: int = 4,
        Nt: int = 1,
        K: int = 192,
        T: int = 24,
        Nu: int = 1,
        power: float = 1.0,
    ):
        super().__init__()
        depths = depths or [2, 2, 6, 2]
        num_heads = num_heads or [3, 6, 12, 24]

        self.Nt, self.K, self.T, self.Nu = Nt, K, T, Nu
        self.power = power
        self.num_symbols = Nt * K * T

        self.patch_embed = PatchEmbed(patch_size, in_channels, embed_dim)

        self.stages = nn.ModuleList()
        for i, (d, nh) in enumerate(zip(depths, num_heads)):
            self.stages.append(EncoderStage(
                dim=embed_dim * (2**i), depth=d, num_heads=nh,
                window_size=window_size, mlp_ratio=mlp_ratio, drop=drop,
                downsample=(i < len(depths) - 1),
            ))

        n_stages = len(depths)
        final_dim = embed_dim * (2 ** (n_stages - 1))
        final_sp = (256 // patch_size) // (2 ** (n_stages - 1))
        feat_size = final_dim * final_sp * final_sp

        self.head = nn.Sequential(
            nn.LayerNorm(feat_size),
            nn.Linear(feat_size, 2 * self.num_symbols),
        )

    def forward(self, D0: torch.Tensor) -> torch.Tensor:
        B = D0.shape[0]
        x, H, W = self.patch_embed(D0)
        for stage in self.stages:
            x, H, W = stage(x, H, W)
        x = x.view(B, -1)
        out = self.head(x)
        real, imag = out[:, :self.num_symbols], out[:, self.num_symbols:]
        X = torch.complex(real, imag).view(B, self.Nt * self.K, self.T)
        X = X.unsqueeze(1).expand(-1, self.Nu, -1, -1)
        # Power normalization
        pwr = (X.abs()**2).mean(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        return X * (self.power / pwr).sqrt()


class DJSCCDecoder(nn.Module):
    """
    DJSCC Decoder g_β: complex channel symbols → reconstructed image.

    X ∈ C^{B×Nu×NtK×T}  →  D̂0 ∈ R^{B×3×256×256}

    Used ONLY during encoder training (not at PVD inference time).
    """

    def __init__(
        self,
        out_channels: int = 3,
        embed_dim: int = 96,
        depths: List[int] = None,
        num_heads: List[int] = None,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        patch_size: int = 4,
        Nt: int = 1,
        K: int = 192,
        T: int = 24,
    ):
        super().__init__()
        depths = depths or [2, 6, 2, 2]
        num_heads = num_heads or [24, 12, 6, 3]

        n = len(depths)
        max_dim = embed_dim * (2 ** (n - 1))
        init_sp = (256 // patch_size) // (2 ** (n - 1))
        num_symbols = Nt * K * T

        self.init_dim = max_dim
        self.init_H = self.init_W = init_sp

        self.sym_proj = nn.Sequential(
            nn.Linear(2 * num_symbols, max_dim * init_sp * init_sp),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(max_dim)

        self.stages = nn.ModuleList()
        for i, (d, nh) in enumerate(zip(depths, num_heads)):
            dim_i = embed_dim * (2 ** (n - 1 - i))
            self.stages.append(DecoderStage(
                dim=dim_i, depth=d, num_heads=nh,
                window_size=window_size, mlp_ratio=mlp_ratio, drop=drop,
                upsample=(i < n - 1),
            ))

        self.final = FinalExpand(embed_dim, patch_size, out_channels)
        self.act = nn.Tanh()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() == 4:
            X = X[:, 0]  # take first user
        B = X.shape[0]
        xf = X.reshape(B, -1)
        x = torch.cat([xf.real, xf.imag], dim=-1)
        x = self.sym_proj(x).view(B, self.init_H * self.init_W, self.init_dim)
        x = self.norm(x)
        H, W = self.init_H, self.init_W
        for stage in self.stages:
            x, H, W = stage(x, H, W)
        return self.act(self.final(x, H, W))
