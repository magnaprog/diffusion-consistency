"""Conditional 1D U-Net denoiser (plan Section 6). No PISCES autoencoder reuse."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from prigm.config import (
    HIDDEN_END,
    HIDDEN_START,
    KERNEL_SIZE,
    N_BINS,
    N_CHANNELS,
    N_GROUPS,
    N_RES_BLOCKS,
    UNET_WIDTHS,
)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat((torch.sin(args), torch.cos(args)), dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


def _groups(channels: int) -> int:
    g = min(N_GROUPS, channels)
    while channels % g != 0:
        g -= 1
    return max(g, 1)


class ResidualBlock(nn.Module):
    """Two kernel-3 convolutions, GroupNorm, SiLU, timestep add."""

    def __init__(self, channels: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        g = _groups(channels)
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv1 = nn.Conv1d(channels, channels, KERNEL_SIZE, padding=KERNEL_SIZE // 2)
        self.norm2 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv1d(channels, channels, KERNEL_SIZE, padding=KERNEL_SIZE // 2)
        self.time_proj = nn.Linear(time_dim, channels)
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(temb))[:, :, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return x + h


class ConditionalUNet(nn.Module):
    """Widths 64/128/128 over length 48 -> 24 -> 12.

    Inputs (B, C, 48): noisy generated state, observed values (hidden zeroed),
    observation mask (1, L), relative positions (1, L), timestep (B,).
    """

    def __init__(
        self,
        n_generated: int = N_CHANNELS,
        widths: tuple[int, ...] = UNET_WIDTHS,
        n_res: int = N_RES_BLOCKS,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(widths) != 3:
            raise ValueError("plan specifies three widths")
        self.n_generated = n_generated
        self.n_observed = N_CHANNELS
        # G5 still conditions on all six observed channels; only noisy state is 5-D.
        in_ch = n_generated + self.n_observed + 1 + 1
        w0, w1, w2 = widths
        time_dim = w0 * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(w0),
            nn.Linear(w0, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv1d(in_ch, w0, KERNEL_SIZE, padding=KERNEL_SIZE // 2)
        self.enc0 = nn.ModuleList([ResidualBlock(w0, time_dim, dropout) for _ in range(n_res)])
        self.down0 = nn.Conv1d(w0, w1, 4, stride=2, padding=1)
        self.enc1 = nn.ModuleList([ResidualBlock(w1, time_dim, dropout) for _ in range(n_res)])
        self.down1 = nn.Conv1d(w1, w2, 4, stride=2, padding=1)
        self.enc2 = nn.ModuleList([ResidualBlock(w2, time_dim, dropout) for _ in range(n_res)])
        self.mid = nn.ModuleList([ResidualBlock(w2, time_dim, dropout) for _ in range(n_res)])
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(w2, w1, KERNEL_SIZE, padding=KERNEL_SIZE // 2),
        )
        self.dec1_in = nn.Conv1d(w1 + w1, w1, KERNEL_SIZE, padding=KERNEL_SIZE // 2)
        self.dec1 = nn.ModuleList([ResidualBlock(w1, time_dim, dropout) for _ in range(n_res)])
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(w1, w0, KERNEL_SIZE, padding=KERNEL_SIZE // 2),
        )
        self.dec0_in = nn.Conv1d(w0 + w0, w0, KERNEL_SIZE, padding=KERNEL_SIZE // 2)
        self.dec0 = nn.ModuleList([ResidualBlock(w0, time_dim, dropout) for _ in range(n_res)])
        self.out_norm = nn.GroupNorm(_groups(w0), w0)
        self.out_conv = nn.Conv1d(w0, n_generated, KERNEL_SIZE, padding=KERNEL_SIZE // 2)

    def forward(
        self,
        noisy: torch.Tensor,
        observed: torch.Tensor,
        mask: torch.Tensor,
        relpos: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if noisy.shape[1] != self.n_generated:
            raise ValueError("noisy generated-channel mismatch")
        if observed.shape[1] != self.n_observed:
            raise ValueError("observed must keep six channels, including G5")
        x = torch.cat((noisy, observed, mask, relpos), dim=1)
        temb = self.time_mlp(t)
        h = self.in_conv(x)
        for blk in self.enc0:
            h = blk(h, temb)
        skip0 = h
        h = self.down0(h)
        for blk in self.enc1:
            h = blk(h, temb)
        skip1 = h
        h = self.down1(h)
        for blk in self.enc2:
            h = blk(h, temb)
        for blk in self.mid:
            h = blk(h, temb)
        h = self.up1(h)
        if h.shape[-1] != skip1.shape[-1]:
            h = F.interpolate(h, size=skip1.shape[-1], mode="nearest")
        h = self.dec1_in(torch.cat((h, skip1), dim=1))
        for blk in self.dec1:
            h = blk(h, temb)
        h = self.up0(h)
        if h.shape[-1] != skip0.shape[-1]:
            h = F.interpolate(h, size=skip0.shape[-1], mode="nearest")
        h = self.dec0_in(torch.cat((h, skip0), dim=1))
        for blk in self.dec0:
            h = blk(h, temb)
        return self.out_conv(F.silu(self.out_norm(h)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def hidden_mask(n_generated: int, length: int = N_BINS, device=None, dtype=None) -> torch.Tensor:
    m = torch.zeros(1, n_generated, length, device=device, dtype=dtype)
    m[:, :, HIDDEN_START:HIDDEN_END] = 1.0
    return m
