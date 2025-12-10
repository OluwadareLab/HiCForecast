import torch
import torch.nn as nn
from .denoiser import Denoiser
from .attention import SimpleSelfAttention
import torch.nn.functional as F

class VoxelFlowStub(nn.Module):
    """A lightweight placeholder for the original voxel/optical-flow blocks.
    In a full implementation, replace this with the original HiCForecast blocks or import them."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)

class HiCForecastModified(nn.Module):
    def __init__(self,
                 in_channels=1,
                 use_denoiser=True,
                 use_attention=True,
                 use_multimodal=False,
                 mc_dropout_p=0.1):
        super().__init__()
        self.use_denoiser = use_denoiser
        self.use_attention = use_attention
        self.use_multimodal = use_multimodal
        self.denoiser = Denoiser(in_channels) if use_denoiser else nn.Identity()
        # fusion block: simple conv to mix channels (if multimodal)
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.voxel = VoxelFlowStub(in_channels)
        self.attn = SimpleSelfAttention(in_channels) if use_attention else nn.Identity()
        # dropout for MC dropout uncertainty
        self.dropout = nn.Dropout2d(mc_dropout_p)
        # prediction head
        self.pred_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        # optional sigma head for aleatoric uncertainty
        self.sigma_head = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x, return_sigma=False, mc_iters=1):
        # x: (B, C, H, W)
        x = self.denoiser(x)
        x = self.fusion(x)
        x = self.voxel(x)
        x = self.attn(x)
        # MC dropout: run multiple passes and average
        preds = []
        sigmas = []
        for _ in range(mc_iters):
            y = self.dropout(x)
            pred = self.pred_head(y)
            preds.append(pred)
            sigma = self.sigma_head(y)
            sigmas.append(sigma)
        preds = torch.stack(preds, dim=0).mean(0)  # (B,1,H,W)
        if return_sigma:
            sigma_mean = torch.stack(sigmas, dim=0).mean(0)
            # total uncertainty estimate: variance across mc passes + predicted sigma
            var_mc = torch.stack(preds, dim=0).var(0) if preds.shape[0] > 1 else torch.zeros_like(preds)
            return preds, sigma_mean, var_mc
        return preds
