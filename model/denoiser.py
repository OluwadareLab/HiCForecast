import torch
import torch.nn as nn

class Denoiser(nn.Module):
    """A small CNN-based denoiser to clean sparse/noisy Hi-C input."""
    def __init__(self, in_channels=1, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1)
        )
    def forward(self, x):
        # residual-style: input + learned residual
        res = self.net(x)
        return x + res
