import torch
import torch.nn as nn

class SimpleSelfAttention(nn.Module):
    """A small self-attention block that flattens spatial dims to attend across bins."""
    def __init__(self, channels, heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.out = nn.Linear(channels, channels)
        self.heads = heads
        assert channels % heads == 0
        self.d_k = channels // heads

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.view(B, C, N).permute(0,2,1)  # (B, N, C)
        x_norm = self.norm(x_flat)
        qkv = self.qkv(x_norm).chunk(3, dim=-1)  # each (B, N, C)
        q, k, v = [t.view(B, N, self.heads, self.d_k).permute(0,2,1,3) for t in qkv]
        # scaled dot-product
        scores = torch.matmul(q, k.transpose(-2,-1)) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, heads, N, d_k)
        out = out.permute(0,2,1,3).contiguous().view(B, N, C)
        out = self.out(out)
        out = out.permute(0,2,1).view(B, C, H, W)
        return out
