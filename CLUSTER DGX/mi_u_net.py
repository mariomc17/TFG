import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device).float() / (half - 1)
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)
    
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, num_groups: int = 32, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))

        scale, shift = self.cond_proj(cond).chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = self.conv2(self.dropout(self.act(h)))

        return h + self.skip(x)
    
class SelfAttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8, num_groups: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1, bias=False)

        self.out_proj = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)

        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        out = F.scaled_dot_product_attention(q, k, v) 

        out = out.permute(0, 1, 3, 2).reshape(B, C, H * W)
        out = self.out_proj(out).view(B, C, H, W)

        return x + out
    
class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

class CustomGalaxyUNet(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 3, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        D = embed_dim

        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(D),
            nn.Linear(D, D * 4),
            nn.SiLU(),
            nn.Linear(D * 4, D),
        )

        self.inc = nn.Conv2d(n_channels, 128, 3, padding=1)
        
        self.enc0_a = ResBlock(128, 128, D, dropout=dropout)
        self.enc0_b = ResBlock(128, 128, D, dropout=dropout)
        self.enc0_dn = Downsample(128)

        self.enc1_a = ResBlock(128, 256, D, dropout=dropout)
        self.enc1_b = ResBlock(256, 256, D, dropout=dropout)
        self.enc1_dn = Downsample(256)

        self.enc2_a = ResBlock(256, 256, D, dropout=dropout)
        self.enc2_b = ResBlock(256, 256, D, dropout=dropout)
        self.enc2_dn = Downsample(256)

        self.enc3_a = ResBlock(256, 512, D, dropout=dropout)
        self.enc3_sa1 = SelfAttentionBlock(512)
        self.enc3_b = ResBlock(512, 512, D, dropout=dropout)
        self.enc3_sa2 = SelfAttentionBlock(512)
        self.enc3_dn = Downsample(512)

        self.mid_a = ResBlock(512, 512, D, dropout=dropout)
        self.mid_sa = SelfAttentionBlock(512)
        self.mid_b = ResBlock(512, 512, D, dropout=dropout)

        self.dec3_up = Upsample(512)
        self.dec3_a = ResBlock(512 + 512, 512, D, dropout=dropout)
        self.dec3_sa1 = SelfAttentionBlock(512)
        self.dec3_b = ResBlock(512, 512, D, dropout=dropout)
        self.dec3_sa2 = SelfAttentionBlock(512)

        self.dec2_up = Upsample(512)
        self.dec2_a = ResBlock(512 + 256, 256, D, dropout=dropout)
        self.dec2_b = ResBlock(256, 256, D, dropout=dropout)

        self.dec1_up = Upsample(256)
        self.dec1_a = ResBlock(256 + 256, 256, D, dropout=dropout)
        self.dec1_b = ResBlock(256, 256, D, dropout=dropout)

        self.dec0_up = Upsample(256)
        self.dec0_a = ResBlock(256 + 128, 128, D, dropout=dropout)
        self.dec0_b = ResBlock(128, 128, D, dropout=dropout)

        self.out_norm = nn.GroupNorm(32, 128)
        self.outc = nn.Conv2d(128, n_classes, 1)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t = self.time_mlp(timesteps)
        cond = t + cond_emb 

        x0 = self.inc(x)
        x0 = self.enc0_a(x0, cond)
        x0 = self.enc0_b(x0, cond)

        x1 = self.enc0_dn(x0)
        x1 = self.enc1_a(x1, cond)
        x1 = self.enc1_b(x1, cond)

        x2 = self.enc1_dn(x1)
        x2 = self.enc2_a(x2, cond)
        x2 = self.enc2_b(x2, cond)

        x3 = self.enc2_dn(x2)
        x3 = self.enc3_a(x3, cond)
        x3 = self.enc3_sa1(x3)
        x3 = self.enc3_b(x3, cond)
        x3 = self.enc3_sa2(x3)

        xm = self.enc3_dn(x3)
        xm = self.mid_a(xm, cond)
        xm = self.mid_sa(xm)
        xm = self.mid_b(xm, cond)

        h = self.dec3_up(xm)
        h = torch.cat([h, x3], dim=1) 
        h = self.dec3_a(h, cond)
        h = self.dec3_sa1(h)
        h = self.dec3_b(h, cond)
        h = self.dec3_sa2(h)

        h = self.dec2_up(h)
        h = torch.cat([h, x2], dim=1)
        h = self.dec2_a(h, cond)
        h = self.dec2_b(h, cond)

        h = self.dec1_up(h)
        h = torch.cat([h, x1], dim=1)
        h = self.dec1_a(h, cond)
        h = self.dec1_b(h, cond)

        h = self.dec0_up(h)
        h = torch.cat([h, x0], dim=1)
        h = self.dec0_a(h, cond)
        h = self.dec0_b(h, cond)

        return self.outc(F.silu(self.out_norm(h)))