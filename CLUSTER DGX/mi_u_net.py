import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Time Embedding (El reloj del ruido)

class SinusoidalEmbedding(nn.Module):
    """MEJORA SOTA: Embeddings sinusoidales puros (Vaswani et al. 2017)."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        # Frecuencias logarítmicas para abarcar todo el espectro temporal
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device).float() / (half - 1)
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)  # [B, dim]
    
# 2. ResBlock con Adaptive Group Normalization Zero-Init (AdaGN-Zero)

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, num_groups: int = 32, dropout: float = 0.1):
        super().__init__()
        # MEJORA SOTA: GroupNorm en lugar de BatchNorm
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        # MEJORA SOTA (AdaGN-Zero): proyectamos la física+tiempo para modular el GN2.
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)
        # INICIALIZACIÓN A CERO: Estabilidad extrema al inicio del training
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # MEJORA SOTA: conexión residual
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU() # SiLU es más suave que ReLU, ideal para difusión

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # 1. Camino principal: normaliza, activa, convoluciona
        h = self.conv1(self.act(self.norm1(x)))

        # 2. Inyección de física (AdaGN): modulamos el GroupNorm 2
        scale, shift = self.cond_proj(cond).chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

        # 3. Final del camino principal
        h = self.conv2(self.dropout(self.act(h)))

        # 4. Entrada + Camino principal
        return h + self.skip(x)
    
# 3. Self-Attention Block (Para los brazos espirales)

class SelfAttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8, num_groups: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(num_groups, channels)
        # Calculamos Query, Key y Value de golpe (más eficiente)
        self.qkv = nn.Conv1d(channels, channels * 3, 1, bias=False)

        # Proyección de salida inicializada a cero (estabilidad SOTA)
        self.out_proj = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Aplanamos la imagen 2D a una secuencia 1D para la atención
        h = self.norm(x).view(B, C, H * W)

        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        # MEJORA SOTA: Flash Attention (Ultra rápido, nativo en PyTorch 2.0)
        out = F.scaled_dot_product_attention(q, k, v) 

        out = out.permute(0, 1, 3, 2).reshape(B, C, H * W)
        out = self.out_proj(out).view(B, C, H, W)

        return x + out # Conexión residual
    
# 4. Módulos de up/downsampling SOTA

class Downsample(nn.Module):
    """MEJORA SOTA: Downsampling aprendido (Stride-2 Conv) en vez de MaxPool."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    """MEJORA SOTA: Interpolación Bilinear + Conv para evitar 'tablero de ajedrez'."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)
