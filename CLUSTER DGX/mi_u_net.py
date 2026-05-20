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

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        else:
            self.time_mlp = None

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, t=None):
        x = self.conv1(x)

        if self.time_mlp is not None and t is not None:
            time_emb = self.time_mlp(t)
            time_emb = time_emb[(..., ) + (None, ) * 2]
            x = x + time_emb

        x = self.conv2(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        # w x h -> w/2 x h/2
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, time_emb_dim)

    def forward(self, x, t=None):
        x = self.maxpool(x)
        return self.conv(x, t)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, time_emb_dim)

    def forward(self, x1, x2, t=None):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, t)


class FiLMBlock(nn.Module):
    def __init__(self, in_channels, embed_dim=256):
        super().__init__()
        # Proyectamos el embedding físico a Gamma (escala) y Beta (desplazamiento)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, in_channels * 2)
        )

    def forward(self, x, physics_context):
        # physics_context forma esperada: [Batch, embed_dim]
        emb = self.proj(physics_context)
        gamma, beta = emb.chunk(2, dim=-1)
        
        # Expandimos dimensiones para que coincidan con [B, C, H, W]
        gamma = gamma[..., None, None]
        beta = beta[..., None, None]
        
        # Aplicamos la modulación
        return x * (1 + gamma) + beta


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # Convierte un número (ej. t=500) en un vector de tamaño 'dim'
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(
            half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CustomGalaxyUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, embed_dim=256, time_dim=256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.inc = DoubleConv(n_channels, 64, time_emb_dim=time_dim)

        # Encoder
        self.down1 = Down(64, 128, time_emb_dim=time_dim)
        self.down2 = Down(128, 256, time_emb_dim=time_dim)
        self.down3 = Down(256, 512, time_emb_dim=time_dim)

        # Cross Attention intermedio (Mejora)
        self.cross_attn_down3 = FiLMBlock(
            in_channels=512, embed_dim=embed_dim)

        # Cuello de botella
        self.down4 = Down(512, 1024, time_emb_dim=time_dim)
        self.cross_attn = FiLMBlock(
            in_channels=1024, embed_dim=embed_dim)

        # Decoder
        self.up1 = Up(1024, 512, time_emb_dim=time_dim)
        self.up2 = Up(512, 256, time_emb_dim=time_dim)

        # Cross Attention intermedio (Mejora)
        self.cross_attn_up2 = FiLMBlock(
            in_channels=256, embed_dim=embed_dim)

        self.up3 = Up(256, 128, time_emb_dim=time_dim)
        self.up4 = Up(128, 64, time_emb_dim=time_dim)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x, context, timesteps):
        t = self.time_mlp(timesteps)
        # Bajada
        x1 = self.inc(x, t)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)

        x4 = self.down3(x3, t)
        x4 = self.cross_attn_down3(x4, context)  # Atención a nivel medio

        # Cuello de botella
        x5 = self.down4(x4, t)
        x5 = self.cross_attn(x5, context)  # Atención a nivel profundo

        # Subida
        x = self.up1(x5, x4, t)
        x = self.up2(x, x3, t)
        x = self.cross_attn_up2(x, context)  # Atención a nivel medio

        x = self.up3(x, x2, t)
        x = self.up4(x, x1, t)

        logits = self.outc(x)
        return logits
