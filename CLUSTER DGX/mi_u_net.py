"""
mi_u_net.py — U-Net de difusión para generación de galaxias condicionada por física.

Mejoras SOTA respecto a la versión anterior:
  ✓ ResBlocks con GroupNorm + AdaGN-Zero (Dhariwal & Nichol 2021, Peebles & Xie 2023)
  ✓ Self-Attention en 16×16 y 8×8 con Flash Attention (F.scaled_dot_product_attention)
  ✓ Stride-2 Conv para downsampling (vs MaxPool): gradientes más ricos
  ✓ Bilinear upsample + Conv (vs ConvTranspose2d): elimina artefactos de tablero de ajedrez
  ✓ Skip connections en todos los ResBlocks (He et al. 2016)
  ✓ SiLU en lugar de ReLU: gradiente más suave, estándar en difusión (DDPM++)
  ✓ Zero-init en proyecciones AdaGN y attention output: estabilidad en training inicial
  ✓ Conditioning unificado: time_emb + phys_emb (ADM, Dhariwal 2021)

Canales por resolución para imagen 512×512:
    512×512 →  32ch (L0)
    256×256 →  64ch (L1)
    128×128 → 128ch (L2)
     64×64  → 256ch (L3)
     32×32  → 256ch (L4)
     16×16  → 512ch (L5) + SelfAttention
      8×8   → 512ch (L6, cuello de botella) + SelfAttention

Firma del forward: (x, cond_emb, timesteps)
    x         : [B, 3, 512, 512]  imagen con ruido
    cond_emb  : [B, embed_dim]    embedding físico del PhysicsProjector
    timesteps : [B]               timestep como entero
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════════════════
# 1. Time embedding sinusoidal
# ══════════════════════════════════════════════════════════════════════════════


class SinusoidalEmbedding(nn.Module):
    """Embeddings sinusoidales de posición para los timesteps (Vaswani et al. 2017)."""

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
        return torch.cat([args.sin(), args.cos()], dim=-1)  # [B, dim]


# ══════════════════════════════════════════════════════════════════════════════
# 2. ResBlock con Adaptive Group Normalization Zero-Init (AdaGN-Zero)
# ══════════════════════════════════════════════════════════════════════════════


class ResBlock(nn.Module):
    """
    Bloque residual con Adaptive Group Normalization (Zero-init).

    Por qué GroupNorm en lugar de BatchNorm:
        BatchNorm normaliza sobre el batch. En difusión, el mismo modelo procesa
        imágenes con 0% de ruido (t=0) y 100% de ruido (t=T) en el mismo batch:
        las estadísticas de activación varían drásticamente con el timestep.
        GroupNorm opera dentro de cada muestra individualmente, por lo que es
        invariante al nivel de ruido. Todos los modelos SOTA (ADM, DDPM++, EDM)
        lo usan exclusivamente.

    AdaGN-Zero (Peebles & Xie 2023 — DiT):
        La proyección del conditioning a escala/shift se inicializa a CERO.
        Esto hace que al inicio del entrenamiento el bloque se comporte como
        identidad pura (el residual domina), lo que estabiliza las primeras
        iteraciones y acelera la convergencia.

    Flujo:
        h = Conv1(SiLU(GN1(x)))          ← primer camino
        h = AdaGN2(h, cond)              ← modulación por física+tiempo
        h = Conv2(Dropout(SiLU(h)))      ← segundo camino
        return h + skip(x)               ← residual
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        num_groups: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        # AdaGN: cond → (escala, desplazamiento) para modular GN2
        # Zero-init para comportamiento de identidad al inicio del training
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # Skip connection con proyección 1×1 si los canales cambian
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Primer camino: normaliza, activa, convoluciona
        h = self.conv1(self.act(self.norm1(x)))

        # Modulación AdaGN: el conditioning dicta escala y shift del segundo GN
        scale, shift = self.cond_proj(cond).chunk(2, dim=1)  # cada uno [B, out_ch]
        h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

        # Segundo camino: activa, dropout, convoluciona
        h = self.conv2(self.dropout(self.act(h)))

        return h + self.skip(x)  # conexión residual


# ══════════════════════════════════════════════════════════════════════════════
# 3. Self-Attention Block
# ══════════════════════════════════════════════════════════════════════════════


class SelfAttentionBlock(nn.Module):
    """
    Multi-head self-attention para feature maps 2D.

    Por qué es crítico para galaxias espirales:
        Los brazos espirales son estructuras de largo alcance que atraviesan
        decenas de píxeles. El campo receptivo efectivo de K capas conv es
        proporcional a K: en 5 niveles, ~2^5=32 píxeles. Insuficiente para
        coherencia global en 128×128. Self-attention conecta directamente
        cualquier par de posiciones en O(1) pasos, habilitando la coherencia
        necesaria para estructuras espirales.

    Implementación:
        - Pre-norm con GroupNorm (pre-norm = más estable que post-norm)
        - QKV en una sola Conv1d (eficiente)
        - F.scaled_dot_product_attention: usa Flash Attention automáticamente
          en PyTorch 2.0+ con CUDA, sin overhead de implementación manual
        - Zero-init en output projection: idem ResBlock

    Colocación: 16×16 (512ch) y 8×8 (512ch).
        - 32×32 con 256ch: secuencia de 1024 tokens → factible pero el beneficio
          marginal es menor con solo 20k imágenes de entrenamiento.
        - 64×64+: demasiado costoso y el modelo necesita datos para aprovecharlo.
    """

    def __init__(self, channels: int, num_heads: int = 8, num_groups: int = 32):
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels ({channels}) debe ser divisible por num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1, bias=False)

        # Zero-init para estabilidad inicial
        self.out_proj = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Pre-norm y aplanar dimensiones espaciales: [B, C, HW]
        h = self.norm(x).view(B, C, H * W)

        # Proyecciones QKV
        q, k, v = self.qkv(h).chunk(3, dim=1)  # cada uno [B, C, HW]

        # Reshape para multi-head: [B, heads, HW, head_dim]
        q = q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        # Flash Attention (automático en PyTorch 2.0+ / CUDA)
        out = F.scaled_dot_product_attention(q, k, v)  # [B, heads, HW, head_dim]

        # Reconstituir [B, C, H, W] y proyectar salida
        out = out.permute(0, 1, 3, 2).reshape(B, C, H * W)
        out = self.out_proj(out).view(B, C, H, W)

        return x + out  # conexión residual


# ══════════════════════════════════════════════════════════════════════════════
# 4. Módulos de up/downsampling
# ══════════════════════════════════════════════════════════════════════════════


class Downsample(nn.Module):
    """
    Downsampling con stride-2 Conv en lugar de MaxPool.

    MaxPool descarta información por diseño (solo toma el máximo).
    Stride-2 Conv aprende qué información preservar, con gradientes completos.
    Estándar en todos los modelos de difusión SOTA.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsampling con interpolación bilinear + Conv 3×3.

    ConvTranspose2d produce artefactos de "tablero de ajedrez" (Odena et al. 2016)
    por la distribución desigual de pesos en el kernel transpuesto. La combinación
    bilinear+conv los elimina por completo, produciendo imágenes más suaves.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ══════════════════════════════════════════════════════════════════════════════
# 5. U-Net principal
# ══════════════════════════════════════════════════════════════════════════════


class CustomGalaxyUNet(nn.Module):
    """
    U-Net de difusión SOTA para generación de galaxias espirales.

    Arquitectura por resolución (imagen 512×512):
        L0: 512×512,  32ch  — 2 ResBlocks
        L1: 256×256,  64ch  — 2 ResBlocks
        L2: 128×128, 128ch  — 2 ResBlocks
        L3:  64×64,  256ch  — 2 ResBlocks
        L4:  32×32,  256ch  — 2 ResBlocks
        L5:  16×16,  512ch  — 2 ResBlocks + 2 SelfAttention
        L6:   8×8,   512ch  — 2 ResBlocks + 1 SelfAttention  (cuello de botella)
        Decoder simétrico con skip connections de U-Net

    Conditioning:
        t_emb   = time_mlp(timesteps)     [B, D]  (D = embed_dim = 256)
        cond    = t_emb + phys_emb        [B, D]  (suma directa, ADM 2021)
        Cada ResBlock modula sus GroupNorms con cond vía AdaGN-Zero.
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 3,
        embed_dim: int = 256,
        dropout: float = 0.1,
        img_size: int = 512,
    ):
        super().__init__()
        D = embed_dim  # Dimensión del conditioning unificado
        if img_size not in (128, 512):
            raise ValueError(
                f"CustomGalaxyUNet solo define topologías para img_size 128 o 512, no {img_size}."
            )
        self.img_size = img_size

        # ── Time embedding ────────────────────────────────────────────────
        # Sinusoidal → MLP con hidden dim 4×D (estándar DDPM)
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(D),
            nn.Linear(D, D * 4),
            nn.SiLU(),
            nn.Linear(D * 4, D),
        )

        if self.img_size == 128:
            self._init_legacy_128(n_channels, n_classes, D, dropout)
            return

        # ── Encoder ──────────────────────────────────────────────────────
        # L0: 512×512, 32ch. GN=16 evita grupos de un solo canal en 32ch.
        self.inc = nn.Conv2d(n_channels, 32, 3, padding=1)
        self.enc0_a = ResBlock(32, 32, D, num_groups=16, dropout=dropout)
        self.enc0_b = ResBlock(32, 32, D, num_groups=16, dropout=dropout)
        self.enc0_dn = Downsample(32)

        # L1: 256×256, 32→64ch
        self.enc1_a = ResBlock(32, 64, D, num_groups=16, dropout=dropout)
        self.enc1_b = ResBlock(64, 64, D, dropout=dropout)
        self.enc1_dn = Downsample(64)

        # L2: 128×128, 64→128ch
        self.enc2_a = ResBlock(64, 128, D, dropout=dropout)
        self.enc2_b = ResBlock(128, 128, D, dropout=dropout)
        self.enc2_dn = Downsample(128)

        # L3: 64×64, 128→256ch
        self.enc3_a = ResBlock(128, 256, D, dropout=dropout)
        self.enc3_b = ResBlock(256, 256, D, dropout=dropout)
        self.enc3_dn = Downsample(256)

        # L4: 32×32, 256ch
        self.enc4_a = ResBlock(256, 256, D, dropout=dropout)
        self.enc4_b = ResBlock(256, 256, D, dropout=dropout)
        self.enc4_dn = Downsample(256)

        # L5: 16×16, 256→512ch + Self-Attention
        self.enc5_a = ResBlock(256, 512, D, dropout=dropout)
        self.enc5_sa1 = SelfAttentionBlock(512)
        self.enc5_b = ResBlock(512, 512, D, dropout=dropout)
        self.enc5_sa2 = SelfAttentionBlock(512)
        self.enc5_dn = Downsample(512)

        # ── Cuello de botella: 8×8, 512ch ────────────────────────────────
        self.mid_a = ResBlock(512, 512, D, dropout=dropout)
        self.mid_sa = SelfAttentionBlock(512)
        self.mid_b = ResBlock(512, 512, D, dropout=dropout)

        # ── Decoder ──────────────────────────────────────────────────────
        # L5 up: 16×16 — cat(512 + skip_enc5[512]) = 1024 → 512
        self.dec5_up = Upsample(512)
        self.dec5_a = ResBlock(512 + 512, 512, D, dropout=dropout)
        self.dec5_sa1 = SelfAttentionBlock(512)
        self.dec5_b = ResBlock(512, 512, D, dropout=dropout)
        self.dec5_sa2 = SelfAttentionBlock(512)

        # L4 up: 32×32 — cat(512 + skip_enc4[256]) = 768 → 256
        self.dec4_up = Upsample(512)
        self.dec4_a = ResBlock(512 + 256, 256, D, dropout=dropout)
        self.dec4_b = ResBlock(256, 256, D, dropout=dropout)

        # L3 up: 64×64 — cat(256 + skip_enc3[256]) = 512 → 256
        self.dec3_up = Upsample(256)
        self.dec3_a = ResBlock(256 + 256, 256, D, dropout=dropout)
        self.dec3_b = ResBlock(256, 256, D, dropout=dropout)

        # L2 up: 128×128 — cat(256 + skip_enc2[128]) = 384 → 128
        self.dec2_up = Upsample(256)
        self.dec2_a = ResBlock(256 + 128, 128, D, dropout=dropout)
        self.dec2_b = ResBlock(128, 128, D, dropout=dropout)

        # L1 up: 256×256 — cat(128 + skip_enc1[64]) = 192 → 64
        self.dec1_up = Upsample(128)
        self.dec1_a = ResBlock(128 + 64, 64, D, dropout=dropout)
        self.dec1_b = ResBlock(64, 64, D, dropout=dropout)

        # L0 up: 512×512 — cat(64 + skip_enc0[32]) = 96 → 32
        self.dec0_up = Upsample(64)
        self.dec0_a = ResBlock(64 + 32, 32, D, num_groups=16, dropout=dropout)
        self.dec0_b = ResBlock(32, 32, D, num_groups=16, dropout=dropout)

        # ── Output ───────────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(16, 32)
        self.outc = nn.Conv2d(32, n_classes, 1)

    def _init_legacy_128(
        self,
        n_channels: int,
        n_classes: int,
        cond_dim: int,
        dropout: float,
    ):
        """Mantiene cargables los checkpoints 128 entrenados antes de los niveles nuevos."""
        self.inc = nn.Conv2d(n_channels, 128, 3, padding=1)
        self.enc0_a = ResBlock(128, 128, cond_dim, dropout=dropout)
        self.enc0_b = ResBlock(128, 128, cond_dim, dropout=dropout)
        self.enc0_dn = Downsample(128)

        self.enc1_a = ResBlock(128, 256, cond_dim, dropout=dropout)
        self.enc1_b = ResBlock(256, 256, cond_dim, dropout=dropout)
        self.enc1_dn = Downsample(256)

        self.enc2_a = ResBlock(256, 256, cond_dim, dropout=dropout)
        self.enc2_b = ResBlock(256, 256, cond_dim, dropout=dropout)
        self.enc2_dn = Downsample(256)

        self.enc3_a = ResBlock(256, 512, cond_dim, dropout=dropout)
        self.enc3_sa1 = SelfAttentionBlock(512)
        self.enc3_b = ResBlock(512, 512, cond_dim, dropout=dropout)
        self.enc3_sa2 = SelfAttentionBlock(512)
        self.enc3_dn = Downsample(512)

        self.mid_a = ResBlock(512, 512, cond_dim, dropout=dropout)
        self.mid_sa = SelfAttentionBlock(512)
        self.mid_b = ResBlock(512, 512, cond_dim, dropout=dropout)

        self.dec3_up = Upsample(512)
        self.dec3_a = ResBlock(512 + 512, 512, cond_dim, dropout=dropout)
        self.dec3_sa1 = SelfAttentionBlock(512)
        self.dec3_b = ResBlock(512, 512, cond_dim, dropout=dropout)
        self.dec3_sa2 = SelfAttentionBlock(512)

        self.dec2_up = Upsample(512)
        self.dec2_a = ResBlock(512 + 256, 256, cond_dim, dropout=dropout)
        self.dec2_b = ResBlock(256, 256, cond_dim, dropout=dropout)

        self.dec1_up = Upsample(256)
        self.dec1_a = ResBlock(256 + 256, 256, cond_dim, dropout=dropout)
        self.dec1_b = ResBlock(256, 256, cond_dim, dropout=dropout)

        self.dec0_up = Upsample(256)
        self.dec0_a = ResBlock(256 + 128, 128, cond_dim, dropout=dropout)
        self.dec0_b = ResBlock(128, 128, cond_dim, dropout=dropout)

        self.out_norm = nn.GroupNorm(32, 128)
        self.outc = nn.Conv2d(128, n_classes, 1)

    def _forward_legacy_128(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
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

    def forward(
        self,
        x: torch.Tensor,
        cond_emb: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x         : [B, 3, H, W]    imagen con ruido
            cond_emb  : [B, embed_dim]  embedding físico (del PhysicsProjector)
            timesteps : [B]             timestep entero

        Returns:
            [B, 3, H, W]  predicción de velocidad v (v-prediction)
        """
        # Combinar time embedding + physics embedding (ADM, Dhariwal 2021)
        t = self.time_mlp(timesteps)  # [B, D]
        cond = t + cond_emb  # [B, D] — suma directa

        if self.img_size == 128:
            return self._forward_legacy_128(x, cond)

        # ── Encoder ──────────────────────────────────────────────────────
        x0 = self.inc(x)  # [B, 32, 512, 512]
        x0 = self.enc0_a(x0, cond)
        x0 = self.enc0_b(x0, cond)  # skip L0

        x1 = self.enc0_dn(x0)  # [B, 32, 256, 256]
        x1 = self.enc1_a(x1, cond)  # 32→64ch
        x1 = self.enc1_b(x1, cond)  # skip L1 (64ch)

        x2 = self.enc1_dn(x1)  # [B, 64, 128, 128]
        x2 = self.enc2_a(x2, cond)  # 64→128ch
        x2 = self.enc2_b(x2, cond)  # skip L2 (128ch)

        x3 = self.enc2_dn(x2)  # [B, 128, 64, 64]
        x3 = self.enc3_a(x3, cond)  # 128→256ch
        x3 = self.enc3_b(x3, cond)  # skip L3 (256ch)

        x4 = self.enc3_dn(x3)  # [B, 256, 32, 32]
        x4 = self.enc4_a(x4, cond)
        x4 = self.enc4_b(x4, cond)  # skip L4 (256ch)

        x5 = self.enc4_dn(x4)  # [B, 256, 16, 16]
        x5 = self.enc5_a(x5, cond)  # 256→512ch
        x5 = self.enc5_sa1(x5)
        x5 = self.enc5_b(x5, cond)
        x5 = self.enc5_sa2(x5)  # skip L5 (512ch)

        # ── Cuello de botella ─────────────────────────────────────────────
        xm = self.enc5_dn(x5)  # [B, 512, 8, 8]
        xm = self.mid_a(xm, cond)
        xm = self.mid_sa(xm)
        xm = self.mid_b(xm, cond)

        # ── Decoder ──────────────────────────────────────────────────────
        h = self.dec5_up(xm)  # [B, 512, 16, 16]
        h = torch.cat([h, x5], dim=1)  # [B, 1024, 16, 16]
        h = self.dec5_a(h, cond)  # 1024→512ch
        h = self.dec5_sa1(h)
        h = self.dec5_b(h, cond)
        h = self.dec5_sa2(h)

        h = self.dec4_up(h)  # [B, 512, 32, 32]
        h = torch.cat([h, x4], dim=1)  # [B, 768, 32, 32]
        h = self.dec4_a(h, cond)  # 768→256ch
        h = self.dec4_b(h, cond)

        h = self.dec3_up(h)  # [B, 256, 64, 64]
        h = torch.cat([h, x3], dim=1)  # [B, 512, 64, 64]
        h = self.dec3_a(h, cond)  # 512→256ch
        h = self.dec3_b(h, cond)

        h = self.dec2_up(h)  # [B, 256, 128, 128]
        h = torch.cat([h, x2], dim=1)  # [B, 384, 128, 128]
        h = self.dec2_a(h, cond)  # 384→128ch
        h = self.dec2_b(h, cond)

        h = self.dec1_up(h)  # [B, 128, 256, 256]
        h = torch.cat([h, x1], dim=1)  # [B, 192, 256, 256]
        h = self.dec1_a(h, cond)  # 192→64ch
        h = self.dec1_b(h, cond)

        h = self.dec0_up(h)  # [B, 64, 512, 512]
        h = torch.cat([h, x0], dim=1)  # [B, 96, 512, 512]
        h = self.dec0_a(h, cond)  # 96→32ch
        h = self.dec0_b(h, cond)

        # ── Output ───────────────────────────────────────────────────────
        return self.outc(F.silu(self.out_norm(h)))  # [B, 3, 512, 512]
