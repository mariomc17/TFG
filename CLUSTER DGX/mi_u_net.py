import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, time_emb_dim)

        def forward(self, x1, x2, t=None):
            x1 = self.up(x1)
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x, t)
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels, embed_dim=256):
        super().__init__()

        self.query_proj = nn.Linear(in_channels, embed_dim)
        
        # Módulo de PyTorch
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        
        self.out_proj = nn.Linear(embed_dim, in_channels)

    def forward(self, x, physics_context):
        # x: es la imagen. Forma: [Batch, Canales, Alto, Ancho]
        # physics_context: es la física. Forma: [Batch, Secuencia (4), Dimensiones (256)]
        
        B, C, H, W = x.shape
        
        # Pasa de [Batch, Canales, Alto, Ancho] a [Batch, Píxeles (Alto*Ancho), Canales]
        x_flat = x.view(B, C, H * W).permute(0, 2, 1) 
        
        # Preguntas Q
        Q = self.query_proj(x_flat) 
        
        # Claves (K) y valores (V)
        attn_out, _ = self.attention(query=Q, key=physics_context, value=physics_context)
        
        attn_out = self.out_proj(attn_out)
        
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        
        return x + attn_out

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # Convierte un número (ej. t=500) en un vector de tamaño 'dim'
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
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
        
        # Cuello de botella
        self.down4 = Down(512, 1024, time_emb_dim=time_dim)
        self.cross_attn = CrossAttentionBlock(in_channels=1024, embed_dim=embed_dim)

        # Decoder
        self.up1 = Up(1024, 512, time_emb_dim=time_dim)
        self.up2 = Up(512, 256, time_emb_dim=time_dim)      
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
        
        # Cuello de botella
        x5 = self.down4(x4, t)
        x5 = self.cross_attn(x5, context)

        # Subida
        x = self.up1(x5, x4, t)
        x = self.up2(x, x3, t)
        x = self.up3(x, x2, t)
        x = self.up4(x, x1, t)
        
        logits = self.outc(x)
        return logits