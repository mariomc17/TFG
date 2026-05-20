import argparse
import copy
import csv
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import torch
import torch.nn as nn
from dataset import GalaxiasFisicasDataset
from diffusers import DDPMScheduler
from mi_u_net import CustomGalaxyUNet
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuración (DataClasses)

@dataclass
class DataCfg:
    hdf5_path: str = "dataset_galaxias_sin_rgb.h5"
    img_size: int = 128
    # MEJORA SOTA: añadimos RADIO_P (tamaño angular real)
    variables: List[str] = field(
        default_factory=lambda: ["ESCALA_KPC_PX", "LOG_MS", "EA", "RADIO_P"]
    )
    num_workers: int = 4
    augment: bool = True

@dataclass
class TrainCfg:
    batch_size: int = 32
    epochs: int = 200 # Más épocas porque el proceso es más suave
    lr: float = 1e-4 # Learning rate más bajo y estable
    save_every: int = 10
    patience: int = 20
    min_delta: float = 1e-5
    amp: bool = True
    compile: bool = True
    dropout: float = 0.1
    cfg_drop_prob: float = 0.15
    # MEJORA SOTA: hiperparámetros para la Media Móvil Exponencial (EMA)
    ema_decay: float = 0.9999
    ema_warmup_steps: int = 1000

@dataclass
class ModelCfg:
    n_channels: int = 3
    n_classes: int = 3
    embed_dim: int = 256

@dataclass
class DiffusionCfg:
    num_train_timesteps: int = 1000

@dataclass
class Config:
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    diffusion: DiffusionCfg = field(default_factory=DiffusionCfg)

# ══════════════════════════════════════════════════════════════════════════════
# Clases SOTA (Projector + EMA)
# ══════════════════════════════════════════════════════════════════════════════
class PhysicsProjector(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 256):
        super().__init__()
        # MEJORA SOTA: Null Tokens aprendibles en lugar de vectores a cero.
        # Se inicializan con randn para simular la distribución Z-Score normal.
        self.null_tokens = nn.Parameter(torch.randn(input_dim))
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x: torch.Tensor, drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            # Si estamos aplicando CFG (15% de las veces), sustituimos la física
            # real por los "null tokens" aprendibles.
            if drop_mask is not None and drop_mask.any():
                null = self.null_tokens.unsqueeze(0).expand(x.shape[0], -1)
                # torch.where: (condicion, valor_si_es_verdad, valor_si_es_falso)
                x = torch.where(drop_mask[:, None], null, x)
            return self.net(x)

    @torch.no_grad()
    def get_null_embedding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Extrae el token incondicional puro para la inferencia."""
        null_vec = self.null_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        return self.net(null_vec)

class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup_steps: int = 1000):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0
        self.shadow: dict = {
            name: param.data.detach().float().cpu().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Mezclamos los pesos viejos con los nuevos suavemente en cada batch."""
        self.step += 1
        # Warm-up de Karras et al.: empieza mezclando rápido y luego se asienta
        decay = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow:
                continue
            self.shadow[name] = (
                decay * self.shadow[name]
                + (1.0 - decay) * param.data.detach().float().cpu()
            )

    def copy_to(self, model: nn.Module):
        """Inyecta los pesos promediados de vuelta a una red para guardarla."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].to(param.device))

    def state_dict(self) -> dict:
        return {
            "shadow": self.shadow, "step": self.step,
            "decay": self.decay, "warmup_steps": self.warmup_steps,
        }

def get_raw_state_dict(model: nn.Module) -> dict:
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    return raw.state_dict()

def load_config(yaml_path: str, cli_overrides: List[str]):
    schema = OmegaConf.structured(Config)
    file_cfg = OmegaConf.load(yaml_path) if yaml_path else OmegaConf.create({})
    cli_cfg = OmegaConf.from_dotlist(cli_overrides) if cli_overrides else OmegaConf.create({})
    return OmegaConf.merge(schema, file_cfg, cli_cfg)

def setup_run_dir() -> str:
    run_dir = os.environ.get("RUN_DIR")
    if run_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(SCRIPT_DIR, "runs", "train", f"local_{ts}")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    return run_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento del diffusion U-Net de galaxias.")
    parser.add_argument("--config", type=str, default="configs/train_baseline.yaml")
    args, overrides = parser.parse_known_args()
    return args, overrides

def make_grad_scaler(device: torch.device, enabled: bool):
    use_amp = bool(enabled and device.type == "cuda")
    if hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            return torch.amp.GradScaler(enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)

def autocast_context(device: torch.device, enabled: bool):
    use_amp = bool(enabled and device.type == "cuda")
    if not use_amp:
        return nullcontext()
    if hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", dtype=torch.float16, enabled=True)
        except TypeError:
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    return torch.cuda.amp.autocast(dtype=torch.float16, enabled=True)

def build_checkpoint(unet, unet_raw, projector, ema, dataset, cfg, epoch, mean_loss, best_loss) -> dict:
    # 1. Copiamos la estructura cruda
    unet_ema_save = copy.deepcopy(unet_raw)
    # 2. Le inyectamos los pesos estabilizados del EMA
    ema.copy_to(unet_ema_save)
    
    return {
        "unet_ema": unet_ema_save.state_dict(), # Pesos de inferencia
        "unet": get_raw_state_dict(unet),       # Pesos para continuar training
        "projector": projector.state_dict(),
        "ema_state": ema.state_dict(),
        "variables": list(cfg.data.variables),
        "norm_stats": dataset.norm_stats,       # MEJORA SOTA: imprescindible para inferencia
        "epoch": epoch,
        "mean_loss": mean_loss,
        "best_loss": best_loss,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

# Bucle principal de Entrenamiento

def main():
    args, overrides = parse_args()
    cfg = load_config(args.config, overrides)

    torch.backends.cudnn.benchmark = True
    run_dir = setup_run_dir()
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    metrics_path = os.path.join(run_dir, "metrics.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"RUN_DIR:    {run_dir}")
    print(f"Device:     {device}")
    print("=" * 70)
    OmegaConf.save(cfg, os.path.join(run_dir, "config_used.yaml"))

    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "mean_mse_loss", "best_loss"])

    dataset = GalaxiasFisicasDataset(
        hdf5_path=cfg.data.hdf5_path,
        img_size=cfg.data.img_size,
        variables_elegidas=list(cfg.data.variables),
        augment=cfg.data.augment,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.data.num_workers > 0),
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.diffusion.num_train_timesteps,
        prediction_type="v_prediction",
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
    )

    unet = CustomGalaxyUNet(
        n_channels=cfg.model.n_channels,
        n_classes=cfg.model.n_classes,
        embed_dim=cfg.model.embed_dim,
        dropout=cfg.train.dropout,
    ).to(device)

    unet_raw = unet
    if cfg.train.compile and hasattr(torch, "compile") and device.type == "cuda":
        unet = torch.compile(unet)

    projector = PhysicsProjector(
        input_dim=len(cfg.data.variables),
        embed_dim=cfg.model.embed_dim,
    ).to(device)

    optimizer = AdamW(
        list(unet.parameters()) + list(projector.parameters()),
        lr=cfg.train.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    scaler = make_grad_scaler(device, cfg.train.amp)

    ema = EMAModel(unet_raw, decay=cfg.train.ema_decay, warmup_steps=cfg.train.ema_warmup_steps)
    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(cfg.train.epochs):
        unet.train()
        projector.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}")

        for batch in progress_bar:
            clean_images, phys_vectors = batch
            clean_images = clean_images.to(device)
            phys_vectors = phys_vectors.to(device)

            bsz = clean_images.shape[0]
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bsz,), device=device,
            ).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            optimizer.zero_grad(set_to_none=True)

            # CFG Dropout con Learnable Null Tokens
            drop_mask = torch.rand(bsz, device=device) < cfg.train.cfg_drop_prob

            with autocast_context(device, cfg.train.amp):
                # El proyector ahora se encarga de inyectar los null tokens si drop_mask es True
                cond_emb = projector(phys_vectors, drop_mask=drop_mask)
                pred = unet(x=noisy_images, cond_emb=cond_emb, timesteps=timesteps)
                target = noise_scheduler.get_velocity(clean_images, noise, timesteps)

                alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
                a_t = alphas_cumprod[timesteps]
                snr = a_t / (1.0 - a_t)
                snr_weight = torch.clamp(snr, max=5.0) / (snr + 1.0)

                loss = nn.functional.mse_loss(pred, target, reduction="none")
                loss = loss.mean(dim=[1, 2, 3])
                loss = (loss * snr_weight).mean()

            scaler.scale(loss).backward()
            # MEJORA SOTA: Gradient Clipping para evitar colapsos matemáticos
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(unet.parameters()) + list(projector.parameters()), max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()

            # MEJORA SOTA: actualizamos la copia fantasma (EMA) en cada paso
            ema.update(unet_raw)

            epoch_loss += loss.item()
            progress_bar.set_postfix({"MSE": f"{loss.item():.4f}"})

        mean_loss = epoch_loss / len(dataloader)
        print(f"Época {epoch + 1} | MSE Loss Medio: {mean_loss:.6f}")

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{mean_loss:.6f}", f"{best_loss:.6f}"])

        improved = mean_loss < best_loss - cfg.train.min_delta
        next_best_loss = mean_loss if improved else best_loss

        ckpt = build_checkpoint(
            unet, unet_raw, projector, ema, dataset, cfg, epoch + 1, mean_loss, next_best_loss
        )
        torch.save(ckpt, os.path.join(ckpt_dir, "last.pt"))

        if cfg.train.save_every > 0 and (epoch + 1) % cfg.train.save_every == 0:
            torch.save(ckpt, os.path.join(ckpt_dir, f"modelo_epoca_{epoch + 1:03d}.pt"))

        if improved:
            best_loss = mean_loss
            epochs_no_improve = 0
            torch.save(ckpt, os.path.join(ckpt_dir, "mejor_modelo.pt"))
            print(f"  → Nuevo mejor modelo guardado (EMA) (loss: {best_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"  → Sin mejora. Paciencia: {epochs_no_improve}/{cfg.train.patience}")
            if epochs_no_improve >= cfg.train.patience:
                print(f"\nEarly stopping en época {epoch + 1}. Mejor loss: {best_loss:.6f}")
                break

    print(f"\nEntrenamiento completado. Resultados en: {run_dir}")

if __name__ == "__main__":
    main()