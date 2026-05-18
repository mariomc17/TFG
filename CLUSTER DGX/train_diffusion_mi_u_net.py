"""
train_diffusion_mi_u_net.py — Entrenamiento del diffusion U-Net de galaxias.

Mejoras respecto a la versión anterior:
  ✓ PhysicsProjector con Learnable Null Tokens por variable
      Los null tokens se aprenden (nn.Parameter), no son vectores de ceros.
      Esto resuelve la ambigüedad "vector nulo ≠ galaxia incondicional".
  ✓ EMA (Exponential Moving Average) con warm-up de decay (Karras et al. 2022)
      Los pesos EMA son los que se usan para generar imágenes en inferencia.
      Se guardan separados de los pesos brutos en el checkpoint.
  ✓ CFG correcto: se usa projector(x, drop_mask) en lugar de zeroing manual
  ✓ Nueva firma del UNet: forward(x, cond_emb, timesteps)
  ✓ norm_stats del dataset guardados en el checkpoint (necesarios en inferencia)
  ✓ Detección automática de errores constantes (no augmentation espurio)
  ✓ Variables por defecto actualizadas: + RADIO_P, sin SFR/G_R/MET/REDSHIFT
"""

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


# ══════════════════════════════════════════════════════════════════════════════
# Schema de configuración
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class DataCfg:
    hdf5_path: str = "dataset_galaxias_sin_rgb.h5"
    img_size: int = 128
    # RADIO_P añadido: encapsula tamaño angular independiente de la distancia
    # SFR/G_R/MET/REDSHIFT eliminados (ver dataset.py para justificación)
    variables: List[str] = field(
        default_factory=lambda: ["ESCALA_KPC_PX", "LOG_MS", "EA", "RADIO_P"]
    )
    num_workers: int = 4
    augment: bool = True


@dataclass
class TrainCfg:
    batch_size: int = 32
    epochs: int = 200
    lr: float = 1e-4
    save_every: int = 10
    patience: int = 20
    min_delta: float = 1e-5
    amp: bool = True
    compile: bool = True
    dropout: float = 0.1
    cfg_drop_prob: float = 0.15  # Probabilidad de dropout de conditioning (CFG)
    ema_decay: float = 0.9999  # Decay EMA (Karras et al. 2022)
    ema_warmup_steps: int = 1000  # Pasos de warm-up del decay EMA


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
# PhysicsProjector con Learnable Null Tokens
# ══════════════════════════════════════════════════════════════════════════════


class PhysicsProjector(nn.Module):
    """
    Proyector de variables físicas al espacio de embedding del conditioning.

    Implementa Learnable Null Tokens por variable (Ho & Salimans 2022, CFG):
        self.null_tokens : nn.Parameter [input_dim]
            Un token nulo aprendible por variable física, inicializado con randn
            (distribución similar a los valores Z-score de las variables reales).
            Durante el training, cuando se aplica CFG dropout a una muestra,
            se sustituye TODA su física por el null token en espacio crudo,
            y la red aprende qué embedding representa "ausencia de conditioning".

    Por qué null tokens y no vector de ceros:
        Con normalización Z-score, ceros significa "valor medio de cada variable"
        → la galaxia de masas medias, edad media y escala media.
        El null token es un vector APRENDIDO que la red optimiza para representar
        "no tengo información de física" → generación incondicional real.

    Por qué tokens por variable y no un embedding global:
        - La red puede aprender representaciones nulas diferenciadas por dimensión
        - Abre la puerta a partial guidance en inferencia (especificar solo masa,
          por ejemplo) sin reentrenar, pasando null_tokens para las dimensiones
          no especificadas
    """

    def __init__(self, input_dim: int, embed_dim: int = 256):
        super().__init__()
        # randn → distribución coherente con los valores Z-score reales
        self.null_tokens = nn.Parameter(torch.randn(input_dim))

        # MLP con dos capas ocultas (suficiente para proyección no lineal)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x         : [B, input_dim]  vectores físicos normalizados (Z-score)
            drop_mask : [B] bool        True = reemplazar con null_tokens (CFG dropout)

        Returns:
            [B, embed_dim]  embedding físico proyectado
        """
        if drop_mask is not None and drop_mask.any():
            null = self.null_tokens.unsqueeze(0).expand(x.shape[0], -1)
            # where(condition, value_if_true, value_if_false)
            x = torch.where(drop_mask[:, None], null, x)
        return self.net(x)

    @torch.no_grad()
    def get_null_embedding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Devuelve el embedding del null token para CFG en inferencia.
        El null token pasa por la misma red que los vectores reales.
        """
        null_vec = self.null_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        return self.net(null_vec)


# ══════════════════════════════════════════════════════════════════════════════
# EMA (Exponential Moving Average)
# ══════════════════════════════════════════════════════════════════════════════


class EMAModel:
    """
    Exponential Moving Average de los pesos del modelo.

    Fundamento (Karras et al. 2022 — EDM):
        Los modelos de difusión tienen dinámicas de gradientes ruidosas,
        especialmente hacia el final del training. Los pesos del último paso
        capturan el ruido de las últimas iteraciones.
        EMA mantiene una copia de los pesos que se actualiza lentamente:
            θ_ema = β · θ_ema + (1 − β) · θ_actual
        Con β=0.9999, los pesos EMA son un promedio de las últimas ~10.000
        iteraciones, mucho más estables. Generar con EMA produce imágenes
        más nítidas y sin artefactos de alta frecuencia.

    Warm-up de decay (Karras et al. 2022):
        En los primeros pasos, el modelo aprende rápidamente.
        Con β fijo en 0.9999 desde el inicio, el EMA queda "pegado" a los
        pesos malos del inicio del training.
        Solución: β_efectivo = min(β, (1 + n) / (10 + n))
        Esto hace que β arranque cerca de 0 (EMA sigue al modelo de cerca)
        y converja asintóticamente al β objetivo a medida que n→∞.

    Uso:
        ema = EMAModel(unet, decay=0.9999, warmup_steps=1000)
        # En el loop de training, al final de cada batch:
        ema.update(unet)
        # Para guardar checkpoint:
        ema_copy = copy.deepcopy(unet_raw)
        ema.copy_to(ema_copy)
        torch.save({'unet_ema': ema_copy.state_dict(), ...}, ckpt_path)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 1000,
    ):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0

        # Copia shadow en float32 en CPU para evitar acumulación de errores
        # de redondeo con float16 o bfloat16
        self.shadow: dict = {
            name: param.data.detach().float().cpu().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Actualiza los pesos EMA con los pesos actuales del modelo."""
        self.step += 1
        # Warm-up: β_efectivo converge suavemente al β objetivo
        decay = min(self.decay, (1.0 + self.step) / (10.0 + self.step))

        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow:
                continue
            self.shadow[name] = (
                decay * self.shadow[name]
                + (1.0 - decay) * param.data.detach().float().cpu()
            )

    def copy_to(self, model: nn.Module):
        """
        Copia los pesos EMA al modelo dado.
        Usar para guardar checkpoints o para inferencia.
        """
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].to(param.device))

    def state_dict(self) -> dict:
        return {
            "shadow": self.shadow,
            "step": self.step,
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
        }

    @classmethod
    def from_state_dict(cls, state: dict, model: nn.Module) -> "EMAModel":
        """Reconstruye EMAModel desde un state_dict guardado."""
        ema = cls(
            model,
            decay=state["decay"],
            warmup_steps=state.get("warmup_steps", 1000),
        )
        ema.shadow = state["shadow"]
        ema.step = state["step"]
        return ema


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def get_raw_state_dict(model: nn.Module) -> dict:
    """Extrae state_dict de un modelo, sea o no compilado con torch.compile."""
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    return raw.state_dict()


def get_raw_model(model: nn.Module) -> nn.Module:
    """Devuelve el modelo sin wrapper de torch.compile."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def load_config(yaml_path: str, cli_overrides: List[str]):
    schema = OmegaConf.structured(Config)
    file_cfg = OmegaConf.load(yaml_path) if yaml_path else OmegaConf.create({})
    cli_cfg = (
        OmegaConf.from_dotlist(cli_overrides) if cli_overrides else OmegaConf.create({})
    )
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
    parser = argparse.ArgumentParser(
        description="Entrenamiento del diffusion U-Net de galaxias."
    )
    parser.add_argument("--config", type=str, default="configs/train_baseline.yaml")
    args, overrides = parser.parse_known_args()
    return args, overrides


# ══════════════════════════════════════════════════════════════════════════════
# AMP helpers
# ══════════════════════════════════════════════════════════════════════════════


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
            return torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=True
            )
    return torch.cuda.amp.autocast(dtype=torch.float16, enabled=True)


def build_checkpoint(
    unet: nn.Module,
    unet_raw: nn.Module,
    projector: nn.Module,
    ema: EMAModel,
    dataset: GalaxiasFisicasDataset,
    cfg,
    epoch: int,
    mean_loss: float,
    best_loss: float,
) -> dict:
    """Construye un checkpoint completo, incluyendo pesos EMA para inferencia."""
    unet_ema_save = copy.deepcopy(unet_raw)
    ema.copy_to(unet_ema_save)
    return {
        # Pesos para inferencia.
        "unet_ema": unet_ema_save.state_dict(),
        # Pesos para continuar training.
        "unet": get_raw_state_dict(unet),
        "projector": projector.state_dict(),
        "ema_state": ema.state_dict(),
        # Metadatos críticos para inferencia.
        "variables": list(cfg.data.variables),
        "norm_stats": dataset.norm_stats,
        "epoch": epoch,
        "mean_loss": mean_loss,
        "best_loss": best_loss,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Entrenamiento
# ══════════════════════════════════════════════════════════════════════════════


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
    print(f"Config:     {args.config}")
    if overrides:
        print(f"Overrides:  {overrides}")
    print(f"Device:     {device}")
    print("Configuración final:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70)

    OmegaConf.save(cfg, os.path.join(run_dir, "config_used.yaml"))

    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "mean_mse_loss", "best_loss"])

    # ── Dataset y DataLoader ──────────────────────────────────────────────
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

    # ── Scheduler de ruido (DDPM, v-prediction) ───────────────────────────
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.diffusion.num_train_timesteps,
        prediction_type="v_prediction",
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
    )

    # ── Modelo ────────────────────────────────────────────────────────────
    unet = CustomGalaxyUNet(
        n_channels=cfg.model.n_channels,
        n_classes=cfg.model.n_classes,
        embed_dim=cfg.model.embed_dim,
        dropout=cfg.train.dropout,
    ).to(device)

    # Guardamos referencia al modelo SIN compilar para EMA y checkpoints
    unet_raw = unet

    if cfg.train.compile and hasattr(torch, "compile") and device.type == "cuda":
        print("Compilando el modelo con torch.compile()...")
        unet = torch.compile(unet)

    projector = PhysicsProjector(
        input_dim=len(cfg.data.variables),
        embed_dim=cfg.model.embed_dim,
    ).to(device)

    # Contar parámetros
    n_unet = sum(p.numel() for p in unet_raw.parameters()) / 1e6
    n_proj = sum(p.numel() for p in projector.parameters()) / 1e6
    print(
        f"Parámetros: U-Net={n_unet:.1f}M, Projector={n_proj:.1f}M, Total={n_unet + n_proj:.1f}M"
    )

    # ── Optimizador y AMP ─────────────────────────────────────────────────
    optimizer = AdamW(
        list(unet.parameters()) + list(projector.parameters()),
        lr=cfg.train.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    scaler = make_grad_scaler(device, cfg.train.amp)

    # ── EMA ───────────────────────────────────────────────────────────────
    ema = EMAModel(
        unet_raw,
        decay=cfg.train.ema_decay,
        warmup_steps=cfg.train.ema_warmup_steps,
    )
    print(
        f"EMA inicializado: decay={cfg.train.ema_decay}, warmup={cfg.train.ema_warmup_steps} pasos"
    )

    best_loss = float("inf")
    epochs_no_improve = 0

    # ── Bucle de entrenamiento ────────────────────────────────────────────
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
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
            ).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            optimizer.zero_grad(set_to_none=True)

            # ── CFG Dropout con Learnable Null Tokens ─────────────────────
            # drop_mask: True para muestras que usarán null token
            # La red aprende simultáneamente:
            #   - condicionamiento físico completo (85% de los pasos)
            #   - generación incondicional (15% de los pasos)
            drop_mask = torch.rand(bsz, device=device) < cfg.train.cfg_drop_prob

            with autocast_context(device, cfg.train.amp):
                # El projector aplica el null token donde drop_mask es True
                cond_emb = projector(phys_vectors, drop_mask=drop_mask)

                # V-prediction (Salimans & Ho 2022)
                pred = unet(
                    x=noisy_images,
                    cond_emb=cond_emb,
                    timesteps=timesteps,
                )

                # Objetivo v-prediction
                target = noise_scheduler.get_velocity(clean_images, noise, timesteps)

                # ── Min-SNR Weighting (Hang et al. 2023) ──────────────────
                # Atenúa la contribución de timesteps con SNR extremo.
                # Específicamente, limita el peso a max=5.0 para evitar que los
                # timesteps de bajo ruido (fáciles) dominen el gradiente.
                # Ajuste v-prediction: peso = snr / (snr + 1)
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
                a_t = alphas_cumprod[timesteps]
                snr = a_t / (1.0 - a_t)
                snr_weight = torch.clamp(snr, max=5.0) / (snr + 1.0)

                # MSE ponderado por Min-SNR
                loss = nn.functional.mse_loss(pred, target, reduction="none")
                loss = loss.mean(dim=[1, 2, 3])  # media espacial → [B]
                loss = (loss * snr_weight).mean()  # ponderación y media batch

            scaler.scale(loss).backward()
            # Gradient clipping para estabilidad
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(unet.parameters()) + list(projector.parameters()),
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()

            # ── Actualización EMA (cada paso, no cada época) ──────────────
            ema.update(unet_raw)

            epoch_loss += loss.item()
            progress_bar.set_postfix({"MSE": f"{loss.item():.4f}"})

        mean_loss = epoch_loss / len(dataloader)
        print(f"Época {epoch + 1} | MSE Loss Medio: {mean_loss:.6f}")

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{mean_loss:.6f}", f"{best_loss:.6f}"])

        # ── Checkpoint & Early Stopping ───────────────────────────────────
        improved = mean_loss < best_loss - cfg.train.min_delta
        if improved:
            next_best_loss = mean_loss
        else:
            next_best_loss = best_loss

        ckpt = build_checkpoint(
            unet=unet,
            unet_raw=unet_raw,
            projector=projector,
            ema=ema,
            dataset=dataset,
            cfg=cfg,
            epoch=epoch + 1,
            mean_loss=mean_loss,
            best_loss=next_best_loss,
        )
        last_path = os.path.join(ckpt_dir, "last.pt")
        torch.save(ckpt, last_path)

        if cfg.train.save_every > 0 and (epoch + 1) % cfg.train.save_every == 0:
            periodic_path = os.path.join(ckpt_dir, f"modelo_epoca_{epoch + 1:03d}.pt")
            torch.save(ckpt, periodic_path)
            print(f"  → Checkpoint periódico guardado: {periodic_path}")

        if improved:
            best_loss = mean_loss
            epochs_no_improve = 0

            ckpt_path = os.path.join(ckpt_dir, "mejor_modelo.pt")
            torch.save(ckpt, ckpt_path)
            print(
                f"  → Nuevo mejor modelo guardado (EMA): {ckpt_path} (loss: {best_loss:.6f})"
            )

        else:
            epochs_no_improve += 1
            print(
                f"  → Sin mejora. Paciencia: {epochs_no_improve}/{cfg.train.patience}"
            )
            if epochs_no_improve >= cfg.train.patience:
                print(
                    f"\nEarly stopping en época {epoch + 1}. Mejor loss: {best_loss:.6f}"
                )
                break

    print(f"\nEntrenamiento completado. Resultados en: {run_dir}")


if __name__ == "__main__":
    main()
