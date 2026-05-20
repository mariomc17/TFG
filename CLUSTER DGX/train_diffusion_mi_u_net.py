import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from torch.optim import AdamW
from tqdm import tqdm
import os
import csv
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from typing import List
from contextlib import nullcontext

from omegaconf import OmegaConf

from mi_u_net import CustomGalaxyUNet
from dataset import GalaxiasFisicasDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =====================================================================
# Schema de configuración (dataclasses tipadas).
# Estos son los DEFAULTS — el YAML los sobreescribe, y la CLI sobre el YAML.
# =====================================================================
@dataclass
class DataCfg:
    hdf5_path: str = "dataset_galaxias_sin_rgb.h5"
    img_size: int = 128
    # MEJORA SOTA: añadimos RADIO_P
    variables: List[str] = field(
        default_factory=lambda: ["ESCALA_KPC_PX", "LOG_MS", "EA", "RADIO_P"]
    )
    num_workers: int = 4
    augment: bool = True

@dataclass
class TrainCfg:
    batch_size: int = 32
    epochs: int = 200 # Más épocas porque el proceso es más suave
    lr: float = 1e-4  # Learning rate más bajo y estable
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
    time_dim: int = 256

@dataclass
class DiffusionCfg:
    num_train_timesteps: int = 1000

@dataclass
class Config:
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    diffusion: DiffusionCfg = field(default_factory=DiffusionCfg)

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


# =====================================================================
# Helpers
# =====================================================================
def load_config(yaml_path: str, cli_overrides: List[str]):
    """Combina: defaults del dataclass + YAML + overrides CLI.

    OmegaConf valida tipos al hacer el merge contra el schema, así que
    si pones 'train.lr: hola' en el YAML, falla aquí y no a las 20 épocas.
    """
    schema = OmegaConf.structured(Config)
    file_cfg = OmegaConf.load(yaml_path) if yaml_path else OmegaConf.create({})
    cli_cfg = OmegaConf.from_dotlist(cli_overrides) if cli_overrides else OmegaConf.create({})
    cfg = OmegaConf.merge(schema, file_cfg, cli_cfg)
    return cfg


def setup_run_dir():
    """Devuelve la carpeta de la run actual.

    Si viene RUN_DIR del entorno (lo inyecta submit_train.sh) la usa;
    si no, crea una local con timestamp para no machacar nada.
    """
    run_dir = os.environ.get("RUN_DIR")
    if run_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(SCRIPT_DIR, "runs", "train", f"local_{ts}")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento del diffusion U-Net de galaxias.")
    parser.add_argument(
        "--config", type=str, default="configs/train_baseline.yaml",
        help="Ruta al YAML de configuración.")
    # parse_known_args() devuelve también los args no reconocidos,
    # que serán los overrides estilo 'train.lr=5e-5' para OmegaConf.
    args, overrides = parser.parse_known_args()
    return args, overrides


# =====================================================================
# Compatibilidad AMP entre versiones de PyTorch.
# =====================================================================
def make_grad_scaler(device, enabled: bool):
    """Devuelve un GradScaler compatible con PyTorch viejo y nuevo."""
    use_amp = bool(enabled and device.type == "cuda")

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            return torch.amp.GradScaler(enabled=use_amp)

    if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=use_amp)

    raise RuntimeError("Esta versión de PyTorch no expone GradScaler compatible con AMP.")


def autocast_context(device, enabled: bool):
    """Context manager autocast compatible con PyTorch viejo y nuevo."""
    use_amp = bool(enabled and device.type == "cuda")

    if not use_amp:
        return nullcontext()

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", dtype=torch.float16, enabled=True)
        except TypeError:
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True)

    if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(dtype=torch.float16, enabled=True)

    return nullcontext()


# =====================================================================
# Entrenamiento
# =====================================================================
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
    print("Configuración final tras merges:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70)

    # Snapshot de la config EXACTA usada (con overrides ya aplicados)
    OmegaConf.save(cfg, os.path.join(run_dir, "config_used.yaml"))

    # Cabecera del CSV de métricas
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "mean_mse_loss"])

    # ---------- Datos ----------
    dataset = GalaxiasFisicasDataset(
        hdf5_path=cfg.data.hdf5_path,
        img_size=cfg.data.img_size,
        variables_elegidas=list(cfg.data.variables),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.diffusion.num_train_timesteps,
        prediction_type="v_prediction",
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing"
    )

    # ---------- Modelo ----------
    unet = CustomGalaxyUNet(
        n_channels=cfg.model.n_channels,
        n_classes=cfg.model.n_classes,
        embed_dim=cfg.model.embed_dim,
        time_dim=cfg.model.time_dim,
    ).to(device)

    if cfg.train.compile and hasattr(torch, 'compile') and device.type == 'cuda':
        print("Compilando el modelo con torch.compile()...")
        unet = torch.compile(unet)

    projector = PhysicsProjector(
        input_dim=len(cfg.data.variables),
        embed_dim=cfg.model.embed_dim,
    ).to(device)

    optimizer = AdamW(
        list(unet.parameters()) + list(projector.parameters()),
        lr=cfg.train.lr,
    )
    criterion = nn.MSELoss()

    # AMP compatible con versiones viejas y nuevas de PyTorch
    scaler = make_grad_scaler(device, cfg.train.amp)

    best_loss = float('inf')
    epochs_no_improve = 0

    # ---------- Bucle de entrenamiento ----------
    for epoch in range(cfg.train.epochs):
        unet.train()
        projector.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}")

        for batch in progress_bar:
            clean_images, phys_vectors = batch
            clean_images = clean_images.to(device)
            phys_vectors = phys_vectors.to(device)

            noise = torch.randn_like(clean_images)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bsz,), device=device).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            optimizer.zero_grad(set_to_none=True)

            # --- IMPLEMENTACIÓN DE CFG (15% de probabilidad de condicionamiento nulo) ---
            drop_prob = 0.15
            mask = torch.rand(bsz, device=device) < drop_prob
            phys_vectors_cfg = phys_vectors.clone()
            phys_vectors_cfg[mask] = 0.0 # Vector nulo para aprender generación incondicional

            with autocast_context(device, cfg.train.amp):
                encoder_hidden_states = projector(phys_vectors_cfg)
                # La U-Net ahora predice la velocidad 'v', no el ruido 'epsilon'
                pred = unet(
                    x=noisy_images,
                    context=encoder_hidden_states,
                    timesteps=timesteps,
                )
                
                # Objetivo de v-prediction
                target = noise_scheduler.get_velocity(clean_images, noise, timesteps)
                
                # --- IMPLEMENTACIÓN MIN-SNR WEIGHTING ---
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
                a_t = alphas_cumprod[timesteps]
                snr = a_t / (1 - a_t)
                snr_weight = torch.clamp(snr, max=5.0) # Recortamos a 5.0
                # Ajuste específico para v-prediction:
                snr_weight = snr_weight / (snr + 1)
                
                # MSE Ponderado
                loss = nn.functional.mse_loss(pred, target, reduction="none")
                # Promediamos sobre canales/pixeles, mantenemos batch
                loss = loss.mean(dim=[1, 2, 3]) 
                loss = (loss * snr_weight).mean() # Ponderación y media final

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"MSE Loss": f"{loss.item():.4f}"})

        mean_loss = epoch_loss / len(dataloader)
        print(f"Época {epoch+1} terminada | MSE Loss Medio: {mean_loss:.4f}")

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{mean_loss:.6f}"])

        # ---------- Checkpoint & Early Stopping ----------
        if mean_loss < best_loss - cfg.train.min_delta:
            best_loss = mean_loss
            epochs_no_improve = 0
            unet_state = (
                unet._orig_mod.state_dict()
                if hasattr(unet, '_orig_mod')
                else unet.state_dict()
            )
            ckpt_name = "mejor_modelo.pt"
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            torch.save({
                'unet': unet_state,
                'projector': projector.state_dict(),
                'variables': list(cfg.data.variables),
                'epoch': epoch + 1,
                'mean_loss': mean_loss,
                'config': OmegaConf.to_container(cfg, resolve=True),
            }, ckpt_path)
            print(f"  -> Nuevo mejor modelo guardado: {ckpt_path} (MSE Loss: {best_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"  -> Sin mejora. Paciencia: {epochs_no_improve}/{cfg.train.patience}")
            if epochs_no_improve >= cfg.train.patience:
                print(f"\nEarly stopping disparado en la época {epoch+1}. Mejor loss: {best_loss:.6f}")
                break

    print(f"\nEntrenamiento completado. Resultados en: {run_dir}")


if __name__ == "__main__":
    main()
