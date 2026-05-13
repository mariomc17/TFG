import torch
import matplotlib.pyplot as plt
import os
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from omegaconf import OmegaConf
from diffusers import DDIMScheduler
from tqdm import tqdm

from mi_u_net import CustomGalaxyUNet
# Reutilizamos PhysicsProjector del script de entrenamiento (no se duplica)
from train_diffusion_mi_u_net import PhysicsProjector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =====================================================================
# Schema del config de generación.
# =====================================================================
@dataclass
class GalaxySpec:
    etiqueta: str = "galaxia"
    escala: float = 0.5
    masa: float = 0.5
    sfr: float = 0.5
    ea: float = 0.5


@dataclass
class GenConfig:
    # Ruta al .pt. Si es null, submit_gen.sh debe pasarla por CLI override.
    checkpoint: Optional[str] = None
    # Tamaño que debe coincidir con el de entrenamiento.
    # Se autocompletará desde el checkpoint si está disponible.
    img_size: int = 128
    inference_steps: int = 50
    galaxies: List[GalaxySpec] = field(default_factory=list)


# =====================================================================
# Helpers
# =====================================================================
def load_config(yaml_path, cli_overrides):
    schema = OmegaConf.structured(GenConfig)
    file_cfg = OmegaConf.load(yaml_path) if yaml_path else OmegaConf.create({})
    cli_cfg = OmegaConf.from_dotlist(cli_overrides) if cli_overrides else OmegaConf.create({})
    cfg = OmegaConf.merge(schema, file_cfg, cli_cfg)
    return cfg


def resolve_output_dir():
    """Prioridad: env var OUTPUT_DIR > cwd."""
    return os.environ.get("OUTPUT_DIR", os.getcwd())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generar galaxias condicionadas por física (batch).")
    parser.add_argument(
        "--config", type=str, default="configs/gen_examples.yaml",
        help="YAML con checkpoint, pasos y la lista 'galaxies'.")
    args, overrides = parser.parse_known_args()
    return args, overrides


# =====================================================================
# Lógica de generación
# =====================================================================
def generar_una(unet, projector, scheduler, device, img_size, spec, output_dir):
    image = torch.randn((1, 3, img_size, img_size)).to(device)
    phys_vector = torch.tensor(
        [[spec.escala, spec.masa, spec.sfr, spec.ea]],
        dtype=torch.float32,
    ).to(device)

    with torch.inference_mode():
        encoder_hidden_states = projector(phys_vector)
        for t in tqdm(scheduler.timesteps, desc=f"Esculpiendo {spec.etiqueta}", leave=False):
            t_batch = torch.tensor([t], dtype=torch.long, device=device)
            noise_pred = unet(
                x=image, context=encoder_hidden_states, timesteps=t_batch)
            image = scheduler.step(noise_pred, t, image).prev_sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(
        f"{spec.etiqueta}\n"
        f"Esc={spec.escala:.2f} | M={spec.masa:.2f} "
        f"| SFR={spec.sfr:.2f} | EA={spec.ea:.2f}"
    )
    nombre_archivo = (
        f"{spec.etiqueta}__esc{spec.escala:.2f}_m{spec.masa:.2f}"
        f"_sfr{spec.sfr:.2f}_ea{spec.ea:.2f}.png"
    )
    ruta = os.path.join(output_dir, nombre_archivo)
    plt.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Guardado: {ruta}")


def main():
    args, overrides = parse_args()
    cfg = load_config(args.config, overrides)

    if cfg.checkpoint is None:
        raise SystemExit(
            "ERROR: no se ha especificado 'checkpoint'. "
            "Pásalo en el YAML o vía CLI: 'checkpoint=/ruta/al/modelo.pt'"
        )
    if not cfg.galaxies:
        raise SystemExit(
            "ERROR: la lista 'galaxies' del YAML está vacía. No hay nada que generar."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        torch.set_num_threads(os.cpu_count() or 4)

    output_dir = resolve_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"Device:        {device}")
    print(f"Checkpoint:    {cfg.checkpoint}")
    print(f"Pasos DDIM:    {cfg.inference_steps}")
    print(f"# de galaxias: {len(cfg.galaxies)}")
    print(f"Output dir:    {output_dir}")
    print("=" * 70)

    # ---- Cargar checkpoint ----
    checkpoint = torch.load(cfg.checkpoint, map_location=device, weights_only=False)

    # Si el checkpoint guardó config, recuperamos img_size y dims de modelo
    ckpt_cfg = checkpoint.get('config', {}) or {}
    data_cfg = ckpt_cfg.get('data', {}) if isinstance(ckpt_cfg, dict) else {}
    model_cfg = ckpt_cfg.get('model', {}) if isinstance(ckpt_cfg, dict) else {}

    img_size = data_cfg.get('img_size', cfg.img_size)
    embed_dim = model_cfg.get('embed_dim', 256)
    time_dim = model_cfg.get('time_dim', 256)
    n_channels = model_cfg.get('n_channels', 3)
    n_classes = model_cfg.get('n_classes', 3)

    print(f"Modelo: {img_size}x{img_size}, embed_dim={embed_dim}, time_dim={time_dim}")

    unet = CustomGalaxyUNet(
        n_channels=n_channels, n_classes=n_classes,
        embed_dim=embed_dim, time_dim=time_dim,
    ).to(device)
    projector = PhysicsProjector(input_dim=4, embed_dim=embed_dim).to(device)

    unet.load_state_dict(checkpoint['unet'])
    projector.load_state_dict(checkpoint['projector'])
    unet.eval()
    projector.eval()

    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(cfg.inference_steps)

    # Guarda la config efectiva al lado de las imágenes
    parent_dir = os.path.dirname(output_dir) if output_dir.endswith("images") else output_dir
    OmegaConf.save(cfg, os.path.join(parent_dir, "config_used.yaml"))

    # ---- Iteración sobre galaxias ----
    print(f"\nGenerando {len(cfg.galaxies)} galaxias...\n")
    for i, spec_dict in enumerate(cfg.galaxies, start=1):
        # Cada spec viene como DictConfig (OmegaConf); lo materializamos a dataclass
        spec = OmegaConf.merge(OmegaConf.structured(GalaxySpec), spec_dict)
        print(f"[{i}/{len(cfg.galaxies)}] {spec.etiqueta}")
        generar_una(unet, projector, scheduler, device,
                    img_size, spec, output_dir)

    print(f"\nListo. {len(cfg.galaxies)} imágenes en {output_dir}")


if __name__ == "__main__":
    main()
