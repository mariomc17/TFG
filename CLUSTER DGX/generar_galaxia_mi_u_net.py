import argparse
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from diffusers import DDIMScheduler
from omegaconf import OmegaConf

from mi_u_net import CustomGalaxyUNet
from train_diffusion_mi_u_net import PhysicsProjector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuración física

@dataclass
class GalaxySpec:
    etiqueta: str = "galaxia"
    escala_kpc_px: Optional[float] = 0.80  # kpc/px
    log_ms: Optional[float] = 10.0         # log10(M_sol)
    ea_gyr: Optional[float] = 2.0          # Gyr
    radio_p_arcsec: Optional[float] = 10.0 # arcsec

@dataclass
class GenConfig:
    checkpoint: Optional[str] = None
    output_dir: Optional[str] = None
    img_size: int = 128
    inference_steps: int = 50
    guidance_scale: float = 5.0 # Escala CFG
    seed: Optional[int] = 42
    galaxies: List[GalaxySpec] = field(default_factory=list)

_SPEC_TO_VAR = {
    "escala_kpc_px": "ESCALA_KPC_PX",
    "log_ms": "LOG_MS",
    "ea_gyr": "EA",
    "radio_p_arcsec": "RADIO_P",
}

_LOG1P_VARS = frozenset({"EA", "ESCALA_KPC_PX", "RADIO_P"})


def normalize_value(value: float, var_name: str, norm_stats: dict) -> float:
    """Aplica la misma transformación matemática que usó el Dataset durante el training."""
    s = norm_stats[var_name]
    value = float(np.clip(value, s["clip_lo"], s["clip_hi"]))
    if s["log_transform"]:
        value = float(np.log1p(max(value, 0.0)))
    return (value - s["mean"]) / (s["std"] + 1e-8)

def build_phys_vector(spec: GalaxySpec, variables: List[str], norm_stats: dict, null_tokens: torch.Tensor, device: torch.device) -> tuple:
    spec_values: Dict[str, Optional[float]] = {}
    for attr, var_name in _SPEC_TO_VAR.items():
        spec_values[var_name] = getattr(spec, attr, None)

    vec = []
    null_mask = []
    for i, var_name in enumerate(variables):
        val = spec_values.get(var_name, None)
        if val is None:
            # Condicionamiento parcial: inyectamos el Null Token aprendido
            vec.append(null_tokens[i].item())
            null_mask.append(True)
        else:
            vec.append(normalize_value(val, var_name, norm_stats))
            null_mask.append(False)

    phys_vector = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
    return phys_vector, null_mask

@torch.no_grad()
def generate_galaxies_batch(specs: List[GalaxySpec], unet: CustomGalaxyUNet, projector: PhysicsProjector, noise_scheduler: DDIMScheduler, variables: List[str], norm_stats: dict, device: torch.device, guidance_scale: float, seed: Optional[int], img_size: int) -> List[torch.Tensor]:
    """Genera todas las galaxias simultáneamente en la GPU (x10 más rápido)."""
    N = len(specs)
    if N == 0: return []

    unet.eval()
    projector.eval()
    if seed is not None: torch.manual_seed(seed)

    phys_vecs = []
    for spec in specs:
        phys_vec, _ = build_phys_vector(spec, variables, norm_stats, projector.null_tokens, device)
        phys_vecs.append(phys_vec)
    phys_vecs = torch.cat(phys_vecs, dim=0)

    # Obtenemos condicionamiento explícito e incondicional (para CFG)
    cond_embs = projector(phys_vecs)
    uncond_embs = projector.get_null_embedding(N, device)
    combined = torch.cat([uncond_embs, cond_embs], dim=0)

    # Ruido inicial
    latents = torch.randn(N, 3, img_size, img_size, device=device)

    for t in noise_scheduler.timesteps:
        latents_in = torch.cat([latents, latents], dim=0)
        t_batch = t.unsqueeze(0).repeat(2 * N).to(device)

        noise_pred = unet(x=latents_in, cond_emb=combined, timesteps=t_batch)
        
        # Extrapolación CFG
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_guided = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        latents = noise_scheduler.step(noise_guided, t, latents).prev_sample

    return [latents[i] for i in range(N)]

def safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return name.strip("._") or "galaxia"

def load_checkpoint(ckpt_path: str, device: torch.device):
    print(f"Cargando checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    variables = checkpoint["variables"]
    norm_stats = checkpoint["norm_stats"]
    cfg_dict = checkpoint.get("config", {})

    print(f"  Variables recuperadas: {variables}")
    
    model_cfg = cfg_dict.get("model", {})
    embed_dim = model_cfg.get("embed_dim", 256)

    # MEJORA SOTA: cargamos unet_ema
    unet = CustomGalaxyUNet(
        n_channels=model_cfg.get("n_channels", 3),
        n_classes=model_cfg.get("n_classes", 3),
        embed_dim=embed_dim, dropout=0.0
    ).to(device)
    unet.load_state_dict(checkpoint["unet_ema"])
    unet.eval()
    print("Pesos EMA cargados en U-Net")

    projector = PhysicsProjector(input_dim=len(variables), embed_dim=embed_dim).to(device)
    projector.load_state_dict(checkpoint["projector"])
    projector.eval()
    print("PhysicsProjector cargado (con null tokens aprendidos)")

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        prediction_type="v_prediction",
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing"
    )

    return unet, projector, noise_scheduler, variables, norm_stats

# Bucle Principal

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gen_examples.yaml")
    args, overrides = parser.parse_known_args()
    
    schema = OmegaConf.structured(GenConfig)
    file_cfg = OmegaConf.load(args.config) if args.config else OmegaConf.create({})
    cli_cfg = OmegaConf.from_dotlist(overrides) if overrides else OmegaConf.create({})
    cfg = OmegaConf.merge(schema, file_cfg, cli_cfg)

    cfg.output_dir = cfg.output_dir or os.environ.get("OUTPUT_DIR", "galaxias_generadas")
    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.checkpoint is None:
        raise SystemExit("ERROR: falta 'checkpoint'. Indícalo en YAML o como override.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"Device:       {device}")
    print(f"Checkpoint:   {cfg.checkpoint}")
    print(f"Guidance CFG: {cfg.guidance_scale}")
    print(f"Pasos DDIM:   {cfg.inference_steps}")
    print("=" * 70)

    unet, projector, noise_scheduler, variables, norm_stats = load_checkpoint(cfg.checkpoint, device)
    noise_scheduler.set_timesteps(cfg.inference_steps)

    # Materializamos las specs
    specs_lista = [OmegaConf.merge(OmegaConf.structured(GalaxySpec), spec) for spec in cfg.galaxies]
    
    # Generamos en un solo batch
    tensors = generate_galaxies_batch(
        specs_lista, unet, projector, noise_scheduler, 
        variables, norm_stats, device, 
        guidance_scale=cfg.guidance_scale, seed=cfg.seed, img_size=cfg.img_size
    )

    # Guardado con formato Matplotlib solicitado
    for i, (spec, tensor) in enumerate(zip(specs_lista, tensors)):
        img_np = tensor.cpu().float().clamp(-1, 1)
        img_np = (img_np + 1.0) / 2.0
        img_np = img_np.permute(1, 2, 0).numpy()

        param_parts = []
        for attr, var_name in _SPEC_TO_VAR.items():
            val = getattr(spec, attr, None)
            short = var_name.split("_")[0] if "_" in var_name else var_name
            if val is None:
                param_parts.append(f"{short}=—")
            else:
                param_parts.append(f"{short}={val:.2f}")
        subtitle = " | ".join(param_parts)

        plt.figure(figsize=(6, 6), dpi=300)
        plt.imshow(img_np)
        plt.axis('off')
        plt.title(f"{spec.etiqueta}\n{subtitle}", fontsize=16)
        
        out_file = os.path.join(cfg.output_dir, f"{i + 1:02d}_{safe_filename(spec.etiqueta)}.png")
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"  → Guardado: {out_file}")

if __name__ == "__main__":
    main()