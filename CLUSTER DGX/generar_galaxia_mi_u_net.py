import argparse
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from diffusers import DDIMScheduler

from mi_u_net import CustomGalaxyUNet
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from train_diffusion_mi_u_net import PhysicsProjector

@dataclass
class GalaxySpec:
    etiqueta: str = "galaxia"

    escala_kpc_px: Optional[float] = 0.80  # kpc / pixel
    log_ms: Optional[float] = 10.0  # log10(M_sol)
    ea_gyr: Optional[float] = 2.0  # Edad estelar [Gyr]
    radio_p_arcsec: Optional[float] = 10.0  # Radio Petrosian [arcsec]


@dataclass
class GenConfig:
    checkpoint: Optional[str] = None
    output_dir: Optional[str] = None
    img_size: int = 128
    inference_steps: int = 50
    guidance_scale: float = 7.5
    mode: str = "examples"
    sweep_var: str = "log_ms"
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
    s = norm_stats[var_name]
    value = float(np.clip(value, s["clip_lo"], s["clip_hi"]))
    if s["log_transform"]:
        value = float(np.log1p(max(value, 0.0)))
    return (value - s["mean"]) / (s["std"] + 1e-8)


def build_phys_vector(
    spec: GalaxySpec,
    variables: List[str],
    norm_stats: dict,
    null_tokens: torch.Tensor,
    device: torch.device,
) -> tuple:

    spec_values: Dict[str, Optional[float]] = {}
    for attr, var_name in _SPEC_TO_VAR.items():
        spec_values[var_name] = getattr(spec, attr, None)

    vec = []
    null_mask = []
    for i, var_name in enumerate(variables):
        val = spec_values.get(var_name, None)
        if val is None:
            vec.append(null_tokens[i].item())
            null_mask.append(True)
        else:
            vec.append(normalize_value(val, var_name, norm_stats))
            null_mask.append(False)

    phys_vector = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
    return phys_vector, null_mask

@torch.no_grad()
def generate_galaxy(
    spec: GalaxySpec,
    unet: CustomGalaxyUNet,
    projector: PhysicsProjector,
    noise_scheduler: DDIMScheduler,
    variables: List[str],
    norm_stats: dict,
    device: torch.device,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    img_size: int = 128,
) -> torch.Tensor:

    unet.eval()
    projector.eval()

    if seed is not None:
        torch.manual_seed(seed)

    phys_vector, null_mask = build_phys_vector(
        spec, variables, norm_stats, projector.null_tokens, device
    )

    cond_emb = projector(phys_vector)  
    uncond_emb = projector.get_null_embedding(1, device)  

    cond_combined = torch.cat([uncond_emb, cond_emb], dim=0)

    latent = torch.randn(1, 3, img_size, img_size, device=device)

    for t in noise_scheduler.timesteps:
        latent_input = torch.cat([latent, latent], dim=0)
        t_batch = t.unsqueeze(0).repeat(2).to(device) 

        noise_pred = unet(
            x=latent_input,
            cond_emb=cond_combined,
            timesteps=t_batch,
        )

        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        latent = noise_scheduler.step(noise_pred, t, latent).prev_sample

    return latent.squeeze(0)


@torch.no_grad()
def generate_galaxies_batch(
    specs: List[GalaxySpec],
    unet: CustomGalaxyUNet,
    projector: PhysicsProjector,
    noise_scheduler: DDIMScheduler,
    variables: List[str],
    norm_stats: dict,
    device: torch.device,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    img_size: int = 128,
) -> List[torch.Tensor]:

    N = len(specs)
    if N == 0:
        return []

    unet.eval()
    projector.eval()

    if seed is not None:
        torch.manual_seed(seed)

    phys_vecs = []
    for spec in specs:
        phys_vec, _ = build_phys_vector(
            spec, variables, norm_stats, projector.null_tokens, device
        )
        phys_vecs.append(phys_vec)
    phys_vecs = torch.cat(phys_vecs, dim=0)

    cond_embs = projector(phys_vecs)
    uncond_embs = projector.get_null_embedding(N, device)

    combined = torch.cat([uncond_embs, cond_embs], dim=0)

    latents = torch.randn(N, 3, img_size, img_size, device=device)

    for t in noise_scheduler.timesteps:
        latents_in = torch.cat([latents, latents], dim=0)  
        t_batch = t.unsqueeze(0).repeat(2 * N).to(device)

        noise_pred = unet(x=latents_in, cond_emb=combined, timesteps=t_batch)

        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_guided = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        latents = noise_scheduler.step(
            noise_guided, t, latents
        ).prev_sample 

    return [latents[i] for i in range(N)]


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    img = tensor.cpu().float().clamp(-1, 1)
    img = (img + 1.0) / 2.0  # → [0, 1]
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


def annotate_image(
    img: Image.Image,
    spec: GalaxySpec,
    variables: List[str],
) -> Image.Image:
    W, H = img.size
    margin = 40
    new_img = Image.new("RGB", (W, H + margin), color=(20, 20, 20))
    new_img.paste(img, (0, margin))
    draw = ImageDraw.Draw(new_img)

    title = spec.etiqueta
    param_parts = []
    for attr, var_name in _SPEC_TO_VAR.items():
        val = getattr(spec, attr, None)
        short = var_name.split("_")[0] if "_" in var_name else var_name
        if val is None:
            param_parts.append(f"{short}=—")
        else:
            param_parts.append(f"{short}={val:.2f}")
    subtitle = " | ".join(param_parts)

    try:
        font_title = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11
        )
        font_sub = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9
        )
    except IOError:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    draw.text((4, 2), title, font=font_title, fill=(230, 230, 230))
    draw.text((4, 16), subtitle, font=font_sub, fill=(180, 180, 180))

    return new_img


def make_grid(images: List[Image.Image], cols: int = 4) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    W, H = images[0].size
    grid = Image.new("RGB", (W * cols, H * rows), color=(10, 10, 10))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(img, (c * W, r * H))
    return grid


def safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return name.strip("._") or "galaxia"


def load_checkpoint(ckpt_path: str, device: torch.device):

    print(f"Cargando checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    variables = checkpoint["variables"]
    norm_stats = checkpoint["norm_stats"]
    cfg_dict = checkpoint.get("config", {})

    print(f"  Variables: {variables}")
    print(f"  Época guardada: {checkpoint.get('epoch', '?')}")
    mean_loss = checkpoint.get("mean_loss")
    if mean_loss is None:
        print("  Loss guardada: ?")
    else:
        print(f"  Loss guardada: {float(mean_loss):.6f}")

    model_cfg = cfg_dict.get("model", {})
    embed_dim = model_cfg.get("embed_dim", 256)
    dropout = cfg_dict.get("train", {}).get("dropout", 0.0)

    unet = CustomGalaxyUNet(
        n_channels=model_cfg.get("n_channels", 3),
        n_classes=model_cfg.get("n_classes", 3),
        embed_dim=embed_dim,
        dropout=dropout,
    ).to(device)

    unet.load_state_dict(checkpoint["unet_ema"])
    unet.eval()
    print("Pesos EMA cargados en U-Net")

    projector = PhysicsProjector(
        input_dim=len(variables),
        embed_dim=embed_dim,
    ).to(device)

    projector.load_state_dict(checkpoint["projector"])
    projector.eval()
    print("PhysicsProjector cargado (con null_tokens aprendidos)")

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=cfg_dict.get("diffusion", {}).get(
            "num_train_timesteps", 1000
        ),
        prediction_type="v_prediction",
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
    )

    return unet, projector, noise_scheduler, variables, norm_stats

def run_examples(unet, projector, noise_scheduler, variables, norm_stats, device, args):
    ejemplo_specs = (
        list(args.galaxies)
        if args.galaxies
        else [
            GalaxySpec(
                "espiral_masiva_vieja",
                log_ms=10.8,
                ea_gyr=7.0,
                escala_kpc_px=0.5,
                radio_p_arcsec=18.0,
            ),
            GalaxySpec(
                "espiral_media_joven",
                log_ms=9.5,
                ea_gyr=1.5,
                escala_kpc_px=0.8,
                radio_p_arcsec=10.0,
            ),
            GalaxySpec(
                "espiral_difusa_activa",
                log_ms=9.0,
                ea_gyr=0.8,
                escala_kpc_px=1.2,
                radio_p_arcsec=8.0,
            ),
            GalaxySpec(
                "compacta_masiva",
                log_ms=11.0,
                ea_gyr=8.0,
                escala_kpc_px=0.2,
                radio_p_arcsec=5.0,
            ),
            GalaxySpec(
                "distante_tipica",
                log_ms=10.2,
                ea_gyr=3.0,
                escala_kpc_px=1.8,
                radio_p_arcsec=6.0,
            ),
            GalaxySpec(
                "cercana_extendida",
                log_ms=9.8,
                ea_gyr=4.0,
                escala_kpc_px=0.3,
                radio_p_arcsec=25.0,
            ),
            GalaxySpec(
                "solo_masa_alta",
                log_ms=11.0,
                ea_gyr=None,
                escala_kpc_px=None,
                radio_p_arcsec=None,
            ),
            GalaxySpec(
                "solo_masa_baja",
                log_ms=8.5,
                ea_gyr=None,
                escala_kpc_px=None,
                radio_p_arcsec=None,
            ),
            GalaxySpec(
                "masa_y_edad_joven",
                log_ms=10.0,
                ea_gyr=0.5,
                escala_kpc_px=None,
                radio_p_arcsec=None,
            ),
            GalaxySpec(
                "masa_y_edad_vieja",
                log_ms=10.0,
                ea_gyr=9.0,
                escala_kpc_px=None,
                radio_p_arcsec=None,
            ),
            GalaxySpec(
                "incondicional_1",
                log_ms=None,
                ea_gyr=None,
                escala_kpc_px=None,
                radio_p_arcsec=None,
            ),
            GalaxySpec(
                "incondicional_2",
                log_ms=None,
                ea_gyr=None,
                escala_kpc_px=None,
                radio_p_arcsec=None,
            ),
        ]
    )

    imagenes = []
    print(
        f"Generando {len(ejemplo_specs)} galaxias en batch (guidance={args.guidance_scale}, pasos={args.inference_steps})..."
    )
    t0 = __import__("time").time()

    tensors = generate_galaxies_batch(
        ejemplo_specs,
        unet,
        projector,
        noise_scheduler,
        variables,
        norm_stats,
        device,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        img_size=args.img_size,
    )

    elapsed = __import__("time").time() - t0
    print(
        f"Generación completada en {elapsed:.1f}s ({elapsed / len(ejemplo_specs):.1f}s/galaxia)"
    )

    for i, (spec, tensor) in enumerate(zip(ejemplo_specs, tensors)):
        img = tensor_to_pil(tensor)
        img = annotate_image(img, spec, variables)
        out_file = os.path.join(
            args.output_dir,
            f"{i + 1:02d}_{safe_filename(spec.etiqueta)}.png",
        )
        os.makedirs(args.output_dir, exist_ok=True)
        img.save(out_file)
        print(f"  → Guardado: {out_file}")
        imagenes.append(img)

    os.makedirs(args.output_dir, exist_ok=True)
    grid = make_grid(imagenes, cols=4)
    out_path = os.path.join(args.output_dir, "ejemplos_galaxias.png")
    grid.save(out_path)
    print(f"\nGrid guardado en: {out_path}")


def run_sweep(unet, projector, noise_scheduler, variables, norm_stats, device, args):
    sweep_var = args.sweep_var
    sweep_range_map = {
        "log_ms": (8.0, 12.0, 8),
        "ea_gyr": (0.5, 10.0, 8),
        "escala_kpc_px": (0.15, 2.5, 8),
        "radio_p_arcsec": (3.0, 30.0, 8),
    }

    if sweep_var not in sweep_range_map:
        print(f"sweep_var debe ser uno de: {list(sweep_range_map.keys())}")
        return

    lo, hi, n = sweep_range_map[sweep_var]
    valores = np.linspace(lo, hi, n)

    sweep_specs = []
    for v in valores:
        kwargs = {
            "log_ms": 10.0,
            "ea_gyr": 3.0,
            "escala_kpc_px": 0.8,
            "radio_p_arcsec": 10.0,
        }
        kwargs[sweep_var] = float(v)
        sweep_specs.append(GalaxySpec(etiqueta=f"{sweep_var}={v:.2f}", **kwargs))

    print(f"Sweep de '{sweep_var}': {lo} → {hi} en {n} pasos, generando en batch...")
    t0 = __import__("time").time()

    tensors = generate_galaxies_batch(
        sweep_specs,
        unet,
        projector,
        noise_scheduler,
        variables,
        norm_stats,
        device,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        img_size=args.img_size,
    )

    elapsed = __import__("time").time() - t0
    print(f"Sweep completado en {elapsed:.1f}s")

    imagenes = [
        annotate_image(tensor_to_pil(tensor), spec, variables)
        for spec, tensor in zip(sweep_specs, tensors)
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    grid = make_grid(imagenes, cols=n)
    out_path = os.path.join(args.output_dir, f"sweep_{sweep_var}.png")
    grid.save(out_path)
    print(f"Sweep guardado en: {out_path}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generación de galaxias con CFG + DDIM."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gen_examples.yaml",
        help="YAML con checkpoint, pasos DDIM, guidance y lista 'galaxies'.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Ruta al checkpoint .pt (debe contener 'unet_ema' y 'norm_stats')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="galaxias_generadas",
        help="Directorio de salida para las imágenes",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Escala CFG. 1.0=sin guidance, 7.5=equilibrio calidad/diversidad",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="examples",
        choices=["examples", "sweep"],
        help="Modo: 'examples' genera un set de specs; 'sweep' barre un parámetro",
    )
    parser.add_argument(
        "--sweep_var",
        type=str,
        default="log_ms",
        choices=["log_ms", "ea_gyr", "escala_kpc_px", "radio_p_arcsec"],
        help="Variable a barrer en modo sweep",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad (None para aleatoria)",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=None,
        help="Alias CLI de inference_steps.",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def load_config(yaml_path: str, cli_overrides: List[str]):
    schema = OmegaConf.structured(GenConfig)
    file_cfg = OmegaConf.load(yaml_path) if yaml_path else OmegaConf.create({})
    cli_cfg = (
        OmegaConf.from_dotlist(cli_overrides) if cli_overrides else OmegaConf.create({})
    )
    return OmegaConf.merge(schema, file_cfg, cli_cfg)


def resolve_config(args, overrides):
    cfg = load_config(args.config, overrides)

    if args.checkpoint is not None:
        cfg.checkpoint = args.checkpoint
    if args.output_dir != "galaxias_generadas":
        cfg.output_dir = args.output_dir
    if args.guidance_scale != 7.5:
        cfg.guidance_scale = args.guidance_scale
    if args.mode != "examples":
        cfg.mode = args.mode
    if args.sweep_var != "log_ms":
        cfg.sweep_var = args.sweep_var
    if args.seed != 42:
        cfg.seed = args.seed
    if args.n_steps is not None:
        cfg.inference_steps = args.n_steps

    if cfg.output_dir is None:
        cfg.output_dir = os.environ.get("OUTPUT_DIR", "galaxias_generadas")

    if cfg.checkpoint is None:
        raise SystemExit(
            "ERROR: falta 'checkpoint'. Indícalo en el YAML, con "
            "--checkpoint o como override: checkpoint=/ruta/modelo.pt"
        )

    cfg.galaxies = [
        OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(GalaxySpec), spec))
        for spec in cfg.galaxies
    ]
    return cfg


def main():
    args, overrides = parse_args()
    cfg = resolve_config(args, overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(cfg.output_dir, exist_ok=True)
    parent_dir = (
        os.path.dirname(cfg.output_dir)
        if os.path.basename(cfg.output_dir) == "images"
        else cfg.output_dir
    )
    os.makedirs(parent_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(parent_dir, "config_used.yaml"))

    print("=" * 70)
    print(f"Dispositivo: {device}")
    print(f"Config:      {args.config}")
    if overrides:
        print(f"Overrides:   {overrides}")
    print(f"Checkpoint:  {cfg.checkpoint}")
    print(f"Output dir:  {cfg.output_dir}")
    print(f"Modo:        {cfg.mode}")
    print(f"Pasos DDIM:  {cfg.inference_steps}")
    print(f"Guidance:    {cfg.guidance_scale}")
    print("=" * 70)

    unet, projector, noise_scheduler, variables, norm_stats = load_checkpoint(
        cfg.checkpoint, device
    )

    noise_scheduler.set_timesteps(cfg.inference_steps)

    if cfg.mode == "examples":
        run_examples(
            unet, projector, noise_scheduler, variables, norm_stats, device, cfg
        )
    elif cfg.mode == "sweep":
        run_sweep(unet, projector, noise_scheduler, variables, norm_stats, device, cfg)
    else:
        raise SystemExit("ERROR: mode debe ser 'examples' o 'sweep'.")


if __name__ == "__main__":
    main()
