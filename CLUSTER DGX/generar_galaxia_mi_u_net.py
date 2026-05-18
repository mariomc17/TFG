"""
generar_galaxia_mi_u_net.py — Inferencia DDIM con CFG para el diffusion U-Net.

Cambios respecto a la versión anterior:
  ✓ Carga pesos EMA ('unet_ema') del checkpoint, no los pesos brutos ('unet')
  ✓ Usa projector.get_null_embedding() para el embedding incondicional (CFG correcto)
  ✓ Lee norm_stats del checkpoint para normalizar los inputs físicos
  ✓ GalaxySpec actualizado a valores físicos reales (unidades naturales, no min-max)
  ✓ Condicionamiento parcial: se puede especificar un subconjunto de variables;
    el resto se rellena con null_tokens para que la red lo ignore
  ✓ guidance_scale como parámetro configurable (no hardcoded)
  ✓ Generación por grids (barrido de un parámetro) para análisis de sensibilidad

Física de los parámetros (unidades reales):
    escala_kpc_px : kpc / pixel           (típico: 0.13 – 2.85 kpc/px)
    log_ms        : log10(M_sol)          (típico: 8.0 – 12.0)
    ea_gyr        : Edad estelar en Gyr   (típico: 0.5 – 10 Gyr)
    radio_p_arcsec: Radio Petrosian en "  (típico: 3 – 30 arcsec)
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from diffusers import DDIMScheduler

# ── Imports locales ───────────────────────────────────────────────────────────
from mi_u_net import CustomGalaxyUNet
from PIL import Image, ImageDraw, ImageFont
from train_diffusion_mi_u_net import PhysicsProjector

# ══════════════════════════════════════════════════════════════════════════════
# Especificación de una galaxia a generar
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class GalaxySpec:
    """
    Especificación física de una galaxia a generar.

    Todos los valores están en sus unidades físicas naturales, NO en min-max.
    El script se encarga de la normalización usando los stats del checkpoint.

    Condicionamiento parcial: cualquier campo puede pasarse como None para
    que la red use el null_token de esa dimensión (no incluye información
    de esa variable). Esto permite, por ejemplo, generar galaxias fijando
    solo la masa y dejando que la red "decida" el resto.
    """

    etiqueta: str = "galaxia"

    # ── Variables físicas (None = usar null token para esa dimensión) ─────
    escala_kpc_px: Optional[float] = 0.80  # kpc / pixel
    log_ms: Optional[float] = 10.0  # log10(M_sol)
    ea_gyr: Optional[float] = 2.0  # Edad estelar [Gyr]
    radio_p_arcsec: Optional[float] = 10.0  # Radio Petrosian [arcsec]


# ══════════════════════════════════════════════════════════════════════════════
# Normalización usando los stats del checkpoint
# ══════════════════════════════════════════════════════════════════════════════

# Correspondencia entre atributos de GalaxySpec y nombres de variable en el HDF5
_SPEC_TO_VAR = {
    "escala_kpc_px": "ESCALA_KPC_PX",
    "log_ms": "LOG_MS",
    "ea_gyr": "EA",
    "radio_p_arcsec": "RADIO_P",
}

# Variables que requieren log1p antes del Z-score (debe coincidir con dataset.py)
_LOG1P_VARS = frozenset({"EA", "ESCALA_KPC_PX", "RADIO_P"})


def normalize_value(value: float, var_name: str, norm_stats: dict) -> float:
    """
    Normaliza un valor físico usando los estadísticos del checkpoint.

    Orden idéntico al de dataset.py:
      1. Clip al rango [clip_lo, clip_hi] del training data
      2. log1p si corresponde
      3. Z-score
    """
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
    """
    Construye el vector físico normalizado a partir de un GalaxySpec.

    Para variables con valor None, inserta el null_token aprendido de esa
    dimensión. Devuelve también una máscara booleana indicando qué dimensiones
    son nulas (para el cálculo del prompt en el CFG).

    Returns:
        phys_vector : [1, N_vars]  tensor normalizado
        null_mask   : [N_vars]     True donde se usa null_token
    """
    # Mapa de variable → valor de spec
    spec_values: Dict[str, Optional[float]] = {}
    for attr, var_name in _SPEC_TO_VAR.items():
        spec_values[var_name] = getattr(spec, attr, None)

    vec = []
    null_mask = []
    for i, var_name in enumerate(variables):
        val = spec_values.get(var_name, None)
        if val is None:
            # Null token en espacio crudo (antes de normalización)
            # El null_tokens[i] ya está en el mismo espacio normalizado
            vec.append(null_tokens[i].item())
            null_mask.append(True)
        else:
            vec.append(normalize_value(val, var_name, norm_stats))
            null_mask.append(False)

    phys_vector = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
    return phys_vector, null_mask


# ══════════════════════════════════════════════════════════════════════════════
# Generación DDIM con CFG
# ══════════════════════════════════════════════════════════════════════════════


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
    """
    Genera una galaxia con DDIM y CFG.

    Args:
        spec            : especificación física de la galaxia
        unet            : modelo U-Net con pesos EMA cargados
        projector       : PhysicsProjector con null_tokens cargados
        noise_scheduler : DDIMScheduler configurado
        variables       : lista de nombres de variables (del checkpoint)
        norm_stats      : estadísticos de normalización (del checkpoint)
        device          : dispositivo de computación
        guidance_scale  : escala de guidance libre de clasificador
                          1.0 = sin guidance; 7.5 = punto de equilibrio calidad/diversidad
        seed            : semilla para reproducibilidad
        img_size        : tamaño de la imagen

    Returns:
        Tensor [3, img_size, img_size] en el rango [-1, 1]
    """
    unet.eval()
    projector.eval()

    if seed is not None:
        torch.manual_seed(seed)

    # ── Vector físico condicional ──────────────────────────────────────────
    phys_vector, null_mask = build_phys_vector(
        spec, variables, norm_stats, projector.null_tokens, device
    )

    # ── Embeddings para CFG ────────────────────────────────────────────────
    cond_emb = projector(phys_vector)  # [1, D]
    uncond_emb = projector.get_null_embedding(1, device)  # [1, D]

    # Batch de 2 para CFG eficiente (una pasada por el UNet)
    cond_combined = torch.cat([uncond_emb, cond_emb], dim=0)  # [2, D]

    # ── Ruido inicial ─────────────────────────────────────────────────────
    latent = torch.randn(1, 3, img_size, img_size, device=device)

    # ── Loop de denoising DDIM ────────────────────────────────────────────
    noise_scheduler.set_timesteps(50)

    for t in noise_scheduler.timesteps:
        latent_input = torch.cat([latent, latent], dim=0)  # [2, 3, H, W]
        t_batch = t.unsqueeze(0).repeat(2).to(device)  # [2]

        noise_pred = unet(
            x=latent_input,
            cond_emb=cond_combined,
            timesteps=t_batch,
        )

        # Guidance libre de clasificador (Classifier-Free Guidance)
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        latent = noise_scheduler.step(noise_pred, t, latent).prev_sample

    return latent.squeeze(0)


# ══════════════════════════════════════════════════════════════════════════════
# Conversión a imagen PIL
# ══════════════════════════════════════════════════════════════════════════════


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convierte tensor [3, H, W] en rango [-1, 1] a imagen PIL RGB."""
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
    """Añade una anotación con los parámetros físicos de la galaxia."""
    W, H = img.size
    margin = 40
    new_img = Image.new("RGB", (W, H + margin), color=(20, 20, 20))
    new_img.paste(img, (0, margin))
    draw = ImageDraw.Draw(new_img)

    title = spec.etiqueta
    # Construir línea de parámetros con valores reales
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
    """Crea un grid de imágenes PIL."""
    rows = (len(images) + cols - 1) // cols
    W, H = images[0].size
    grid = Image.new("RGB", (W * cols, H * rows), color=(10, 10, 10))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(img, (c * W, r * H))
    return grid


# ══════════════════════════════════════════════════════════════════════════════
# Carga del checkpoint
# ══════════════════════════════════════════════════════════════════════════════


def load_checkpoint(ckpt_path: str, device: torch.device):
    """
    Carga el checkpoint y devuelve los modelos y metadatos.

    Carga los pesos EMA (unet_ema), no los pesos brutos (unet),
    ya que el EMA produce imágenes de mayor calidad.

    Returns:
        unet, projector, noise_scheduler, variables, norm_stats
    """
    print(f"Cargando checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Extraer metadatos del checkpoint
    variables = checkpoint["variables"]
    norm_stats = checkpoint["norm_stats"]
    cfg_dict = checkpoint.get("config", {})

    print(f"  Variables: {variables}")
    print(f"  Época guardada: {checkpoint.get('epoch', '?')}")
    print(f"  Mejor loss: {checkpoint.get('mean_loss', '?'):.6f}")

    # Reconstruir configuración del modelo
    model_cfg = cfg_dict.get("model", {})
    embed_dim = model_cfg.get("embed_dim", 256)
    dropout = cfg_dict.get("train", {}).get("dropout", 0.0)

    # ── U-Net (pesos EMA) ─────────────────────────────────────────────────
    unet = CustomGalaxyUNet(
        n_channels=model_cfg.get("n_channels", 3),
        n_classes=model_cfg.get("n_classes", 3),
        embed_dim=embed_dim,
        dropout=dropout,  # dropout=0 en inferencia; nn.Dropout lo maneja con eval()
    ).to(device)

    unet.load_state_dict(checkpoint["unet_ema"])
    unet.eval()
    print("  ✓ Pesos EMA cargados en U-Net")

    # ── Projector ─────────────────────────────────────────────────────────
    projector = PhysicsProjector(
        input_dim=len(variables),
        embed_dim=embed_dim,
    ).to(device)

    projector.load_state_dict(checkpoint["projector"])
    projector.eval()
    print("  ✓ PhysicsProjector cargado (con null_tokens aprendidos)")

    # ── Scheduler DDIM ────────────────────────────────────────────────────
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=cfg_dict.get("diffusion", {}).get(
            "num_train_timesteps", 1000
        ),
        prediction_type="v_prediction",
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
    )

    return unet, projector, noise_scheduler, variables, norm_stats


# ══════════════════════════════════════════════════════════════════════════════
# Modos de generación
# ══════════════════════════════════════════════════════════════════════════════


def run_examples(unet, projector, noise_scheduler, variables, norm_stats, device, args):
    """Genera galaxias de ejemplo según un conjunto de specs predefinidas."""

    ejemplo_specs = [
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
        # Condicionamiento parcial: solo masa
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
        # Condicionamiento parcial: masa + edad
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
        # Completamente incondicional
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

    imagenes = []
    for i, spec in enumerate(ejemplo_specs):
        print(f"Generando {i + 1}/{len(ejemplo_specs)}: {spec.etiqueta}")
        tensor = generate_galaxy(
            spec,
            unet,
            projector,
            noise_scheduler,
            variables,
            norm_stats,
            device,
            guidance_scale=args.guidance_scale,
            seed=args.seed + i if args.seed is not None else None,
        )
        img = tensor_to_pil(tensor)
        img = annotate_image(img, spec, variables)
        imagenes.append(img)

    os.makedirs(args.output_dir, exist_ok=True)
    grid = make_grid(imagenes, cols=4)
    out_path = os.path.join(args.output_dir, "ejemplos_galaxias.png")
    grid.save(out_path)
    print(f"\nGrid guardado en: {out_path}")


def run_sweep(unet, projector, noise_scheduler, variables, norm_stats, device, args):
    """
    Barrido de un parámetro físico, fijando los demás a valores típicos.
    Útil para analizar la sensibilidad del modelo al condicionamiento.
    """
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

    imagenes = []
    for v in valores:
        kwargs = {
            "log_ms": 10.0,
            "ea_gyr": 3.0,
            "escala_kpc_px": 0.8,
            "radio_p_arcsec": 10.0,
        }
        kwargs[sweep_var] = float(v)
        spec = GalaxySpec(etiqueta=f"{sweep_var}={v:.2f}", **kwargs)
        print(f"  {spec.etiqueta}")
        tensor = generate_galaxy(
            spec,
            unet,
            projector,
            noise_scheduler,
            variables,
            norm_stats,
            device,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
        img = tensor_to_pil(tensor)
        img = annotate_image(img, spec, variables)
        imagenes.append(img)

    os.makedirs(args.output_dir, exist_ok=True)
    grid = make_grid(imagenes, cols=n)
    out_path = os.path.join(args.output_dir, f"sweep_{sweep_var}.png")
    grid.save(out_path)
    print(f"Sweep guardado en: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════════════════════


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generación de galaxias con CFG + DDIM."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
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
        default=50,
        help="Número de pasos DDIM (más pasos = mayor calidad, más lento)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    print(f"Guidance scale: {args.guidance_scale}")

    unet, projector, noise_scheduler, variables, norm_stats = load_checkpoint(
        args.checkpoint, device
    )

    # Configurar número de pasos DDIM
    noise_scheduler.set_timesteps(args.n_steps)

    if args.mode == "examples":
        run_examples(
            unet, projector, noise_scheduler, variables, norm_stats, device, args
        )
    elif args.mode == "sweep":
        run_sweep(unet, projector, noise_scheduler, variables, norm_stats, device, args)


if __name__ == "__main__":
    main()
