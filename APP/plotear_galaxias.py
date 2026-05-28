import os
import sys
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# --- 1. AJUSTE DE RUTAS ---
ruta_actual = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
ruta_cluster = os.path.abspath(os.path.join(ruta_actual, "..", "CLUSTER DGX"))
sys.path.append(ruta_cluster)

from generar_galaxia_mi_u_net import load_checkpoint, generate_galaxy, GalaxySpec, tensor_to_pil

# --- 2. CONFIGURACIÓN GLOBAL ---
RUTA_CHECKPOINT = os.path.expanduser("~/Downloads/mejor_modelo.pt")
IMG_SIZE = 128
INFERENCE_STEPS = 100
GUIDANCE_SCALE = 3.0
SEED = 42
# CONFIGURACIÓN UNIFICADA: Este tamaño afectará ahora a todas las generaciones (1x1, 2x1 y 1x4)
TAMANO_FUENTE_PANORAMICA = 8.2

# --- 3. FUNCIÓN DE ANOTACIÓN ---
def annotate_image(image, text, fill_color=(255, 255, 255), font_size=TAMANO_FUENTE_PANORAMICA):
    width, height = image.size
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

    try:
        text_bbox = font.getbbox("Sample Text\nLine 2")
        text_height = text_bbox[3] - text_bbox[1]
    except AttributeError:
        text_width, text_height = font.getsize("Sample Text\nLine 2")
        
    margin = 4
    bar_height = text_height + 2 * margin
    
    new_image = Image.new('RGB', (width, height + bar_height), color=(0, 0, 0))
    new_image.paste(image, (0, bar_height))
    
    draw = ImageDraw.Draw(new_image)
    draw.text((margin, margin), text, fill=fill_color, font=font, spacing=2)
    
    return new_image, (width, height + bar_height)

# --- 4. FUNCIÓN GENÉRICA PARA CONSTRUIR CUALQUIER TAMAÑO DE MOSAICO ---
def crear_mosaico(lista_escenarios, columnas, filas, unet, projector, noise_scheduler, variables, norm_stats, device):
    # Generamos una muestra efímera para medir las dimensiones reales usando la fuente configurada
    sample_spec = GalaxySpec(etiqueta="sample", escala_kpc_px=0.5, log_ms=10, ea_gyr=1, radio_p_arcsec=10)
    sample_tensor = generate_galaxy(spec=sample_spec, unet=unet, projector=projector, noise_scheduler=noise_scheduler, variables=variables, norm_stats=norm_stats, device=device, guidance_scale=GUIDANCE_SCALE, seed=SEED, img_size=IMG_SIZE)
    sample_img = tensor_to_pil(sample_tensor)
    
    # CORRECCIÓN: Ahora pasamos explícitamente TAMANO_FUENTE_PANORAMICA para la simulación del tamaño del lienzo
    _, (w_final, h_final) = annotate_image(sample_img, "ID: 00\nE:0 M:0 EA:0 R:0", font_size=TAMANO_FUENTE_PANORAMICA)
    
    mosaico_img = Image.new('RGB', (w_final * columnas, h_final * filas), color=(0, 0, 0))

    for i, esc in enumerate(lista_escenarios):
        spec = GalaxySpec(
            etiqueta=esc['id'],
            escala_kpc_px=esc['p'][0],
            log_ms=esc['p'][1],
            ea_gyr=esc['p'][2],
            radio_p_arcsec=esc['p'][3]
        )
        
        tensor = generate_galaxy(
            spec=spec, unet=unet, projector=projector, noise_scheduler=noise_scheduler,
            variables=variables, norm_stats=norm_stats, device=device,
            guidance_scale=GUIDANCE_SCALE, seed=SEED, img_size=IMG_SIZE
        )
        
        imagen_cruda = tensor_to_pil(tensor)
        texto_etiqueta = f"{esc['id']}\nE:{esc['p'][0]} M:{esc['p'][1]} EA:{esc['p'][2]} R:{esc['p'][3]}"
        
        # CORRECCIÓN: Forzamos el uso del tamaño corregido en el renderizado real de cada celda
        imagen_anotada, _ = annotate_image(imagen_cruda, texto_etiqueta, font_size=TAMANO_FUENTE_PANORAMICA)

        col = i % columnas
        fila = i // columnas
        mosaico_img.paste(imagen_anotada, (col * w_final, fila * h_final))
        
    return mosaico_img

# --- 5. EJECUCIÓN Y PRUEBAS ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cargando modelo en {device} para generaciones personalizadas...")
    unet, projector, noise_scheduler, variables, norm_stats = load_checkpoint(RUTA_CHECKPOINT, device)
    noise_scheduler.set_timesteps(INFERENCE_STEPS)

    # Parámetros para el Plot de 1 Galaxia (Malla 1x1)
    params_1 = [
        {"id": "ID: 1","p": [0.43, 9.12, 0.4042, 27.9]} # RGB 1
        # {"id": "ID: 2", "p": [0.53, 10.75, 2.4, 15]} # RGB 2
        # {"id": "ID: 1","p": [0.98, 10.41, 1.434, 13.75]} # 1237665532792537288
        # {"id": "ID: 2", "p": [1.16, 10.21, 1.13, 6.64]}, # OBJID: 1237668292835475673
        # {"id": "ID: 3", "p": [0.78, 9.90, 0.90, 7.14]}, # OBJID: 1237668298215325864
        # {"id": "ID: 4", "p": [0.43, 9.62, 2.0, 15.38]}, # OBJID: 1237668315383005343


    
    ]
    
    # Parámetros para el Plot de 2 Galaxias (Malla 2x1 -> Dos en horizontal)
    params_2 = [
        # {"id": "ID:1",      "p": [1, 10, 3, 6.0]},
        # {"id": "ID:2",      "p": [0.9, 9.5, 3, 6.0]}
    ]
    # Parámetros para el Plot de 4 Galaxias (Malla 2x2 -> Matriz)
    params_4 = [
        # {"id": "ID:1",              "p": [1, 10.0, 3, 6.0]},
        # {"id": "ID:2",              "p": [1, 10.0, 3, 9.0]},
        # {"id": "ID:3",              "p": [1, 10.0, 3, 11.0]},
        # {"id": "ID:4",              "p": [1, 10.0, 3, 20.0]}
    ]


    # =========================================================================
    # EJECUCIÓN DE LOS PLOTS Y GUARDADO FÍSICO (Afecta a todos por igual)
    # =========================================================================
    if params_1:
        print("\nGenerando plot de 1 galaxia...")
        img_1 = crear_mosaico(params_1, columnas=1, filas=1, unet=unet, projector=projector, noise_scheduler=noise_scheduler, variables=variables, norm_stats=norm_stats, device=device)
        img_1.save(os.path.join(ruta_actual, "galaxia_1.png"))
    
    if params_2:
        print("Generando plot de 2 galaxias...")
        img_2 = crear_mosaico(params_2, columnas=2, filas=1, unet=unet, projector=projector, noise_scheduler=noise_scheduler, variables=variables, norm_stats=norm_stats, device=device)
        img_2.save(os.path.join(ruta_actual, "galaxias_2.png"))
    
    if params_4:
        print("Generando tira panorámica de 4 galaxias inéditas (1x4)...")
        img_4 = crear_mosaico(params_4, columnas=4, filas=1, unet=unet, projector=projector, noise_scheduler=noise_scheduler, variables=variables, norm_stats=norm_stats, device=device)
        img_4.save(os.path.join(ruta_actual, "galaxias_4.png"))
    
        # Visualización panorámica optimizada para Overleaf
        plt.figure(figsize=(18, 5), facecolor='#121212') 
        plt.imshow(img_4)
        plt.axis('off')
        plt.tight_layout()
    
    print(f"\n[ÉXITO] Archivos guardados en: {ruta_actual}")
    print("Todos los mosaicos han sido procesados con tipografía uniforme.")
    plt.show()

if __name__ == "__main__":
    main()