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

# --- 2. RUTAS Y PARÁMETROS EXACTOS DEL YAML ---
RUTA_CHECKPOINT = os.path.expanduser("~/Downloads/mejor_modelo.pt")
IMG_SIZE = 128
INFERENCE_STEPS = 100
GUIDANCE_SCALE = 3.0
SEED = 42

# NUEVA CONFIGURACIÓN: Sincronizado con el tamaño de fuente optimizado de plotear_galaxias
TAMANO_FUENTE_PANORAMICA = 8.5

# --- 3. LOS 12 ESCENARIOS ---
escenarios = [
    {"id": "01 Típica mediana",            "p": [0.9, 10.2, 1.4, 9.0]}, # "0.9 (media) / 10.2 (media) / 1.4 (media) / 9.0 (media)"
    {"id": "02 Baja masa compacta",       "p": [0.65, 9.2, 0.7, 6.0]}, # "1.0 (media) / 9.2 (pequeña) / 0.7 (azul) / 6.0 (pequeña)"
    {"id": "03 Masa media extendida",     "p": [0.75, 10.0, 0.8, 14.0]}, # "0.75 (grande) / 10.0 (media) / 0.8 (azul) / 14.0 (grande)"
    {"id": "04 Masiva madura grande",     "p": [1.1, 10.9, 3.2, 18.0]},  # "1.1 (media) / 10.9 (media) / 3.2 (poco roja) / 18.0 (grandísima)"
    {"id": "05 Masiva muy vieja",          "p": [1.0, 11.3, 6.0, 16.0]}, # "1.0 (media) / 11.3 (alta) / 6.0 (valor extremo) (roja) / 16.0 (grande)"
    {"id": "06 Cercana extendida",         "p": [0.6, 10.1, 4.5, 20.0]}, # "0.6 (grande) / 10.1 (media) / 6.0 (roja) / 20.0 (grande)"
    {"id": "07 Distante compacta",         "p": [1.2, 10.3, 1.5, 12.0]}, # "2.1 (pequeñísima) / 10.3 (media) / 1.5 (azul) / 5.0 (pequeña)"
    {"id": "08 Alta escala límite",        "p": [2.5, 10.4, 1.6, 8.0]}, # "2.5 (valor extremo) (pequeñísima) / 10.4 (media) / 1.6 (azul) / 8.0 (media)"
    {"id": "09 Radio grande outlier",      "p": [0.6, 10.5, 2.0, 55.0]}, # "0.6 (grande) / 10.5 (media) / 2.0 (media) / (valor extremo) 55.0 (grandísima)"
    {"id": "10 Masa baja extrema",         "p": [0.8, 8.9, 1.0, 7.0]}, # "0.8 (media) / 8.9 (valor extremo) (baja) / 1.0 (azul) / 7.0 (pequeña)"
    {"id": "11 Masa alta extrema",         "p": [1.0, 10.6, 2.5, 15.0]}, # "1.0 (media) / 8.9 (valor extremo) (alta) / 2.5 (media) / 15.0 (grande)"
    {"id": "12 Incondicional",             "p": [None, None, None, None]}
]


# --- 4. FUNCIÓN DE ANOTACIÓN ---
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

# --- 5. FUNCIÓN PARA CREAR EL MOSAICO REESTRUCTURADO (2x6) ---
def crear_mosaico_matriz(lista_escenarios, unet, projector, noise_scheduler, variables, norm_stats, device):
    # Forzamos la reconfiguración estricta a 2 filas por 6 columnas
    columnas = 6
    filas = 2
    
    # Simulación base con el tamaño de fuente corregido para medir el lienzo sin clipping
    sample_spec = GalaxySpec(etiqueta="sample", escala_kpc_px=0.5, log_ms=10, ea_gyr=1, radio_p_arcsec=10)
    sample_tensor = generate_galaxy(spec=sample_spec, unet=unet, projector=projector, noise_scheduler=noise_scheduler, variables=variables, norm_stats=norm_stats, device=device, guidance_scale=GUIDANCE_SCALE, seed=SEED, img_size=IMG_SIZE)
    sample_img = tensor_to_pil(sample_tensor)
    
    _, (measured_width, measured_height) = annotate_image(sample_img, "ID: 00\nE:0 M:0 EA:0 R:0", font_size=TAMANO_FUENTE_PANORAMICA)
    
    mosaico_img = Image.new('RGB', (measured_width * columnas, measured_height * filas), color=(0, 0, 0))

    for i, esc in enumerate(lista_escenarios):
        num_id = esc['id'].split()[0]
        
        spec = GalaxySpec(
            etiqueta=esc['id'],
            escala_kpc_px=esc['p'][0],
            log_ms=esc['p'][1],
            ea_gyr=esc['p'][2],
            radio_p_arcsec=esc['p'][3]
        )
        
        # Selección del pipeline de inferencia condicionado vs incondicionado
        if esc['id'] == "12 Incondicional":
            tensor = generate_galaxy(
                spec=spec, unet=unet, projector=projector, noise_scheduler=noise_scheduler,
                variables=variables, norm_stats=norm_stats, device=device,
                guidance_scale=0.0, seed=SEED, img_size=IMG_SIZE
            )
        else:
            tensor = generate_galaxy(
                spec=spec, unet=unet, projector=projector, noise_scheduler=noise_scheduler,
                variables=variables, norm_stats=norm_stats, device=device,
                guidance_scale=GUIDANCE_SCALE, seed=SEED, img_size=IMG_SIZE
            )
        
        imagen_cruda = tensor_to_pil(tensor)
        
        # Formatear la cadena de texto limpia
        if esc['id'] == "12 Incondicional":
            texto_etiqueta = f"ID: {num_id}\nIncondicional"
        else:
            linea_1 = f"ID: {num_id}"
            linea_2 = f"E:{esc['p'][0]} M:{esc['p'][1]} EA:{esc['p'][2]} R:{esc['p'][3]}"
            texto_etiqueta = f"{linea_1}\n{linea_2}"
            
        # Forzar el dibujado con la tipografía pequeña controlada
        imagen_anotada, _ = annotate_image(imagen_cruda, texto_etiqueta, fill_color=(255, 255, 255), font_size=TAMANO_FUENTE_PANORAMICA)

        col = i % columnas
        fila = i // columnas
        mosaico_img.paste(imagen_anotada, (col * measured_width, fila * measured_height))

    return mosaico_img

# --- 6. PROCESAMIENTO PRINCIPAL ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando entorno en {device} para la matriz estructural completa...")
    
    unet, projector, noise_scheduler, variables, norm_stats = load_checkpoint(RUTA_CHECKPOINT, device)
    noise_scheduler.set_timesteps(INFERENCE_STEPS)

    print("\n[PROCESANDO] Generando matriz panorámica de 2x6 escenarios galácticos...")
    mosaico_final = crear_mosaico_matriz(escenarios, unet, projector, noise_scheduler, variables, norm_stats, device)
    
    # Guardar matriz física resultante
    ruta_salida = os.path.join(ruta_actual, "matriz_escenarios_2x6.png")
    mosaico_final.save(ruta_salida)
    print(f"[ÉXITO] Matriz compacta exportada correctamente en:\n-> {ruta_salida}")

    # Ventana de visualización con estética panorámica adaptada a las nuevas dimensiones
    plt.figure(figsize=(18, 7), facecolor='#121212') 
    plt.imshow(mosaico_final)
    plt.title(f"", color='white', fontsize=14, pad=15)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()