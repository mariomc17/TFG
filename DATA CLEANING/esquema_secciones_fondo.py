import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from PIL import Image

###########################################################################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DIR_IMAGENES = os.path.join(REPO_ROOT, "jpg_sdss")

FIG_SIZE = (12, 12)
DIVISOR_AREA = 50
DEFAULT_SIZE = 512
###########################################################################################

def main():
    if not os.path.exists(DIR_IMAGENES):
        print(f"Error: no se encuentra la carpeta {DIR_IMAGENES}")
        return

    archivos = [f for f in os.listdir(DIR_IMAGENES) if f.lower().endswith(('.jpg', '.png'))]
    if not archivos:
        print(f"Error: no hay imágenes en {DIR_IMAGENES}")
        return

    archivo_elegido = random.choice(archivos)
    obj_id = archivo_elegido.split('.')[0]
    ruta_imagen = os.path.join(DIR_IMAGENES, archivo_elegido)

    try:
        img = Image.open(ruta_imagen)
        img_array = np.array(img).astype(float)
        h, w, _ = img_array.shape
        brightness = img_array.sum(axis=2)
    except Exception as e:
        print(f"Error cargando la imagen: {e}")
        h, w = DEFAULT_SIZE, DEFAULT_SIZE
        img_array = np.zeros((h, w, 3), dtype=np.uint8)
        brightness = np.zeros((h, w))

    lado_teorico = int(np.sqrt((h * w) / DIVISOR_AREA))
    n_divisiones = max(1, round(min(h, w) / lado_teorico)) 
    lado = min(h, w) // n_divisiones

    secciones_cielo = []
    for i in range(n_divisiones):
        secciones_cielo.append((0, i * lado))          
        secciones_cielo.append((h - lado, i * lado))   
    for j in range(1, n_divisiones - 1):
        secciones_cielo.append((j * lado, 0))          
        secciones_cielo.append((j * lado, w - lado))   

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.imshow(img_array.astype(np.uint8))

    for idx, (y, x) in enumerate(secciones_cielo, start=1):
        parche = brightness[y:y+lado, x:x+lado]
        media_parche = np.mean(parche)
        
        rect = patches.Rectangle((x, y), lado, lado, linewidth=2.5, edgecolor='crimson', facecolor='none', alpha=0.9)
        ax.add_patch(rect)
        
        centro_x = x + (lado / 2)
        centro_y = y + (lado / 2)
        texto_etiqueta = f"Sec {idx}\n$\\mu$={media_parche:.1f}"
        
        ax.text(centro_x, centro_y, texto_etiqueta, color='white', fontsize=13, fontweight='bold', ha='center', va='center', path_effects=[pe.withStroke(linewidth=3, foreground="black")])

    ax.set_title(f"Esquema de la división de la imagen - {obj_id}", fontsize=16)
    ax.axis('off') 

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()