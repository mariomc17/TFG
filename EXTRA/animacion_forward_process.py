import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

###########################################################################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DIR_IMAGENES = os.path.join(REPO_ROOT, "galaxias_elipse")

BETA_START = 0.0001
BETA_END = 0.02
TIMESTEPS = 1000
INSTANTES = [0, 100, 250, 500, 750, 999]
IMG_SIZE = 512
###########################################################################################

def linear_beta_schedule(timesteps):
    return np.linspace(BETA_START, BETA_END, timesteps)

def forward_diffusion(x_0, t, betas):
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alpha_bar_t = alphas_cumprod[t]
    ruido = np.random.randn(*x_0.shape)
    x_t = np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1 - alpha_bar_t) * ruido
    return x_t

def main():
    if not os.path.exists(DIR_IMAGENES):
        print(f"Error: no se encuentra la carpeta {DIR_IMAGENES}")
        return

    archivos = [f for f in os.listdir(DIR_IMAGENES) if f.lower().endswith(('.jpg', '.png'))]
    if not archivos:
        print(f"Error: no hay imágenes en {DIR_IMAGENES}")
        return

    archivo_elegido = random.choice(archivos)
    ruta_imagen = os.path.join(DIR_IMAGENES, archivo_elegido)
    
    try:
        img = Image.open(ruta_imagen).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        return
    
    x_0 = np.array(img).astype(np.float32)
    x_0 = (x_0 / 127.5) - 1.0

    betas = linear_beta_schedule(TIMESTEPS)

    fig, axes = plt.subplots(1, len(INSTANTES), figsize=(18, 4))
    fig.suptitle(f"Proceso de difusión (Forward Process) - {archivo_elegido}", fontsize=16)

    for i, t in enumerate(INSTANTES):
        if t == 0:
            x_t = x_0
        else:
            x_t = forward_diffusion(x_0, t, betas)
        
        img_mostrar = (x_t + 1.0) / 2.0
        img_mostrar = np.clip(img_mostrar, 0, 1) 
        
        ax = axes[i]
        ax.imshow(img_mostrar)
        ax.axis('off')
        ax.set_title(f"t = {t}", fontsize=14, pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.85])
    plt.show()

if __name__ == "__main__":
    main()