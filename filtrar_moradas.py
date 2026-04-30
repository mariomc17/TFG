import os
import time
import pandas as pd
import numpy as np
import threading
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

###########################################################################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DIR_IMAGENES = os.path.join(REPO_ROOT, "jpg_sdss")
DIR_TXT = os.path.join(REPO_ROOT, "txt")
NOMBRE_TXT = "galaxias_moradas.txt"
TXT_PATH = os.path.join(DIR_TXT, NOMBRE_TXT)

if not os.path.exists(DIR_TXT):
    os.makedirs(DIR_TXT)

UMBRAL_VERDE = 0.48
MAX_WORKERS = 8
###########################################################################################

VALORES = []
LOCK = threading.Lock()

def analizar_imagen(archivo):
    try:
        ruta = os.path.join(DIR_IMAGENES, archivo)
        obj_id = os.path.basename(archivo).split(".")[0]
        
        img = Image.open(ruta)
        img_array = np.array(img).astype(float)
        
        h, w, _ = img_array.shape
        c_y, c_x = h // 2, w // 2
        r_n = 64

        nucleo = img_array[c_y-r_n : c_y+r_n, c_x-r_n : c_x+r_n]
        
        m_r = np.mean(nucleo[:,:,0])
        m_g = np.mean(nucleo[:,:,1])
        m_b = np.mean(nucleo[:,:,2])
        
        rv = m_g / (m_r + m_b + 1e-6)
        
        with LOCK:
            VALORES.append({'OBJID': obj_id, 'Archivo': archivo, 'Ratio_Verde': rv})
        return True
    except Exception as e:
        return False

def main():
    inicio = time.time()
    
    if not os.path.exists(DIR_IMAGENES):
        print(f"Error: no se encuentra la carpeta de imágenes en {DIR_IMAGENES}")
        return

    archivos = sorted([f for f in os.listdir(DIR_IMAGENES) if f.lower().endswith(('.jpg', '.png'))])
    total = len(archivos)

    if total > 0:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            list(tqdm(executor.map(analizar_imagen, archivos), total=total, desc="Procesando", unit=" galaxias"))

    df = pd.DataFrame(VALORES)
    df_moradas = df[df['Ratio_Verde'] < UMBRAL_VERDE]
    lista_borrar = df_moradas['OBJID'].tolist()

    with open(TXT_PATH, 'w') as f:
        for item in lista_borrar:
            f.write(f"{item}\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Distribución del Ratio Verde (RV)', fontsize=16)
    
    data = df['Ratio_Verde'].dropna()

    c1, b1, p1 = ax1.hist(data, bins=60, color='mediumseagreen', edgecolor='black', alpha=0.8)
    ax1.axvline(x=UMBRAL_VERDE, color='crimson', linestyle='dashed', linewidth=2)
    ax1.set_title('Escala Lineal')
    for count, edge, patch in zip(c1, b1, p1):
        if edge < UMBRAL_VERDE:
            patch.set_facecolor('crimson')

    c2, b2, p2 = ax2.hist(data, bins=60, color='mediumseagreen', edgecolor='black', alpha=0.8)
    ax2.axvline(x=UMBRAL_VERDE, color='crimson', linestyle='dashed', linewidth=2)
    ax2.set_yscale('log')
    ax2.set_title('Escala Logarítmica')
    for count, edge, patch in zip(c2, b2, p2):
        if edge < UMBRAL_VERDE:
            patch.set_facecolor('crimson')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    fin = time.time()
    print(f"Proceso finalizado. Galaxias moradas detectadas: {len(lista_borrar)}")
    print(f"Archivo generado en: {TXT_PATH}")
    print(f"Tiempo total: {fin - inicio:.2f} segundos")

if __name__ == "__main__":
    main()