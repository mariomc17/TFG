import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

###########################################################################################

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..\.."))
DIR_IMAGENES = os.path.join(REPO_ROOT, "galaxias_sdss_filtradas")
DIR_CSV = os.path.join(REPO_ROOT, "csv")
RUTA_CSV = os.path.join(DIR_CSV, "estadisticas_cielo.csv")

OBJIDS_PRIMERA_TANDA = [
    "1237648674511454344", "1237648674529739178", "1237648675066675556",
    "1237648702966988820", "1237648702967054406", "1237648702967251093",
    "1237648702968103043", "1237648702968954880", "1237648702972297472",
    "1237648704042434653"
]

MAX_ARCHIVOS = 100
MAX_WORKERS = 8
UMBRAL_MULT = 1.2
LADO_P = 8
LOTE_VISUALIZACION = 10

###########################################################################################

def analizar_galaxia_rectangulo(file, cielo_dict):
    try:
        obj_id = os.path.basename(file).split(".")[0] 
        if obj_id not in cielo_dict: return None
            
        datos_cielo = cielo_dict[obj_id]
        umbral = UMBRAL_MULT * datos_cielo['LC_Final']
        
        img_path = os.path.join(DIR_IMAGENES, file)
        img = Image.open(img_path)
        img_array = np.array(img).astype(float) 
        
        brightness = img_array.sum(axis=2) 
        h, w = brightness.shape 
        c_y, c_x = h // 2, w // 2
        
        rx, ry = 1, 1
        max_pasos = (h // 2) // LADO_P 
        crecer_x, crecer_y = True, True

        while (crecer_x or crecer_y) and max(rx, ry) < max_pasos:
            brillos_temp_x, brillos_temp_y = [], []
            for i in range(-ry, ry + 1):
                for j in range(-rx, rx + 1):
                    es_borde_y = (abs(i) == ry)
                    es_borde_x = (abs(j) == rx)
                    if es_borde_y or es_borde_x:
                        py = max(0, min(c_y - (LADO_P // 2) + (i * LADO_P), h - LADO_P))
                        px = max(0, min(c_x - (LADO_P // 2) + (j * LADO_P), w - LADO_P))
                        b_medio = np.mean(brightness[py:py+LADO_P, px:px+LADO_P]) 
                        if es_borde_y and crecer_y: brillos_temp_y.append(b_medio)
                        if es_borde_x and crecer_x: brillos_temp_x.append(b_medio)

            if crecer_y and brillos_temp_y and np.mean(brillos_temp_y) <= umbral: crecer_y = False
            if crecer_x and brillos_temp_x and np.mean(brillos_temp_x) <= umbral: crecer_x = False
            if crecer_x: rx += 1
            if crecer_y: ry += 1
            
        rx_px, ry_px = rx * LADO_P, ry * LADO_P
        mascara_rectangulo = np.zeros((h, w), dtype=bool)
        mascara_rectangulo[max(0, c_y - ry_px):min(h, c_y + ry_px), max(0, c_x - rx_px):min(w, c_x + rx_px)] = True

        return (obj_id, img_array, mascara_rectangulo, c_x, c_y, rx, ry)
    except Exception:
        return None

def main():
    start_time = time.time()
    
    if not os.path.exists(RUTA_CSV):
        print(f"Error: no se encuentra {RUTA_CSV}")
        return

    df_cielo = pd.read_csv(RUTA_CSV)
    df_cielo['OBJID'] = df_cielo['OBJID'].astype(str)
    cielo_dict = df_cielo.set_index('OBJID').to_dict('index')

    if not os.path.exists(DIR_IMAGENES):
        print(f"Error: no se encuentra la carpeta {DIR_IMAGENES}")
        return

    all_files = sorted([f for f in os.listdir(DIR_IMAGENES) if f.lower().endswith(('.jpg', '.png'))])
    archivos_especificos = [f"{objid}.jpg" for objid in OBJIDS_PRIMERA_TANDA if f"{objid}.jpg" in all_files]
    archivos_resto = [f for f in all_files if f not in archivos_especificos]
    files = archivos_especificos + archivos_resto[:MAX_ARCHIVOS - len(archivos_especificos)]
    total_files = len(files)

    visualizacion_data = [] 

    if total_files > 0:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            resultados = tqdm(executor.map(lambda f: analizar_galaxia_rectangulo(f, cielo_dict), files), total=total_files, desc="Recorte Rectángulo")
            for res in resultados:
                if res is not None:
                    visualizacion_data.append(res)

    T_SUPERTITULO = 19
    T_SUBTITULO = 13

    for i in range(0, len(visualizacion_data), LOTE_VISUALIZACION):
        batch = visualizacion_data[i:i + LOTE_VISUALIZACION]
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle("Recorte mediante rectángulos", fontsize=T_SUPERTITULO, y=0.98)
        axes = axes.flatten()
        
        for idx, (obj_id, img_array, mask_rect, c_x, c_y, rx, ry) in enumerate(batch):
            ax = axes[idx]
            h, w = img_array.shape[:2]
            ax.imshow(img_array.astype(np.uint8))
            ax.set_title(f"({chr(97 + idx)}) ID: {obj_id}", fontsize=T_SUBTITULO)
            ax.axis('off')
            
            for fila in range(-ry, ry + 1):
                for col in range(-rx, rx + 1):
                    if abs(fila) == ry or abs(col) == rx:
                        py = max(0, min(c_y - (LADO_P // 2) + (fila * LADO_P), h - LADO_P))
                        px = max(0, min(c_x - (LADO_P // 2) + (col * LADO_P), w - LADO_P))
                        cuadradito = patches.Rectangle((px, py), LADO_P, LADO_P, linewidth=1, edgecolor='yellow', facecolor='none', alpha=0.7)
                        ax.add_patch(cuadradito)
            
            ax.plot(c_x, c_y, marker='+', color='red', markersize=5)
                        
        for idx in range(len(batch), len(axes)):
            axes[idx].axis('off')
            
        plt.tight_layout()
        plt.show()

    print(f"Tiempo total: {time.time() - start_time:.2f} segundos")

if __name__ == "__main__":
    main()