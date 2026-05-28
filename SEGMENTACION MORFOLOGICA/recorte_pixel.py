import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from photutils.segmentation import detect_sources
from scipy.ndimage import binary_fill_holes

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
UMBRAL_MULT = 1.3
LOTE_VISUALIZACION = 10

###########################################################################################

def analizar_galaxia_pixel(file, cielo_dict):
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

        segmentacion = detect_sources(brightness, umbral, npixels=10) 
        mascara_pixel = np.zeros((h, w), dtype=bool)
        
        if segmentacion is not None:
            id_galaxia = segmentacion.data[c_y, c_x]
            if id_galaxia != 0:
                silueta_cruda = (segmentacion.data == id_galaxia)
                mascara_pixel = binary_fill_holes(silueta_cruda)

        return (obj_id, img_array, mascara_pixel, c_x, c_y)
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
            resultados = tqdm(executor.map(lambda f: analizar_galaxia_pixel(f, cielo_dict), files), total=total_files, desc="Recorte Píxeles")
            for res in resultados:
                if res is not None:
                    visualizacion_data.append(res)

    T_SUPERTITULO = 19
    T_SUBTITULO = 13

    for i in range(0, len(visualizacion_data), LOTE_VISUALIZACION):
        batch = visualizacion_data[i:i + LOTE_VISUALIZACION]
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle("Recorte píxel a píxel", fontsize=T_SUPERTITULO, y=0.98)
        axes = axes.flatten()
        
        for idx, (obj_id, img_array, mask_pixel, c_x, c_y) in enumerate(batch):
            ax = axes[idx]
            ax.imshow(img_array.astype(np.uint8))
            ax.set_title(f"({chr(97 + idx)}) ID: {obj_id}", fontsize=T_SUBTITULO)
            ax.axis('off')
            
            if np.any(mask_pixel):
                ax.contour(mask_pixel, levels=[0.5], colors='magenta', linewidths=1.5, alpha=0.9)
            
            ax.plot(c_x, c_y, marker='+', color='red', markersize=5)

        for j in range(len(batch), len(axes)): 
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()      

if __name__ == "__main__":
    main()