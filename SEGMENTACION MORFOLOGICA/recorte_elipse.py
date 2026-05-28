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

def analizar_galaxia_elipse(file, cielo_dict):
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
        mascara_galaxia_aislada = np.zeros((h, w), dtype=bool)
        
        if segmentacion is not None:
            id_galaxia = segmentacion.data[c_y, c_x]
            if id_galaxia != 0:
                silueta_cruda = (segmentacion.data == id_galaxia)
                mascara_galaxia_aislada = binary_fill_holes(silueta_cruda)

        y_coords, x_coords = np.nonzero(mascara_galaxia_aislada)
        mascara_elipse = np.zeros((h, w), dtype=bool)

        if len(x_coords) >= 50:
            x_c_diff, y_c_diff = x_coords - c_x, y_coords - c_y
            mu20, mu02, mu11 = np.sum(x_c_diff**2), np.sum(y_c_diff**2), np.sum(x_c_diff * y_c_diff)
            temp = np.sqrt((mu20 - mu02)**2 + 4 * mu11**2)
            a_rel = np.sqrt(0.5 * (mu20 + mu02 + temp))
            b_rel = max(1e-6, np.sqrt(0.5 * (mu20 + mu02 - temp)))
            theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

            Y, X = np.ogrid[:h, :w]
            X_rot = (X - c_x) * np.cos(theta) + (Y - c_y) * np.sin(theta)
            Y_rot = -(X - c_x) * np.sin(theta) + (Y - c_y) * np.cos(theta)
            distancia_eliptica = np.sqrt(X_rot**2 + (Y_rot * (a_rel / b_rel))**2)

            radio_actual, paso_radial, radio_maximo = 0, 2, min(h, w)
            crecer = True
            
            while crecer and radio_actual < radio_maximo:
                anillo_mask = (distancia_eliptica >= radio_actual) & (distancia_eliptica < radio_actual + paso_radial)
                brillo_anillo = brightness[anillo_mask]

                if len(brillo_anillo) > 0 and np.mean(brillo_anillo) <= umbral:
                    crecer = False
                else:
                    radio_actual += paso_radial

            mascara_elipse = distancia_eliptica <= radio_actual

        return (obj_id, img_array, mascara_elipse, mascara_galaxia_aislada, c_x, c_y)
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
        print(f"Error: no se encuentra la carpeta de imágenes en {DIR_IMAGENES}")
        return

    all_files = sorted([f for f in os.listdir(DIR_IMAGENES) if f.lower().endswith(('.jpg', '.png'))])
    archivos_especificos = [f"{objid}.jpg" for objid in OBJIDS_PRIMERA_TANDA if f"{objid}.jpg" in all_files]
    archivos_resto = [f for f in all_files if f not in archivos_especificos]
    files = archivos_especificos + archivos_resto[:MAX_ARCHIVOS - len(archivos_especificos)]
    total_files = len(files)

    visualizacion_data = [] 

    if total_files > 0:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            resultados = tqdm(executor.map(lambda f: analizar_galaxia_elipse(f, cielo_dict), files), total=total_files, desc="Recorte Elipse")
            for res in resultados:
                if res is not None:
                    visualizacion_data.append(res)

    T_SUPERTITULO = 19
    T_SUBTITULO = 13

    for i in range(0, len(visualizacion_data), LOTE_VISUALIZACION):
        batch = visualizacion_data[i:i + LOTE_VISUALIZACION]
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle("Recorte mediante ajuste elíptico", fontsize=T_SUPERTITULO, y=0.98)
        axes = axes.flatten()
        
        for idx, (obj_id, img_array, mask_elipse, mask_aislada, c_x, c_y) in enumerate(batch):
            ax = axes[idx]
            ax.imshow(img_array.astype(np.uint8))
            ax.set_title(f"({chr(97 + idx)}) ID: {obj_id}", fontsize=T_SUBTITULO)
            ax.axis('off')
            
            if np.any(mask_aislada):
                ax.contour(mask_aislada, levels=[0.5], colors='magenta', linewidths=1.5, alpha=0.8)
            if np.any(mask_elipse):
                ax.contour(mask_elipse, levels=[0.5], colors='cyan', linewidths=1.5, alpha=0.9)
            
            ax.plot(c_x, c_y, marker='+', color='red', markersize=5)

        for j in range(len(batch), len(axes)): 
            axes[j].axis('off')
        plt.tight_layout()
        plt.show()
        
    print(f"Tiempo total: {time.time() - start_time:.2f} segundos")

if __name__ == "__main__":
    main()