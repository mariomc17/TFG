import os
import time
import pandas as pd
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from photutils.segmentation import detect_sources
from scipy.ndimage import binary_fill_holes

###########################################################################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DIR_IMAGENES = os.path.join(REPO_ROOT, "jpg_sdss") # Aquí ya se han filtrado las defectuosas
DIR_CSV = os.path.join(REPO_ROOT, "csv")
RUTA_CSV = os.path.join(DIR_CSV, "estadisticas_cielo.csv")
DIR_SALIDA = os.path.join(REPO_ROOT, "galaxias_elipse")

if not os.path.exists(DIR_SALIDA):
    os.makedirs(DIR_SALIDA)

MAX_WORKERS = 8
UMBRAL_MULT = 1.3
MAX_ARCHIVOS = None 
###########################################################################################

def analizar_galaxia_elipse(file, cielo_dict):
    try:
        obj_id = os.path.basename(file).split(".")[0] 
        if obj_id not in cielo_dict: return False
            
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

        bg_color = [datos_cielo['LC_R'], datos_cielo['LC_G'], datos_cielo['LC_B']]
        img_recortada = img_array.copy()
        img_recortada[~mascara_elipse] = bg_color
        img_recortada_uint8 = np.clip(img_recortada, 0, 255).astype(np.uint8)

        ruta_guardado = os.path.join(DIR_SALIDA, f"{obj_id}.png")
        Image.fromarray(img_recortada_uint8).save(ruta_guardado, format="PNG")

        return True
    except Exception:
        return False

def main():
    start_time = time.time()
    
    if not os.path.exists(RUTA_CSV):
        print(f"Error: mo se encuentra {RUTA_CSV}")
        return

    df_cielo = pd.read_csv(RUTA_CSV)
    
    if 'LC_R' not in df_cielo.columns:
        print("Error: el archivo CSV no tiene columnas LC_R, LC_G, LC_B.")
        return

    df_cielo['OBJID'] = df_cielo['OBJID'].astype(str)
    cielo_dict = df_cielo.set_index('OBJID').to_dict('index')

    if not os.path.exists(DIR_IMAGENES):
        print(f"Error: no se encuentra la carpeta {DIR_IMAGENES}")
        return

    files = sorted([f for f in os.listdir(DIR_IMAGENES) if f.lower().endswith(('.jpg', '.png'))])
    
    if MAX_ARCHIVOS is not None:
        files = files[:MAX_ARCHIVOS]
        
    total_files = len(files)
    procesadas = 0na

    if total_files > 0:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            resultados = tqdm(executor.map(lambda f: analizar_galaxia_elipse(f, cielo_dict), files), total=total_files, desc="Recorte Elipse")
            for res in resultados:
                if res: procesadas += 1

    print(f"Galaxias procesadas y guardadas: {procesadas}/{total_files}")
    print(f"Tiempo total: {time.time() - start_time:.2f} segundos")

if __name__ == "__main__":
    main()