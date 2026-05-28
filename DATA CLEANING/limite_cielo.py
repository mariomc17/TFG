import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

###########################################################################################

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..\.."))
DIR_IMAGENES = os.path.join(REPO_ROOT, "galaxias_sdss_sin_filtrar")
DIR_CSV = os.path.join(REPO_ROOT, "csv")
NOMBRE_CSV = "estadisticas_cielo.csv"
CSV_PATH = os.path.join(DIR_CSV, NOMBRE_CSV)

if not os.path.exists(DIR_CSV):
    os.makedirs(DIR_CSV)

MAX_WORKERS = 8

###########################################################################################

def analizar_cielo(file):
    try:
        img_path = os.path.join(DIR_IMAGENES, file)
        obj_id = os.path.basename(file).split(".")[0] 
        
        img = Image.open(img_path)
        img_array = np.array(img).astype(float) \
        
        brightness = img_array.sum(axis=2) 
        h, w = brightness.shape 
        
        lado_teorico = int(np.sqrt((h * w) / 50))
        n_divisiones = max(1, round(min(h, w) / lado_teorico)) 
        lado = min(h, w) // n_divisiones
        
        secciones_cielo = []
        for i in range(n_divisiones):
            secciones_cielo.append((0, i * lado))
            secciones_cielo.append((h - lado, i * lado))
        for j in range(1, n_divisiones - 1):
            secciones_cielo.append((j * lado, 0))
            secciones_cielo.append((j * lado, w - lado))
        
        secciones_mean = [np.mean(brightness[y:y+lado, x:x+lado]) for y, x in secciones_cielo]

        cielo_mean_bruto = np.mean(secciones_mean)
        cielo_std_bruto = np.std(secciones_mean)
        
        limite_bruto = cielo_mean_bruto + (3 * cielo_std_bruto)
    
        q1 = np.percentile(secciones_mean, 25)
        q3 = np.percentile(secciones_mean, 75)
        iqr = q3 - q1 
        
        limite_inf = q1 - (1.5 * iqr)
        limite_sup = q3 + (1.5 * iqr)
        
        secciones_limpias = [val for val in secciones_mean if limite_inf <= val <= limite_sup]
        indices_limpios = [i for i, val in enumerate(secciones_mean) if limite_inf <= val <= limite_sup]
            
        if len(secciones_limpias) == 0:
            secciones_limpias = secciones_mean

        cielo_mean_limpia = np.mean(secciones_limpias)          
        cielo_std_limpia = np.std(secciones_limpias)

        limite_total = cielo_mean_limpia + (3 * cielo_std_limpia)

        R_means = [np.mean(img_array[y:y+lado, x:x+lado, 0]) for y, x in secciones_cielo]
        G_means = [np.mean(img_array[y:y+lado, x:x+lado, 1]) for y, x in secciones_cielo]
        B_means = [np.mean(img_array[y:y+lado, x:x+lado, 2]) for y, x in secciones_cielo]

        R_limpias = [R_means[i] for i in indices_limpios]
        G_limpias = [G_means[i] for i in indices_limpios]
        B_limpias = [B_means[i] for i in indices_limpios]

        LC_R = np.mean(R_limpias) + (3 * np.std(R_limpias))
        LC_G = np.mean(G_limpias) + (3 * np.std(G_limpias))
        LC_B = np.mean(B_limpias) + (3 * np.std(B_limpias))

        return {
            "OBJID": obj_id, 
            "LC_Bruto": limite_bruto,
            "LC_Final": limite_total,
            "LC_R": LC_R,
            "LC_G": LC_G,
            "LC_B": LC_B
        }
    except Exception:
        return None

def main():
    start_time = time.time()
    datos_totales = []

    if not os.path.exists(DIR_IMAGENES):
        print(f"Error: no se encuentra la carpeta de imágenes en {DIR_IMAGENES}")
        return

    files = sorted([f for f in os.listdir(DIR_IMAGENES) if f.lower().endswith(('.png', '.jpg'))])
    total_files = len(files)

    if total_files > 0:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            resultados_iterador = tqdm(executor.map(analizar_cielo, files), total=total_files, desc="Procesando", unit=" galaxias")
            for resultado in resultados_iterador:
                if resultado is not None:
                    datos_totales.append(resultado)

    df_cielo = pd.DataFrame(datos_totales)

    if not df_cielo.empty:
        df_cielo.to_csv(CSV_PATH, index=False)
        print(f"Archivo generado en: {CSV_PATH}")
        
        T_SUPERTITULO = 19
        T_EJES = 14.5
        T_TICKS_LEYENDA = 13

        plt.figure(figsize=(10, 6))
        
        plt.hist(df_cielo['LC_Bruto'], bins=80, range=(0, 600), color='indianred', 
                 edgecolor='darkred', alpha=0.6, label='$LC_{bruto}$ (Sin filtro de Tukey)', zorder=3)
        
        plt.hist(df_cielo['LC_Final'], bins=80, range=(0, 600), color='mediumseagreen', 
                 edgecolor='darkgreen', alpha=0.8, label='$LC_{final}$ (Con filtro de Tukey)', zorder=3)
        
        plt.title('Comparativa de estimadores del Límite de Cielo ($LC$)', fontsize=T_SUPERTITULO)
        plt.xlabel('Intensidad ($LC$)', fontsize=T_EJES)
        plt.ylabel('Número de galaxias', fontsize=T_EJES)

        plt.xlim(50, 400)
        
        plt.grid(axis='y', linestyle=':', color='gray', linewidth=0.7, alpha=0.7, zorder=1)
        plt.tick_params(axis='both', labelsize=T_TICKS_LEYENDA)
        plt.legend(fontsize=T_TICKS_LEYENDA, loc='upper right')
                
        plt.tight_layout()
        plt.show()

    end_time = time.time()
    print(f"Tiempo total: {end_time - start_time:.2f} segundos")

if __name__ == "__main__":
    main()