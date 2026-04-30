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
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DIR_IMAGENES = os.path.join(REPO_ROOT, "jpg_sdss") # Aquí ya se han filtrado las defectuosas
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
        img_array = np.array(img).astype(float) 
        
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
        cielo_median_bruto = np.median(secciones_mean)
        cielo_std_bruto = np.std(secciones_mean)
        cielo_var_bruto = np.var(secciones_mean) 
        
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
        cielo_median_limpia = np.median(secciones_limpias)      
        cielo_std_limpia = np.std(secciones_limpias)
        cielo_var_limpia = np.var(secciones_limpias)

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
            "Mean_Bruta": cielo_mean_bruto,
            "Median_Bruta": cielo_median_bruto,
            "Desviación_Bruta": cielo_std_bruto,
            "Varianza_Bruta": cielo_var_bruto,
            "LC_Bruto": limite_bruto,
            "Mean_Limpia": cielo_mean_limpia,
            "Median_Limpia": cielo_median_limpia, 
            "Desviación_Limpia": cielo_std_limpia,
            "Varianza_Limpia": cielo_var_limpia,
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
        print("Iniciando análisis de cielo...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            resultados_iterador = tqdm(executor.map(analizar_cielo, files), total=total_files, desc="Procesando", unit=" galaxias")
            for resultado in resultados_iterador:
                if resultado is not None:
                    datos_totales.append(resultado)

    df_cielo = pd.DataFrame(datos_totales)

    if not df_cielo.empty:
        df_cielo.to_csv(CSV_PATH, index=False)
        print(f"Archivo generado en: {CSV_PATH}")
        
        fig, axs = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("Evolución de estadísticas del cielo", fontsize=16)

        color_bruto = 'indianred'
        axs[0, 0].hist(df_cielo['Median_Bruta'], bins=50, color=color_bruto, edgecolor='black', alpha=0.8)
        axs[0, 0].set_title("Mediana Bruta")
        axs[0, 0].set_xlabel("Brillo")
        axs[0, 0].set_ylabel("Nº de galaxias")

        axs[0, 1].hist(df_cielo['Mean_Bruta'], bins=50, color=color_bruto, edgecolor='black', alpha=0.8)
        axs[0, 1].set_title("Media Bruta")
        axs[0, 1].set_xlabel("Brillo")

        axs[0, 2].hist(df_cielo['Desviación_Bruta'], bins=50, color=color_bruto, edgecolor='black', alpha=0.8)
        axs[0, 2].set_title("Desviación Bruta")
        axs[0, 2].set_xlabel("Desviación")
        axs[0, 2].set_yscale('log')

        color_limpio = 'mediumseagreen'
        axs[1, 0].hist(df_cielo['Median_Limpia'], bins=50, color=color_limpio, edgecolor='black', alpha=0.8)
        axs[1, 0].set_title("Mediana Limpia")
        axs[1, 0].set_xlabel("Brillo")
        axs[1, 0].set_ylabel("Nº de galaxias")

        axs[1, 1].hist(df_cielo['Mean_Limpia'], bins=50, color=color_limpio, edgecolor='black', alpha=0.8)
        axs[1, 1].set_title("Media Limpia")
        axs[1, 1].set_xlabel("Brillo")

        axs[1, 2].hist(df_cielo['Desviación_Limpia'], bins=80, color=color_limpio, edgecolor='black', alpha=0.8)
        axs[1, 2].set_title("Desviación Limpia")
        axs[1, 2].set_xlabel("Desviación")
        axs[1, 2].set_yscale('log')

        for ax in axs.flat:
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        fig2, axs2 = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle("Evolución de $LC$", fontsize=16)

        axs2[0].hist(df_cielo['LC_Bruto'], bins=50, color=color_bruto, edgecolor='black', alpha=0.8)
        axs2[0].set_title("$LC$ Bruto (Antes de Tukey)")
        axs2[0].set_xlabel("Brillo de Límite de Cielo")
        axs2[0].set_ylabel("Nº de galaxias")

        axs2[1].hist(df_cielo['LC_Final'], bins=50, color=color_limpio, edgecolor='black', alpha=0.8)
        axs2[1].set_title("$LC$ Final (Después de Tukey)")
        axs2[1].set_xlabel("Brillo de Límite de Cielo")
        axs2[1].set_ylabel("Nº de galaxias")
        
        for ax in axs2.flat:
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

    end_time = time.time()
    print(f"Tiempo total: {end_time - start_time:.2f} segundos")

if __name__ == "__main__":
    main()