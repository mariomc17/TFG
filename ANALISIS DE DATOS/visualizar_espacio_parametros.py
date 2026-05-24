import os
import h5py
import pandas as pd
import json
import seaborn as plt_sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

VARIABLES = ['ESCALA_KPC_PX', 'LOG_MS', 'SFR', 'EA', 'MET', 'RADIO_P', 'G_R']

def main():
    ruta_datos = r"C:\Users\mario\OneDrive\Escritorio\GitHub\h5_sin_rgb\dataset_galaxias_sin_rgb.h5"
    df_raw = None
    stats = None
    
    if os.path.exists(ruta_datos):
        print(f"Cargando {ruta_datos}...")
        try:
            with h5py.File(ruta_datos, 'r') as f:
                if 'fisica' in f and 'columnas_fisicas' in f.attrs and 'stats' in f.attrs:
                    columnas = json.loads(f.attrs['columnas_fisicas'])
                    stats = json.loads(f.attrs['stats'])
                    datos_fisica = f['fisica'][:]
                    
                    df_raw = pd.DataFrame(datos_fisica, columns=columnas)
        except Exception as e:
            print(f"Error al leer el HDF5: {e}")

    if df_raw is None or df_raw.empty:
        print("No se pudo cargar el HDF5.")
        return

    df_raw = df_raw[VARIABLES].dropna()
    n_vars = len(VARIABLES)

    print(f"Generando visualización con valores físicos para {len(df_raw)} galaxias...")
    
    with tqdm(total=n_vars + 1, desc="Generando histogramas", unit="paso") as pbar:
        n_cols = min(3, n_vars) 
        n_rows = math.ceil(n_vars / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
        axes = axes.flatten()

        for i, var in enumerate(VARIABLES):
            pbar.set_description(f"Procesando {var}")
            plt_sns.histplot(df_raw[var], bins=50, kde=True, ax=axes[i])
            
            axes[i].set_title(f"{var}", fontsize=12)
            axes[i].set_xlabel(f"Valor original ({var})", fontsize=11)
            axes[i].set_ylabel("Número de Galaxias", fontsize=11)
            axes[i].tick_params(labelsize=9)
            
            pbar.update(1)

        for j in range(n_vars, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f"Análisis de variables astrofísicas", y=0.96, fontsize=18)
        fig.subplots_adjust(top=0.90, bottom=0.10, left=0.08, right=0.96, wspace=0.25, hspace=0.35)

        pbar.set_description("Mostrando gráfico")
        plt.show()
        pbar.update(1)

    print("¡Visualización completada exitosamente!")

if __name__ == "__main__":
    main()