import os
import h5py
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import math  # <-- Importante para calcular filas y columnas

# Definir variables del modelo de difusión
VARIABLES = ['ESCALA_KPC_PX', 'LOG_MS', 'SFR', 'EA', 'MET', 'RADIO_P', 'G_R']

def main():
    ruta_datos = r"C:\Users\mario\OneDrive\Escritorio\GitHub\h5_sin_rgb\dataset_galaxias_sin_rgb.h5"
    df = None
    
    if os.path.exists(ruta_datos):
        print(f"Cargando {ruta_datos}...")
        try:
            with h5py.File(ruta_datos, 'r') as f:
                if 'fisica' in f and 'columnas_fisicas' in f.attrs and 'stats' in f.attrs:
                    columnas = json.loads(f.attrs['columnas_fisicas'])
                    stats = json.loads(f.attrs['stats'])
                    datos_fisica = f['fisica'][:]
                    
                    df_completo = pd.DataFrame(datos_fisica, columns=columnas)
                    
                    # --- APLICAR NORMALIZACIÓN MIN-MAX AQUÍ ---
                    for col in columnas:
                        vmin = stats[col]['min']
                        vmax = stats[col]['max']
                        denominador = (vmax - vmin) if (vmax - vmin) > 0 else 1e-8
                        df_completo[col] = (df_completo[col] - vmin) / denominador
                    # ------------------------------------------

                    cols_presentes = [v for v in VARIABLES if v in df_completo.columns]
                    if cols_presentes:
                        df = df_completo[cols_presentes]
        except Exception as e:
            print(f"Error al leer el HDF5: {e}")

    if df is None or df.empty:
        print("No se pudo cargar el HDF5 o las variables no se encontraron.")
        print(f"Asegúrate de ejecutar este script en la misma carpeta que '{ruta_datos}'.")
        return

    # Filtrar solo las variables de interés y quitar NaNs
    df = df[VARIABLES].dropna()
    n_vars = len(VARIABLES)

    print(f"Generando visualización para {len(df)} galaxias y {n_vars} variables...")
    
    with tqdm(total=n_vars + 1, desc="Generando histogramas", unit="paso") as pbar:
        # Configuración visual
        sns.set_theme(style="whitegrid", palette="deep")
        
        # --- CÁLCULO DINÁMICO DE LA CUADRÍCULA ---
        # Fijamos un máximo de 3 columnas para que las gráficas no queden muy estrechas
        n_cols = min(3, n_vars) 
        n_rows = math.ceil(n_vars / n_cols)
        
        # Ajustar el tamaño de la figura dinámicamente en función de las filas y columnas
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
        axes = axes.flatten()

        for i, var in enumerate(VARIABLES):
            pbar.set_description(f"Procesando {var}")
            sns.histplot(df[var], bins=50, color="#00A390", kde=True, ax=axes[i], alpha=0.7)
            
            axes[i].set_title(f"Distribución: {var}", fontsize=14, fontweight='bold')
            axes[i].set_xlabel("Valor Normalizado [0, 1]", fontsize=12)
            axes[i].set_ylabel("Número de Galaxias", fontsize=12)
            axes[i].set_xlim(-0.05, 1.05)
            
            pbar.update(1)

        # --- ELIMINAR LOS GRÁFICOS SOBRANTES (VACÍOS) ---
        for j in range(n_vars, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f"Análisis de Parámetros de Entrada U-Net ({n_vars} Variables)", fontsize=18, fontweight='bold', y=0.98)
        
        # Ajuste para evitar que el título general pise a los títulos de las gráficas
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        pbar.set_description("Mostrando gráfico")
        plt.show()
        pbar.update(1)

    print("¡Visualización completada exitosamente!")
if __name__ == "__main__":
    main()