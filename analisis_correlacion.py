import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns

###########################################################################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DIR_CSV = os.path.join(REPO_ROOT, "csv")
RUTA_CSV_SDSS = os.path.join(DIR_CSV, "galaxias_sdss.csv")
###########################################################################################

def estudio_correlacion_estocastica(df):
    todas_variables = [
        'REDSHIFT', 'LOG_MS', 'SFR', 'EA', 'MET', 'RADIO_P', 'G_R',
        'REDSHIFT_ERR', 'LOG_MS_ERR', 'SFR_ERR', 'EA_ERR', 'MET_ERR', 'RADIO_P_ERR', 'G_R_ERR'
    ]
    
    columnas_validas = [col for col in todas_variables if col in df.columns]
    df_corr = df[columnas_validas]
    
    columnas_no_constantes = df_corr.columns[df_corr.nunique() > 1]
    columnas_constantes = set(df_corr.columns) - set(columnas_no_constantes)
    
    if columnas_constantes:
        print(f"\nColumnas excluidas por ser constantes: {', '.join(columnas_constantes)}")
            
    df_final = df_corr[columnas_no_constantes]
    matriz_corr = df_final.corr(method='pearson').abs()
    mask = np.tril(np.ones_like(matriz_corr, dtype=bool), k=-1)
    
    plt.figure(figsize=(14, 11))
    ax = plt_sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap='Reds', vmin=0, vmax=1, square=True, linewidths=.5, mask=mask, cbar_kws={"shrink": .8, "label": "Valor absoluto de Pearson (|r|)"})
    
    num_nominales = len([col for col in columnas_no_constantes if not col.endswith('_ERR')])
    ax.axhline(num_nominales, color='black', linewidth=1.5, alpha=0.8, linestyle="--")
    ax.axvline(num_nominales, color='black', linewidth=1.5, alpha=0.8, linestyle="--")
    
    plt.title("Matriz de correlación (parámetros y errores)", fontsize=16, pad=5)
    plt.tight_layout()
    plt.show()

def main():
    if not os.path.exists(RUTA_CSV_SDSS):
        print(f"Error: no se encuentra {RUTA_CSV_SDSS}")
        return

    df_sdss = pd.read_csv(RUTA_CSV_SDSS)
    df_limpio = df_sdss[(df_sdss > -100).all(axis=1)].dropna()

    estudio_correlacion_estocastica(df_limpio)

if __name__ == "__main__":
    main()