import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..\.."))
DIR_CSV = os.path.join(REPO_ROOT, "csv")
RUTA_CSV_SDSS = os.path.join(DIR_CSV, "galaxias_sdss.csv")

def estudio_correlacion_estocastica(df, pbar):
    variables_nominales = [
        'REDSHIFT', 'LOG_MS', 'SFR', 'EA', 'MET', 'RADIO_P', 'G_R', 'ESCALA_KPC_PX'
    ]
    
    columnas_validas = [col for col in variables_nominales if col in df.columns]
    df_corr = df[columnas_validas]
    
    columnas_no_constantes = df_corr.columns[df_corr.nunique() > 1]
    columnas_constantes = set(df_corr.columns) - set(columnas_no_constantes)
    
    if columnas_constantes:
        print(f"\nColumnas excluidas por ser constantes: {', '.join(columnas_constantes)}")
            
    df_final = df_corr[columnas_no_constantes]
    pbar.update(1)

    limite_inf = df_final.quantile(0.01)
    limite_sup = df_final.quantile(0.99)
    
    mascara = ~((df_final < limite_inf) | (df_final > limite_sup)).any(axis=1)
    df_filtrado = df_final[mascara]
    print(f"Galaxias originales: {len(df_final)} | Galaxias tras el filtro: {len(df_filtrado)}")
    
    matriz_corr = df_filtrado.corr(method='pearson').abs()
    mask = np.tril(np.ones_like(matriz_corr, dtype=bool), k=-1)
    
    plt.figure(figsize=(10, 8))
    ax = plt_sns.heatmap(
        matriz_corr, 
        annot=True, 
        fmt=".2f", 
        cmap='Reds', 
        vmin=0, 
        vmax=1, 
        square=True, 
        linewidths=.5, 
        mask=mask, 
        cbar_kws={"shrink": .8, "label": "Valor absoluto de Pearson (|r|)"}
    )
    
    plt.title("Matriz de correlación absoluta", fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    pbar.update(1)   

    df_sample = df_filtrado 
        
    pair_plot = plt_sns.pairplot(
        df_sample, 
        kind="hist",           
        diag_kind="kde",       
        corner=True,
        height=2.5,
        aspect=1.1,
    )
    
    pair_plot.fig.suptitle("Análisis distribucional multidimensional de variables astrofísicas", y=0.96, fontsize=18)
    
    for ax in pair_plot.axes.flatten():
        if ax is not None:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.xaxis.label.set_size(11)
            ax.yaxis.label.set_size(11)
            ax.tick_params(labelsize=9)
            
    pair_plot.fig.subplots_adjust(top=0.90, bottom=0.12, left=0.12, right=0.96, wspace=0.22, hspace=0.22)
    
    plt.show()
    pbar.update(1)

def main():
    if not os.path.exists(RUTA_CSV_SDSS):
        print(f"Error: no se encuentra {RUTA_CSV_SDSS}")
        return

    with tqdm(total=4, desc="Progreso del análisis", unit="fase") as pbar:
        df_sdss = pd.read_csv(RUTA_CSV_SDSS)

        df_limpio = df_sdss[(df_sdss > -100).all(axis=1)].dropna()
        pbar.update(1)

        estudio_correlacion_estocastica(df_limpio, pbar)

if __name__ == "__main__":
    main()