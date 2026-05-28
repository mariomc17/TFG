import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
import matplotlib.gridspec as gridspec
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DIR_CSV = os.path.join(REPO_ROOT, "csv")
RUTA_CSV_SDSS = os.path.join(DIR_CSV, "galaxias_sdss.csv")

def estudio_correlacion_estocastica(df, pbar):
    variables_nominales = [
        'REDSHIFT', 'LOG_MS', 'SFR', 'EA', 'MET', 'RADIO_P', 'G_R', 'ESCALA_KPC_PX'
    ]
    
    columnas_validas = [col for col in variables_nominales if col in df.columns]
    df_corr = df[columnas_validas]
    
    columnas_no_constantes = df_corr.columns[df_corr.nunique() > 1]
    
    df_final = df_corr[columnas_no_constantes]
    pbar.update(1)

    limite_inf = df_final.quantile(0.01)
    limite_sup = df_final.quantile(0.99)
    
    mascara = ~((df_final < limite_inf) | (df_final > limite_sup)).any(axis=1)
    df_filtrado = df_final[mascara]
    print(f"Galaxias originales: {len(df_final)} | Galaxias tras el filtro: {len(df_filtrado)}")
    
    matriz_corr = df_filtrado.corr(method='pearson').abs()
    mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
    
    T_SUPERTITULO = 19
    T_SUBTITULO = 14
    T_EJES = 11
    T_TICKS_LEYENDA = 9

    # --- FIGURA 1: MATRIZ DE CORRELACIÓN ---
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    fig1.suptitle("Matriz de correlación absoluta", fontsize=T_SUPERTITULO, y=0.96)
    
    plt_sns.heatmap(
        matriz_corr, 
        annot=True, 
        fmt=".2f", 
        cmap='Reds', 
        vmin=0, 
        vmax=1, 
        square=True, 
        linewidths=.5, 
        mask=mask, 
        cbar_kws={"shrink": .8, "label": "Valor absoluto de Pearson (|r|)"},
        ax=ax1
    )
    
    ax1.tick_params(axis='both', labelsize=T_TICKS_LEYENDA)
    plt.tight_layout()
    plt.subplots_adjust(top=0.91)
    plt.show()
    pbar.update(1)   

    # --- FIGURA 2: ANÁLISIS DISTRIBUCIONAL (PAIRPLOT + HISTOGRAMAS) ---
    df_sample = df_filtrado 
    variables = df_sample.columns.tolist()
    n_variables = len(variables)
    
    # Crear una figura grande para albergar ambas secciones
    fig2 = plt.figure(figsize=(22, 12))
    fig2.suptitle("Análisis distribucional multidimensional de variables astrofísicas", y=0.98, fontsize=T_SUPERTITULO)
    
    # Dividir la figura en 2 grandes bloques (Izquierda: Dispersión | Derecha: Histogramas)
    gs_main = gridspec.GridSpec(1, 2, figure=fig2, width_ratios=[1.2, 1], wspace=0.25)
    
    # --- SUBPLOT IZQUIERDO: Pairplot manual (Solo triángulo inferior, diagonal vacía) ---
    gs_left = gridspec.GridSpecFromSubplotSpec(n_variables, n_variables, subplot_spec=gs_main[0], wspace=0.1, hspace=0.1)
    
    for i in range(n_variables):
        for j in range(n_variables):
            if j < i: # Solo generamos el triángulo inferior
                ax = fig2.add_subplot(gs_left[i, j])
                plt_sns.scatterplot(
                    data=df_sample, x=variables[j], y=variables[i], 
                    alpha=0.4, s=10, color='#1f77b4', edgecolor='none', ax=ax
                )
                
                # Configurar etiquetas solo en los bordes exteriores de la matriz
                if j == 0:
                    ax.set_ylabel(variables[i], fontsize=T_EJES)
                    ax.tick_params(labelleft=True, labelsize=T_TICKS_LEYENDA)
                else:
                    ax.set_ylabel('')
                    ax.tick_params(labelleft=False)
                    
                if i == n_variables - 1:
                    ax.set_xlabel(variables[j], fontsize=T_EJES)
                    ax.tick_params(labelbottom=True, labelsize=T_TICKS_LEYENDA)
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                else:
                    ax.set_xlabel('')
                    ax.tick_params(labelbottom=False)
                    
                ax.grid(True, linestyle=':', color='gray', linewidth=0.5, alpha=0.5)

    # --- SUBPLOT DERECHO: Cuadrícula de Histogramas ---
    # Calculamos cuántas filas necesitamos para 2 columnas de histogramas
    cols_hist = 2
    rows_hist = int(np.ceil(n_variables / cols_hist))
    gs_right = gridspec.GridSpecFromSubplotSpec(rows_hist, cols_hist, subplot_spec=gs_main[1], wspace=0.3, hspace=0.5)
    
    for idx, var in enumerate(variables):
        r = idx // cols_hist
        c = idx % cols_hist
        ax = fig2.add_subplot(gs_right[r, c])
        
        plt_sns.histplot(
            data=df_sample, x=var, kde=True, bins=50, 
            color='mediumseagreen', edgecolor='black', alpha=0.8, ax=ax
        )
        
        ax.set_ylabel("Número de galaxias", fontsize=T_EJES)
        ax.set_xlabel(f"Valor original ({var})", fontsize=T_EJES)
        ax.tick_params(axis='both', labelsize=T_TICKS_LEYENDA)
        ax.grid(axis='y', linestyle=':', color='gray', linewidth=0.7, alpha=0.7)

    # Ajustes finales y guardado
    fig2.subplots_adjust(top=0.90, bottom=0.10, left=0.05, right=0.98)
    
    ruta_salida_plot = os.path.join(SCRIPT_DIR, "analisis_distribucional.png")
    fig2.savefig(ruta_salida_plot, dpi=300, bbox_inches='tight')
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