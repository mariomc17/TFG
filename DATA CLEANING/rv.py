import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

###########################################################################################

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DIR_FILTRADAS = os.path.join(REPO_ROOT, "galaxias_sdss_filtradas")
DIR_MORADAS = os.path.join(REPO_ROOT, "galaxias_moradas")

NOMBRE_NORMAL = "1237648704054886645.jpg"
NOMBRE_ANOMALA = "1237648703503794332.jpg"

RUTA_GALAXIA_NORMAL = os.path.join(DIR_FILTRADAS, NOMBRE_NORMAL)
RUTA_GALAXIA_ANOMALA = os.path.join(DIR_MORADAS, NOMBRE_ANOMALA)

###########################################################################################

def procesar_galaxia(ruta_imagen):
    img = np.array(Image.open(ruta_imagen).convert('RGB'))
    
    h, w, _ = img.shape
    cy, cx = h // 2, w // 2
    
    recorte = img[cy-64:cy+64, cx-64:cx+64]
    
    R_media = np.mean(recorte[:, :, 0])
    G_media = np.mean(recorte[:, :, 1])
    B_media = np.mean(recorte[:, :, 2])
    
    epsilon = 1e-6
    RV = G_media / (R_media + B_media + epsilon)
    
    return img, cx, cy, R_media, G_media, B_media, RV

def main():
    T_SUPERTITULO = 19
    T_SUBTITULO = 16
    T_EJES = 14.5
    T_TICKS_LEYENDA = 13

    if not os.path.exists(RUTA_GALAXIA_NORMAL) or not os.path.exists(RUTA_GALAXIA_ANOMALA):
        print(f"Error: mo se encuentran las imágenes.")
        return

    img_norm, cx_norm, cy_norm, R_norm, G_norm, B_norm, RV_norm = procesar_galaxia(RUTA_GALAXIA_NORMAL)
    img_anom, cx_anom, cy_anom, R_anom, G_anom, B_anom, RV_anom = procesar_galaxia(RUTA_GALAXIA_ANOMALA)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle('Análisis fotométrico del núcleo galáctico (recorte de 128x128 px)', fontsize=T_SUPERTITULO, y=1.05)
    
    ax_img_norm = axes[0]
    ax_bar_norm = axes[1]
    ax_img_anom = axes[2]
    ax_bar_anom = axes[3]

    colores = ['#d62728', '#2ca02c', '#1f77b4']
    canales = ['Rojo (R)', 'Verde (G)', 'Azul (B)']

    ax_img_norm.imshow(img_norm)
    ax_img_norm.set_title('Imagen válida', fontsize=T_SUBTITULO, pad=10)
    ax_img_norm.axis('off')
    
    rect_norm = patches.Rectangle((cx_norm - 64, cy_norm - 64), 128, 128, linewidth=2.5, edgecolor='red', facecolor='none')
    ax_img_norm.add_patch(rect_norm)

    ax_bar_norm.bar(canales, [R_norm, G_norm, B_norm], color=colores, alpha=0.85, edgecolor='black', linewidth=1.2)
    ax_bar_norm.set_title('Intensidad promedio en el centro', fontsize=T_SUBTITULO, pad=15)
    ax_bar_norm.set_ylabel('Intensidad fotométrica', fontsize=T_EJES)
    ax_bar_norm.tick_params(axis='both', labelsize=T_TICKS_LEYENDA)
    
    texto_norm = f'RV = {RV_norm:.2f}'
    ax_bar_norm.text(1, max(R_norm, G_norm, B_norm) + 15, texto_norm, ha='center', fontsize=T_TICKS_LEYENDA,
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.6'))

    ax_img_anom.imshow(img_anom)
    ax_img_anom.set_title('Imagen saturada en morado', fontsize=T_SUBTITULO, pad=10)
    ax_img_anom.axis('off')
    
    rect_anom = patches.Rectangle((cx_anom - 64, cy_anom - 64), 128, 128, linewidth=2.5, edgecolor='red', facecolor='none')
    ax_img_anom.add_patch(rect_anom)

    ax_bar_anom.bar(canales, [R_anom, G_anom, B_anom], color=colores, alpha=0.85, edgecolor='black', linewidth=1.2)
    ax_bar_anom.set_title('Intensidad promedio en el centro', fontsize=T_SUBTITULO, pad=15)
    ax_bar_anom.tick_params(axis='both', labelsize=T_TICKS_LEYENDA)
    
    color_borde = 'darkred' if RV_anom < 0.48 else 'gray'
    color_texto = 'darkred' if RV_anom < 0.48 else 'black'
    texto_anom = f'RV = {RV_anom:.2f}\n(Descartada)'
    
    ax_bar_anom.text(1, max(R_anom, G_anom, B_anom) + 15, texto_anom, ha='center', fontsize=T_TICKS_LEYENDA, color=color_texto,
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor=color_borde, boxstyle='round,pad=0.6'))

    y_max_dinamico = min(max(R_norm, G_norm, B_norm, R_anom, G_anom, B_anom) + 50, 265)
    
    for ax in [ax_bar_norm, ax_bar_anom]:
        ax.set_ylim(0, y_max_dinamico)
        ax.grid(axis='y', linestyle=':', color='gray', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, wspace=0.25) 
    
    ruta_salida = os.path.join(SCRIPT_DIR, "analisis_completo_ratio_verde_1x4.png")
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()