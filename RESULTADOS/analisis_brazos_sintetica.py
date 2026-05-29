import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

###########################################################################################

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..\.."))
DIR_APP = os.path.join(REPO_ROOT, "figuras")
RUTA_GALAXIA_SINTETICA = os.path.join(DIR_APP, "rgb1_sintetica.png")

LADO_P = 2 

T_SUPERTITULO = 19
T_SUBTITULO = 16
T_EJES = 14.5
T_TICKS_LEYENDA = 13

###########################################################################################

plt.rcParams['axes.edgecolor'] = "#000000"
plt.rcParams['axes.linewidth'] = 1.2

def analizar_galaxia_sintetica(img_path):
    if not os.path.exists(img_path):
        print(f"Error: no se encuentra el archivo.")
        return

    nombre_archivo = os.path.basename(img_path)

    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)
    h_img, w_img = img_array.shape[:2]
    c_y, c_x = h_img // 2, w_img // 2

    brillos_borde = []
    
    margen_inferior = h_img - (LADO_P * 2) 
    
    for y in range(margen_inferior, h_img, LADO_P):
        for x in range(0, w_img, LADO_P):
            parche = img_array[y:y+LADO_P, x:x+LADO_P]
            brillo_parche = np.mean(np.sum(parche, axis=-1))
            brillos_borde.append(brillo_parche)

    media_borde = np.mean(brillos_borde)
    std_borde = np.std(brillos_borde)
    
    umbral_cielo = max(media_borde + (3 * std_borde), 15.0)

    espiral_relativa = []
    gx, gy = 0, 0
    dx, dy = 1, 0
    pasos_tramo, longitud_tramo, giros = 0, 1, 0
    total_parches_max = (h_img // LADO_P) * (w_img // LADO_P)

    while len(espiral_relativa) < total_parches_max:
        espiral_relativa.append((gy, gx))
        gx += dx
        gy += dy
        pasos_tramo += 1
        if pasos_tramo == longitud_tramo:
            pasos_tramo = 0
            dx, dy = -dy, dx
            giros += 1
            if giros % 2 == 0:
                longitud_tramo += 1

    r_means, g_means, b_means = [], [], []
    posiciones_reales = []
    
    limite_y_texto = int(h_img * 0.18) 

    for (gy_rel, gx_rel) in espiral_relativa:
        py, px = int(c_y - (LADO_P // 2) + (gy_rel * LADO_P)), int(c_x - (LADO_P // 2) + (gx_rel * LADO_P))
        
        if limite_y_texto <= py <= h_img - LADO_P and 0 <= px <= w_img - LADO_P:
            sec_rgb = img_array[py:py+LADO_P, px:px+LADO_P]
            brillo_actual = np.mean(np.sum(sec_rgb, axis=-1))
            
            if brillo_actual > umbral_cielo:
                posiciones_reales.append((py, px))
                r_means.append(np.mean(sec_rgb[:,:,0]))
                g_means.append(np.mean(sec_rgb[:,:,1]))
                b_means.append(np.mean(sec_rgb[:,:,2]))

    n_p = len(posiciones_reales)

    total_int = sum(r_means) + sum(g_means) + sum(b_means)
    p_r = (sum(r_means) / (total_int + 1e-6)) * 100
    p_g = (sum(g_means) / (total_int + 1e-6)) * 100
    p_b = (sum(b_means) / (total_int + 1e-6)) * 100

    info_text = f"Canales:\n● Rojo: {p_r:.1f}%\n● Verde: {p_g:.1f}%\n● Azul: {p_b:.1f}%"
    bw = 0.3
    x = np.arange(1, n_p + 1)
    
    id_plot = f"Sintética ({nombre_archivo})"

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={'width_ratios': [1, 1.5]})
    fig1.suptitle(f"Análisis RGB - Sintética 1", fontsize=T_SUPERTITULO)
    
    ax1.imshow(img); ax1.axis('off')
    ax1.set_title("Morfología y muestreo", fontsize=T_SUBTITULO)
    
    for py, px in posiciones_reales:
        ax1.add_patch(patches.Rectangle((px, py), LADO_P, LADO_P, linewidth=0.4, edgecolor='cyan', facecolor='none', alpha=0.3))

    ax2.bar(x - bw, r_means, width=bw, color="#EB1224", label='Canal Rojo (R)', alpha=0.9, zorder=3)
    ax2.bar(x, g_means, width=bw, color="#13BB5F", label='Canal Verde (G)', alpha=0.9, zorder=3)
    ax2.bar(x + bw, b_means, width=bw, color="#1E9AE7", label='Canal Azul (B)', alpha=0.9, zorder=3)
    
    ax2.set_title("Intensidad por parche", fontsize=T_SUBTITULO)
    ax2.set_xlabel('Nº de parche', fontsize=T_EJES)
    ax2.set_ylabel('Intensidad', fontsize=T_EJES)
    ax2.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax2.tick_params(axis='both', which='major', labelsize=T_TICKS_LEYENDA)
    ax2.legend(frameon=True, fontsize=T_TICKS_LEYENDA, loc='upper right', bbox_to_anchor=(0.98, 0.80))
    
    ax2.text(0.98, 0.97, info_text, transform=ax2.transAxes, fontsize=T_TICKS_LEYENDA, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    
    ruta_fig1 = os.path.join(SCRIPT_DIR, f"analisis_rgb_sintetica.png")
    fig1.savefig(ruta_fig1, dpi=300, bbox_inches='tight')
    print(f"\nGráfica guardada: {ruta_fig1}")
    plt.show(block=False)

    p_zoom = min(100, n_p)
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={'width_ratios': [1, 1.5]})
    fig2.suptitle(f"Detalle primeros {p_zoom} parches - {id_plot}", fontsize=T_SUPERTITULO, y=0.98)
    
    ax3.set_title("Morfología y muestreo", fontsize=T_SUBTITULO)
    ax3.imshow(img)
    mz = 30
    ax3.set_xlim(c_x - mz, c_x + mz); ax3.set_ylim(c_y + mz, c_y - mz)
    
    for idx, (py, px) in enumerate(posiciones_reales[:p_zoom]):
        ax3.add_patch(patches.Rectangle((px, py), LADO_P, LADO_P, linewidth=1, edgecolor='black', facecolor='none'))
        ax3.text(px+(LADO_P/2.0), py+(LADO_P/2.0), str(idx+1), color='white', fontsize=4, ha='center', va='center',
                 path_effects=[path_effects.withStroke(linewidth=1, foreground='black')])
    ax3.axis('off')

    xz = np.arange(1, p_zoom + 1)
    ax4.bar(xz - bw, r_means[:p_zoom], width=bw, color="#EB1224", label='Canal Rojo (R)', alpha=0.9, zorder=3)
    ax4.bar(xz, g_means[:p_zoom], width=bw, color="#13BB5F", label='Canal Verde (G)', alpha=0.9, zorder=3)
    ax4.bar(xz + bw, b_means[:p_zoom], width=bw, color="#1E9AE7", label='Canal Azul (B)', alpha=0.9, zorder=3)
    
    ax4.set_title("Intensidad por parche", fontsize=T_SUBTITULO)
    ax4.set_xlabel("Nº de parche", fontsize=T_EJES)
    ax4.set_ylabel("Intensidad", fontsize=T_EJES)
    ax4.grid(True, alpha=0.3, zorder=0)
    ax4.tick_params(axis='both', which='major', labelsize=T_TICKS_LEYENDA)
    ax4.legend(frameon=True, fontsize=T_TICKS_LEYENDA, loc='upper right', bbox_to_anchor=(0.98, 0.85))

    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    
    ruta_fig2 = os.path.join(SCRIPT_DIR, f"detalle_zoom_sintetica.png")
    fig2.savefig(ruta_fig2, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada: {ruta_fig2}")
    plt.show(block=False)

    fig3, axs = plt.subplots(2, 2, figsize=(20, 18), gridspec_kw={'width_ratios': [1, 1.5]})
    fig3.suptitle(f"Análisis RGB - Sintética 1", fontsize=T_SUPERTITULO + 2, y=0.95)
    
    axs[0, 0].imshow(img)
    axs[0, 0].axis('off')
    axs[0, 0].set_title("Morfología y muestreo (completo)", fontsize=T_SUBTITULO)
    for py, px in posiciones_reales:
        axs[0, 0].add_patch(patches.Rectangle((px, py), LADO_P, LADO_P, linewidth=0.4, edgecolor='cyan', facecolor='none', alpha=0.3))

    axs[0, 1].bar(x - bw, r_means, width=bw, color="#EB1224", label='Canal Rojo (R)', alpha=0.9, zorder=3)
    axs[0, 1].bar(x, g_means, width=bw, color="#13BB5F", label='Canal Verde (G)', alpha=0.9, zorder=3)
    axs[0, 1].bar(x + bw, b_means, width=bw, color="#1E9AE7", label='Canal Azul (B)', alpha=0.9, zorder=3)
    axs[0, 1].set_title("Intensidad por parche (completo)", fontsize=T_SUBTITULO)
    axs[0, 1].set_xlabel('Nº de parche', fontsize=T_EJES)
    axs[0, 1].set_ylabel('Intensidad', fontsize=T_EJES)
    axs[0, 1].grid(True, linestyle='--', alpha=0.4, axis='y')
    axs[0, 1].tick_params(axis='both', which='major', labelsize=T_TICKS_LEYENDA)
    axs[0, 1].legend(frameon=True, fontsize=T_TICKS_LEYENDA, loc='upper right', bbox_to_anchor=(0.98, 0.80))
    axs[0, 1].text(0.98, 0.97, info_text, transform=axs[0, 1].transAxes, fontsize=T_TICKS_LEYENDA, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    axs[1, 0].imshow(img)
    axs[1, 0].set_xlim(c_x - mz, c_x + mz)
    axs[1, 0].set_ylim(c_y + mz, c_y - mz)
    axs[1, 0].set_title("Morfología y muestreo (zoom)", fontsize=T_SUBTITULO)
    for idx, (py, px) in enumerate(posiciones_reales[:p_zoom]):
        axs[1, 0].add_patch(patches.Rectangle((px, py), LADO_P, LADO_P, linewidth=1, edgecolor='black', facecolor='none'))
        axs[1, 0].text(px+(LADO_P/2.0), py+(LADO_P/2.0), str(idx+1), color='white', fontsize=4, ha='center', va='center',
                 path_effects=[path_effects.withStroke(linewidth=1, foreground='black')])
    axs[1, 0].axis('off')

    axs[1, 1].bar(xz - bw, r_means[:p_zoom], width=bw, color="#EB1224", label='Canal Rojo (R)', alpha=0.9, zorder=3)
    axs[1, 1].bar(xz, g_means[:p_zoom], width=bw, color="#13BB5F", label='Canal Verde (G)', alpha=0.9, zorder=3)
    axs[1, 1].bar(xz + bw, b_means[:p_zoom], width=bw, color="#1E9AE7", label='Canal Azul (B)', alpha=0.9, zorder=3)
    axs[1, 1].set_title("Intensidad por parche (zoom)", fontsize=T_SUBTITULO)
    axs[1, 1].set_xlabel("Nº de parche", fontsize=T_EJES)
    axs[1, 1].set_ylabel("Intensidad", fontsize=T_EJES)
    axs[1, 1].grid(True, alpha=0.3, zorder=0)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=T_TICKS_LEYENDA)
    axs[1, 1].legend(frameon=True, fontsize=T_TICKS_LEYENDA, loc='upper right', bbox_to_anchor=(0.98, 0.85))

    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    
    ruta_fig3 = os.path.join(SCRIPT_DIR, f"combinado_sintetica.png")
    fig3.savefig(ruta_fig3, dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    analizar_galaxia_sintetica(RUTA_GALAXIA_SINTETICA)