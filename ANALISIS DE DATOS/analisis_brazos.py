import os
import pandas as pd
from PIL import Image
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.lines as mlines
import random
import json 
import seaborn as sns


###########################################################################################

OBJID_OBJETIVO = "1237648704057573661" 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..\.."))

DIR_IMAGENES = os.path.join(REPO_ROOT, "galaxias_elipse") 
DIR_CSV = os.path.join(REPO_ROOT, "csv")
DIR_GRAFICAS = os.path.join(REPO_ROOT, "graficas")

RUTA_CSV_SDSS = os.path.join(DIR_CSV, "galaxias_sdss.csv")
RUTA_CSV_CIELO = os.path.join(DIR_CSV, "estadisticas_cielo.csv")
RUTA_CSV_SALIDA = os.path.join(DIR_CSV, "estadisticas_galaxias.csv")
RUTA_CSV_UNET = os.path.join(DIR_CSV, "analisis_brazos.csv")

if not os.path.exists(DIR_CSV):
    os.makedirs(DIR_CSV)

if not os.path.exists(DIR_GRAFICAS):
    os.makedirs(DIR_GRAFICAS)

MAX_ARCHIVOS = 10
LADO_P = 8

###########################################################################################

T_SUPERTITULO = 19
T_SUBTITULO = 16
T_EJES = 14.5
T_TICKS_LEYENDA = 13

plt.rcParams['axes.edgecolor'] = "#000000"
plt.rcParams['axes.linewidth'] = 1.2

def main():
    start_time = time.time()

    if not os.path.exists(DIR_IMAGENES):
        print(f"Error: no se encuentra la carpeta {DIR_IMAGENES}")
        return

    archivos_en_carpeta = [f for f in os.listdir(DIR_IMAGENES) if f.lower().endswith('.png')]
    print(f"Hay {len(archivos_en_carpeta)} imágenes PNG en la carpeta.")

    if not os.path.exists(RUTA_CSV_CIELO) or not os.path.exists(RUTA_CSV_SDSS):
        print("Faltan los archivos CSV necesarios para cruzar los datos.")
        return

    df_mario = pd.read_csv(RUTA_CSV_SDSS, dtype={'OBJID': str})
    df_cielo = pd.read_csv(RUTA_CSV_CIELO, dtype={'OBJID': str})
    df_cielo['OBJID'] = df_cielo['OBJID'].astype(str)
    cielo_dict = df_cielo.set_index('OBJID').to_dict('index')

    df_validos = df_cielo[df_cielo['OBJID'].apply(lambda x: os.path.exists(os.path.join(DIR_IMAGENES, f"{x}.png")))]
    
    if MAX_ARCHIVOS is not None:
        df_validos = df_validos.head(MAX_ARCHIVOS)
        
    total_imgs = len(df_validos)
    print(f"Se procesarán {total_imgs} imágenes.")

    total_parches_max = 64 * 64 
    espiral_relativa = []
    gx, gy = 0, 0
    dx, dy = 1, 0
    pasos_tramo, longitud_tramo, giros = 0, 1, 0

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

    def extraer_datos_galaxia(objid, use_tqdm=False):
        img_path = os.path.join(DIR_IMAGENES, f"{objid}.png")
        datos_cielo = cielo_dict[objid]
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)
        h_img, w_img = img_array.shape[:2]
        c_y, c_x = h_img // 2, w_img // 2
        
        bg_rgb = np.clip([datos_cielo['LC_R'], datos_cielo['LC_G'], datos_cielo['LC_B']], 0, 255).astype(np.uint8)
        mascara_galaxia = ~np.all(img_array == bg_rgb, axis=-1)
        
        r_m, g_m, b_m = [], [], []
        pos = []
        
        iterador = tqdm(espiral_relativa, desc=f"Analizando parches de {objid}") if use_tqdm else espiral_relativa
        
        for (gy_rel, gx_rel) in iterador:
            py, px = int(c_y - (LADO_P // 2) + (gy_rel * LADO_P)), int(c_x - (LADO_P // 2) + (gx_rel * LADO_P))
            if 0 <= py <= h_img - LADO_P and 0 <= px <= w_img - LADO_P:
                if np.any(mascara_galaxia[py:py+LADO_P, px:px+LADO_P]):
                    pos.append((py, px))
                    sec_rgb = img_array[py:py+LADO_P, px:px+LADO_P]
                    r_m.append(np.mean(sec_rgb[:,:,0]))
                    g_m.append(np.mean(sec_rgb[:,:,1]))
                    b_m.append(np.mean(sec_rgb[:,:,2]))
        
        return r_m, g_m, b_m, pos, img, bg_rgb, img_path

    if OBJID_OBJETIVO in cielo_dict and os.path.exists(os.path.join(DIR_IMAGENES, f"{OBJID_OBJETIVO}.png")):
        
        r_vals, g_vals, b_vals, posiciones, img_plot, _, _ = extraer_datos_galaxia(OBJID_OBJETIVO, use_tqdm=True)
        n_p = len(posiciones)
        
        if n_p > 0:
            total_int = sum(r_vals) + sum(g_vals) + sum(b_vals)
            p_r = (sum(r_vals) / (total_int + 1e-6)) * 100
            p_g = (sum(g_vals) / (total_int + 1e-6)) * 100
            p_b = (sum(b_vals) / (total_int + 1e-6)) * 100

            fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={'width_ratios': [1, 1.5]})
            fig3.suptitle(f"Análisis RGB - {OBJID_OBJETIVO}", fontsize=T_SUPERTITULO)
            ax1.set_title("Morfología y muestreo", fontsize=T_SUBTITULO)

            ax1.imshow(img_plot); ax1.axis('off')
            for py, px in posiciones:
                ax1.add_patch(patches.Rectangle((px, py), LADO_P, LADO_P, linewidth=0.4, edgecolor='cyan', facecolor='none', alpha=0.3))

            x = np.arange(1, n_p + 1); bw = 0.3
            ax2.bar(x - bw, r_vals, width=bw, color="#EB1224", label='Canal Rojo (R)', alpha=0.9, zorder=3)
            ax2.bar(x, g_vals, width=bw, color="#13BB5F", label='Canal Verde (G)', alpha=0.9, zorder=3)
            ax2.bar(x + bw, b_vals, width=bw, color="#1E9AE7", label='Canal Azul (B)', alpha=0.9, zorder=3)
            
            ax2.set_title("Intensidad por parche", fontsize=T_SUBTITULO)
            ax2.set_xlabel('Nº de parche', fontsize=T_EJES)
            ax2.set_ylabel('Intensidad', fontsize=T_EJES)
            ax2.grid(True, linestyle='--', alpha=0.4, axis='y')
            ax2.tick_params(axis='both', which='major', labelsize=T_TICKS_LEYENDA)
            ax2.legend(frameon=True, fontsize=T_TICKS_LEYENDA, loc='upper right', bbox_to_anchor=(0.98, 0.80))
            
            info_text = f"Canales:\n● Rojo: {p_r:.1f}%\n● Verde: {p_g:.1f}%\n● Azul: {p_b:.1f}%"
            ax2.text(0.98, 0.97, info_text, transform=ax2.transAxes, fontsize=T_TICKS_LEYENDA, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            plt.tight_layout()
            ruta_fig3 = os.path.join(DIR_GRAFICAS, f"analisis_rgb_{OBJID_OBJETIVO}.png")
            fig3.savefig(ruta_fig3, dpi=300, bbox_inches='tight')
            print(f"Gráfica guardada: {ruta_fig3}")
            plt.show(block=False)

            p_zoom = min(100, n_p)
            fig4, (ax3, ax4) = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={'width_ratios': [1, 1.5]})
            fig4.suptitle(f"Detalle primeros {p_zoom} parches - {OBJID_OBJETIVO}", fontsize=T_SUPERTITULO)
            ax3.set_title("Morfología y muestreo", fontsize=T_SUBTITULO)

            ax3.imshow(img_plot)
            c_x_img, c_y_img = img_plot.width // 2, img_plot.height // 2
            mz = 60
            ax3.set_xlim(c_x_img - mz, c_x_img + mz); ax3.set_ylim(c_y_img + mz, c_y_img - mz)
            
            for idx, (py, px) in enumerate(posiciones[:p_zoom]):
                ax3.add_patch(patches.Rectangle((px, py), LADO_P, LADO_P, linewidth=1, edgecolor='black', facecolor='none'))
                ax3.text(px+4, py+4, str(idx+1), color='white', fontsize=7, ha='center', va='center',
                         path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
            ax3.axis('off')

            xz = np.arange(1, p_zoom + 1)
            ax4.bar(xz - bw, r_vals[:p_zoom], width=bw, color="#EB1224", label='Canal Rojo (R)', alpha=0.9, zorder=3)
            ax4.bar(xz, g_vals[:p_zoom], width=bw, color="#13BB5F", label='Canal Verde (G)', alpha=0.9, zorder=3)
            ax4.bar(xz + bw, b_vals[:p_zoom], width=bw, color="#1E9AE7", label='Canal Azul (B)', alpha=0.9, zorder=3)
            
            ax4.set_title("Intensidad por parche", fontsize=T_SUBTITULO)
            ax4.set_xlabel("Nº de parche", fontsize=T_EJES)
            ax4.set_ylabel("Intensidad", fontsize=T_EJES)
            ax4.grid(True, alpha=0.3, zorder=0)
            ax4.tick_params(axis='both', which='major', labelsize=T_TICKS_LEYENDA)
            ax4.legend(frameon=True, fontsize=T_TICKS_LEYENDA, loc='upper right', bbox_to_anchor=(0.98, 0.85))

            plt.tight_layout()
            ruta_fig4 = os.path.join(DIR_GRAFICAS, f"detalle_zoom_{OBJID_OBJETIVO}.png")
            fig4.savefig(ruta_fig4, dpi=300, bbox_inches='tight')
            print(f"Gráfica guardada: {ruta_fig4}")
            plt.show(block=False)


            fig5, axs = plt.subplots(2, 2, figsize=(20, 18), gridspec_kw={'width_ratios': [1, 1.5]})
            fig5.suptitle(f"Análisis RGB - {OBJID_OBJETIVO}", fontsize=T_SUPERTITULO + 2, y=1.001)
            
            axs[0, 0].imshow(img_plot)
            axs[0, 0].axis('off')
            axs[0, 0].set_title("Morfología y muestreo (completo)", fontsize=T_SUBTITULO)
            for py, px in posiciones:
                axs[0, 0].add_patch(patches.Rectangle((px, py), LADO_P, LADO_P, linewidth=0.4, edgecolor='cyan', facecolor='none', alpha=0.3))

            axs[0, 1].bar(x - bw, r_vals, width=bw, color="#EB1224", label='Canal Rojo (R)', alpha=0.9, zorder=3)
            axs[0, 1].bar(x, g_vals, width=bw, color="#13BB5F", label='Canal Verde (G)', alpha=0.9, zorder=3)
            axs[0, 1].bar(x + bw, b_vals, width=bw, color="#1E9AE7", label='Canal Azul (B)', alpha=0.9, zorder=3)
            axs[0, 1].set_title("Intensidad por parche (completo)", fontsize=T_SUBTITULO)
            axs[0, 1].set_xlabel('Nº de parche', fontsize=T_EJES)
            axs[0, 1].set_ylabel('Intensidad', fontsize=T_EJES)
            axs[0, 1].grid(True, linestyle='--', alpha=0.4, axis='y')
            axs[0, 1].tick_params(axis='both', which='major', labelsize=T_TICKS_LEYENDA)
            axs[0, 1].legend(frameon=True, fontsize=T_TICKS_LEYENDA, loc='upper right', bbox_to_anchor=(0.98, 0.80))
            axs[0, 1].text(0.98, 0.97, info_text, transform=axs[0, 1].transAxes, fontsize=T_TICKS_LEYENDA, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            axs[1, 0].imshow(img_plot)
            axs[1, 0].set_xlim(c_x_img - mz, c_x_img + mz)
            axs[1, 0].set_ylim(c_y_img + mz, c_y_img - mz)
            axs[1, 0].set_title("Morfología y muestreo (zoom)", fontsize=T_SUBTITULO)
            for idx, (py, px) in enumerate(posiciones[:p_zoom]):
                axs[1, 0].add_patch(patches.Rectangle((px, py), LADO_P, LADO_P, linewidth=1, edgecolor='black', facecolor='none'))
                axs[1, 0].text(px+4, py+4, str(idx+1), color='white', fontsize=7, ha='center', va='center',
                         path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
            axs[1, 0].axis('off')

            axs[1, 1].bar(xz - bw, r_vals[:p_zoom], width=bw, color="#EB1224", label='Canal Rojo (R)', alpha=0.9, zorder=3)
            axs[1, 1].bar(xz, g_vals[:p_zoom], width=bw, color="#13BB5F", label='Canal Verde (G)', alpha=0.9, zorder=3)
            axs[1, 1].bar(xz + bw, b_vals[:p_zoom], width=bw, color="#1E9AE7", label='Canal Azul (B)', alpha=0.9, zorder=3)
            axs[1, 1].set_title("Intensidad por parche (zoom)", fontsize=T_SUBTITULO)
            axs[1, 1].set_xlabel("Nº de parche", fontsize=T_EJES)
            axs[1, 1].set_ylabel("Intensidad", fontsize=T_EJES)
            axs[1, 1].grid(True, alpha=0.3, zorder=0)
            axs[1, 1].tick_params(axis='both', which='major', labelsize=T_TICKS_LEYENDA)
            axs[1, 1].legend(frameon=True, fontsize=T_TICKS_LEYENDA, loc='upper right', bbox_to_anchor=(0.98, 0.85))

            fig5.tight_layout()
            ruta_fig5 = os.path.join(DIR_GRAFICAS, f"combinado_{OBJID_OBJETIVO}.png")
            fig5.savefig(ruta_fig5, dpi=300, bbox_inches='tight')
            
            plt.show(block=False)
            plt.pause(0.1)

    else:
        print(f"Error: el OBJID '{OBJID_OBJETIVO}' no se encuentra en el dataset o no tiene imagen.")
    
    resultados_todas = []
    resultados_unet = [] 

    for index, row in tqdm(df_validos.iterrows(), total=total_imgs, desc="Analizando siluetas"):
        objid = str(row['OBJID'])
        if objid not in cielo_dict: continue
            
        try:
            r_means, g_means, b_means, posiciones_reales, _, bg_rgb, img_path = extraer_datos_galaxia(objid)
                
            if posiciones_reales:
                total_int = sum(r_means) + sum(g_means) + sum(b_means)
                p_r = (sum(r_means) / (total_int + 1e-6)) * 100
                p_g = (sum(g_means) / (total_int + 1e-6)) * 100
                p_b = (sum(b_means) / (total_int + 1e-6)) * 100

                resultados_todas.append({
                    'OBJID': objid, 
                    'r_means': json.dumps(r_means),
                    'g_means': json.dumps(g_means), 
                    'b_means': json.dumps(b_means),
                    'img_path': img_path,
                    'posiciones': posiciones_reales, 
                    'total_parches': len(posiciones_reales),
                    'p_r': p_r, 'p_g': p_g, 'p_b': p_b
                })

                pad_length = total_parches_max - len(r_means)
                r_unet = r_means + [float(bg_rgb[0])] * pad_length
                g_unet = g_means + [float(bg_rgb[1])] * pad_length
                b_unet = b_means + [float(bg_rgb[2])] * pad_length
                
                resultados_unet.append({
                    'OBJID': objid,
                    'R_array': json.dumps(r_unet),
                    'G_array': json.dumps(g_unet),
                    'B_array': json.dumps(b_unet)
                })

        except Exception as e:
            print(f"Error procesando {objid}: {e}")

    if resultados_todas:
        columnas_estadisticas = [{
            'OBJID': d['OBJID'], 
            'total_parches': d['total_parches'], 
            'p_r': d['p_r'], 
            'p_g': d['p_g'], 
            'p_b': d['p_b']
        } for d in resultados_todas]
        
        df_global = pd.DataFrame(columnas_estadisticas)
        df_global.to_csv(RUTA_CSV_SALIDA, index=False)
        print(f"\nDatos estadísticos exportados a: {RUTA_CSV_SALIDA}")

        df_unet = pd.DataFrame(resultados_unet)
        df_unet.to_csv(RUTA_CSV_UNET, index=False)
        print(f"Tensores RGB exportados a: {RUTA_CSV_UNET}")
        
        df_global['OBJID'] = df_global['OBJID'].astype(str)
        df_mario['OBJID'] = df_mario['OBJID'].astype(str)
        df = pd.merge(df_global, df_mario, on='OBJID', how='inner')
        
        fig1, (ax_scat1, ax_box) = plt.subplots(1, 2, figsize=(18, 8))
        fig1.suptitle("Estadísticas globales de la muestra", fontsize=T_SUPERTITULO)

        colors_rgb = np.array([df['p_r'], df['p_g'], df['p_b']]).T / 100.0
        ax_scat1.scatter(df['p_r'], df['p_b'], c=colors_rgb, s=60, edgecolors='black', linewidth=0.3, alpha=0.8)
        
        ax_scat1.set_title("Mapa de la secuencia de Hubble", fontsize=T_SUBTITULO)
        ax_scat1.set_xlabel("% Intensidad en el canal Rojo (R)", fontsize=T_EJES)
        ax_scat1.set_ylabel("% Intensidad en canal Azul (B)", fontsize=T_EJES)
        ax_scat1.tick_params(axis='both', which='major', labelsize=T_TICKS_LEYENDA)
        ax_scat1.grid(True, linestyle='--', alpha=0.5)
        ax_scat1.text(0.05, 0.95, "Galaxias tardías (azules)", transform=ax_scat1.transAxes, color="#1E9AE7", va='top', fontsize=T_TICKS_LEYENDA)
        ax_scat1.text(0.95, 0.05, "Galaxias tempranas (rojas)", transform=ax_scat1.transAxes, color="#EB1224", ha='right', fontsize=T_TICKS_LEYENDA)

        bp = ax_box.boxplot([df['p_r'], df['p_g'], df['p_b']], patch_artist=True, tick_labels=['% Rojo', '% Verde', '% Azul'])
        colors = ["#EB1224", "#13BB5F", "#1E9AE7"]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax_box.set_title("Diagrama $boxplot$", fontsize=T_SUBTITULO)
        ax_box.set_ylabel("Porcentaje de contribución (%)", fontsize=T_EJES)
        ax_box.tick_params(axis='both', which='major', labelsize=T_TICKS_LEYENDA)
        ax_box.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        leyenda_caja = patches.Rectangle((0,0),1,1, edgecolor='black')
        leyenda_mediana = mlines.Line2D([], [], color='#ff7f0e', linewidth=2) 
        leyenda_bigotes = mlines.Line2D([], [], color='black', linewidth=1.5)
        leyenda_outliers = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markerfacecolor='none')
        
        ax_box.legend(
            [leyenda_caja, leyenda_mediana, leyenda_bigotes, leyenda_outliers],
            ['Caja: IQR', 'Línea: mediana', 'Bigotes (Tukey): extremos típicos', 'Círculos: outliers'],
            loc='best', fontsize=T_TICKS_LEYENDA-2, framealpha=0.9
        )
        fig1.tight_layout()

        ruta_fig1 = os.path.join(DIR_GRAFICAS, "estadisticas_globales.png")
        fig1.savefig(ruta_fig1, dpi=300, bbox_inches='tight')
        print(f"\nGráfica guardada: {ruta_fig1}")

        plt.show()

    print(f"Proceso finalizado en {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()