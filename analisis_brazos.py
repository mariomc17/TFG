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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DIR_IMAGENES = os.path.join(REPO_ROOT, "galaxias_elipse") 
DIR_CSV = os.path.join(REPO_ROOT, "csv")

RUTA_CSV_SDSS = os.path.join(DIR_CSV, "galaxias_sdss.csv")
RUTA_CSV_CIELO = os.path.join(DIR_CSV, "estadisticas_cielo.csv")
RUTA_CSV_SALIDA = os.path.join(DIR_CSV, "estadisticas_galaxias.csv")
RUTA_CSV_UNET = os.path.join(DIR_CSV, "analisis_brazos.csv")

if not os.path.exists(DIR_CSV):
    os.makedirs(DIR_CSV)

MAX_ARCHIVOS = 100
LADO_P = 8
###########################################################################################

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

def main():
    start_time = time.time()

    if not os.path.exists(DIR_IMAGENES):
        print(f"Error: No se encuentra la carpeta {DIR_IMAGENES}")
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

    resultados_todas = []
    resultados_unet = [] 

    for index, row in tqdm(df_validos.iterrows(), total=total_imgs, desc="Analizando siluetas"):
        objid = str(row['OBJID'])
        img_path = os.path.join(DIR_IMAGENES, f"{objid}.png")
        
        if objid not in cielo_dict: continue
            
        try:
            datos_cielo = cielo_dict[objid]
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            h_img, w_img = img_array.shape[:2]
            c_y, c_x = h_img // 2, w_img // 2
            
            bg_rgb = np.clip([datos_cielo['LC_R'], datos_cielo['LC_G'], datos_cielo['LC_B']], 0, 255).astype(np.uint8)
            mascara_galaxia = ~np.all(img_array == bg_rgb, axis=-1)
            
            r_means, g_means, b_means = [], [], []
            posiciones_reales = []
            
            for (gy_rel, gx_rel) in espiral_relativa:
                py, px = int(c_y - (LADO_P // 2) + (gy_rel * LADO_P)), int(c_x - (LADO_P // 2) + (gx_rel * LADO_P))
                if 0 <= py <= h_img - LADO_P and 0 <= px <= w_img - LADO_P:
                    if np.any(mascara_galaxia[py:py+LADO_P, px:px+LADO_P]):
                        posiciones_reales.append((py, px))
                        sec_rgb = img_array[py:py+LADO_P, px:px+LADO_P]
                        r_means.append(np.mean(sec_rgb[:,:,0]))
                        g_means.append(np.mean(sec_rgb[:,:,1]))
                        b_means.append(np.mean(sec_rgb[:,:,2]))
                
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

        colors_rgb = np.array([df['p_r'], df['p_g'], df['p_b']]).T / 100.0
        ax_scat1.scatter(df['p_r'], df['p_b'], c=colors_rgb, s=60, edgecolors='black', linewidth=0.3, alpha=0.8)
        
        ax_scat1.set_title("Mapa de la secuencia de Hubble", fontsize=16)
        ax_scat1.set_xlabel("% Intensidad en el canal Rojo (R)", fontsize=13)
        ax_scat1.set_ylabel("% Intensidad en canal Azul (B)", fontsize=13)
        ax_scat1.grid(True, linestyle='--', alpha=0.5)
        ax_scat1.text(0.05, 0.95, "Galaxias Tardías (Azules)", transform=ax_scat1.transAxes, color="#1E9AE7", fontweight='bold', va='top')
        ax_scat1.text(0.95, 0.05, "Galaxias Tempranas (Rojas)", transform=ax_scat1.transAxes, color="#EB1224", fontweight='bold', ha='right')

        bp = ax_box.boxplot([df['p_r'], df['p_g'], df['p_b']], patch_artist=True, tick_labels=['% Rojo', '% Verde', '% Azul'])
        colors = ["#EB1224", "#13BB5F", "#1E9AE7"]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax_box.set_title("Diagrama Boxplot", fontsize=16)
        ax_box.set_ylabel("Porcentaje de contribución (%)", fontsize=13)
        ax_box.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        leyenda_caja = patches.Rectangle((0,0),1,1, edgecolor='black')
        leyenda_mediana = mlines.Line2D([], [], color='#ff7f0e', linewidth=2) 
        leyenda_bigotes = mlines.Line2D([], [], color='black', linewidth=1.5)
        leyenda_outliers = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markerfacecolor='none')
        
        ax_box.legend(
            [leyenda_caja, leyenda_mediana, leyenda_bigotes, leyenda_outliers],
            ['Caja: IQR', 'Línea: mediana', 'Bigotes (Tukey): extremos típicos ($1.5 \, IQR$)', 'Círculos: outliers'],
            loc='best', fontsize=11, framealpha=0.9
        )
        fig1.tight_layout()

        fig2, ax_scat2 = plt.subplots(figsize=(10, 8)) 

        scatter = ax_scat2.scatter(df['p_r'], df['p_b'], c=df['SFR'], cmap='Spectral_r', 
                                s=70, edgecolors='black', linewidth=0.4, alpha=0.85)
        
        ax_scat2.set_title("Secuencia de Hubble condicionada por SFR", fontsize=16, pad=15)
        ax_scat2.set_xlabel("% Intensidad en el canal Rojo (R)", fontsize=13)
        ax_scat2.set_ylabel("% Intensidad en canal Azul (B)", fontsize=13)
        ax_scat2.grid(True, linestyle='--', alpha=0.5)
        
        cbar = plt.colorbar(scatter, ax=ax_scat2)
        cbar.set_label('Tasa de Formación Estelar ($\log_{10}(\text{SFR})$) [$M_\odot / yr$]', fontsize=12)
        
        fig2.tight_layout()
        plt.show(block=False) 

        print("\n" + "="*60)
        objid_deseado = input("Escribe el OBJID para detalle individual (o pulsa Enter para una al azar): ").strip()
        
        g = next((res for res in resultados_todas if res['OBJID'] == objid_deseado), None)
        if g:
            print(f"Graficando detalle de galaxia ID: {objid_deseado}...")
        else:
            g = random.choice(resultados_todas)
            print(f"Graficando galaxia al azar: ID {g['OBJID']}")

        img_plot = Image.open(g['img_path']).convert("RGB")
        n_p = g['total_parches']
        r_vals, g_vals, b_vals = json.loads(g['r_means']), json.loads(g['g_means']), json.loads(g['b_means'])
        
        fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={'width_ratios': [1, 1.5]})
        ax1.imshow(img_plot); ax1.axis('off')
        ax1.set_title(f"Morfología y muestreo\nID: {g['OBJID']}", fontsize=16, pad=15)
        for py, px in g['posiciones']:
            ax1.add_patch(patches.Rectangle((px, py), LADO_P, LADO_P, linewidth=0.4, edgecolor='cyan', facecolor='none', alpha=0.3))

        x = np.arange(1, n_p + 1); bw = 0.3
        ax2.bar(x - bw, r_vals, width=bw, color="#EB1224", label='Canal Rojo (R)', alpha=0.9, zorder=3)
        ax2.bar(x, g_vals, width=bw, color="#13BB5F", label='Canal Verde (G)', alpha=0.9, zorder=3)
        ax2.bar(x + bw, b_vals, width=bw, color="#1E9AE7", label='Canal Azul (B)', alpha=0.9, zorder=3)
        ax2.set_xlabel('Nº de parche'); ax2.set_ylabel('Intensidad'); ax2.grid(True, linestyle='--', alpha=0.4, axis='y')
        ax2.legend(frameon=True, fontsize=11, loc='upper right', bbox_to_anchor=(0.98, 0.80))
        
        info_text = f"Canales:\n● Rojo: {g['p_r']:.1f}%\n● Verde: {g['p_g']:.1f}%\n● Azul: {g['p_b']:.1f}%"
        ax2.text(0.98, 0.97, info_text, transform=ax2.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), family='monospace')
        plt.tight_layout(); plt.show(block=False)

        p_zoom = min(100, n_p)
        fig4, (ax3, ax4) = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={'width_ratios': [1, 1.5]})
        
        ax3.imshow(img_plot)
        c_x, c_y = img_plot.width // 2, img_plot.height // 2
        mz = 60
        ax3.set_xlim(c_x - mz, c_x + mz); ax3.set_ylim(c_y + mz, c_y - mz)
        ax3.set_title(f"Detalle primeros {p_zoom} parches", fontsize=16)
        
        for idx, (py, px) in enumerate(g['posiciones'][:p_zoom]):
            ax3.add_patch(patches.Rectangle((px, py), LADO_P, LADO_P, linewidth=1, edgecolor='black', facecolor='none'))
            ax3.text(px+4, py+4, str(idx+1), color='white', fontsize=7, ha='center', va='center', weight='bold',
                     path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
        
        ax3.axis('off')

        xz = np.arange(1, p_zoom + 1)
        ax4.bar(xz - bw, r_vals[:p_zoom], width=bw, color="#EB1224", label='Canal Rojo (R)', alpha=0.9, zorder=3)
        ax4.bar(xz, g_vals[:p_zoom], width=bw, color="#13BB5F", label='Canal Verde (G)', alpha=0.9, zorder=3)
        ax4.bar(xz + bw, b_vals[:p_zoom], width=bw, color="#1E9AE7", label='Canal Azul (B)', alpha=0.9, zorder=3)
        
        ax4.set_xlabel("Nº de parche"); ax4.set_ylabel("Intensidad")
        ax4.grid(True, alpha=0.3, zorder=0)
        ax4.legend(frameon=True, fontsize=11, loc='upper right', bbox_to_anchor=(0.98, 0.85))

        plt.tight_layout()
        plt.show() 

    print(f"Procesamiento finalizado en {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()