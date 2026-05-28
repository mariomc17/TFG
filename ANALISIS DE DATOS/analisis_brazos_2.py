import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

###########################################################################################
# CONFIGURACIÓN DE RUTAS AUTOMÁTICAS E INMUTABLES
###########################################################################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DIR_APP = os.path.join(REPO_ROOT, "APP")
RUTA_GALAXIA_SINTETICA = os.path.join(DIR_APP, "galaxia_1.png")

# Tamaño de parche para 128x128
LADO_P = 2 
###########################################################################################

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

def analizar_galaxia_sintetica(img_path):
    if not os.path.exists(img_path):
        print(f"Error Crítico: No se encuentra el archivo en la ruta esperada:\n{img_path}")
        return

    nombre_archivo = os.path.basename(img_path)
    print(f"\n" + "="*60)
    print(f"ANALIZANDO GALAXIA SINTÉTICA: {nombre_archivo}")
    print("="*60)
    
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)
    h_img, w_img = img_array.shape[:2]
    c_y, c_x = h_img // 2, w_img // 2

    # =========================================================================
    # CÁLCULO DEL FONDO DE CIELO (MÉTODO BLINDADO)
    # IGNORAMOS LA PARTE SUPERIOR DONDE ESTÁ EL TEXTO PARA NO CONTAMINAR
    # =========================================================================
    brillos_borde = []
    
    # Muestreamos EXCLUSIVAMENTE las dos últimas filas de píxeles (el borde inferior absoluto)
    # donde sabemos con 100% de seguridad que hay vacío (negro real) y no hay letras.
    margen_inferior = h_img - (LADO_P * 2) 
    
    for y in range(margen_inferior, h_img, LADO_P):
        for x in range(0, w_img, LADO_P):
            parche = img_array[y:y+LADO_P, x:x+LADO_P]
            brillo_parche = np.mean(np.sum(parche, axis=-1))
            brillos_borde.append(brillo_parche)

    media_borde = np.mean(brillos_borde)
    std_borde = np.std(brillos_borde)
    
    # Definimos el umbral de corte. 
    # El valor 15.0 es un "suelo" de seguridad para asegurar que el ruido del fondo negro no entre.
    umbral_cielo = max(media_borde + (3 * std_borde), 15.0)
    
    print(f"-> Brillo Medio del Cielo (Borde inferior limpio): {media_borde:.2f}")
    print(f"-> Umbral estricto de recorte (Galaxia vs Fondo): {umbral_cielo:.2f}")

    # =========================================================================
    # LÓGICA DE ESCANEO EN ESPIRAL
    # =========================================================================
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
    
    # IMPORTANTE: Definimos a qué altura Y empieza la imagen real (descartando la franja de texto)
    # Asumimos que el texto ocupa más o menos un 15% - 20% de la parte superior.
    # Así el escáner no intentará medir las letras como si fuesen galaxia.
    limite_y_texto = int(h_img * 0.18) 

    for (gy_rel, gx_rel) in espiral_relativa:
        py, px = int(c_y - (LADO_P // 2) + (gy_rel * LADO_P)), int(c_x - (LADO_P // 2) + (gx_rel * LADO_P))
        
        # Comprobamos que el parche esté dentro de los límites Y por DEBAJO de las letras
        if limite_y_texto <= py <= h_img - LADO_P and 0 <= px <= w_img - LADO_P:
            sec_rgb = img_array[py:py+LADO_P, px:px+LADO_P]
            brillo_actual = np.mean(np.sum(sec_rgb, axis=-1))
            
            # FILTRO: Si el brillo supera el umbral limpio del cielo, es estructura galáctica.
            if brillo_actual > umbral_cielo:
                posiciones_reales.append((py, px))
                r_means.append(np.mean(sec_rgb[:,:,0]))
                g_means.append(np.mean(sec_rgb[:,:,1]))
                b_means.append(np.mean(sec_rgb[:,:,2]))

    n_p = len(posiciones_reales)
    if n_p == 0:
        print("Error Crítico: El escáner no detectó luz galáctica por encima del umbral del cielo.")
        return

    # Cálculos de composición termodinámica
    total_int = sum(r_means) + sum(g_means) + sum(b_means)
    p_r = (sum(r_means) / (total_int + 1e-6)) * 100
    p_g = (sum(g_means) / (total_int + 1e-6)) * 100
    p_b = (sum(b_means) / (total_int + 1e-6)) * 100

    print(f"-> Muestreo morfológico completado: {n_p} parches estructurales capturados.")
    print(f"-> Balance térmico integrado: R={p_r:.1f}%, G={p_g:.1f}%, B={p_b:.1f}%")

    # =========================================================================
    # PLOT 1: DESPLIEGUE COMPLETO 
    # =========================================================================
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={'width_ratios': [1, 1.5]})
    ax1.imshow(img); ax1.axis('off')
    ax1.set_title(f"Morfología y muestreo\n ID: Sintética 2", fontsize=15)
    
    for py, px in posiciones_reales:
        # Aumentamos ligeramente la opacidad (alpha=0.4) para que se vea mejor el recorte sobre el borde tenue
        ax1.add_patch(patches.Rectangle((px, py), LADO_P, LADO_P, linewidth=0.3, edgecolor='cyan', facecolor='none', alpha=0.4))

    x = np.arange(1, n_p + 1); bw = 0.3
    ax2.bar(x - bw, r_means, width=bw, color="#EB1224", label='Canal Rojo (R)', alpha=0.9, zorder=3)
    ax2.bar(x, g_means, width=bw, color="#13BB5F", label='Canal Verde (G)', alpha=0.9, zorder=3)
    ax2.bar(x + bw, b_means, width=bw, color="#1E9AE7", label='Canal Azul (B)', alpha=0.9, zorder=3)
    ax2.set_xlabel('Nº de parche)', fontsize=12); ax2.set_ylabel('Intensidad', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax2.legend(frameon=True, fontsize=11, loc='upper right', bbox_to_anchor=(0.98, 0.80))
    
    info_text = f"Canales:\n● R: {p_r:.1f}%\n● G: {p_g:.1f}%\n● B: {p_b:.1f}%"
    ax2.text(0.98, 0.97, info_text, transform=ax2.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), family='monospace')
    plt.tight_layout()
    
    ruta_salida_completo = os.path.join(DIR_APP, f"analisis_completo_{nombre_archivo}")
    plt.savefig(ruta_salida_completo, dpi=200)

    # =========================================================================
    # PLOT 2: DETALLE PRIMEROS 100 PARCHES
    # =========================================================================
    p_zoom = min(100, n_p)
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={'width_ratios': [1, 1.5]})
    
    ax3.imshow(img)
    mz = 30 
    ax3.set_xlim(c_x - mz, c_x + mz); ax3.set_ylim(c_y + mz, c_y - mz)
    ax3.set_title(f"Detalle central (Primeros {p_zoom} parches)", fontsize=15)
    
    for idx, (py, px) in enumerate(posiciones_reales[:p_zoom]):
        ax3.add_patch(patches.Rectangle((px, py), LADO_P, LADO_P, linewidth=0.5, edgecolor='black', facecolor='none'))
        ax3.text(px+(LADO_P/2.0), py+(LADO_P/2.0), str(idx+1), color='white', fontsize=4, ha='center', va='center', weight='bold',
                 path_effects=[path_effects.withStroke(linewidth=1, foreground='black')])
    
    ax3.axis('off')

    xz = np.arange(1, p_zoom + 1)
    ax4.bar(xz - bw, r_means[:p_zoom], width=bw, color="#EB1224", label='Canal Rojo (R)', alpha=0.9, zorder=3)
    ax4.bar(xz, g_means[:p_zoom], width=bw, color="#13BB5F", label='Canal Verde (G)', alpha=0.9, zorder=3)
    ax4.bar(xz + bw, b_means[:p_zoom], width=bw, color="#1E9AE7", label='Canal Azul (B)', alpha=0.9, zorder=3)
    
    ax4.set_xlabel("Nº de parche", fontsize=12); ax4.set_ylabel("Intensidad fotométrica", fontsize=12)
    ax4.grid(True, alpha=0.3, zorder=0)
    ax4.legend(frameon=True, fontsize=11, loc='upper right', bbox_to_anchor=(0.98, 0.85))

    plt.tight_layout()
    f
    ruta_salida_zoom = os.path.join(DIR_APP, f"analisis_zoom_{nombre_archivo}")
    plt.savefig(ruta_salida_zoom, dpi=200)
    
    print(f"\n[ÉXITO] Gráficas exportadas directamente en la carpeta APP:")
    print(f"1. {ruta_salida_completo}")
    print(f"2. {ruta_salida_zoom}\n")
    plt.show()

if __name__ == "__main__":
    analizar_galaxia_sintetica(RUTA_GALAXIA_SINTETICA)