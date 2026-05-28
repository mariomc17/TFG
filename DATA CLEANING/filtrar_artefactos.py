import os
import cv2
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

###########################################################################################

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..\.."))
DIR_IMAGENES = os.path.join(REPO_ROOT, "galaxias_sdss_sin_filtrar")
DIR_TXT = os.path.join(REPO_ROOT, "txt")
NOMBRE_TXT = "galaxias_artefactos.txt"
TXT_PATH = os.path.join(DIR_TXT, NOMBRE_TXT)

if not os.path.exists(DIR_TXT):
    os.makedirs(DIR_TXT)

UMBRAL_SATURACION = 210
UMBRAL_VALOR = 200
AREA_MINIMA_COLOR = 150
LONGITUD_LINEA = 215
GAP_LINEA = 20
MAX_WORKERS = 8

###########################################################################################

LISTA_TRAZOS = []
LOCK = threading.Lock()

def analizar_imagen(archivo):
    try:
        ruta = os.path.join(DIR_IMAGENES, archivo)
        obj_id = os.path.basename(archivo).split(".")[0]
        
        img = cv2.imread(ruta)
        if img is None: return
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, np.array([0, UMBRAL_SATURACION, UMBRAL_VALOR]), np.array([179, 255, 255]))
        contours, _ = cv2.findContours(mask_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > AREA_MINIMA_COLOR:
                with LOCK:
                    LISTA_TRAZOS.append(obj_id)
                return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 200, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=150, 
                                minLineLength=LONGITUD_LINEA, maxLineGap=GAP_LINEA)
        
        if lines is not None:
            with LOCK:
                LISTA_TRAZOS.append(obj_id)
                
    except Exception:
        pass

def main():
    inicio = time.time()
    
    if not os.path.exists(DIR_IMAGENES):
        print(f"Error: no se encuentra la carpeta de imágenes en {DIR_IMAGENES}")
        return

    archivos = sorted([f for f in os.listdir(DIR_IMAGENES) if f.lower().endswith(('.jpg', '.png'))])
    total = len(archivos)

    if total > 0:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            list(tqdm(executor.map(analizar_imagen, archivos), total=total, desc="Escaneando", unit=" img"))

    if LISTA_TRAZOS:
        with open(TXT_PATH, 'w') as f:
            for objid in LISTA_TRAZOS:
                f.write(f"{objid}\n")

    fin = time.time()
    print(f"Trazos y artefactos de color encontrados: {len(LISTA_TRAZOS)} galaxias")
    print(f"Archivo generado en: {TXT_PATH}")
    print(f"Tiempo total: {fin - inicio:.2f} segundos")

if __name__ == "__main__":
    main()