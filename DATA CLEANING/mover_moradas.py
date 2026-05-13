import os
import shutil

###########################################################################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DIR_ORIGEN = os.path.join(REPO_ROOT, "jpg_sdss")
DIR_DESTINO = os.path.join(REPO_ROOT, "galaxias_moradas")
TXT_PATH = os.path.join(REPO_ROOT, "txt", "galaxias_moradas.txt")

os.makedirs(DIR_DESTINO, exist_ok=True)
###########################################################################################

def main():
    movidas = 0
    no_encontradas = 0

    if not os.path.exists(TXT_PATH):
        print(f"Error: No se encuentra el archivo {TXT_PATH}")
        return

    with open(TXT_PATH, 'r') as f:
        lineas = [l.strip() for l in f.readlines() if l.strip()]

    for linea in lineas:
        nombre = linea if linea.lower().endswith('.jpg') else f"{linea}.jpg"
        ruta_origen = os.path.join(DIR_ORIGEN, nombre)
        ruta_destino = os.path.join(DIR_DESTINO, nombre)

        if os.path.exists(ruta_origen):
            shutil.move(ruta_origen, ruta_destino)
            movidas += 1
        else:
            no_encontradas += 1

    print(f"Galaxias moradas movidas: {movidas}")
    if no_encontradas > 0:
        print(f"No encontradas en origen: {no_encontradas}")

if __name__ == "__main__":
    main()