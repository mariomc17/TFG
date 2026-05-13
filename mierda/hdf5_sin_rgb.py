import os
import pandas as pd
import numpy as np
import h5py
from PIL import Image
import json
from tqdm import tqdm

###########################################################################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DIR_CSV = os.path.join(REPO_ROOT, "csv")
CSV_FISICA = os.path.join(DIR_CSV, "galaxias_sdss.csv")
DIR_IMAGENES = os.path.join(REPO_ROOT, "galaxias_elipse") 
DIR_H5 = os.path.join(REPO_ROOT, "h5_sin_rgb")
HDF5_SALIDA = os.path.join(DIR_H5, "dataset_galaxias_sin_rgb.h5")

if not os.path.exists(DIR_H5):
    os.makedirs(DIR_H5)

COLUMNAS_FISICAS = [
    'RA', 'DEC', 'REDSHIFT', 'REDSHIFT_ERR', 'LOG_MS', 'LOG_MS_ERR',
    'SFR', 'SFR_ERR', 'EA', 'EA_ERR', 'MET', 'MET_ERR',
    'RADIO_P', 'RADIO_P_ERR', 'G_R', 'G_R_ERR', 'ESCALA_KPC_PX', 'ESCALA_KPC_PX_ERR'
]

IMG_SIZE = 512
###########################################################################################

def main():
    if not os.path.exists(CSV_FISICA):
        print("Error: falta el archivo CSV de física.")
        return

    df_crudo = pd.read_csv(CSV_FISICA, dtype={'OBJID': str})
    print(f"Filas totales en galaxias_sdss.csv: {len(df_crudo)}")
    
    validas = []
    for idx, row in df_crudo.iterrows():
        ruta_img = os.path.join(DIR_IMAGENES, f"{row['OBJID']}.png")
        if os.path.exists(ruta_img):
            validas.append(row)
            
    if not validas:
        ejemplo = os.path.join(DIR_IMAGENES, f"{df_crudo.iloc[0]['OBJID']}.png")
        print(f"Error: no se encontraron imágenes válidas.")
        print(f"El script ha intentado buscar, por ejemplo: {ejemplo}")
        print("Comprueba si la carpeta es correcta")
        return

    df = pd.DataFrame(validas).reset_index(drop=True)
    df = df.head(20000)
    N = len(df)
    print(f"Se van a empaquetar {N} galaxias.")

    columnas_presentes = [col for col in COLUMNAS_FISICAS if col in df.columns]
    num_fisicas = len(columnas_presentes)

    stats = {}
    for col in columnas_presentes:
        stats[col] = {'min': float(df[col].min()), 'max': float(df[col].max())}

    with h5py.File(HDF5_SALIDA, 'w') as f:
        f.attrs['stats'] = json.dumps(stats)
        f.attrs['columnas_fisicas'] = json.dumps(columnas_presentes)
        
        dset_imgs = f.create_dataset('images', shape=(N, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8, chunks=True)
        dset_phys = f.create_dataset('fisica', shape=(N, num_fisicas), dtype=np.float32)

        for i, row in tqdm(df.iterrows(), total=N, desc="Empaquetando en HDF5"):
            img_path = os.path.join(DIR_IMAGENES, f"{row['OBJID']}.png")
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
            dset_imgs[i] = img_np
            
            vals_fisicos = [max(0.0, float(row[col])) if 'ERR' in col else float(row[col]) for col in columnas_presentes]
            dset_phys[i] = vals_fisicos

    print(f"HDF5 creado con éxito. Tamaño: {os.path.getsize(HDF5_SALIDA) / (1024**3):.2f} GB")

if __name__ == "__main__":
    main()