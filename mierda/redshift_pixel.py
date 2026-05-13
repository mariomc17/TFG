import os
import pandas as pd
import numpy as np
from astropy.cosmology import Planck18
import astropy.units as u
from tqdm import tqdm

###########################################################################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DIR_CSV = os.path.join(REPO_ROOT, "csv")
RUTA_CSV_SDSS = os.path.join(DIR_CSV, "galaxias_sdss.csv")

BINNING_FACTOR = 4
ESCALA_NATIVA = 0.15
###########################################################################################

def calcular_escala_fisica(redshift: float, binning_factor: int = 4) -> float:
    if pd.isna(redshift) or redshift <= 0:
        return np.nan

    escala_imagen = ESCALA_NATIVA * binning_factor
    kpc_por_arcsec = Planck18.kpc_proper_per_arcmin(redshift).to(u.kpc / u.arcsec)
    kpc_por_pixel = escala_imagen * kpc_por_arcsec.value

    return kpc_por_pixel

def calcular_error_escala(redshift: float, redshift_err: float, binning_factor: int = 4) -> float:
    if pd.isna(redshift) or pd.isna(redshift_err) or redshift <= 0:
        return np.nan
        
    z_plus = redshift + redshift_err
    z_minus = max(1e-5, redshift - redshift_err) 
    
    escala_plus = calcular_escala_fisica(z_plus, binning_factor)
    escala_minus = calcular_escala_fisica(z_minus, binning_factor)
    
    return abs(escala_plus - escala_minus) / 2.0

def main():
    if not os.path.exists(RUTA_CSV_SDSS):
        print(f"Error: no se encuentra el archivo en {RUTA_CSV_SDSS}")
        return

    df = pd.read_csv(RUTA_CSV_SDSS)

    if 'REDSHIFT' not in df.columns or 'REDSHIFT_ERR' not in df.columns:
        print("Error: faltan las columnas 'REDSHIFT' o 'REDSHIFT_ERR' en el catálogo.")
        return

    tqdm.pandas(desc="Calculando escalas físicas y errores")

    df['ESCALA_KPC_PX'] = df['REDSHIFT'].progress_apply(
        lambda z: calcular_escala_fisica(z, binning_factor=BINNING_FACTOR)
    )

    df['ESCALA_KPC_PX_ERR'] = df.progress_apply(
        lambda row: calcular_error_escala(row['REDSHIFT'], row['REDSHIFT_ERR'], binning_factor=BINNING_FACTOR),
        axis=1
    )

    df.to_csv(RUTA_CSV_SDSS, index=False)

if __name__ == "__main__":
    main()