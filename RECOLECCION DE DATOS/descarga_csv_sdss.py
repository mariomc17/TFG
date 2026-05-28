import os
import time
import pandas as pd
from astroquery.sdss import SDSS

###########################################################################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..\..", "csv"))
OUTPUT_FILE = "galaxias_sdss.csv"
CSV_PATH = os.path.join(BASE_DIR, OUTPUT_FILE)

if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)
###########################################################################################

query = """
SELECT TOP 30000
    p.objid AS OBJID,
    p.ra AS RA,
    p.dec AS DEC,
    
    -- 1. REDSHIFT
    s.z AS REDSHIFT,
    s.zErr AS REDSHIFT_ERR,

    -- 2. MASA ESTELAR
    m.logMass AS LOG_MS, 
    (m.maxLogMass - m.minLogMass) / 2.0 AS LOG_MS_ERR,
    
    -- 3. TASA DE FORMACIÓN ESTELAR
    e.sfr_tot_p50 AS SFR,
    (e.sfr_tot_p84 - e.sfr_tot_p16) / 2.0 AS SFR_ERR,

    -- 4. EDAD ESTELAR
    m.age AS EA,
    1.5 AS EA_ERR,

    -- 5. METALICIDAD
    m.metallicity AS MET,
    0.15 AS MET_ERR,

    -- 6. RADIO DE PETROSIAN
    p.petroRad_r AS RADIO_P,
    p.petroRadErr_r AS RADIO_P_ERR,
    
    -- 7. ÍNDICE DE COLOR (g-r)
    (p.g - p.r) AS G_R,
    SQRT(POWER(p.err_g, 2) + POWER(p.err_r, 2)) AS G_R_ERR

FROM SpecObj s
JOIN PhotoObj p ON s.bestObjID = p.objid
JOIN stellarMassStarformingPort m ON s.specObjID = m.specObjID
JOIN galSpecExtra e ON s.specObjID = e.specObjID
JOIN zooSpec zoo ON s.specObjID = zoo.specobjid

WHERE s.class = 'GALAXY'
  AND zoo.p_cs_debiased > 0.90
  AND zoo.p_edge < 0.10
  AND s.zWarning = 0
"""

def main():
    start_time = time.time()
    
    try:
        res = SDSS.query_sql(query, data_release=16)
        
        if res is not None:
            df = res.to_pandas()
            
            for col in df.select_dtypes([object]).columns:
                df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

            df.to_csv(CSV_PATH, index=False)
            
            print(f"Archivo guardado en: {CSV_PATH}")
            print(f"Total de galaxias: {len(df)}")
            print(f"Variables: {', '.join(df.columns)}")
        else:
            print("No se encontraron resultados o el servidor no respondió.")
            
    except Exception as e:
        print(f"Error durante la descarga: {e}")

    end_time = time.time()
    print(f"Tiempo total: {end_time - start_time:.2f} segundos")

if __name__ == "__main__":
    main()