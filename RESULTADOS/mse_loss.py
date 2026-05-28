import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

###########################################################################################

ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_csv = os.path.abspath(os.path.join(ruta_actual, "..\..", "entrenamiento", "metrics.csv"))

###########################################################################################

def plotear_metricas():
    if not os.path.exists(ruta_csv):
        print(f"No se encuentra el archivo en la ruta.")

        return

    df = pd.read_csv(ruta_csv)

    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['mean_mse_loss'], label='Mean MSE Loss', color="#2b42c0", linewidth=2)
    plt.plot(df['epoch'], df['best_loss'], label='Best Loss', color='#e74c3c', linestyle='--', linewidth=2)

    plt.xlabel('Época', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Evolución del entrenamiento', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plotear_metricas()