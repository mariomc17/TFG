"""
dataset.py — Dataset de galaxias con normalización robusta.

Mejoras respecto a la versión anterior:
  ✓ Z-score en lugar de Min-Max (más robusto ante outliers, distribuciones gaussianas
    internas más coherentes con los null tokens inicializados con randn)
  ✓ log1p transform previa al Z-score para variables log-normales:
        EA           (skewness +3.45, lognormal)
        ESCALA_KPC_PX (skewness +0.52)
        RADIO_P      (skewness +2.96, lognormal)
  ✓ Estadísticos calculados en init desde los datos crudos (no desde attrs del HDF5),
    robustecidos mediante recorte al percentil 0.5–99.5 antes de calcular mean/std
  ✓ Detección automática de errores constantes (EA_ERR=1.5, MET_ERR=0.15):
    se omite el augmentation por error cuando el error es idéntico para todas las muestras
  ✓ Augmentación de imagen: RandomHorizontalFlip + RandomVerticalFlip + RandomRotation(180°)
    Las galaxias no tienen orientación preferida en el plano del cielo.
  ✓ Clamp post-normalización: [−3, +3] en lugar de [0, 1]
  ✓ norm_stats accesible como atributo para guardarlo en el checkpoint

Variables de condicionamiento recomendadas (por defecto):
    ['ESCALA_KPC_PX', 'LOG_MS', 'EA', 'RADIO_P']

Variables descartadas y por qué:
    SFR          — valores centinela −9999, Q1 negativa (artefacto de fitting SED)
    G_R          — valores centinela −10016, distribución patológica
    MET          — distribución cuantizada (grid SED), error constante, redundante con LOG_MS
    REDSHIFT     — correlación 1.00 con ESCALA_KPC_PX (misma variable por construcción)
"""

import json

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

# Variables que requieren transformación log1p antes del Z-score.
# El resultado es una distribución más simétrica (próxima a gaussiana),
# coherente con las distribuciones que la red neural espera internamente.
_LOG1P_VARS = frozenset({"EA", "ESCALA_KPC_PX", "RADIO_P"})


class GalaxiasFisicasDataset(Dataset):
    """
    Dataset HDF5 de galaxias con normalización Z-score robusta.

    Args:
        hdf5_path          : ruta al fichero HDF5
        img_size           : tamaño de la imagen de salida (cuadrada)
        variables_elegidas : lista de nombres de variables físicas a usar
        augment            : si True, aplica flip/rotación a las imágenes
                             y augmentation por errores de medición a la física
    """

    def __init__(
        self,
        hdf5_path: str,
        img_size: int,
        variables_elegidas: list,
        augment: bool = True,
    ):
        self.hdf5_path = hdf5_path
        self.img_size = img_size
        self.variables_elegidas = variables_elegidas
        self.augment = augment
        self.h5_file = None  # Lazy open (necesario para DataLoader multi-worker)

        # ── Carga inicial de metadatos y estadísticas ────────────────────
        with h5py.File(self.hdf5_path, "r") as f:
            self.length = len(f["images"])
            self.columnas_fisicas = json.loads(f.attrs["columnas_fisicas"])
            # Cargar todos los datos físicos para calcular estadísticos de normalización.
            # ~20k × N_vars × 8 bytes ≈ pocos MB, totalmente asumible en RAM.
            fisica_all = f["fisica"][:]  # [N, num_vars_total]

        print(f"Dataset HDF5 cargado: {self.length} galaxias.")
        print(f"Variables elegidas: {self.variables_elegidas}")

        # ── Índices de variables y sus errores ───────────────────────────
        self.indices_vars = []
        self.indices_errs = []
        for var in self.variables_elegidas:
            self.indices_vars.append(self.columnas_fisicas.index(var))
            err_col = f"{var}_ERR"
            if err_col in self.columnas_fisicas:
                self.indices_errs.append(self.columnas_fisicas.index(err_col))
            else:
                self.indices_errs.append(-1)

        # ── Calcular estadísticos de normalización ────────────────────────
        self.norm_stats = self._compute_norm_stats(fisica_all)
        # Detectar qué variables tienen errores constantes (placeholder) →
        # para esas NO aplicamos augmentation por error de medición
        self.use_err_aug = self._detect_usable_errors(fisica_all)

        # Imprimir resumen de normalización
        print("\nEstadísticos de normalización:")
        for var in self.variables_elegidas:
            s = self.norm_stats[var]
            transf = "log1p → Z-score" if s["log_transform"] else "Z-score"
            aug_str = (
                "con aug. de error"
                if self.use_err_aug[var]
                else "sin aug. (error constante)"
            )
            print(
                f"  {var:20s}: μ={s['mean']:.4f}, σ={s['std']:.4f} | {transf} | {aug_str}"
            )
        print()

        # ── Transformaciones de imagen ────────────────────────────────────
        aug_transforms = []
        if self.augment:
            aug_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # RandomRotation rellena las esquinas con 0 (negro = fondo oscuro del cielo),
                # coherente con el fondo real de las imágenes SDSS.
                transforms.RandomRotation(
                    degrees=180,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
            ]

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                *aug_transforms,
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    # ── Métodos de normalización ──────────────────────────────────────────

    def _compute_norm_stats(self, fisica_all: np.ndarray) -> dict:
        """
        Calcula mean/std para cada variable elegida.

        Proceso:
          1. Extraer la columna de la variable del array global
          2. Recortar al percentil 0.5–99.5 para robustez ante outliers extremos
             (variables como SFR tienen valores centinela −9999 que destruirían
             cualquier estadístico sin este paso)
          3. Aplicar log1p si la variable está en _LOG1P_VARS
          4. Calcular mean y std del resultado transformado

        Los clip_lo/clip_hi se guardan en espacio CRUDO (antes de log1p) para
        aplicarlos correctamente en __getitem__.
        """
        stats = {}
        for var_name, idx in zip(self.variables_elegidas, self.indices_vars):
            vals = fisica_all[:, idx].astype(np.float64)

            # Recorte robusto en espacio crudo
            clip_lo, clip_hi = np.percentile(vals, [0.5, 99.5])
            vals_clipped = np.clip(vals, clip_lo, clip_hi)

            use_log = var_name in _LOG1P_VARS
            if use_log:
                vals_transformed = np.log1p(np.maximum(vals_clipped, 0.0))
            else:
                vals_transformed = vals_clipped

            stats[var_name] = {
                "mean": float(np.mean(vals_transformed)),
                "std": float(np.std(vals_transformed)),
                "log_transform": use_log,
                "clip_lo": float(clip_lo),
                "clip_hi": float(clip_hi),
            }
        return stats

    def _detect_usable_errors(self, fisica_all: np.ndarray) -> dict:
        """
        Detecta qué variables tienen errores con variación real (no constantes).

        Si std(error) ≈ 0, el error es un placeholder (e.g. EA_ERR=1.5 para todas
        las galaxias, MET_ERR=0.15 para todas). Aplicar augmentation con esos
        valores añade ruido uniforme a TODAS las muestras por igual, lo que no
        aporta nada y puede dañar el condicionamiento.
        """
        usable = {}
        for var_name, idx_err in zip(self.variables_elegidas, self.indices_errs):
            if idx_err == -1:
                usable[var_name] = False
            else:
                err_vals = fisica_all[:, idx_err].astype(np.float64)
                # Consideramos usable si la std del error es > 1% de su media
                err_std = np.std(err_vals)
                err_mean = np.abs(np.mean(err_vals))
                usable[var_name] = (err_std > 1e-6) and (err_std > 0.01 * err_mean)
        return usable

    def _normalize(self, value: float, var_name: str) -> float:
        """
        Normaliza un valor crudo de una variable física.

        Orden de operaciones (crítico mantenerlo consistente):
          1. Clip al rango del training (en espacio crudo)
          2. log1p si corresponde
          3. Z-score
        """
        s = self.norm_stats[var_name]
        value = float(np.clip(value, s["clip_lo"], s["clip_hi"]))
        if s["log_transform"]:
            value = float(np.log1p(max(value, 0.0)))
        return (value - s["mean"]) / (s["std"] + 1e-8)

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        # Apertura lazy del HDF5 (necesario para DataLoader multi-worker:
        # cada worker abre su propio handle)
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, "r")

        # ── Imagen ────────────────────────────────────────────────────────
        img_np = self.h5_file["images"][idx]
        image = Image.fromarray(img_np)
        image = self.transform(image)

        # ── Vector físico ─────────────────────────────────────────────────
        phys_data = self.h5_file["fisica"][idx]
        vect_final = []

        for i, var_name in enumerate(self.variables_elegidas):
            val = float(phys_data[self.indices_vars[i]])
            idx_err = self.indices_errs[i]

            # Augmentation por error de medición (solo si el error es significativo
            # y estamos en modo entrenamiento)
            if self.augment and self.use_err_aug[var_name] and idx_err != -1:
                err = abs(float(phys_data[idx_err]))
                if err > 0:
                    val = float(np.random.normal(loc=val, scale=err))

            val_norm = self._normalize(val, var_name)
            vect_final.append(val_norm)

        fisica_vector = torch.tensor(vect_final, dtype=torch.float32)

        # Clamp suave: ±3σ (coherente con Z-score; valores fuera son outliers extremos
        # o augmentation muy agresiva. No clampar a [0,1] que no tiene sentido con Z-score)
        fisica_vector = torch.clamp(fisica_vector, -3.0, 3.0)

        return image, fisica_vector
