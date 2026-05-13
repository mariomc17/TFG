#!/bin/bash
# submit_train.sh — wrapper para lanzar entrenamientos con organización por run.
#
# Uso:
#   ./submit_train.sh [TAG] [CONFIG] [OVERRIDES...]
#
# Argumentos:
#   TAG       Nombre legible de la run (default: baseline)
#   CONFIG    YAML de configuración    (default: configs/train_baseline.yaml)
#   OVERRIDES Pares clave=valor estilo OmegaConf, opcionales
#             ej.: train.lr=5e-5 train.batch_size=64
#
# Ejemplos:
#   ./submit_train.sh
#   ./submit_train.sh experimento_lr_alto configs/train_lr_alto.yaml
#   ./submit_train.sh quick_test configs/train_baseline.yaml train.epochs=3

set -euo pipefail

TAG="${1:-baseline}"
CONFIG="${2:-configs/train_baseline.yaml}"
shift 2 2>/dev/null || shift $# || true
OVERRIDES="$*"   # resto de argumentos como string

if [ ! -f "$CONFIG" ]; then
  echo "ERROR: no existe el config '$CONFIG'"
  exit 1
fi

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RUN_PARENT="$(pwd)/runs/train"
RUN_DIR_TMP="${RUN_PARENT}/${TIMESTAMP}__pending__${TAG}"
mkdir -p "${RUN_DIR_TMP}/logs" "${RUN_DIR_TMP}/checkpoints"

echo "Preparando run:"
echo "  Carpeta:    ${RUN_DIR_TMP}"
echo "  Config:     ${CONFIG}"
[ -n "$OVERRIDES" ] && echo "  Overrides:  ${OVERRIDES}"

sbatch \
  --job-name="${TAG}" \
  --output="${RUN_DIR_TMP}/logs/train.out" \
  --error="${RUN_DIR_TMP}/logs/train.err" \
  --export=ALL,RUN_DIR_TMP="${RUN_DIR_TMP}",RUN_TAG="${TAG}",RUN_TS="${TIMESTAMP}",CONFIG_PATH="${CONFIG}",OVERRIDES="${OVERRIDES}" \
  lanzar_galaxias.slurm
