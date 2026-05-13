#!/bin/bash
# submit_gen.sh — wrapper para lanzar generación de galaxias por SLURM.
#
# Uso:
#   ./submit_gen.sh [CKPT] [CONFIG] [OVERRIDES...]
#
# Argumentos:
#   CKPT       Ruta al .pt. Si está vacío, busca el last.pt de la run más reciente.
#   CONFIG     YAML con la lista de galaxias  (default: configs/gen_examples.yaml)
#   OVERRIDES  Pares clave=valor estilo OmegaConf
#
# Ejemplos:
#   ./submit_gen.sh
#   ./submit_gen.sh runs/train/<run>/checkpoints/modelo_epoca_050.pt
#   ./submit_gen.sh "" configs/gen_examples.yaml inference_steps=100

set -euo pipefail

CKPT="${1:-}"
CONFIG="${2:-configs/gen_examples.yaml}"
shift 2 2>/dev/null || shift $# || true
OVERRIDES="$*"

if [ ! -f "$CONFIG" ]; then
  echo "ERROR: no existe el config '$CONFIG'"; exit 1
fi

# --- Autodescubrir checkpoint si no se ha pasado ---
if [ -z "$CKPT" ]; then
  echo "No se pasó checkpoint, buscando el más reciente con last.pt..."
  CKPT=""
  for d in $(ls -1dt runs/train/*/ 2>/dev/null); do
    if [ -f "${d}checkpoints/last.pt" ]; then
      CKPT="${d}checkpoints/last.pt"
      break
    fi
  done
  if [ -z "$CKPT" ]; then
    echo "ERROR: no encuentro ningún last.pt bajo runs/train/. Lanza un entrenamiento primero o pasa una ruta explícita."
    exit 1
  fi
fi

if [ ! -e "$CKPT" ]; then
  echo "ERROR: checkpoint no existe: $CKPT"; exit 1
fi

# Resolver symlink (si es last.pt) para que en metadatos quede la ruta real
CKPT_REAL=$(readlink -f "$CKPT")

# De qué run de train viene este checkpoint (para etiquetar la carpeta de gen)
PARENT_RUN=$(basename "$(dirname "$(dirname "$CKPT_REAL")")")
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
GEN_PARENT="$(pwd)/runs/gen"
GEN_DIR_TMP="${GEN_PARENT}/${TIMESTAMP}__pending__from_${PARENT_RUN}"
mkdir -p "${GEN_DIR_TMP}/logs" "${GEN_DIR_TMP}/images"

echo "Preparando generación:"
echo "  Carpeta:    ${GEN_DIR_TMP}"
echo "  Checkpoint: ${CKPT_REAL}"
echo "  Config:     ${CONFIG}"
[ -n "$OVERRIDES" ] && echo "  Overrides:  ${OVERRIDES}"

sbatch \
  --job-name="gen_galaxias" \
  --output="${GEN_DIR_TMP}/logs/gen.out" \
  --error="${GEN_DIR_TMP}/logs/gen.err" \
  --export=ALL,GEN_DIR_TMP="${GEN_DIR_TMP}",GEN_TS="${TIMESTAMP}",PARENT_RUN="${PARENT_RUN}",CKPT_HOST="${CKPT_REAL}",CONFIG_PATH="${CONFIG}",OVERRIDES="${OVERRIDES}" \
  generar_galaxias.slurm
