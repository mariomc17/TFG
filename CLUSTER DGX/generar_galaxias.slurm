#!/bin/bash
#SBATCH --account=slurmuser
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
# --job-name, --output y --error los inyecta submit_gen.sh

set -euo pipefail

GEN_DIR="$(dirname "${GEN_DIR_TMP}")/${GEN_TS}__job${SLURM_JOB_ID}__from_${PARENT_RUN}"
mv "${GEN_DIR_TMP}" "${GEN_DIR}"

echo "================================================================"
echo "Job:        ${SLURM_JOB_ID}"
echo "Checkpoint: ${CKPT_HOST}"
echo "Config:     ${CONFIG_PATH}"
[ -n "${OVERRIDES:-}" ] && echo "Overrides:  ${OVERRIDES}"
echo "Host:       $(hostname)"
echo "Inicio:     $(date)"
echo "GEN_DIR:    ${GEN_DIR}"
echo "================================================================"

R=$(pwd)
NOMBRECONT="${SLURM_JOB_ID}_generador"

# Traducir rutas host -> contenedor (monta $R como /app)
CKPT_IN_CONT="/app/${CKPT_HOST#$R/}"
GEN_DIR_IN_CONT="/app/${GEN_DIR#$R/}"

# ── Build inteligente ─────────────────────────────────────────────────────────
# Los scripts .py llegan vía -v $R:/app, por lo que el COPY del Dockerfile queda
# sobreescrito en ejecución. La imagen solo necesita reconstruirse cuando cambian
# Dockerfile o requirements.txt (instalación de paquetes del sistema/pip).
# Estrategia: hash de ambos ficheros almacenado como label en la imagen Docker.
# Si el label coincide → saltamos el build (ahorra 10-18 min por job).
DEPS_HASH=$(md5sum Dockerfile requirements.txt 2>/dev/null | md5sum | cut -d' ' -f1)
CURRENT_LABEL=$(docker inspect --format='{{index .Config.Labels "deps_hash"}}' galaxias-ml 2>/dev/null || echo "none")

if [ "$CURRENT_LABEL" != "$DEPS_HASH" ]; then
    echo "Imagen inexistente o deps modificadas (hash=${DEPS_HASH}). Construyendo..."
    docker build --label "deps_hash=${DEPS_HASH}" -t galaxias-ml .
    echo "Build completado en $(date)."
else
    echo "Imagen galaxias-ml ya actualizada (hash=${DEPS_HASH}). Build saltado."
fi

GPU_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

docker run --gpus "device=${GPU_DEVICES}" --name $NOMBRECONT --rm \
  --ipc=host \
  -e HF_HOME=/app/.cache \
  -e TORCH_HOME=/app/.cache \
  -e TORCHINDUCTOR_CACHE_DIR=/app/.cache/inductor \
  -e TRITON_CACHE_DIR=/app/.cache/triton \
  -e TRITON_HOME=/app/.cache/triton_home \
  -e MPLCONFIGDIR=/tmp \
  -e OUTPUT_DIR="${GEN_DIR_IN_CONT}/images" \
  -u $(id -u):$(id -g) \
  -v "$R":/app \
  galaxias-ml bash -c "python generar_galaxia_mi_u_net.py --config '${CONFIG_PATH}' checkpoint='${CKPT_IN_CONT}' ${OVERRIDES:-}"

echo "Generación finalizada a $(date)"
