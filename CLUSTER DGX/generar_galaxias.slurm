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

echo "Construyendo/verificando la imagen Docker..."
docker build -t galaxias-ml .

GPU_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

docker run --gpus "device=${GPU_DEVICES}" --name $NOMBRECONT --rm \
  --ipc=host \
  -e HF_HOME=/app/.cache \
  -e TORCH_HOME=/app/.cache \
  -e MPLCONFIGDIR=/tmp \
  -e OUTPUT_DIR="${GEN_DIR_IN_CONT}/images" \
  -u $(id -u):$(id -g) \
  -v "$R":/app \
  galaxias-ml bash -c "python generar_galaxia_mi_u_net.py --config '${CONFIG_PATH}' checkpoint='${CKPT_IN_CONT}' ${OVERRIDES:-}"

echo "Generación finalizada a $(date)"
