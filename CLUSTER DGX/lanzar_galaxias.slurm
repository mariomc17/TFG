#!/bin/bash
#SBATCH --account=slurmuser
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --partition=dgx
#SBATCH --gres=gpu:4
#SBATCH --time=1-23:00:00
# --job-name, --output y --error los inyecta submit_train.sh

set -euo pipefail

# Renombrar la carpeta de pending -> con job_id ahora que SLURM lo asignó
RUN_PARENT=$(dirname "${RUN_DIR_TMP}")
RUN_DIR="${RUN_PARENT}/${RUN_TS}__job${SLURM_JOB_ID}__${RUN_TAG}"
mv "${RUN_DIR_TMP}" "${RUN_DIR}"

echo "================================================================"
echo "Job:        ${SLURM_JOB_ID}"
echo "Tag:        ${RUN_TAG}"
echo "Config:     ${CONFIG_PATH}"
echo "GPUs:       ${TRAIN_GPUS:-4}"
[ -n "${OVERRIDES:-}" ] && echo "Overrides:  ${OVERRIDES}"
echo "Host:       $(hostname)"
echo "Inicio:     $(date)"
echo "RUN_DIR:    ${RUN_DIR}"
echo "================================================================"

R=$(pwd)
NOMBRECONT="${SLURM_JOB_ID}_galaxias"

# Snapshot del código que se ejecuta (reproducibilidad)
mkdir -p "${RUN_DIR}/code_snapshot"
cp -t "${RUN_DIR}/code_snapshot/" \
   train_diffusion_mi_u_net.py \
   generar_galaxia_mi_u_net.py \
   mi_u_net.py \
   dataset.py \
   requirements.txt \
   Dockerfile
# También el YAML original
cp "${CONFIG_PATH}" "${RUN_DIR}/code_snapshot/"

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

# Traducir ruta de RUN_DIR del host al interior del contenedor
RUN_DIR_IN_CONT="/app/${RUN_DIR#$R/}"
GPU_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${TRAIN_GPUS:-4}"

echo "Lanzando el entrenamiento..."
docker run --gpus "device=${GPU_DEVICES}" --name $NOMBRECONT --rm \
  --ipc=host \
  -e HOME=/app \
  -e HF_HOME=/app/.cache \
  -e TORCH_HOME=/app/.cache \
  -e TORCHINDUCTOR_CACHE_DIR=/app/.cache/inductor \
  -e TRITON_CACHE_DIR=/app/.cache/triton \
  -e TRITON_HOME=/app/.cache/triton_home \
  -e RUN_DIR="${RUN_DIR_IN_CONT}" \
  -u $(id -u):$(id -g) \
  -v "$R":/app \
  galaxias-ml /bin/bash -c "torchrun --standalone --nproc_per_node='${NPROC_PER_NODE}' train_diffusion_mi_u_net.py --config '${CONFIG_PATH}' ${OVERRIDES:-}"

echo "Trabajo finalizado a las $(date)"
