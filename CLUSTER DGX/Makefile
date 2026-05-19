# Makefile — atajos para el flujo de trabajo del proyecto Galaxias U-Net.
# Uso rápido: 'make' o 'make help' para ver todo.
.DEFAULT_GOAL := help
SHELL := /bin/bash

# Colores ANSI
B := \033[1m
G := \033[32m
Y := \033[33m
C := \033[36m
R := \033[0m

# Variables sobreescribibles desde la línea de comandos:
#   make train TAG=lr_alto CONFIG=configs/train_lr_alto.yaml
TAG    ?= baseline
CONFIG ?= configs/train_baseline.yaml
GEN_CONFIG ?= configs/gen_examples.yaml
CKPT   ?=
EXTRA  ?=

# Parámetros SLURM para comandos interactivos/cortos que necesitan Docker/GPU.
# Se usan en build, shell, show-config y monitorización.
PARTITION ?= dgx
ACCOUNT   ?= slurmuser
GPUS      ?= 1
CPUS      ?= 8
MEM       ?= 64G
TIME      ?= 00:30:00
IMAGE     ?= galaxias-ml

# Para targets de monitorización. JOB=latest toma el último job RUNNING del usuario.
JOB       ?= latest
INTERVAL  ?= 2

SRUN_BASE = srun --account=$(ACCOUNT) --partition=$(PARTITION) --gres=gpu:$(GPUS) --cpus-per-task=$(CPUS) --mem=$(MEM) --time=$(TIME)

.PHONY: help build build-force build-local train gen gen-sweep status list-runs tail tail-err clean-pending shell show-config resources resources-watch gpu gpu-watch nvtop htop

help: ## Muestra esta ayuda
	@printf "$(B)Galaxias U-Net — comandos$(R)\n\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  $(G)%-16s$(R) %s\n", $$1, $$2}'
	@printf "\n$(B)Variables principales:$(R)\n"
	@printf "  $(Y)TAG$(R)         nombre legible de la run         (default: $(TAG))\n"
	@printf "  $(Y)CONFIG$(R)      YAML de entrenamiento            (default: $(CONFIG))\n"
	@printf "  $(Y)GEN_CONFIG$(R)  YAML de generación               (default: $(GEN_CONFIG))\n"
	@printf "  $(Y)CKPT$(R)        checkpoint para gen              (default: autodescubre)\n"
	@printf "  $(Y)EXTRA$(R)       overrides OmegaConf, entre comillas\n"
	@printf "\n$(B)Variables SLURM/Docker:$(R)\n"
	@printf "  $(Y)PARTITION$(R)   partición SLURM                  (default: $(PARTITION))\n"
	@printf "  $(Y)ACCOUNT$(R)     cuenta SLURM                     (default: $(ACCOUNT))\n"
	@printf "  $(Y)GPUS$(R)        GPUs para srun                   (default: $(GPUS))\n"
	@printf "  $(Y)CPUS$(R)        CPUs para srun                   (default: $(CPUS))\n"
	@printf "  $(Y)MEM$(R)         memoria para srun                (default: $(MEM))\n"
	@printf "  $(Y)TIME$(R)        tiempo para srun                 (default: $(TIME))\n"
	@printf "  $(Y)IMAGE$(R)       imagen Docker                    (default: $(IMAGE))\n"
	@printf "  $(Y)JOB$(R)         job a monitorizar o latest        (default: $(JOB))\n"
	@printf "  $(Y)INTERVAL$(R)    segundos entre nvidia-smi         (default: $(INTERVAL))\n"
	@printf "\n$(B)Ejemplos:$(R)\n"
	@printf "  $(C)make build$(R)                               # smart: solo si cambiaron deps\n"
	@printf "  $(C)make build-force$(R)                         # fuerza reconstruccion completa\n"
	@printf "  $(C)make shell$(R)\n"
	@printf "  $(C)make show-config CONFIG=configs/train_lr_alto.yaml EXTRA=\"train.epochs=3\"$(R)\n"
	@printf "  $(C)make train TAG=quick_test EXTRA=\"train.epochs=3 train.compile=false\"$(R)\n"
	@printf "  $(C)make gen EXTRA=\"inference_steps=100 guidance_scale=9.0\"$(R)\n"
	@printf "  $(C)make gen-sweep EXTRA=\"sweep_var=ea_gyr\"$(R)\n"
	@printf "  $(C)make resources$(R)\n"
	@printf "  $(C)make resources-watch JOB=52549 INTERVAL=5$(R)\n"
	@printf "  $(C)make gpu$(R)\n"
	@printf "  $(C)make gpu-watch JOB=52549 INTERVAL=5$(R)\n"
	@printf "  $(C)make htop$(R)\n"

build: ## Construye la imagen Docker en un nodo dgx (solo si cambiaron Dockerfile/requirements.txt)
	$(SRUN_BASE) bash -c '\
	  DEPS_HASH=$$(md5sum Dockerfile requirements.txt | md5sum | cut -d" " -f1); \
	  LABEL=$$(docker inspect --format="{{index .Config.Labels \"deps_hash\"}}" $(IMAGE) 2>/dev/null || echo none); \
	  if [ "$$LABEL" != "$$DEPS_HASH" ]; then \
	    echo "Construyendo imagen (hash=$$DEPS_HASH)..."; \
	    docker build --label "deps_hash=$$DEPS_HASH" -t $(IMAGE) .; \
	  else \
	    echo "Imagen $(IMAGE) ya actualizada (hash=$$DEPS_HASH). Usa build-force para forzar."; \
	  fi'

build-force: ## Fuerza la reconstrucción completa de la imagen Docker (ignora caché de deps)
	$(SRUN_BASE) docker build --no-cache \
	  --label "deps_hash=$$(md5sum Dockerfile requirements.txt | md5sum | cut -d' ' -f1)" \
	  -t $(IMAGE) .

build-local: ## Construye la imagen Docker en el nodo local (solo si cambiaron las deps)
	@DEPS_HASH=$$(md5sum Dockerfile requirements.txt | md5sum | cut -d' ' -f1); \
	 LABEL=$$(docker inspect --format='{{index .Config.Labels "deps_hash"}}' $(IMAGE) 2>/dev/null || echo none); \
	 if [ "$$LABEL" != "$$DEPS_HASH" ]; then \
	   echo "Construyendo imagen local (hash=$$DEPS_HASH)..."; \
	   docker build --label "deps_hash=$$DEPS_HASH" -t $(IMAGE) .; \
	 else \
	   echo "Imagen $(IMAGE) ya actualizada (hash=$$DEPS_HASH). Usa docker rmi $(IMAGE) para forzar."; \
	 fi

train: ## Lanza un entrenamiento por SLURM
	@./submit_train.sh "$(TAG)" "$(CONFIG)" $(EXTRA)

gen: ## Genera galaxias por SLURM; usa el último ckpt si no se da CKPT
	@./submit_gen.sh "$(CKPT)" "$(GEN_CONFIG)" $(EXTRA)

gen-sweep: ## Barrido paramétrico vía SLURM; EXTRA="sweep_var=ea_gyr" para elegir variable
	@./submit_gen.sh "$(CKPT)" "$(GEN_CONFIG)" mode=sweep $(EXTRA)

status: ## Jobs SLURM propios en cola/ejecución
	@squeue -u $$USER -o "%.10i %.12P %.20j %.8T %.10M %R"

list-runs: ## Lista runs de entrenamiento/generación, más reciente primero
	@printf "$(B)Runs de entrenamiento:$(R)\n"
	@for d in $$(ls -1dt runs/train/*/ 2>/dev/null | head -20); do \
	  ckpt="$$d/checkpoints/mejor_modelo.pt"; \
	  if [ -f "$$ckpt" ]; then \
	    info=$$(python3 -c "import torch; c=torch.load('$$ckpt',map_location='cpu'); print(f\"ep{c.get('epoch','?')} loss={c.get('mean_loss',0):.5f}\")" 2>/dev/null || echo "?"); \
	    printf "  %s  [$(G)mejor_modelo.pt$(R) %s]\n" "$$d" "$$info"; \
	  else \
	    printf "  %s  [$(Y)sin checkpoint$(R)]\n" "$$d"; \
	  fi; \
	done
	@printf "\n$(B)Runs de generación:$(R)\n"
	@for d in $$(ls -1dt runs/gen/*/ 2>/dev/null | head -10); do \
	  n=$$(ls -1 $$d/images/*.png 2>/dev/null | wc -l); \
	  printf "  %s  [%d imágenes]\n" "$$d" "$$n"; \
	done

tail: ## tail -f del .out del último entrenamiento
	@d=$$(ls -1dt runs/train/*/ 2>/dev/null | head -n1); \
	  [ -n "$$d" ] || { echo "No hay runs en runs/train/"; exit 1; }; \
	  echo "Tailing $$d/logs/train.out"; \
	  tail -f $$d/logs/train.out

tail-err: ## tail -f del .err del último entrenamiento
	@d=$$(ls -1dt runs/train/*/ 2>/dev/null | head -n1); \
	  [ -n "$$d" ] || { echo "No hay runs en runs/train/"; exit 1; }; \
	  echo "Tailing $$d/logs/train.err"; \
	  tail -f $$d/logs/train.err

clean-pending: ## Borra carpetas __pending__ de jobs que nunca arrancaron
	@find runs -maxdepth 3 -type d -name "*__pending__*" -print -exec rm -rf {} +

show-config: ## Imprime la config efectiva tras merges dentro del contenedor, vía srun
	$(SRUN_BASE) docker run --rm -v "$$(pwd)":/app -w /app $(IMAGE) \
	  python -c "from train_diffusion_mi_u_net import load_config; from omegaconf import OmegaConf; \
	  cfg = load_config('$(CONFIG)', '$(EXTRA)'.split()); print(OmegaConf.to_yaml(cfg))"

shell: ## bash interactivo dentro del contenedor en un nodo dgx vía srun
	$(SRUN_BASE) --pty docker run --gpus all --rm -it --ipc=host \
	  -u $$(id -u):$$(id -g) -v "$$(pwd)":/app -w /app $(IMAGE) bash

resources: ## Lectura puntual de CPU/RAM/VRAM del job RUNNING más reciente o JOB=<id>
	@job="$(JOB)"; \
	if [ "$$job" = "latest" ]; then \
	  job=$$(squeue -h -u $$USER -t RUNNING -o "%i" | tail -n1); \
	fi; \
	[ -n "$$job" ] || { echo "No encuentro ningún job RUNNING de $$USER. Usa make status o JOB=<id>."; exit 1; }; \
	node=$$(squeue -h -j "$$job" -o "%N" | awk '{print $$1}'); \
	[ -n "$$node" ] || { echo "No encuentro el nodo del job $$job."; exit 1; }; \
	echo "$(B)Job SLURM $$job$(R)"; \
	squeue -j "$$job" -o "%.10i %.12P %.24j %.8T %.10M %.10l %.6D %C %m %R"; \
	echo; \
	echo "$(B)CPU/RAM según SLURM accounting$(R)"; \
	if command -v sstat >/dev/null 2>&1; then \
	  sstat -j "$$job.batch" --format=JobID,AveCPU,MaxRSS,AveRSS,MaxVMSize,NTasks 2>/dev/null || \
	  sstat -j "$$job" --format=JobID,AveCPU,MaxRSS,AveRSS,MaxVMSize,NTasks 2>/dev/null || true; \
	else \
	  echo "sstat no está disponible en este login node."; \
	fi; \
	echo; \
	echo "$(B)Docker + GPU/VRAM en $$node$(R)"; \
	ssh "$$node" "job='$$job'; \
	  echo Host:; hostname; \
	  echo; \
	  echo 'Docker stats del contenedor del job:'; \
	  docker stats --no-stream \\\"\\$${job}_galaxias\\\" \\\"\\$${job}_generador\\\" 2>/dev/null || true; \
	  echo; \
	  echo 'GPU asignada y procesos CUDA:'; \
	  nvidia-smi; \
	  echo; \
	  echo 'Procesos del usuario relacionados con este proyecto:'; \
	  ps -u \\\"\\$$USER\\\" -o pid,ppid,pcpu,pmem,rss,vsz,cmd --sort=-rss | grep -E \\\"\\$${job}|train_diffusion|generar_galaxia|galaxias-ml\\\" | grep -v grep || true"

resources-watch: ## Monitoriza CPU/RAM/VRAM periódicamente sin detener el job original
	@job="$(JOB)"; \
	if [ "$$job" = "latest" ]; then \
	  job=$$(squeue -h -u $$USER -t RUNNING -o "%i" | tail -n1); \
	fi; \
	[ -n "$$job" ] || { echo "No encuentro ningún job RUNNING de $$USER. Usa make status o JOB=<id>."; exit 1; }; \
	node=$$(squeue -h -j "$$job" -o "%N" | awk '{print $$1}'); \
	[ -n "$$node" ] || { echo "No encuentro el nodo del job $$job."; exit 1; }; \
	echo "Monitorizando job $$job cada $(INTERVAL)s; Ctrl-C detiene solo este monitor."; \
	while true; do \
	  clear; \
	  date; \
	  echo; \
	  echo "$(B)Job SLURM $$job$(R)"; \
	  squeue -j "$$job" -o "%.10i %.12P %.24j %.8T %.10M %.10l %.6D %C %m %R"; \
	  echo; \
	  echo "$(B)CPU/RAM según SLURM accounting$(R)"; \
	  if command -v sstat >/dev/null 2>&1; then \
	    sstat -j "$$job.batch" --format=JobID,AveCPU,MaxRSS,AveRSS,MaxVMSize,NTasks 2>/dev/null || \
	    sstat -j "$$job" --format=JobID,AveCPU,MaxRSS,AveRSS,MaxVMSize,NTasks 2>/dev/null || true; \
	  else \
	    echo "sstat no está disponible en este login node."; \
	  fi; \
	  echo; \
	  echo "$(B)Docker + GPU/VRAM en $$node$(R)"; \
	  ssh "$$node" "job='$$job'; \
	    echo Host:; hostname; \
	    echo; \
	    docker stats --no-stream \\\"\\$${job}_galaxias\\\" \\\"\\$${job}_generador\\\" 2>/dev/null || true; \
	    echo; \
	    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv; \
	    echo; \
	    nvidia-smi pmon -c 1 2>/dev/null || true"; \
	  sleep $(INTERVAL); \
	done

gpu: ## Ejecuta una lectura de nvidia-smi en el nodo del job RUNNING más reciente o JOB=<id>
	@job="$(JOB)"; \
	if [ "$$job" = "latest" ]; then \
	  job=$$(squeue -h -u $$USER -t RUNNING -o "%i" | tail -n1); \
	fi; \
	[ -n "$$job" ] || { echo "No encuentro ningún job RUNNING de $$USER. Usa make status o JOB=<id>."; exit 1; }; \
	echo "nvidia-smi en job $$job"; \
	srun --jobid=$$job --overlap --pty bash -lc 'hostname; nvidia-smi'

gpu-watch: ## Monitoriza nvidia-smi periódicamente en el nodo del job RUNNING más reciente o JOB=<id>
	@job="$(JOB)"; \
	if [ "$$job" = "latest" ]; then \
	  job=$$(squeue -h -u $$USER -t RUNNING -o "%i" | tail -n1); \
	fi; \
	[ -n "$$job" ] || { echo "No encuentro ningún job RUNNING de $$USER. Usa make status o JOB=<id>."; exit 1; }; \
	echo "watch nvidia-smi en job $$job; Ctrl-C para salir"; \
	srun --jobid=$$job --overlap --pty bash -lc 'hostname; if command -v watch >/dev/null 2>&1; then watch -n $(INTERVAL) nvidia-smi; else while true; do clear; date; nvidia-smi; sleep $(INTERVAL); done; fi'

nvtop: ## Abre nvtop dentro del nodo del job, si está instalado
	@job="$(JOB)"; \
	if [ "$$job" = "latest" ]; then \
	  job=$$(squeue -h -u $$USER -t RUNNING -o "%i" | tail -n1); \
	fi; \
	[ -n "$$job" ] || { echo "No encuentro ningún job RUNNING de $$USER. Usa make status o JOB=<id>."; exit 1; }; \
	srun --jobid=$$job --overlap --pty bash -lc 'if command -v nvtop >/dev/null 2>&1; then nvtop; else echo "nvtop no está instalado en este nodo; usa make gpu-watch."; exit 1; fi'

htop: ## Abre htop dentro del nodo del job, si está instalado; si no, cae a top
	@job="$(JOB)"; \
	if [ "$$job" = "latest" ]; then \
	  job=$$(squeue -h -u $$USER -t RUNNING -o "%i" | tail -n1); \
	fi; \
	[ -n "$$job" ] || { echo "No encuentro ningún job RUNNING de $$USER. Usa make status o JOB=<id>."; exit 1; }; \
	echo "Monitor CPU/RAM en job $$job; pulsa q o Ctrl-C para salir"; \
	srun --jobid=$$job --overlap --pty bash -lc 'hostname; if command -v htop >/dev/null 2>&1; then htop; else echo "htop no está instalado; usando top."; top; fi'
	