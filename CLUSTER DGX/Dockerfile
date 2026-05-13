# Imagen base con PyTorch + CUDA
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Compilador de C/C++ que necesita torch.compile / Inductor
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cacheado del layer de dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Scripts del proyecto
COPY train_diffusion_mi_u_net.py .
COPY generar_galaxia_mi_u_net.py .
COPY mi_u_net.py .
COPY dataset.py .

CMD ["python", "train_diffusion_mi_u_net.py"]
