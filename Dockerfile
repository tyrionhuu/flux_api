# Use a CUDA 12.4 + cuDNN runtime base that is available
# If this tag is unavailable in your registry mirror, try:
#   nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04
#   nvidia/cuda:12.4.0-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# System packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    ca-certificates \
    git \
    lsof \
    nginx \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Make python alias available
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

WORKDIR /app

# Use the PyTorch CUDA 12.8 index for GPU wheels (matches setup.sh)
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128
ENV PYTHONUNBUFFERED=1

# Copy the repository (respects .dockerignore)
COPY . .

# Install core libs following setup.sh (conda replaced by system Python)
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 \
 && python -m pip install --no-cache-dir \
    https://github.com/nunchaku-tech/nunchaku/releases/download/v0.3.2/nunchaku-0.3.2+torch2.8-cp312-cp312-linux_x86_64.whl \
 && python -m pip install --no-cache-dir \
    fastapi uvicorn pillow numpy opencv-python-headless scipy scikit-image matplotlib aiofiles python-multipart redis celery einops xformers peft nvitop gpustat \
 && python -m pip install --no-cache-dir \
    diffusers safetensors transformers huggingface-hub

# Create required runtime directories
RUN mkdir -p \
    logs/multi_gpu \
    generated_images \
    uploads/lora_files \
    static

# Expose ports for service instances and nginx LB
EXPOSE 8080 23333-23340

# Default entrypoint starts the multi-GPU launcher
ENTRYPOINT ["bash", "start_multi_gpu.sh"]
CMD ["-m", "fp4_sekai"]
