# Universal Dockerfile for FLUX API - Works with all deployment modes
FROM nvidia/cuda:12.4.0-cudnn9-runtime-ubuntu22.04

# Install Python 3.12 and essential system packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    lsof \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    https://github.com/nunchaku-tech/nunchaku/releases/download/v0.3.2/nunchaku-0.3.2+torch2.8-cp312-cp312-linux_x86_64.whl

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    opencv-python-headless \
    scipy \
    scikit-image \
    matplotlib \
    aiofiles \
    python-multipart \
    redis \
    celery \
    einops \
    xformers \
    peft \
    nvitop \
    gpustat \
    diffusers \
    safetensors \
    transformers \
    accelerate

# Copy all Python files and directories
COPY . .

# Create required directories
RUN mkdir -p \
    logs/multi_gpu \
    generated_images \
    uploads/lora_files \
    static

# Environment variables (can be overridden at runtime)
ENV PYTHONUNBUFFERED=1

# Expose common ports (can add more as needed)
EXPOSE 8000 8001 8002 23333-23340 8080

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "main_fp4:app", "--host", "0.0.0.0", "--port", "8000"]