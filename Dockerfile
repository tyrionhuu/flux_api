# Use NVIDIA CUDA 12.8 base image with Ubuntu 24.04
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# Production labels
LABEL maintainer="Diffusion API Team"
LABEL version="txt2img-backend-v1"
LABEL description="Diffusion Text-to-Image API with LoRA Fusion - Backend Only"
LABEL service="txt2img-api"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Production environment variables
ENV SERVICE_NAME="txt2img-api"
ENV SERVICE_VERSION="txt2img-backend-v1"
ENV PORT=8000
ENV HOST=0.0.0.0
ENV MODEL_TYPE=flux
ENV FUSION_MODE=false
ENV LORA_NAME=""
ENV LORA_WEIGHT=1.0
ENV LORAS_CONFIG=""
ENV LOG_LEVEL=INFO
ENV MAX_WORKERS=1

# Set Hugging Face cache directory
ENV HF_HOME=/data/hf_cache
ENV HUGGINGFACE_HUB_CACHE=/data/hf_cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libgl1 \
    libsm6 \
    libxrender1 \
    mercurial \
    subversion \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean --all --yes && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Set working directory
WORKDIR /app

# Copy environment file first for better caching
COPY environment.yml .

# Accept conda Terms of Service and configure channels
RUN conda config --set always_yes true && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda config --add channels conda-forge && \
    conda config --add channels pytorch && \
    conda config --add channels nvidia && \
    conda config --set channel_priority strict

# Create conda environment from environment.yml
RUN conda env create -f environment.yml

# Install PyTorch with CUDA support from PyTorch index
RUN conda run -n txt2img pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
RUN conda run -n txt2img pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.0/nunchaku-1.0.0+torch2.8-cp312-cp312-linux_x86_64.whl

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "txt2img", "/bin/bash", "-c"]

# Activate the environment
RUN echo "conda activate txt2img" >> ~/.bashrc
ENV CONDA_DEFAULT_ENV=txt2img
ENV PATH=/opt/conda/envs/txt2img/bin:$PATH

# Copy application code
COPY . .

# Create necessary directories (no frontend directories)
RUN mkdir -p logs generated_images uploads/lora_files cache/merged_loras cache/nunchaku_loras /data/hf_cache

# Expose the API port
EXPOSE 8000

# Enhanced health check for production
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD conda run -n txt2img curl -f http://localhost:${PORT:-8000}/health || exit 1

# Production-ready entry point (direct to Python script)
ENTRYPOINT ["/opt/conda/envs/txt2img/bin/python", "main.py"]
