# Use NVIDIA CUDA 12.8 base image with Ubuntu 24.04
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

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

# Set Hugging Face cache directory
ENV HF_HOME=/data/hf_cache
ENV HUGGINGFACE_HUB_CACHE=/data/hf_cache

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

# Install PyTorch with CUDA support from PyTorch index (stable version)
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

# Create necessary directories
RUN mkdir -p logs generated_images uploads/lora_files cache/merged_loras cache/nunchaku_loras /data/hf_cache

# Set proper permissions
RUN chmod +x start_api.sh docker-start.sh

# Expose the API port
EXPOSE 8200 9000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD conda run -n txt2img curl -f http://localhost:${FP4_API_PORT:-9001}/health || exit 1

# Default command - use docker-start.sh for container execution
# Use exec form to ensure proper signal handling
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate txt2img && exec ./docker-start.sh"]