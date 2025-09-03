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
RUN conda run -n img2img pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
RUN conda run -n img2img pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v0.3.2/nunchaku-0.3.2+torch2.8-cp312-cp312-linux_x86_64.whl

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "img2img", "/bin/bash", "-c"]

# Activate the environment
RUN echo "conda activate img2img" >> ~/.bashrc
ENV CONDA_DEFAULT_ENV=img2img
ENV PATH=/opt/conda/envs/img2img/bin:$PATH

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs generated_images uploads/lora_files cache/merged_loras cache/nunchaku_loras

# Set proper permissions
RUN chmod +x start_flux_api.sh docker-start.sh

# Expose the API port
EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD conda run -n img2img curl -f http://localhost:9000/health || exit 1

# Default command - use the startup script with conda environment
CMD ["conda", "run", "-n", "img2img", "./docker-start.sh"]
