# Use NVIDIA CUDA 12.8 base image with Ubuntu 24.04
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# Set environment variables
ENV HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}
ENV PYTHONUNBUFFERED=1

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

# Install PyTorch with CUDA support from PyTorch index (stable version)
RUN pip3 install --no-cache-dir --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
RUN pip3 install --no-cache-dir https://github.com/nunchaku-tech/nunchaku/releases/download/v0.3.2/nunchaku-0.3.2+torch2.8-cp312-cp312-linux_x86_64.whl

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs generated_images uploads/lora_files cache/merged_loras cache/nunchaku_loras uploads/images

# Set proper permissions
RUN chmod +x start_flux_api.sh

# Expose the API port
EXPOSE 9001 9002 9000 9100

# Default command - run the application directly
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9100"]