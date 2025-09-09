# Docker Usage Guide

This guide explains how to run the FLUX API using Docker.

## Prerequisites

- Docker installed on your system
- NVIDIA Docker runtime (for GPU support)
- Hugging Face token for model access

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t img2img-api .
```

### 2. Run the Container

```bash
# Basic usage with Hugging Face token
docker run -d \
  --gpus all \
  -p 9200:9200 \
  -e HUGGINGFACE_HUB_TOKEN=your_token_here \
  --name img2img-api \
  img2img-api
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGINGFACE_HUB_TOKEN` | Required | Your Hugging Face token for model access |
| `CUDA_VISIBLE_DEVICES` | all | GPU IDs to use (comma-separated) |

### Command-Line Arguments

The Docker container runs with these default arguments:
- `--port 9200` - API port (can be overridden with custom CMD)
- `--cleanup` - Enable cleanup of existing processes
- `--no-frontend` - Disable frontend (backend-only mode) - frontend is enabled by default

### Port Mapping

The container exposes multiple ports:
- `9200` - Main API port (default)

### Direct Usage (Outside Docker)

You can also run the service directly using command-line arguments:

```bash
# Basic usage (port 9200, frontend enabled)
HUGGINGFACE_HUB_TOKEN=your_token /opt/conda/envs/img2img/bin/python start_api_service.py

# Custom port
HUGGINGFACE_HUB_TOKEN=your_token /opt/conda/envs/img2img/bin/python start_api_service.py --port 9001

# Backend-only mode
HUGGINGFACE_HUB_TOKEN=your_token /opt/conda/envs/img2img/bin/python start_api_service.py --no-frontend

# With cleanup enabled
HUGGINGFACE_HUB_TOKEN=your_token /opt/conda/envs/img2img/bin/python start_api_service.py --cleanup

# All options
HUGGINGFACE_HUB_TOKEN=your_token /opt/conda/envs/img2img/bin/python start_api_service.py --port 9001 --cleanup --no-frontend
```

## Usage Examples

### Basic Usage

```bash
# Run with default settings
docker run -d \
  --gpus all \
  -p 9200:9200 \
  -e HUGGINGFACE_HUB_TOKEN=your_token_here \
  --name img2img-api \
  img2img-api
```

### Custom Port and GPU

```bash
# Use specific GPU and port
docker run -d \
  --gpus '"device=0"' \
  -p 9002:9002 \
  -e HUGGINGFACE_HUB_TOKEN=your_token_here \
  -e FP4_API_PORT=9002 \
  -e CUDA_VISIBLE_DEVICES=0 \
  --name img2img-api-gpu0 \
  img2img-api
```

### Multiple GPUs

```bash
# Use multiple GPUs
docker run -d \
  --gpus all \
  -p 9200:9200 \
  -e HUGGINGFACE_HUB_TOKEN=your_token_here \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  --name img2img-api-multi-gpu \
  img2img-api
```

### With Volume Mounts

```bash
# Mount directories for persistent storage
docker run -d \
  --gpus all \
  -p 9200:9200 \
  -e HUGGINGFACE_HUB_TOKEN=your_token_here \
  -v $(pwd)/generated_images:/app/generated_images \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/uploads:/app/uploads \
  --name img2img-api \
  img2img-api
```

### Custom Configuration

To override the default arguments, use a custom CMD:

```bash
# Custom port and backend-only mode
docker run -d \
  --gpus all \
  -e HUGGINGFACE_HUB_TOKEN=your_token_here \
  -p 9001:9001 \
  --name img2img-api \
  img2img-api \
  /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate img2img && /opt/conda/envs/img2img/bin/python start_api_service.py --port 9001 --cleanup --no-frontend"
```

## Container Management

### View Logs

```bash
# View container logs
docker logs img2img-api

# Follow logs in real-time
docker logs -f img2img-api
```

### Stop Container

```bash
# Stop the container
docker stop img2img-api

# Stop and remove the container
docker stop img2img-api && docker rm img2img-api
```

### Restart Container

```bash
# Restart the container
docker restart img2img-api
```

### Access Container Shell

```bash
# Access the container shell
docker exec -it img2img-api /bin/bash
```

## Complete Command Reference

### Comprehensive Docker Command

```bash
docker run -d \
  --gpus all \
  -p 9200:9200 \
  -e HUGGINGFACE_HUB_TOKEN=your_token_here \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e FP4_API_PORT=9200 \
  -e CUDA_HOME=/usr/local/cuda \
  -e PYTHONUNBUFFERED=1 \
  -e CONDA_DEFAULT_ENV=img2img \
  -v /path/to/local/cache:/root/.cache \
  -v /path/to/local/logs:/app/logs \
  -v /path/to/local/generated:/app/generated_images \
  -v /path/to/local/uploads:/app/uploads \
  --name img2img-api \
  --restart unless-stopped \
  --memory=32g \
  --memory-swap=64g \
  --shm-size=16g \
  img2img-api \
  --port 9200 \
  --cleanup \
  --no-frontend
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGINGFACE_HUB_TOKEN` | **Required** | Your Hugging Face token for model access |
| `CUDA_VISIBLE_DEVICES` | all | GPU IDs to use (comma-separated: 0,1,2) |
| `FP4_API_PORT` | 9001 | API port (overridden by --port argument) |
| `CUDA_HOME` | /usr/local/cuda | CUDA installation path |
| `PYTHONUNBUFFERED` | 1 | Python output buffering |
| `CONDA_DEFAULT_ENV` | img2img | Conda environment name |

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--port` | 9200 | API port number |
| `--cleanup` | False | Enable cleanup of existing processes |
| `--no-frontend` | False | Disable frontend (backend-only mode) |